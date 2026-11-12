//! Shared page plumbing: the per-page context used by the layout template, flash messages
//! (a short-lived cookie set before a redirect), rendering helpers and themed error pages.

use askama::Template;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{Html, IntoResponse, Redirect, Response};

use crate::SharedState;
use crate::config::Theme;
use crate::error::ErrorInfo;

pub const FLASH_COOKIE: &str = "las_flash";

#[derive(Clone, Debug)]
pub struct Flash {
    /// `ok`, `error` or `info`.
    pub kind: String,
    pub message: String,
}

/// Values every page template needs (navigation, theme, notifications).
#[derive(Clone, Debug)]
pub struct PageCtx {
    pub title: String,
    pub nav: &'static str,
    pub theme: &'static str,
    pub theme_label: &'static str,
    pub dark: bool,
    pub flash: Option<Flash>,
    pub active_downloads: usize,
    pub generating: usize,
    pub version: &'static str,
    /// Content hash of the stylesheets, for cache-busting their URLs.
    pub css_rev: &'static str,
    /// Auto-refresh interval (seconds) for pages showing live progress.
    pub refresh: Option<u32>,
    pub network_exposed: bool,
    pub themes: Vec<ThemeOption>,
    pub current_path: String,
}

#[derive(Clone, Debug)]
pub struct ThemeOption {
    pub slug: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub dark: bool,
    pub active: bool,
}

impl PageCtx {
    pub fn new(state: &SharedState, headers: &HeaderMap, title: &str, nav: &'static str) -> Self {
        let settings = state.settings();
        let theme = settings.ui.theme;
        Self {
            title: title.to_string(),
            nav,
            theme: theme.slug(),
            theme_label: theme.label(),
            dark: theme.is_dark(),
            flash: read_flash(headers),
            active_downloads: state.downloads.active_count(),
            generating: state.chat.active_count() + state.jobs.active_count(),
            version: crate::APP_VERSION,
            css_rev: crate::web::assets::stylesheet_rev(),
            refresh: None,
            network_exposed: state.server.addr().map(|a| !a.ip().is_loopback()).unwrap_or(false),
            themes: Theme::ALL
                .iter()
                .map(|t| ThemeOption {
                    slug: t.slug(),
                    label: t.label(),
                    description: t.description(),
                    dark: t.is_dark(),
                    active: *t == theme,
                })
                .collect(),
            current_path: String::new(),
        }
    }

    pub fn with_refresh(mut self, secs: Option<u32>) -> Self {
        self.refresh = secs;
        self
    }

    pub fn at(mut self, path: &str) -> Self {
        self.current_path = path.to_string();
        self
    }

    pub fn is(&self, nav: &str) -> bool {
        self.nav == nav
    }
}

fn read_flash(headers: &HeaderMap) -> Option<Flash> {
    let raw = headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .map(str::trim)
        .find_map(|kv| kv.strip_prefix(FLASH_COOKIE).and_then(|r| r.strip_prefix('=')))?;
    let decoded = percent_encoding::percent_decode_str(raw).decode_utf8().ok()?;
    let (kind, message) = decoded.split_once('|')?;
    let kind = match kind {
        "ok" | "error" | "info" => kind,
        _ => "info",
    };
    if message.is_empty() {
        return None;
    }
    Some(Flash { kind: kind.to_string(), message: crate::util::truncate_chars(message, 600) })
}

fn flash_cookie(kind: &str, message: &str) -> Option<HeaderValue> {
    let msg = crate::util::truncate_chars(message, 600);
    let value =
        percent_encoding::utf8_percent_encode(&format!("{kind}|{msg}"), percent_encoding::NON_ALPHANUMERIC).to_string();
    HeaderValue::from_str(&format!("{FLASH_COOKIE}={value}; Path=/; Max-Age=60; SameSite=Strict; HttpOnly")).ok()
}

/// Redirects (303) and shows a message on the next page.
pub fn redirect_flash(to: &str, kind: &str, message: &str) -> Response {
    let mut r = Redirect::to(to).into_response();
    if let Some(v) = flash_cookie(kind, message) {
        r.headers_mut().append(header::SET_COOKIE, v);
    }
    r
}

pub fn redirect(to: &str) -> Response {
    Redirect::to(to).into_response()
}

/// Expires the flash cookie once a page has shown its message, so it isn't shown again.
pub fn clear_flash(ctx: &PageCtx, r: &mut Response) {
    if ctx.flash.is_some() {
        r.headers_mut().append(
            header::SET_COOKIE,
            HeaderValue::from_static("las_flash=; Path=/; Max-Age=0; SameSite=Strict; HttpOnly"),
        );
    }
}

/// Renders a template; clears the flash cookie when the page displayed it.
pub fn render<T: Template>(ctx: &PageCtx, t: &T) -> Response {
    match t.render() {
        Ok(html) => {
            let mut r = Html(html).into_response();
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
            clear_flash(ctx, &mut r);
            r
        }
        Err(e) => {
            tracing::error!("template error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Template error: {e}")).into_response()
        }
    }
}

#[derive(Template)]
#[template(path = "error.html")]
pub struct ErrorPage {
    pub ctx: PageCtx,
    pub status: u16,
    pub heading: String,
    pub message: String,
    pub back: String,
}

/// Replaces bare error responses with a themed page (HTML) or JSON (`/api`, `/v1`).
pub async fn error_pages(State(state): State<SharedState>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    let headers = req.headers().clone();
    let resp = next.run(req).await;
    let Some(info) = resp.extensions().get::<ErrorInfo>().cloned() else {
        return resp;
    };
    if path.starts_with("/api/") || path.starts_with("/v1/") {
        let body = serde_json::json!({ "error": { "message": info.message, "status": info.status.as_u16() } });
        return (info.status, axum::Json(body)).into_response();
    }
    let back = headers
        .get(header::REFERER)
        .and_then(|r| r.to_str().ok())
        .and_then(|r| r.split_once("://").map(|(_, rest)| rest))
        .and_then(|rest| rest.find('/').map(|i| rest[i..].to_string()))
        .filter(|p| p.starts_with('/') && !p.starts_with("//") && *p != path)
        .unwrap_or_else(|| "/".to_string());
    let ctx = PageCtx::new(&state, &headers, &info.title, "");
    let page = ErrorPage {
        ctx: ctx.clone(),
        status: info.status.as_u16(),
        heading: info.title.clone(),
        message: info.message.clone(),
        back,
    };
    let mut r = render(&ctx, &page);
    *r.status_mut() = info.status;
    r
}

/// Reads an optional string field from a form, trimming whitespace.
pub fn trimmed(s: &Option<String>) -> String {
    s.as_deref().unwrap_or("").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_round_trip() {
        let v = flash_cookie("ok", "Saved — all good ✓").unwrap();
        let cookie = v.to_str().unwrap().split(';').next().unwrap().to_string();
        let mut h = HeaderMap::new();
        h.insert(header::COOKIE, HeaderValue::from_str(&cookie).unwrap());
        let f = read_flash(&h).unwrap();
        assert_eq!(f.kind, "ok");
        assert_eq!(f.message, "Saved — all good ✓");
    }

    #[test]
    fn bad_flash_kinds_are_normalised() {
        let mut h = HeaderMap::new();
        h.insert(header::COOKIE, HeaderValue::from_static("las_flash=evil%7Chello"));
        assert_eq!(read_flash(&h).unwrap().kind, "info");
    }
}
