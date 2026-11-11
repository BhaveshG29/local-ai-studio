//! Request guard: DNS-rebinding protection (Host check), cross-site request blocking (Origin
//! check), access control for LAN clients when network access is enabled, and security
//! headers (a CSP that forbids all scripts).

use std::net::{IpAddr, SocketAddr};

use axum::Form;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{Html, IntoResponse, Redirect, Response};
use serde::Deserialize;

use crate::SharedState;

pub const ACCESS_COOKIE: &str = "las_access";

pub const CSP: &str = "default-src 'self'; script-src 'none'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self'; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; form-action 'self'; base-uri 'none'; object-src 'none'";

fn host_without_port(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    match host.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) => h,
        _ => host,
    }
}

/// Whether a Host header value names this machine's loopback interface.
pub fn is_loopback_host(host: &str) -> bool {
    let h = host_without_port(host);
    h.eq_ignore_ascii_case("localhost") || h.parse::<IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(false)
}

fn is_ip_literal(host: &str) -> bool {
    host_without_port(host).parse::<IpAddr>().is_ok()
}

/// Decides whether a Host header is acceptable for a request from `peer_loopback`.
pub fn host_allowed(host: Option<&str>, peer_loopback: bool, network_enabled: bool) -> bool {
    let Some(host) = host else {
        // Non-browser clients may omit Host on HTTP/1.0; browsers always send it.
        return true;
    };
    if is_loopback_host(host) {
        return true;
    }
    if !network_enabled {
        return false;
    }
    // In network mode, LAN clients (authenticated by token) may use any name; local browsers
    // must use an IP literal so a rebinding domain can't reach the app.
    !peer_loopback || is_ip_literal(host)
}

/// Whether an unsafe (state-changing) request comes from our own pages.
pub fn same_origin_ok(headers: &HeaderMap) -> bool {
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok());
    if let Some(origin) = headers.get(header::ORIGIN).and_then(|o| o.to_str().ok()) {
        if origin == "null" {
            return false;
        }
        let origin_host = origin.strip_prefix("http://").or_else(|| origin.strip_prefix("https://")).unwrap_or(origin);
        return match host {
            Some(h) => origin_host.eq_ignore_ascii_case(h),
            None => false,
        };
    }
    if let Some(site) = headers.get("sec-fetch-site").and_then(|s| s.to_str().ok()) {
        return matches!(site, "same-origin" | "none");
    }
    if let Some(referer) = headers.get(header::REFERER).and_then(|r| r.to_str().ok()) {
        let rest = referer.strip_prefix("http://").or_else(|| referer.strip_prefix("https://")).unwrap_or("");
        let rhost = rest.split('/').next().unwrap_or("");
        return host.map(|h| rhost.eq_ignore_ascii_case(h)).unwrap_or(false);
    }
    // No browser headers at all: a script or CLI tool (curl). Browsers always send Origin
    // or Sec-Fetch-Site on cross-site POSTs, so this cannot be a CSRF from a web page.
    true
}

fn cookie_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .map(str::trim)
        .find_map(|kv| kv.strip_prefix(name).and_then(|r| r.strip_prefix('=')))
}

fn has_access(headers: &HeaderMap, token: &str) -> bool {
    if let Some(c) = cookie_value(headers, ACCESS_COOKIE)
        && crate::util::secure_eq(c, token)
    {
        return true;
    }
    headers
        .get(header::AUTHORIZATION)
        .and_then(|a| a.to_str().ok())
        .and_then(|a| a.strip_prefix("Bearer "))
        .map(|t| crate::util::secure_eq(t.trim(), token))
        .unwrap_or(false)
}

fn blocked(status: StatusCode, msg: &str) -> Response {
    let mut r = (status, msg.to_string()).into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    r
}

pub fn apply_security_headers(headers: &mut HeaderMap) {
    // Handlers may set a stricter policy (e.g. sandboxed user files); keep it.
    headers.entry("content-security-policy").or_insert(HeaderValue::from_static(CSP));
    headers.insert("x-content-type-options", HeaderValue::from_static("nosniff"));
    headers.insert("referrer-policy", HeaderValue::from_static("same-origin"));
    headers.insert("x-frame-options", HeaderValue::from_static("SAMEORIGIN"));
    headers.insert("cross-origin-opener-policy", HeaderValue::from_static("same-origin"));
    headers.insert("cross-origin-resource-policy", HeaderValue::from_static("same-origin"));
    headers.insert(
        "permissions-policy",
        HeaderValue::from_static("camera=(), microphone=(), geolocation=(), payment=(), usb=()"),
    );
}

/// Request extension: whether the client is on this computer (loopback).
#[derive(Clone, Copy, Debug)]
pub struct ClientLocal(pub bool);

/// The middleware applied to every request.
pub async fn guard(State(state): State<SharedState>, mut req: Request, next: Next) -> Response {
    let peer: Option<SocketAddr> = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let peer_loopback = peer.map(|p| p.ip().is_loopback()).unwrap_or(true);
    // Binding a non-loopback address requires an explicit opt-in (setting or CLI flag), so
    // "network mode" is simply whether the server is actually reachable from the network.
    let network_enabled = state.server.addr().map(|a| !a.ip().is_loopback()).unwrap_or(false);
    let headers = req.headers();
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok());

    if !host_allowed(host, peer_loopback, network_enabled) {
        tracing::warn!(?host, ?peer, "blocked request with unexpected Host header");
        return blocked(
            StatusCode::MISDIRECTED_REQUEST,
            "Blocked: this request used an unexpected host name. Open Local AI Studio at http://127.0.0.1 or http://localhost instead.",
        );
    }
    if !peer_loopback {
        if !network_enabled {
            return blocked(StatusCode::FORBIDDEN, "Local AI Studio only accepts connections from this computer.");
        }
        let path = req.uri().path();
        let open = path == "/unlock" || path.starts_with("/assets/") || path == "/favicon.ico" || path == "/api/health";
        if !open && !has_access(headers, &state.server.access_token) {
            if path.starts_with("/api/") || path.starts_with("/v1/") {
                return blocked(
                    StatusCode::UNAUTHORIZED,
                    "Missing or invalid access token (send Authorization: Bearer <token>).",
                );
            }
            return Redirect::to("/unlock").into_response();
        }
    }
    let unsafe_method = !matches!(*req.method(), Method::GET | Method::HEAD | Method::OPTIONS);
    if unsafe_method && !same_origin_ok(headers) {
        tracing::warn!(path = %req.uri().path(), "blocked cross-site request");
        return blocked(
            StatusCode::FORBIDDEN,
            "Blocked: this action was requested by another website. If you clicked a button in Local AI Studio, reload the page and try again.",
        );
    }
    req.extensions_mut().insert(ClientLocal(peer_loopback));
    let mut resp = next.run(req).await;
    apply_security_headers(resp.headers_mut());
    resp
}

/// Page where LAN clients enter the access token.
pub async fn unlock_page() -> Html<String> {
    Html(unlock_html(None))
}

fn unlock_html(error: Option<&str>) -> String {
    format!(
        "<!doctype html><html lang=\"en\" data-theme=\"midnight\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Unlock · {app}</title><link rel=\"stylesheet\" href=\"/assets/app.css\"><link rel=\"stylesheet\" href=\"/assets/themes.css\"></head><body class=\"unlock\"><main class=\"unlock-card card\"><h1>Unlock {app}</h1><p>This computer's Local AI Studio is shared on the network. Enter the access token shown in <b>Settings → Server</b> on the host computer.</p>{err}<form method=\"post\" action=\"/unlock\" class=\"stack\"><label class=\"field\"><span>Access token</span><input class=\"input\" type=\"password\" name=\"token\" autocomplete=\"off\" required autofocus></label><button class=\"btn btn-primary\" type=\"submit\">Unlock</button></form></main></body></html>",
        app = crate::APP_NAME,
        err = error
            .map(|e| format!("<p class=\"notice notice-error\">{}</p>", crate::util::html_escape(e)))
            .unwrap_or_default()
    )
}

#[derive(Deserialize)]
pub struct UnlockForm {
    #[serde(default)]
    token: String,
}

pub async fn unlock(State(state): State<SharedState>, Form(f): Form<UnlockForm>) -> Response {
    if crate::util::secure_eq(f.token.trim(), &state.server.access_token) {
        let mut r = Redirect::to("/").into_response();
        let cookie = format!(
            "{ACCESS_COOKIE}={}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000",
            state.server.access_token
        );
        if let Ok(v) = HeaderValue::from_str(&cookie) {
            r.headers_mut().insert(header::SET_COOKIE, v);
        }
        r
    } else {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        (StatusCode::UNAUTHORIZED, Html(unlock_html(Some("That token is not correct.")))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_are_checked() {
        assert!(host_allowed(Some("127.0.0.1:8765"), true, false));
        assert!(host_allowed(Some("localhost:8765"), true, false));
        assert!(host_allowed(Some("[::1]:8765"), true, false));
        assert!(!host_allowed(Some("evil.example:8765"), true, false));
        assert!(!host_allowed(Some("192.168.1.5:8765"), true, false));
        // network mode
        assert!(host_allowed(Some("192.168.1.5:8765"), false, true));
        assert!(host_allowed(Some("mypc.local:8765"), false, true));
        assert!(!host_allowed(Some("rebind.example:8765"), true, true));
        assert!(host_allowed(None, true, false));
    }

    fn hm(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn origins_are_checked() {
        assert!(same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("origin", "http://127.0.0.1:8765")])));
        assert!(!same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("origin", "http://evil.example")])));
        assert!(!same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("origin", "http://127.0.0.1:9999")])));
        assert!(!same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("origin", "null")])));
        assert!(!same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("sec-fetch-site", "cross-site")])));
        assert!(!same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("sec-fetch-site", "same-site")])));
        assert!(same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("sec-fetch-site", "same-origin")])));
        assert!(same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("referer", "http://127.0.0.1:8765/chat")])));
        assert!(!same_origin_ok(&hm(&[("host", "127.0.0.1:8765"), ("referer", "http://evil/x")])));
        assert!(same_origin_ok(&hm(&[("host", "127.0.0.1:8765")])));
    }

    #[test]
    fn access_tokens_are_checked() {
        let t = "abc123";
        assert!(has_access(&hm(&[("cookie", "x=1; las_access=abc123")]), t));
        assert!(has_access(&hm(&[("authorization", "Bearer abc123")]), t));
        assert!(!has_access(&hm(&[("cookie", "las_access=nope")]), t));
        assert!(!has_access(&hm(&[]), t));
    }
}
