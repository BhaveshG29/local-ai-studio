//! Streaming HTML without JavaScript.
//!
//! The chat page is rendered by the template with a marker where the live answer goes. The
//! part before the marker is sent immediately; then the answer is streamed as escaped text
//! while it is generated (browsers render partial HTML progressively); finally the formatted
//! message and a small `<style>` element that switches the composer back from "Stop" to
//! "Send" are sent, followed by the rest of the page.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use tokio_util::sync::CancellationToken;

use crate::inference::chat::LiveGeneration;
use crate::util::html_escape;

pub const LIVE_MARKER: &str = "<!--LAS-LIVE-STREAM-->";

/// Builds the final HTML for a finished message (provided by the caller, e.g. a rendered
/// message partial).
pub type FinalHtml = Box<dyn FnOnce(&crate::inference::chat::LiveSnapshot) -> String + Send>;

pub fn stream_page(
    page_html: String,
    live: Arc<LiveGeneration>,
    model_label: String,
    final_html: FinalHtml,
    shutdown: CancellationToken,
) -> Response {
    let (head, tail) = match page_html.split_once(LIVE_MARKER) {
        Some((h, t)) => (h.to_string(), t.to_string()),
        None => (page_html, String::new()),
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(256);
    let mid = live.message_id.clone();
    tokio::spawn(async move {
        let send = |s: String| {
            let tx = tx.clone();
            async move { tx.send(Ok(Bytes::from(s))).await.is_ok() }
        };
        if !send(head).await {
            return;
        }
        let opening = format!(
            "<div class=\"live-wrap\" id=\"live-{mid}\"><article class=\"msg msg-assistant is-live\" aria-live=\"polite\"><header class=\"msg-head\"><span class=\"msg-who\">{}</span><span class=\"msg-status\"><span class=\"spinner\" aria-hidden=\"true\"></span>Writing…</span></header><div class=\"msg-body\"><div class=\"live-text\">",
            html_escape(&model_label)
        );
        if !send(opening).await {
            return;
        }
        let mut versions = live.subscribe();
        let mut sent = 0usize;
        let mut last_phase: Option<String> = None;
        let mut phases_hidden = false;
        loop {
            // The first snapshot includes text generated before this viewer connected
            // (page reload, or "Continue"), so nothing is lost.
            let snap = live.snapshot();
            if snap.phase != last_phase {
                if let Some(p) = snap.phase.as_deref().filter(|p| !p.is_empty())
                    && !send(format!("<p class=\"phase\">{}</p>", html_escape(p))).await
                {
                    return;
                }
                last_phase = snap.phase.clone();
            }
            if snap.text.len() > sent {
                if !phases_hidden {
                    phases_hidden = true;
                    if !send(format!("<style>#live-{mid} .phase{{display:none}}</style>")).await {
                        return;
                    }
                }
                let delta = snap.text.get(sent..).unwrap_or("").to_string();
                sent = snap.text.len();
                if !send(html_escape(&delta)).await {
                    return;
                }
            }
            if snap.finished.is_some() {
                let fin = final_html(&snap);
                let closing = format!(
                    "</div></div></article><style>#live-{mid}>.is-live{{display:none}}.composer.is-generating .btn-stop,.composer.is-generating .gen-note{{display:none}}.composer.is-generating .btn-send{{display:inline-flex}}</style>{fin}</div>"
                );
                let _ = send(closing).await;
                break;
            }
            tokio::select! {
                changed = versions.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
                _ = shutdown.cancelled() => {
                    let _ = send("</div></div></article><p class=\"notice notice-info\">The app is shutting down; the answer so far has been saved.</p></div>".to_string()).await;
                    break;
                }
                _ = tx.closed() => return,
            }
        }
        let _ = send(tail).await;
    });
    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    let mut r = body.into_response();
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    r
}
