//! HTTP server: HTML pages (server-rendered, no JavaScript), static assets, uploaded and
//! generated files, and a small JSON API (including an OpenAI-compatible chat endpoint).

pub mod api;
pub mod assets;
pub mod page;
pub mod pages;
pub mod security;
pub mod streaming;

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::trace::TraceLayer;

use crate::SharedState;

/// Builds the complete router.
pub fn router(state: SharedState) -> Router {
    use pages::*;
    let uploads_limit = DefaultBodyLimit::max(crate::media::MAX_UPLOAD_BYTES + 1024 * 1024);
    Router::new()
        .route("/", get(home::page))
        // Discover
        .route("/discover", get(discover::page))
        .route("/discover/index/refresh", post(discover::refresh_index))
        .route("/discover/hf/{owner}/{name}", get(discover::hf_model))
        .route("/discover/hf/{owner}/{name}/download", post(discover::hf_download))
        .route("/discover/ollama/pull", post(discover::ollama_pull))
        .route("/discover/ollama/{*name}", get(discover::ollama_model))
        // Downloads
        .route("/downloads", get(downloads::page))
        .route("/downloads/clear", post(downloads::clear))
        .route("/downloads/{id}/{action}", post(downloads::action))
        // Library
        .route("/library", get(library::page))
        .route("/library/import", post(library::import))
        .route("/library/model/{*model_ref}", get(library::detail))
        .route("/library/action", post(library::action))
        // Chat
        .route("/chat", get(chat::index))
        .route("/chat/new", post(chat::create).layer(uploads_limit))
        .route("/chat/import", post(chat::import).layer(DefaultBodyLimit::max(20 * 1024 * 1024)))
        .route("/chat/{id}", get(chat::view))
        .route("/chat/{id}/send", post(chat::send).layer(uploads_limit))
        .route("/chat/{id}/stop", post(chat::stop))
        .route("/chat/{id}/regenerate", post(chat::regenerate))
        .route("/chat/{id}/continue", post(chat::continue_answer))
        .route("/chat/{id}/rename", post(chat::rename))
        .route("/chat/{id}/delete", post(chat::delete))
        .route("/chat/{id}/settings", post(chat::settings))
        .route("/chat/{id}/export", get(chat::export))
        .route("/chat/{id}/messages/{mid}/edit", post(chat::edit_message))
        .route("/chat/{id}/messages/{mid}/raw", get(chat::raw_message))
        // Image generation
        .route("/create", get(create::page).post(create::start))
        .route("/create/{id}/progress", get(create::progress))
        .route("/create/{id}/cancel", post(create::cancel))
        .route("/create/{id}/delete", post(create::delete))
        .route("/create/{id}/save", post(create::save_to_folder))
        // Speech to text
        .route("/transcribe", get(transcribe::page).post(transcribe::start).layer(uploads_limit))
        .route("/transcribe/{id}", get(transcribe::detail))
        .route("/transcribe/{id}/cancel", post(transcribe::cancel))
        .route("/transcribe/{id}/delete", post(transcribe::delete))
        .route("/transcribe/{id}/export", get(transcribe::export))
        .route("/transcribe/{id}/to-chat", post(transcribe::to_chat))
        // System & settings
        .route("/system", get(system::page))
        .route("/system/unload", post(system::unload))
        .route("/settings", get(settings::page))
        .route("/settings/theme", post(settings::theme))
        .route("/settings/save/{section}", post(settings::save))
        .route("/settings/hf-token", post(settings::hf_token))
        .route("/settings/data/{action}", post(settings::data_action))
        .route("/settings/shutdown", post(settings::shutdown))
        .route("/about/licenses", get(settings::licenses))
        .route("/unlock", get(security::unlock_page).post(security::unlock))
        // Files and assets
        .route("/files/{kind}/{*path}", get(files::serve))
        .route("/assets/{*path}", get(assets::serve))
        .route("/favicon.ico", get(assets::favicon))
        // JSON API
        .route("/api/health", get(api::health))
        .route("/api/v1/system", get(api::system))
        .route("/api/v1/models", get(api::models))
        .route("/api/v1/downloads", get(api::downloads))
        .route("/api/v1/chat", post(api::chat))
        .route("/v1/models", get(api::openai_models))
        .route("/v1/chat/completions", post(api::openai_chat))
        .fallback(pages::not_found)
        .layer(axum::middleware::from_fn_with_state(state.clone(), page::error_pages))
        .layer(axum::middleware::from_fn_with_state(state.clone(), security::guard))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .layer(CatchPanicLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Outcome of trying to bind the configured address.
pub enum BindOutcome {
    Bound(tokio::net::TcpListener),
    /// Another Local AI Studio is already serving at this URL.
    AlreadyRunning(String),
}

/// Binds `host:port`. If the port is taken by another Local AI Studio, reports it; if it is
/// taken by something else and the port was not explicitly requested, tries the next ports.
pub async fn bind(host: &str, port: u16, port_explicit: bool) -> anyhow::Result<BindOutcome> {
    let ip: IpAddr = if host.eq_ignore_ascii_case("localhost") {
        IpAddr::from([127, 0, 0, 1])
    } else {
        host.trim_start_matches('[')
            .trim_end_matches(']')
            .parse()
            .map_err(|_| anyhow::anyhow!("\"{host}\" is not a valid IP address to listen on"))?
    };
    let attempts = if port_explicit { 1 } else { 10 };
    let mut last_err = None;
    for offset in 0..attempts {
        let p = port.saturating_add(offset);
        let addr = SocketAddr::new(ip, p);
        match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => {
                if offset > 0 {
                    tracing::warn!("port {port} is in use by another program; using {p} instead");
                }
                return Ok(BindOutcome::Bound(l));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                if offset == 0
                    && let Some(url) = probe_existing(ip, p).await
                {
                    return Ok(BindOutcome::AlreadyRunning(url));
                }
                last_err = Some(e);
            }
            Err(e) => {
                return Err(anyhow::anyhow!("could not listen on {addr}: {e}"));
            }
        }
    }
    Err(anyhow::anyhow!(
        "port {port}{} is already used by another program ({}). Choose another port with --port or in Settings → Server.",
        if attempts > 1 { format!(" and the next {} ports are", attempts - 1) } else { " is".into() },
        last_err.map(|e| e.to_string()).unwrap_or_default()
    ))
}

/// Checks whether the program on a busy port is Local AI Studio.
async fn probe_existing(ip: IpAddr, port: u16) -> Option<String> {
    let probe_ip = if ip.is_unspecified() { IpAddr::from([127, 0, 0, 1]) } else { ip };
    let url = format!("http://{}/api/health", SocketAddr::new(probe_ip, port));
    let client = reqwest::Client::builder().timeout(Duration::from_secs(2)).no_proxy().build().ok()?;
    let v: serde_json::Value = client.get(&url).send().await.ok()?.json().await.ok()?;
    (v.get("app").and_then(|a| a.as_str()) == Some(crate::APP_NAME))
        .then(|| format!("http://{}", SocketAddr::new(probe_ip, port)))
}

/// Serves until Ctrl+C / SIGTERM / the in-app "Stop server" button.
pub async fn serve(state: SharedState, listener: tokio::net::TcpListener) -> anyhow::Result<()> {
    let addr = listener.local_addr()?;
    *state.server.addr.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(addr);
    let app = router(state.clone()).into_make_service_with_connect_info::<SocketAddr>();
    let shutdown = state.shutdown.clone();
    let st = state.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = shutdown_signal() => {},
                _ = shutdown.cancelled() => {},
            }
            tracing::info!("shutting down…");
            // Ends streaming responses and stops background work so the server can drain.
            st.shutdown_gracefully().await;
        })
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
