//! Settings: appearance, storage, Hugging Face token, Ollama, inference, generation
//! defaults, server, logging, media, cache & data management, about.

use std::collections::HashMap;
use std::path::PathBuf;

use askama::Template;
use axum::Form;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, header};
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;

use crate::SharedState;
use crate::config::{Acceleration, BackendPreference, LOG_LEVELS, Settings, Theme};
use crate::error::AppError;
use crate::secrets::TokenSource;
use crate::util::human_bytes;
use crate::web::page::{PageCtx, redirect_flash, render};

pub struct SelectOpt {
    pub value: String,
    pub label: String,
    pub selected: bool,
}

fn select<T: PartialEq + Copy>(list: &[(T, &str, &str)], current: T) -> Vec<SelectOpt> {
    list.iter()
        .map(|(v, slug, label)| SelectOpt {
            value: (*slug).to_string(),
            label: (*label).to_string(),
            selected: *v == current,
        })
        .collect()
}

#[derive(Template)]
#[template(path = "settings.html")]
pub struct SettingsPage {
    pub ctx: PageCtx,
    pub s: Settings,
    pub models_dir: String,
    pub partial_dir: String,
    pub outputs_dir: String,
    pub models_size: String,
    pub free_space: String,
    pub token_label: &'static str,
    pub token_masked: String,
    pub token_set: bool,
    pub token_from_env: bool,
    pub keychain_available: bool,
    pub cli_token_available: bool,
    pub ollama_running: bool,
    pub ollama_version: String,
    pub ollama_os: &'static str,
    pub ollama_steps: Vec<&'static str>,
    pub backends: Vec<SelectOpt>,
    pub accels: Vec<SelectOpt>,
    pub accel_notes: Vec<String>,
    pub log_levels: Vec<SelectOpt>,
    pub logs_dir: String,
    pub ffmpeg_found: String,
    pub ffmpeg_hint: &'static str,
    pub cache_entries: String,
    pub cache_size: String,
    pub index_counts: String,
    pub uploads_size: String,
    pub conversation_count: String,
    pub data_dir: String,
    pub config_file: String,
    pub server_url: String,
    pub access_token: String,
    pub show_token: bool,
    pub seed_text: String,
    pub threads_text: String,
    /// The folder overrides as typed (empty = default location).
    pub models_dir_set: String,
    pub downloads_dir_set: String,
    pub outputs_dir_set: String,
    pub ffmpeg_path_set: String,
    pub default_models_dir: String,
    pub default_outputs_dir: String,
}

fn path_text(p: &Option<PathBuf>) -> String {
    p.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_default()
}

fn dir_size(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                match e.file_type() {
                    Ok(t) if t.is_dir() => stack.push(e.path()),
                    Ok(t) if t.is_file() => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                    _ => {}
                }
            }
        }
    }
    total
}

pub async fn page(
    State(state): State<SharedState>,
    headers: HeaderMap,
    local: Option<axum::Extension<crate::web::security::ClientLocal>>,
) -> Response {
    let peer_is_local = local.map(|l| l.0.0).unwrap_or(true);
    let s = state.settings();
    let ctx = PageCtx::new(&state, &headers, "Settings", "settings").at("/settings");
    let models_dir = s.models_dir(&state.paths);
    let md = models_dir.clone();
    let up = state.paths.uploads_dir.clone();
    let (models_size, uploads_size) =
        tokio::task::spawn_blocking(move || (dir_size(&md), dir_size(&up))).await.unwrap_or((0, 0));
    let token = state.secrets.status().await;
    let ollama = state.ollama.status().await;
    let (os, steps) = crate::ollama::install_instructions();
    let accel = state.system.accel().clone();
    let (cache_n, cache_bytes) = state.db.call(crate::db::catalog::cache_stats).await.unwrap_or((0, 0));
    let counts = state.db.call(crate::db::catalog::count_by_source).await.unwrap_or_default();
    let convs = state
        .db
        .call(|c| c.query_row("SELECT count(*) FROM conversations", [], |r| r.get::<_, i64>(0)))
        .await
        .unwrap_or(0);
    let page = SettingsPage {
        ctx: ctx.clone(),
        models_dir: models_dir.to_string_lossy().to_string(),
        partial_dir: s.partial_dir(&state.paths).to_string_lossy().to_string(),
        outputs_dir: s.outputs_dir(&state.paths).to_string_lossy().to_string(),
        models_size: human_bytes(models_size),
        free_space: state
            .system
            .available_space(&models_dir)
            .await
            .map(human_bytes)
            .unwrap_or_else(|| "unknown".into()),
        token_label: token.source.label(),
        token_masked: token.masked.clone().unwrap_or_default(),
        token_set: token.source != TokenSource::None,
        token_from_env: token.source == TokenSource::Environment,
        keychain_available: token.keychain_available,
        cli_token_available: crate::secrets::read_hf_cli_token().is_some(),
        ollama_running: ollama.running,
        ollama_version: ollama.version.clone().unwrap_or_default(),
        ollama_os: os,
        ollama_steps: steps,
        backends: select(
            &[
                (BackendPreference::Auto, "auto", "Automatic (the one that can use the graphics card)"),
                (BackendPreference::Builtin, "builtin", "Built-in engine first"),
                (BackendPreference::Ollama, "ollama", "Ollama first"),
            ],
            s.inference.preferred_backend,
        ),
        accels: select(
            &[
                (Acceleration::Auto, "auto", "Automatic (use a GPU when available)"),
                (Acceleration::Cpu, "cpu", "CPU only"),
                (Acceleration::Cuda, "cuda", "NVIDIA CUDA"),
                (Acceleration::Metal, "metal", "Apple Metal"),
            ],
            s.inference.acceleration,
        ),
        // Only the kind of GPU this platform can have: Metal on macOS, CUDA elsewhere.
        accel_notes: vec![if cfg!(target_os = "macos") {
            accel.metal_detail.clone()
        } else {
            accel.cuda_detail.clone()
        }],
        log_levels: LOG_LEVELS
            .iter()
            .map(|l| SelectOpt { value: (*l).into(), label: (*l).into(), selected: *l == s.logging.level })
            .collect(),
        logs_dir: state.paths.logs_dir.to_string_lossy().to_string(),
        ffmpeg_found: crate::media::ffmpeg_path(&s).map(|p| p.to_string_lossy().to_string()).unwrap_or_default(),
        ffmpeg_hint: crate::media::ffmpeg_install_hint(),
        cache_entries: cache_n.to_string(),
        cache_size: human_bytes(cache_bytes.max(0) as u64),
        index_counts: if counts.is_empty() {
            "empty".into()
        } else {
            counts
                .iter()
                .map(|(src, n)| format!("{n} from {}", if src == "ollama" { "Ollama" } else { "Hugging Face" }))
                .collect::<Vec<_>>()
                .join(", ")
        },
        uploads_size: human_bytes(uploads_size),
        conversation_count: convs.to_string(),
        data_dir: state.paths.data_dir.to_string_lossy().to_string(),
        config_file: state.paths.config_file.to_string_lossy().to_string(),
        server_url: state.server.url(),
        access_token: state.server.access_token.clone(),
        show_token: peer_is_local && state.server.addr().map(|a| !a.ip().is_loopback()).unwrap_or(false),
        seed_text: s.generation.seed.map(|v| v.to_string()).unwrap_or_default(),
        threads_text: s.inference.threads.map(|v| v.to_string()).unwrap_or_default(),
        models_dir_set: path_text(&s.storage.models_dir),
        downloads_dir_set: path_text(&s.storage.downloads_dir),
        outputs_dir_set: path_text(&s.storage.outputs_dir),
        ffmpeg_path_set: path_text(&s.media.ffmpeg_path),
        default_models_dir: state.paths.default_models_dir.to_string_lossy().to_string(),
        default_outputs_dir: state.paths.outputs_dir.to_string_lossy().to_string(),
        s,
    };
    render(&ctx, &page)
}

#[derive(Deserialize, Default)]
pub struct ThemeForm {
    #[serde(default)]
    theme: String,
    #[serde(default)]
    back: Option<String>,
}

pub async fn theme(State(state): State<SharedState>, Form(f): Form<ThemeForm>) -> Response {
    let back = f.back.filter(|b| b.starts_with('/') && !b.starts_with("//")).unwrap_or_else(|| "/settings".into());
    let Ok(t) = f.theme.parse::<Theme>() else {
        return redirect_flash(&back, "error", "Unknown theme.");
    };
    match state.update_settings(|s| s.ui.theme = t) {
        Ok(_) => redirect_flash(&back, "ok", &format!("Theme changed to {}.", t.label())),
        Err(e) => redirect_flash(&back, "error", &format!("Couldn't save: {e:#}")),
    }
}

fn opt_path(v: Option<&String>) -> Result<Option<PathBuf>, String> {
    match v.map(|s| s.trim()).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(p) => crate::config::validate_writable_dir(p).map(Some),
    }
}

fn num<T: std::str::FromStr>(f: &HashMap<String, String>, key: &str, label: &str) -> Result<Option<T>, String> {
    match f.get(key).map(|s| s.trim()).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(v) => v.parse::<T>().map(Some).map_err(|_| format!("{label} must be a number.")),
    }
}

fn apply_section(s: &mut Settings, section: &str, f: &HashMap<String, String>) -> Result<&'static str, String> {
    let checked = |k: &str| f.contains_key(k);
    match section {
        "ui" => {
            s.ui.restore_last_chat = checked("restore_last_chat");
            Ok("Appearance saved.")
        }
        "storage" => {
            s.storage.models_dir = opt_path(f.get("models_dir"))?;
            s.storage.downloads_dir = opt_path(f.get("downloads_dir"))?;
            s.storage.outputs_dir = opt_path(f.get("outputs_dir"))?;
            if let Some(n) = num::<u8>(f, "max_concurrent_downloads", "Parallel downloads")? {
                s.storage.max_concurrent_downloads = n;
            }
            Ok("Storage saved. A new download limit applies after a restart.")
        }
        "huggingface" => {
            let ep = f.get("endpoint").map(|e| e.trim().to_string()).unwrap_or_default();
            let valid = url::Url::parse(&ep).is_ok_and(|u| matches!(u.scheme(), "http" | "https") && u.has_host());
            if !ep.is_empty() && !valid {
                return Err("The endpoint must be a web address such as https://huggingface.co".into());
            }
            s.huggingface.endpoint = if ep.is_empty() { "https://huggingface.co".into() } else { ep };
            s.huggingface.show_gated = checked("show_gated");
            Ok("Hugging Face settings saved.")
        }
        "ollama" => {
            s.ollama.enabled = checked("enabled");
            let url = f.get("base_url").map(|u| u.trim().to_string()).unwrap_or_default();
            if !url.is_empty() {
                let parsed = url::Url::parse(&url)
                    .map_err(|_| "The Ollama address must look like http://127.0.0.1:11434".to_string())?;
                if !matches!(parsed.scheme(), "http" | "https") {
                    return Err("The Ollama address must start with http:// or https://".into());
                }
                s.ollama.base_url = url;
            }
            if let Some(k) = f.get("keep_alive").map(|k| k.trim()).filter(|k| !k.is_empty()) {
                let ok = k.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') && k.len() <= 10;
                if !ok {
                    return Err("Keep-alive must look like 5m, 1h or -1.".into());
                }
                s.ollama.keep_alive = k.to_string();
            }
            Ok("Ollama settings saved.")
        }
        "inference" => {
            s.inference.preferred_backend = match f.get("preferred_backend").map(String::as_str) {
                Some("builtin") => BackendPreference::Builtin,
                Some("ollama") => BackendPreference::Ollama,
                _ => BackendPreference::Auto,
            };
            s.inference.acceleration = match f.get("acceleration").map(String::as_str) {
                Some("cpu") => Acceleration::Cpu,
                Some("cuda") => Acceleration::Cuda,
                Some("metal") => Acceleration::Metal,
                _ => Acceleration::Auto,
            };
            s.inference.threads = num::<usize>(f, "threads", "Threads")?.filter(|t| *t > 0);
            if let Some(m) = num::<u32>(f, "unload_after_minutes", "Unload after")? {
                s.inference.unload_after_minutes = m;
            }
            if let Some(m) = num::<u8>(f, "max_loaded_models", "Models kept in memory")? {
                s.inference.max_loaded_models = m;
            }
            Ok(
                "Inference settings saved. Thread count applies after a restart; the device applies to models loaded from now on.",
            )
        }
        "generation" => {
            let g = &mut s.generation;
            if let Some(v) = num::<f32>(f, "temperature", "Temperature")? {
                g.temperature = v;
            }
            if let Some(v) = num::<f32>(f, "top_p", "Top-p")? {
                g.top_p = v;
            }
            if let Some(v) = num::<u32>(f, "top_k", "Top-k")? {
                g.top_k = v;
            }
            // Empty = no limit.
            g.max_tokens = num::<u32>(f, "max_tokens", "Maximum answer length")?.unwrap_or(0);
            if let Some(v) = num::<f32>(f, "repeat_penalty", "Repetition penalty")? {
                g.repeat_penalty = v;
            }
            if let Some(v) = num::<u32>(f, "context_length", "Context length")? {
                g.context_length = v;
            }
            g.seed = num::<u64>(f, "seed", "Seed")?;
            if let Some(sp) = f.get("system_prompt") {
                g.system_prompt = sp.trim().to_string();
            }
            Ok("Defaults saved. New chats will use them.")
        }
        "server" => {
            let host = f.get("host").map(|h| h.trim().to_string()).unwrap_or_default();
            if !host.is_empty() {
                if host != "localhost" && host.trim_matches(['[', ']']).parse::<std::net::IpAddr>().is_err() {
                    return Err("The address must be an IP address such as 127.0.0.1 (or localhost).".into());
                }
                s.server.host = host;
            }
            if let Some(p) = num::<u16>(f, "port", "Port")? {
                if p < 1024 {
                    return Err("Choose a port between 1024 and 65535.".into());
                }
                s.server.port = p;
            }
            s.server.open_browser = checked("open_browser");
            s.server.allow_network = checked("allow_network");
            if !crate::config::is_loopback_host(&s.server.host) && !s.server.allow_network {
                return Err("To listen on a network address, also tick \"Allow access from other devices\".".into());
            }
            Ok("Server settings saved. Restart Local AI Studio to apply them.")
        }
        "logging" => {
            let lvl = f.get("level").cloned().unwrap_or_else(|| "info".into());
            if !LOG_LEVELS.contains(&lvl.as_str()) {
                return Err("Unknown log level.".into());
            }
            s.logging.level = lvl;
            s.logging.to_file = checked("to_file");
            Ok("Logging settings saved. They apply after a restart.")
        }
        "media" => {
            s.media.ffmpeg_path = match f.get("ffmpeg_path").map(|p| p.trim()).filter(|p| !p.is_empty()) {
                None => None,
                Some(p) => {
                    let pb = PathBuf::from(p);
                    if !pb.is_file() {
                        return Err(format!("{p} doesn't exist."));
                    }
                    Some(pb)
                }
            };
            if let Some(n) = num::<u8>(f, "video_frames", "Video frames")? {
                s.media.video_frames = n;
            }
            Ok("Media settings saved.")
        }
        _ => Err("Unknown settings section.".into()),
    }
}

pub async fn save(
    State(state): State<SharedState>,
    Path(section): Path<String>,
    Form(f): Form<HashMap<String, String>>,
) -> Response {
    // The Appearance section is saved as "ui".
    let anchor = if section == "ui" { "appearance" } else { section.as_str() };
    let back = format!("/settings#{anchor}");
    let mut next = state.settings();
    match apply_section(&mut next, &section, &f) {
        Ok(msg) => match state.update_settings(|s| *s = next) {
            Ok(_) => redirect_flash(&back, "ok", msg),
            Err(e) => redirect_flash(&back, "error", &format!("Couldn't save config.toml: {e:#}")),
        },
        Err(e) => redirect_flash(&back, "error", &e),
    }
}

#[derive(Deserialize, Default)]
pub struct TokenForm {
    #[serde(default)]
    action: String,
    #[serde(default)]
    token: String,
}

pub async fn hf_token(State(state): State<SharedState>, Form(f): Form<TokenForm>) -> Response {
    let back = "/settings#huggingface";
    match f.action.as_str() {
        "save" | "import-cli" => {
            let token = if f.action == "import-cli" {
                match crate::secrets::read_hf_cli_token() {
                    Some(t) => t,
                    None => return redirect_flash(back, "error", "No token from the Hugging Face CLI was found."),
                }
            } else {
                f.token.trim().to_string()
            };
            if let Err(e) = crate::secrets::validate_token_format(&token) {
                return redirect_flash(back, "error", &e);
            }
            // Verify before storing, so typos are caught immediately.
            let who = match state.hub.whoami(&token).await {
                Ok(w) => Some(w.name),
                Err(AppError::BadRequest(m)) => return redirect_flash(back, "error", &m),
                Err(_) => None,
            };
            match state.secrets.set_hf_token(&token).await {
                Ok(src) => {
                    let place = match src {
                        TokenSource::Keychain => "in your system keychain",
                        _ => "for this session only (no system keychain is available)",
                    };
                    let msg = match who {
                        Some(n) => format!("Token saved {place}. Signed in to Hugging Face as {n}."),
                        None => format!("Token saved {place}. It couldn't be verified right now (offline?)."),
                    };
                    redirect_flash(back, "ok", &msg)
                }
                Err(e) => redirect_flash(back, "error", &e),
            }
        }
        "verify" => match state.secrets.hf_token().await {
            None => redirect_flash(back, "error", "No token is set."),
            Some(t) => match state.hub.whoami(&t).await {
                Ok(w) => redirect_flash(back, "ok", &format!("The token works: signed in as {}.", w.name)),
                Err(e) => redirect_flash(back, "error", &e.user_message()),
            },
        },
        "clear" => match state.secrets.clear_hf_token().await {
            Ok(()) => redirect_flash(back, "ok", "Token removed."),
            Err(e) => redirect_flash(back, "error", &e),
        },
        _ => redirect_flash(back, "error", "Unknown action."),
    }
}

pub async fn data_action(State(state): State<SharedState>, Path(action): Path<String>) -> Response {
    let back = "/settings#data";
    match action.as_str() {
        "clear-cache" => match state.db.call(crate::db::catalog::cache_clear).await {
            Ok(n) => redirect_flash(back, "ok", &format!("Cleared {n} cached pages.")),
            Err(e) => redirect_flash(back, "error", &format!("{e:#}")),
        },
        "clear-index" => match state.db.call(crate::db::catalog::clear).await {
            Ok(n) => redirect_flash(back, "ok", &format!("Removed {n} entries from the offline index.")),
            Err(e) => redirect_flash(back, "error", &format!("{e:#}")),
        },
        "clear-uploads" => {
            // Files still used by a chat or a transcript are kept. If either list can't be read,
            // nothing is deleted: an empty list would mean "delete everything".
            let lists = async {
                let referenced = state.db.call(crate::db::conversations::referenced_uploads).await?;
                let transcripts = state.db.call(|c| crate::db::jobs::list_transcriptions(c, 100_000)).await?;
                Ok::<_, AppError>((referenced, transcripts))
            };
            let (referenced, transcripts) = match lists.await {
                Ok(l) => l,
                Err(e) => return redirect_flash(back, "error", &format!("Nothing was deleted: {}", e.user_message())),
            };
            let transcripts: Vec<String> = transcripts.into_iter().map(|t| t.stored_file).collect();
            let keep: std::collections::HashSet<String> = referenced.into_iter().chain(transcripts).collect();
            let base = state.paths.uploads_dir.clone();
            let removed = tokio::task::spawn_blocking(move || {
                let mut removed = 0usize;
                let mut stack = vec![base.clone()];
                while let Some(d) = stack.pop() {
                    if let Ok(rd) = std::fs::read_dir(&d) {
                        for e in rd.flatten() {
                            let p = e.path();
                            if p.is_dir() {
                                stack.push(p);
                            } else if let Ok(rel) = p.strip_prefix(&base) {
                                let rel = rel.to_string_lossy().replace('\\', "/");
                                if !keep.contains(&rel) && std::fs::remove_file(&p).is_ok() {
                                    removed += 1;
                                }
                            }
                        }
                    }
                }
                removed
            })
            .await
            .unwrap_or(0);
            redirect_flash(
                back,
                "ok",
                &format!(
                    "Deleted {}.",
                    crate::util::count_noun(removed as u64, "unused uploaded file", "unused uploaded files")
                ),
            )
        }
        "delete-conversations" => {
            state.chat.stop_all();
            match state.db.call(crate::db::conversations::delete_all_conversations).await {
                Ok(n) => redirect_flash(
                    back,
                    "ok",
                    &format!("Deleted {}.", crate::util::count_noun(n as u64, "conversation", "conversations")),
                ),
                Err(e) => redirect_flash(back, "error", &format!("{e:#}")),
            }
        }
        "export-all" => {
            let res = state
                .db
                .call(|c| {
                    let list = crate::db::conversations::list_conversations(c, usize::MAX >> 1)?;
                    let mut out = Vec::new();
                    for s in list {
                        if let Some(conv) = crate::db::conversations::get_conversation(c, &s.id)? {
                            let msgs = crate::db::conversations::messages(c, &s.id)?;
                            out.push(crate::web::pages::chat::export_json(&conv, &msgs));
                        }
                    }
                    Ok(out)
                })
                .await;
            match res {
                Ok(all) => {
                    let body = serde_json::to_string_pretty(&all).unwrap_or_else(|_| "[]".into());
                    let mut r = body.into_response();
                    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
                    r.headers_mut().insert(
                        header::CONTENT_DISPOSITION,
                        HeaderValue::from_static("attachment; filename=\"local-ai-studio-conversations.json\""),
                    );
                    r
                }
                Err(e) => redirect_flash(back, "error", &format!("{e:#}")),
            }
        }
        "reset-settings" => match state.update_settings(|s| *s = Settings::default()) {
            Ok(_) => redirect_flash("/settings", "ok", "Settings reset to defaults. Restart to apply server settings."),
            Err(e) => redirect_flash(back, "error", &format!("{e:#}")),
        },
        "open-data" => match crate::util::open_in_file_manager(&state.paths.data_dir) {
            Ok(()) => redirect_flash(back, "ok", "Opened the data folder."),
            Err(e) => redirect_flash(back, "error", &format!("Couldn't open it: {e}")),
        },
        "open-outputs" => {
            let dir = state.settings().outputs_dir(&state.paths);
            let _ = std::fs::create_dir_all(&dir);
            match crate::util::open_in_file_manager(&dir) {
                Ok(()) => redirect_flash("/create", "ok", "Opened the pictures folder."),
                Err(e) => redirect_flash("/create", "error", &format!("Couldn't open it: {e}")),
            }
        }
        "open-logs" => match crate::util::open_in_file_manager(&state.paths.logs_dir) {
            Ok(()) => redirect_flash("/settings#logging", "ok", "Opened the logs folder."),
            Err(e) => redirect_flash("/settings#logging", "error", &format!("Couldn't open it: {e}")),
        },
        _ => redirect_flash(back, "error", "Unknown action."),
    }
}

pub async fn shutdown(State(state): State<SharedState>) -> Html<String> {
    let st = state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        st.shutdown.cancel();
    });
    Html(format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Stopped · {app}</title><link rel=\"stylesheet\" href=\"/assets/app.css\"><link rel=\"stylesheet\" href=\"/assets/themes.css\"></head><body data-theme=\"{theme}\" class=\"stopped\"><main class=\"card unlock-card\"><h1>{app} has stopped</h1><p>Your conversations, models and settings are saved. To start again, open Local AI Studio from your applications menu or run <code>local-ai-studio</code>.</p></main></body></html>",
        app = crate::APP_NAME,
        theme = state.settings().ui.theme.slug()
    ))
}

#[derive(Template)]
#[template(path = "licenses.html")]
pub struct LicensesPage {
    pub ctx: PageCtx,
    pub third_party_html: String,
}

/// The license list is compiled in and never changes, so it is rendered once.
static LICENSES_HTML: std::sync::OnceLock<String> = std::sync::OnceLock::new();

pub async fn licenses(State(state): State<SharedState>, headers: HeaderMap) -> Response {
    let ctx = PageCtx::new(&state, &headers, "Licenses", "settings").at("/about/licenses");
    let page = LicensesPage {
        ctx: ctx.clone(),
        third_party_html: LICENSES_HTML
            .get_or_init(|| {
                crate::render::model_card_html(include_str!("../../../THIRD_PARTY_LICENSES.md"), usize::MAX)
            })
            .clone(),
    };
    render(&ctx, &page)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn server_section_requires_explicit_network_opt_in() {
        let mut s = Settings::default();
        let err = apply_section(&mut s, "server", &form(&[("host", "0.0.0.0"), ("port", "9000")])).unwrap_err();
        assert!(err.contains("Allow access"));
        let mut s = Settings::default();
        apply_section(&mut s, "server", &form(&[("host", "0.0.0.0"), ("port", "9000"), ("allow_network", "on")]))
            .unwrap();
        assert!(s.server.allow_network);
        assert!(apply_section(&mut Settings::default(), "server", &form(&[("port", "80")])).is_err());
        assert!(apply_section(&mut Settings::default(), "server", &form(&[("host", "evil.com")])).is_err());
    }

    #[test]
    fn generation_numbers_are_validated() {
        let mut s = Settings::default();
        assert!(apply_section(&mut s, "generation", &form(&[("temperature", "hot")])).is_err());
        apply_section(&mut s, "generation", &form(&[("temperature", "0.3"), ("seed", "")])).unwrap();
        assert_eq!(s.generation.temperature, 0.3);
        assert_eq!(s.generation.seed, None);
    }

    #[test]
    fn ollama_url_is_validated() {
        let mut s = Settings::default();
        assert!(apply_section(&mut s, "ollama", &form(&[("base_url", "ftp://x")])).is_err());
        apply_section(&mut s, "ollama", &form(&[("enabled", "on"), ("base_url", "http://127.0.0.1:11500")])).unwrap();
        assert_eq!(s.ollama.base_url, "http://127.0.0.1:11500");
        assert!(s.ollama.enabled);
    }
}
