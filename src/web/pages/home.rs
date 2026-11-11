//! The home dashboard: getting started, status at a glance, recent chats.

use askama::Template;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;

use crate::SharedState;
use crate::util::{human_bytes, relative_time};
use crate::web::page::{PageCtx, render};

pub struct RecentChat {
    pub id: String,
    pub title: String,
    pub when: String,
}

pub struct ActiveDownload {
    pub title: String,
    pub percent: u32,
    pub detail: String,
}

/// A suggested first model. Links to the model page, where size, compatibility and memory
/// fit are shown before anything is downloaded.
pub struct Starter {
    pub title: &'static str,
    pub description: &'static str,
    pub href: &'static str,
    pub engine: &'static str,
    pub needs_ollama: bool,
}

pub const STARTERS: &[Starter] = &[
    Starter {
        title: "Qwen 2.5 · 1.5B Instruct",
        description: "Small, fast, multilingual chat model. A good first download on any computer.",
        href: "/discover/hf/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        engine: "Built-in",
        needs_ollama: false,
    },
    Starter {
        title: "Llama 3.2 · 3B Instruct",
        description: "Meta's compact assistant. Better answers, needs about twice the memory.",
        href: "/discover/hf/bartowski/Llama-3.2-3B-Instruct-GGUF",
        engine: "Built-in",
        needs_ollama: false,
    },
    Starter {
        title: "Gemma 3 · 4B",
        description: "Google's model that can also look at pictures you attach.",
        href: "/discover/ollama/gemma3",
        engine: "Ollama",
        needs_ollama: true,
    },
    Starter {
        title: "Whisper Base",
        description: "Turns speech in audio files into text, in many languages.",
        href: "/discover/hf/openai/whisper-base",
        engine: "Built-in",
        needs_ollama: false,
    },
    Starter {
        title: "SD-Turbo",
        description: "Creates images from a text description in a single step.",
        href: "/discover/hf/stabilityai/sd-turbo",
        engine: "Built-in",
        needs_ollama: false,
    },
];

#[derive(Template)]
#[template(path = "home.html")]
pub struct HomePage {
    pub ctx: PageCtx,
    pub first_run: bool,
    pub recent: Vec<RecentChat>,
    pub chat_model_count: usize,
    pub model_count: usize,
    pub ollama_running: bool,
    pub ollama_installed: bool,
    pub ollama_version: String,
    pub ollama_os: &'static str,
    pub ollama_steps: Vec<&'static str>,
    pub engine_label: String,
    pub ram_total: String,
    pub ram_free: String,
    pub ram_used_pct: u32,
    pub disk_free: String,
    pub models_dir: String,
    pub downloads: Vec<ActiveDownload>,
    pub starters: &'static [Starter],
}

pub async fn page(State(state): State<SharedState>, headers: HeaderMap) -> Response {
    let ctx = PageCtx::new(&state, &headers, "Home", "home").at("/");
    let (items, _) = super::library_items(&state).await;
    let ollama = state.ollama.status().await;
    let snap = state.system.snapshot().await;
    let settings = state.settings();
    let models_dir = settings.models_dir(&state.paths);
    let disk_free = state.system.available_space(&models_dir).await;
    let recent = state
        .db
        .call(|c| crate::db::conversations::list_conversations(c, 6))
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|c| RecentChat { when: relative_time(&c.updated_at), id: c.id, title: c.title })
        .collect();
    let downloads = state
        .downloads
        .list()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|j| j.row.state.is_active())
        .take(3)
        .map(|j| ActiveDownload {
            percent: j.percent.round() as u32,
            detail: format!("{} of {} {}", human_bytes(j.done_bytes), human_bytes(j.total_bytes), j.speed_label()),
            title: j.row.title,
        })
        .collect();
    let (os, steps) = crate::ollama::install_instructions();
    let engine_label = match state.inference.device() {
        Ok((_, l)) => l,
        Err(e) => format!("unavailable: {e}"),
    };
    let used_pct = if snap.total_ram > 0 {
        ((snap.total_ram - snap.available_ram.min(snap.total_ram)) as f64 / snap.total_ram as f64 * 100.0).round()
            as u32
    } else {
        0
    };
    let page = HomePage {
        first_run: items.iter().all(|i| !i.runnable),
        chat_model_count: items.iter().filter(|i| i.is_chat() && i.runnable).count(),
        model_count: items.len(),
        ollama_running: ollama.running,
        ollama_installed: ollama.installed(),
        ollama_version: ollama.version.unwrap_or_default(),
        ollama_os: os,
        ollama_steps: steps,
        engine_label,
        ram_total: human_bytes(snap.total_ram),
        ram_free: human_bytes(snap.available_ram),
        ram_used_pct: used_pct,
        disk_free: disk_free.map(human_bytes).unwrap_or_else(|| "unknown".into()),
        models_dir: models_dir.to_string_lossy().to_string(),
        recent,
        downloads,
        starters: STARTERS,
        ctx: ctx.clone(),
    };
    render(&ctx, &page)
}
