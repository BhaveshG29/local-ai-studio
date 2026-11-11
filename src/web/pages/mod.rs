//! HTML page handlers.

pub mod chat;
pub mod create;
pub mod discover;
pub mod downloads;
pub mod files;
pub mod home;
pub mod library;
pub mod settings;
pub mod system;
pub mod transcribe;

use axum::http::Uri;
use axum::response::Response;

use crate::SharedState;
use crate::error::AppError;
use crate::library::LibraryItem;

pub async fn not_found(uri: Uri) -> Response {
    axum::response::IntoResponse::into_response(AppError::not_found(format!("There is no page at {}.", uri.path())))
}

/// A model the user can pick in a dropdown.
#[derive(Clone, Debug)]
pub struct ModelChoice {
    pub value: String,
    pub label: String,
    pub group: &'static str,
    pub vision: bool,
    pub selected: bool,
}

/// All library items plus an Ollama error message (if Ollama couldn't be reached).
pub async fn library_items(state: &SharedState) -> (Vec<LibraryItem>, Option<String>) {
    let mut loaded = state.inference.loaded_refs();
    let mut caps = std::collections::HashMap::new();
    if state.ollama.enabled() {
        if let Ok(models) = state.ollama.list_local().await {
            caps = state.ollama.capabilities(&models).await;
        }
        if let Ok(running) = state.ollama.running().await {
            for r in running {
                loaded.insert(format!("ollama:{}", r.name));
            }
        }
    }
    crate::library::list_all(&state.db, &state.ollama, &loaded, &caps).await
}

/// Chat-capable, runnable models for model pickers.
pub async fn chat_choices(state: &SharedState, selected: Option<&str>) -> (Vec<ModelChoice>, Option<String>) {
    let (items, err) = library_items(state).await;
    let mut out: Vec<ModelChoice> = items
        .into_iter()
        .filter(|i| i.is_chat() && i.runnable)
        .map(|i| ModelChoice {
            selected: selected == Some(i.model_ref.as_str()),
            vision: i.supports_images(),
            group: if i.engine == "Ollama" { "Ollama" } else { "Built-in engine" },
            label: if i.supports_images() { format!("{} · sees images", i.name) } else { i.name.clone() },
            value: i.model_ref,
        })
        .collect();
    out.sort_by(|a, b| a.group.cmp(b.group).then(a.label.to_lowercase().cmp(&b.label.to_lowercase())));
    (out, err)
}

/// Models of a given library kind (`transcription`, `image-generation`).
pub async fn models_of_kind(state: &SharedState, kind: &str) -> Vec<crate::db::models::ModelRow> {
    state.db.call(crate::db::models::list).await.unwrap_or_default().into_iter().filter(|m| m.kind == kind).collect()
}
