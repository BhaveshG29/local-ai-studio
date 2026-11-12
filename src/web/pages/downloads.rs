//! The download manager page. Auto-refreshes every second while something is downloading.

use askama::Template;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::Response;

use crate::SharedState;
use crate::db::downloads::{DownloadKind, DownloadState};
use crate::util::{human_bytes, relative_time};
use crate::web::page::{PageCtx, redirect_flash, render};

pub struct JobRow {
    pub id: String,
    pub title: String,
    pub source: &'static str,
    pub state: &'static str,
    pub state_class: &'static str,
    pub percent: u32,
    pub percent_exact: String,
    pub done: String,
    pub total: String,
    pub speed: String,
    pub eta: String,
    pub current: String,
    pub error: String,
    pub files: usize,
    pub created: String,
    pub can_pause: bool,
    pub can_resume: bool,
    pub can_cancel: bool,
    pub can_remove: bool,
    pub model_href: String,
    pub chat_href: String,
}

#[derive(Template)]
#[template(path = "downloads.html")]
pub struct DownloadsPage {
    pub ctx: PageCtx,
    pub jobs: Vec<JobRow>,
    pub active: usize,
    pub any_finished: bool,
    pub models_dir: String,
    pub partial_dir: String,
    pub free_space: String,
    pub paused_count: usize,
    /// The download whose "Cancel" was pressed: it asks for confirmation inline, and the
    /// page stops refreshing meanwhile (a refresh would close a pop-up immediately).
    pub confirm: String,
}

#[derive(serde::Deserialize, Default)]
pub struct DownloadsQuery {
    confirm: Option<String>,
}

pub async fn page(State(state): State<SharedState>, headers: HeaderMap, Query(q): Query<DownloadsQuery>) -> Response {
    let jobs = state.downloads.list().await.unwrap_or_default();
    let active = jobs.iter().filter(|j| j.row.state.is_active()).count();
    let ctx = PageCtx::new(&state, &headers, "Downloads", "downloads")
        .at("/downloads")
        .with_refresh((active > 0 && q.confirm.is_none()).then_some(1));
    let settings = state.settings();
    let models_dir = settings.models_dir(&state.paths);
    let free = state.system.available_space(&models_dir).await;
    let rows: Vec<JobRow> = jobs
        .iter()
        .map(|j| {
            let st = j.row.state;
            let (label, class) = match st {
                DownloadState::Queued => ("Waiting", "muted"),
                DownloadState::Downloading => ("Downloading", "info"),
                DownloadState::Verifying => ("Verifying", "info"),
                DownloadState::Paused => ("Paused", "warn"),
                DownloadState::Completed => ("Done", "ok"),
                DownloadState::Failed => ("Failed", "danger"),
                DownloadState::Cancelled => ("Cancelled", "muted"),
            };
            let chat_href = match (&j.row.kind, st, &j.row.model_id) {
                (DownloadKind::Ollama, DownloadState::Completed, _) => format!("/chat?model=ollama:{}", j.row.repo_id),
                (DownloadKind::HuggingFace, DownloadState::Completed, Some(id)) => format!("/library/model/local:{id}"),
                _ => String::new(),
            };
            JobRow {
                id: j.row.id.clone(),
                title: j.row.title.clone(),
                source: if j.row.kind == DownloadKind::Ollama { "Ollama" } else { "Hugging Face" },
                state: label,
                state_class: class,
                percent: j.percent.floor() as u32,
                percent_exact: format!("{:.1}", j.percent),
                done: human_bytes(j.done_bytes),
                total: if j.total_bytes > 0 { human_bytes(j.total_bytes) } else { "unknown size".into() },
                speed: j.speed_label(),
                eta: j.eta_label(),
                current: j.current_file.clone(),
                error: j.row.error.clone().unwrap_or_default(),
                files: j.row.files.len(),
                created: relative_time(&j.row.created_at),
                can_pause: st.is_active(),
                can_resume: matches!(st, DownloadState::Paused | DownloadState::Failed),
                can_cancel: !st.is_finished(),
                can_remove: !st.is_active(),
                model_href: match (&j.row.kind, &j.row.model_id) {
                    (DownloadKind::HuggingFace, Some(id)) => format!("/library/model/local:{id}"),
                    _ => String::new(),
                },
                chat_href,
            }
        })
        .collect();
    let page = DownloadsPage {
        ctx: ctx.clone(),
        active,
        any_finished: rows.iter().any(|r| r.can_remove),
        paused_count: jobs.iter().filter(|j| j.row.state == DownloadState::Paused).count(),
        jobs: rows,
        models_dir: models_dir.to_string_lossy().to_string(),
        partial_dir: settings.partial_dir(&state.paths).to_string_lossy().to_string(),
        free_space: free.map(human_bytes).unwrap_or_else(|| "unknown".into()),
        confirm: q.confirm.unwrap_or_default(),
    };
    render(&ctx, &page)
}

pub async fn action(State(state): State<SharedState>, Path((id, action)): Path<(String, String)>) -> Response {
    let res = match action.as_str() {
        "pause" => state.downloads.pause(&id).await.map(|_| "Paused. Press Resume to continue where it stopped."),
        "resume" | "retry" => state.downloads.resume(&id).await.map(|_| "Resuming…"),
        "cancel" => state.downloads.cancel(&id).await.map(|_| "Cancelled. Partial files were removed."),
        "remove" => state.downloads.remove(&id).await.map(|_| "Removed from the list."),
        _ => return redirect_flash("/downloads", "error", "Unknown action."),
    };
    match res {
        Ok(msg) => redirect_flash("/downloads", "ok", msg),
        Err(e) => redirect_flash("/downloads", "error", &e.user_message()),
    }
}

pub async fn clear(State(state): State<SharedState>) -> Response {
    match state.downloads.clear_finished().await {
        Ok(n) => redirect_flash(
            "/downloads",
            "ok",
            &format!("Cleared {}.", crate::util::count_noun(n as u64, "finished item", "finished items")),
        ),
        Err(e) => redirect_flash("/downloads", "error", &e.user_message()),
    }
}
