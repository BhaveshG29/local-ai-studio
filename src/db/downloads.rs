//! Persistent download jobs, so downloads survive restarts and can be resumed.

use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};

use super::{from_json, to_json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadKind {
    HuggingFace,
    Ollama,
}

impl DownloadKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DownloadKind::HuggingFace => "huggingface",
            DownloadKind::Ollama => "ollama",
        }
    }
    pub fn parse(s: &str) -> Self {
        if s == "ollama" { DownloadKind::Ollama } else { DownloadKind::HuggingFace }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadState {
    Queued,
    Downloading,
    Verifying,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl DownloadState {
    pub fn as_str(self) -> &'static str {
        match self {
            DownloadState::Queued => "queued",
            DownloadState::Downloading => "downloading",
            DownloadState::Verifying => "verifying",
            DownloadState::Paused => "paused",
            DownloadState::Completed => "completed",
            DownloadState::Failed => "failed",
            DownloadState::Cancelled => "cancelled",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "queued" => DownloadState::Queued,
            "downloading" => DownloadState::Downloading,
            "verifying" => DownloadState::Verifying,
            "paused" => DownloadState::Paused,
            "completed" => DownloadState::Completed,
            "cancelled" => DownloadState::Cancelled,
            _ => DownloadState::Failed,
        }
    }
    pub fn is_active(self) -> bool {
        matches!(self, DownloadState::Queued | DownloadState::Downloading | DownloadState::Verifying)
    }
    pub fn is_finished(self) -> bool {
        matches!(self, DownloadState::Completed | DownloadState::Failed | DownloadState::Cancelled)
    }
    pub fn label(self) -> &'static str {
        match self {
            DownloadState::Queued => "Waiting",
            DownloadState::Downloading => "Downloading",
            DownloadState::Verifying => "Verifying",
            DownloadState::Paused => "Paused",
            DownloadState::Completed => "Done",
            DownloadState::Failed => "Failed",
            DownloadState::Cancelled => "Cancelled",
        }
    }
}

/// One file of a Hugging Face download job.
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct DownloadFile {
    /// Repository the file comes from (companion files may come from another repo).
    pub repo_id: String,
    pub revision: String,
    /// Path inside the repository.
    pub path: String,
    /// Path inside the local model directory (usually equal to `path`).
    pub local_path: String,
    pub size: u64,
    /// SHA-256 of LFS files.
    pub sha256: Option<String>,
    /// Git blob id (SHA-1) of small, non-LFS files.
    pub git_oid: Option<String>,
    #[serde(default)]
    pub done: bool,
}

/// Extra options of a job.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct DownloadOptions {
    /// Name to give the model in the library.
    pub display_name: Option<String>,
    /// Main weights file (GGUF) for single-file models.
    pub primary_file: Option<String>,
    /// Commit sha the files were resolved at.
    pub commit: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DownloadRow {
    pub id: String,
    pub kind: DownloadKind,
    pub title: String,
    /// HF repo id, or Ollama model name (e.g. `llama3.2:3b`).
    pub repo_id: String,
    pub revision: Option<String>,
    pub files: Vec<DownloadFile>,
    pub state: DownloadState,
    pub total_bytes: i64,
    pub done_bytes: i64,
    pub error: Option<String>,
    pub model_id: Option<String>,
    pub options: DownloadOptions,
    pub created_at: String,
    pub updated_at: String,
}

fn from_row(r: &Row<'_>) -> rusqlite::Result<DownloadRow> {
    let kind: String = r.get("kind")?;
    let state: String = r.get("state")?;
    let files: String = r.get("files_json")?;
    let options: String = r.get("options_json")?;
    Ok(DownloadRow {
        id: r.get("id")?,
        kind: DownloadKind::parse(&kind),
        title: r.get("title")?,
        repo_id: r.get("repo_id")?,
        revision: r.get("revision")?,
        files: from_json(&files),
        state: DownloadState::parse(&state),
        total_bytes: r.get("total_bytes")?,
        done_bytes: r.get("done_bytes")?,
        error: r.get("error")?,
        model_id: r.get("model_id")?,
        options: from_json(&options),
        created_at: r.get("created_at")?,
        updated_at: r.get("updated_at")?,
    })
}

pub fn insert(c: &Connection, d: &DownloadRow) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO downloads (id, kind, title, repo_id, revision, files_json, state, total_bytes, done_bytes,
                                error, model_id, options_json, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            d.id,
            d.kind.as_str(),
            d.title,
            d.repo_id,
            d.revision,
            to_json(&d.files),
            d.state.as_str(),
            d.total_bytes,
            d.done_bytes,
            d.error,
            d.model_id,
            to_json(&d.options),
            d.created_at,
            d.updated_at
        ],
    )?;
    Ok(())
}

pub fn save(c: &Connection, d: &DownloadRow) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE downloads SET files_json = ?2, state = ?3, total_bytes = ?4, done_bytes = ?5, error = ?6,
                model_id = ?7, options_json = ?8, updated_at = ?9 WHERE id = ?1",
        params![
            d.id,
            to_json(&d.files),
            d.state.as_str(),
            d.total_bytes,
            d.done_bytes,
            d.error,
            d.model_id,
            to_json(&d.options),
            d.updated_at
        ],
    )
}

pub fn get(c: &Connection, id: &str) -> rusqlite::Result<Option<DownloadRow>> {
    c.query_row("SELECT * FROM downloads WHERE id = ?1", params![id], from_row).optional()
}

pub fn list(c: &Connection) -> rusqlite::Result<Vec<DownloadRow>> {
    let mut stmt = c.prepare("SELECT * FROM downloads ORDER BY created_at DESC")?;
    let rows = stmt.query_map([], from_row)?;
    rows.collect()
}

pub fn delete(c: &Connection, id: &str) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM downloads WHERE id = ?1", params![id])
}

pub fn delete_finished(c: &Connection) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM downloads WHERE state IN ('completed', 'failed', 'cancelled')", [])
}

/// Jobs that were running when the app stopped become paused (resumable).
pub fn pause_interrupted(c: &Connection, now: &str) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE downloads SET state = 'paused', updated_at = ?1 WHERE state IN ('queued', 'downloading', 'verifying')",
        params![now],
    )
}
