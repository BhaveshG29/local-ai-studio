//! Locally stored models (downloaded from Hugging Face or imported from disk) and
//! per-model user metadata for Ollama models.

use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};

use super::{from_json, to_json};
use crate::library::ModelAnalysis;

/// A file belonging to a stored model.
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct StoredFile {
    /// Path relative to the model directory.
    pub path: String,
    pub size: u64,
    /// SHA-256 when known (LFS files are verified against it after download).
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRow {
    pub id: String,
    /// `huggingface` or `local` (imported from disk).
    pub source: String,
    pub repo_id: Option<String>,
    pub revision: Option<String>,
    pub display_name: String,
    /// Directory containing the model files.
    pub path: String,
    /// Main weights file for single-file formats (GGUF).
    pub primary_file: Option<String>,
    pub format: String,
    pub architecture: Option<String>,
    /// `chat`, `transcription`, `image-generation` or `unknown`.
    pub kind: String,
    pub size_bytes: i64,
    pub quantization: Option<String>,
    pub parameters: Option<i64>,
    pub context_length: Option<i64>,
    pub analysis: ModelAnalysis,
    pub files: Vec<StoredFile>,
    pub created_at: String,
    pub last_used_at: Option<String>,
}

fn from_row(r: &Row<'_>) -> rusqlite::Result<ModelRow> {
    let analysis: String = r.get("analysis_json")?;
    let files: String = r.get("files_json")?;
    Ok(ModelRow {
        id: r.get("id")?,
        source: r.get("source")?,
        repo_id: r.get("repo_id")?,
        revision: r.get("revision")?,
        display_name: r.get("display_name")?,
        path: r.get("path")?,
        primary_file: r.get("primary_file")?,
        format: r.get("format")?,
        architecture: r.get("architecture")?,
        kind: r.get("kind")?,
        size_bytes: r.get("size_bytes")?,
        quantization: r.get("quantization")?,
        parameters: r.get("parameters")?,
        context_length: r.get("context_length")?,
        analysis: from_json(&analysis),
        files: from_json(&files),
        created_at: r.get("created_at")?,
        last_used_at: r.get("last_used_at")?,
    })
}

pub fn upsert(c: &Connection, m: &ModelRow) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO models (id, source, repo_id, revision, display_name, path, primary_file, format, architecture,
                             kind, size_bytes, quantization, parameters, context_length, analysis_json, files_json,
                             created_at, last_used_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
         ON CONFLICT(id) DO UPDATE SET
            source = excluded.source, repo_id = excluded.repo_id, revision = excluded.revision,
            path = excluded.path, primary_file = excluded.primary_file, format = excluded.format,
            architecture = excluded.architecture, kind = excluded.kind, size_bytes = excluded.size_bytes,
            quantization = excluded.quantization, parameters = excluded.parameters,
            context_length = excluded.context_length, analysis_json = excluded.analysis_json,
            files_json = excluded.files_json",
        params![
            m.id,
            m.source,
            m.repo_id,
            m.revision,
            m.display_name,
            m.path,
            m.primary_file,
            m.format,
            m.architecture,
            m.kind,
            m.size_bytes,
            m.quantization,
            m.parameters,
            m.context_length,
            to_json(&m.analysis),
            to_json(&m.files),
            m.created_at,
            m.last_used_at
        ],
    )?;
    Ok(())
}

pub fn list(c: &Connection) -> rusqlite::Result<Vec<ModelRow>> {
    let mut stmt =
        c.prepare("SELECT * FROM models ORDER BY COALESCE(last_used_at, created_at) DESC, display_name ASC")?;
    let rows = stmt.query_map([], from_row)?;
    rows.collect()
}

pub fn get(c: &Connection, id: &str) -> rusqlite::Result<Option<ModelRow>> {
    c.query_row("SELECT * FROM models WHERE id = ?1", params![id], from_row).optional()
}

/// Finds a model stored from the same repository and primary file (duplicate detection).
pub fn find_by_repo(c: &Connection, repo_id: &str, primary_file: Option<&str>) -> rusqlite::Result<Option<ModelRow>> {
    match primary_file {
        Some(f) => c
            .query_row(
                "SELECT * FROM models WHERE repo_id = ?1 AND primary_file = ?2 LIMIT 1",
                params![repo_id, f],
                from_row,
            )
            .optional(),
        None => c
            .query_row(
                "SELECT * FROM models WHERE repo_id = ?1 AND primary_file IS NULL LIMIT 1",
                params![repo_id],
                from_row,
            )
            .optional(),
    }
}

pub fn rename(c: &Connection, id: &str, name: &str) -> rusqlite::Result<usize> {
    c.execute("UPDATE models SET display_name = ?2 WHERE id = ?1", params![id, name])
}

pub fn touch(c: &Connection, id: &str, now: &str) -> rusqlite::Result<usize> {
    c.execute("UPDATE models SET last_used_at = ?2 WHERE id = ?1", params![id, now])
}

pub fn delete(c: &Connection, id: &str) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM models WHERE id = ?1", params![id])
}

/// User metadata for a model managed by Ollama (custom display name, last use).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelMeta {
    pub model_ref: String,
    pub display_name: Option<String>,
    pub last_used_at: Option<String>,
}

pub fn all_meta(c: &Connection) -> rusqlite::Result<Vec<ModelMeta>> {
    let mut stmt = c.prepare("SELECT model_ref, display_name, last_used_at FROM model_meta")?;
    let rows = stmt
        .query_map([], |r| Ok(ModelMeta { model_ref: r.get(0)?, display_name: r.get(1)?, last_used_at: r.get(2)? }))?;
    rows.collect()
}

pub fn set_meta_name(c: &Connection, model_ref: &str, name: Option<&str>) -> rusqlite::Result<usize> {
    c.execute(
        "INSERT INTO model_meta (model_ref, display_name) VALUES (?1, ?2)
         ON CONFLICT(model_ref) DO UPDATE SET display_name = excluded.display_name",
        params![model_ref, name],
    )
}

pub fn touch_meta(c: &Connection, model_ref: &str, now: &str) -> rusqlite::Result<usize> {
    c.execute(
        "INSERT INTO model_meta (model_ref, last_used_at) VALUES (?1, ?2)
         ON CONFLICT(model_ref) DO UPDATE SET last_used_at = excluded.last_used_at",
        params![model_ref, now],
    )
}

pub fn delete_meta(c: &Connection, model_ref: &str) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM model_meta WHERE model_ref = ?1", params![model_ref])
}
