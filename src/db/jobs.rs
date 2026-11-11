//! Image generation history and transcription history.

use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};

use super::{from_json, to_json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Done => "done",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "queued" => JobStatus::Queued,
            "running" => JobStatus::Running,
            "done" => JobStatus::Done,
            "cancelled" => JobStatus::Cancelled,
            _ => JobStatus::Failed,
        }
    }
    pub fn is_active(self) -> bool {
        matches!(self, JobStatus::Queued | JobStatus::Running)
    }
}

/// Parameters of an image generation request (stored with each history item).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ImageParams {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance_scale: f64,
    pub seed: u64,
    pub count: u32,
}

impl Default for ImageParams {
    fn default() -> Self {
        Self { width: 512, height: 512, steps: 1, guidance_scale: 0.0, seed: 0, count: 1 }
    }
}

/// One generated image. `file` is relative to the outputs directory.
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct ImageOutput {
    pub file: String,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationRow {
    pub id: String,
    pub kind: String,
    pub model_id: String,
    pub model_name: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub params: ImageParams,
    pub outputs: Vec<ImageOutput>,
    pub status: JobStatus,
    pub error: Option<String>,
    pub duration_ms: Option<i64>,
    pub created_at: String,
}

fn generation_from_row(r: &Row<'_>) -> rusqlite::Result<GenerationRow> {
    let params: String = r.get("params_json")?;
    let outputs: String = r.get("outputs_json")?;
    let status: String = r.get("status")?;
    Ok(GenerationRow {
        id: r.get("id")?,
        kind: r.get("kind")?,
        model_id: r.get("model_id")?,
        model_name: r.get("model_name")?,
        prompt: r.get("prompt")?,
        negative_prompt: r.get("negative_prompt")?,
        params: from_json(&params),
        outputs: from_json(&outputs),
        status: JobStatus::parse(&status),
        error: r.get("error")?,
        duration_ms: r.get("duration_ms")?,
        created_at: r.get("created_at")?,
    })
}

pub fn insert_generation(c: &Connection, g: &GenerationRow) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO generations (id, kind, model_id, model_name, prompt, negative_prompt, params_json, outputs_json,
                                  status, error, duration_ms, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            g.id,
            g.kind,
            g.model_id,
            g.model_name,
            g.prompt,
            g.negative_prompt,
            to_json(&g.params),
            to_json(&g.outputs),
            g.status.as_str(),
            g.error,
            g.duration_ms,
            g.created_at
        ],
    )?;
    Ok(())
}

pub fn update_generation(c: &Connection, g: &GenerationRow) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE generations SET outputs_json = ?2, status = ?3, error = ?4, duration_ms = ?5 WHERE id = ?1",
        params![g.id, to_json(&g.outputs), g.status.as_str(), g.error, g.duration_ms],
    )
}

pub fn get_generation(c: &Connection, id: &str) -> rusqlite::Result<Option<GenerationRow>> {
    c.query_row("SELECT * FROM generations WHERE id = ?1", params![id], generation_from_row).optional()
}

pub fn list_generations(c: &Connection, limit: usize) -> rusqlite::Result<Vec<GenerationRow>> {
    let mut stmt = c.prepare("SELECT * FROM generations ORDER BY created_at DESC LIMIT ?1")?;
    let rows = stmt.query_map(params![limit as i64], generation_from_row)?;
    rows.collect()
}

pub fn delete_generation(c: &Connection, id: &str) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM generations WHERE id = ?1", params![id])
}

/// A timestamped piece of a transcript.
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptionRow {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub file_name: String,
    /// Stored upload, relative to the uploads directory.
    pub stored_file: String,
    pub language: Option<String>,
    pub task: String,
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub status: JobStatus,
    pub error: Option<String>,
    pub duration_ms: Option<i64>,
    pub created_at: String,
}

fn transcription_from_row(r: &Row<'_>) -> rusqlite::Result<TranscriptionRow> {
    let segments: String = r.get("segments_json")?;
    let status: String = r.get("status")?;
    Ok(TranscriptionRow {
        id: r.get("id")?,
        model_id: r.get("model_id")?,
        model_name: r.get("model_name")?,
        file_name: r.get("file_name")?,
        stored_file: r.get("stored_file")?,
        language: r.get("language")?,
        task: r.get("task")?,
        text: r.get("text")?,
        segments: from_json(&segments),
        status: JobStatus::parse(&status),
        error: r.get("error")?,
        duration_ms: r.get("duration_ms")?,
        created_at: r.get("created_at")?,
    })
}

pub fn insert_transcription(c: &Connection, t: &TranscriptionRow) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO transcriptions (id, model_id, model_name, file_name, stored_file, language, task, text,
                                     segments_json, status, error, duration_ms, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            t.id,
            t.model_id,
            t.model_name,
            t.file_name,
            t.stored_file,
            t.language,
            t.task,
            t.text,
            to_json(&t.segments),
            t.status.as_str(),
            t.error,
            t.duration_ms,
            t.created_at
        ],
    )?;
    Ok(())
}

pub fn update_transcription(c: &Connection, t: &TranscriptionRow) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE transcriptions SET language = ?2, text = ?3, segments_json = ?4, status = ?5, error = ?6,
                duration_ms = ?7 WHERE id = ?1",
        params![t.id, t.language, t.text, to_json(&t.segments), t.status.as_str(), t.error, t.duration_ms],
    )
}

pub fn get_transcription(c: &Connection, id: &str) -> rusqlite::Result<Option<TranscriptionRow>> {
    c.query_row("SELECT * FROM transcriptions WHERE id = ?1", params![id], transcription_from_row).optional()
}

pub fn list_transcriptions(c: &Connection, limit: usize) -> rusqlite::Result<Vec<TranscriptionRow>> {
    let mut stmt = c.prepare("SELECT * FROM transcriptions ORDER BY created_at DESC LIMIT ?1")?;
    let rows = stmt.query_map(params![limit as i64], transcription_from_row)?;
    rows.collect()
}

pub fn delete_transcription(c: &Connection, id: &str) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM transcriptions WHERE id = ?1", params![id])
}

/// Marks jobs interrupted by a shutdown as failed.
pub fn fail_interrupted(c: &Connection) -> rusqlite::Result<usize> {
    let msg = "Interrupted: the app was closed before this finished.";
    let a = c.execute(
        "UPDATE generations SET status = 'failed', error = ?1 WHERE status IN ('queued', 'running')",
        params![msg],
    )?;
    let b = c.execute(
        "UPDATE transcriptions SET status = 'failed', error = ?1 WHERE status IN ('queued', 'running')",
        params![msg],
    )?;
    Ok(a + b)
}
