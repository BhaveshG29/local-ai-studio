//! Speech to text with Whisper: upload audio, pick a language, get a transcript (TXT/SRT).

use std::time::Instant;

use askama::Template;
use axum::extract::{Multipart, Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::SharedState;
use crate::db::conversations::AttachmentKind;
use crate::db::jobs::{JobStatus, TranscriptionRow};
use crate::error::{AppError, AppResult};
use crate::inference::whisper::{LANGUAGES, language_name};
use crate::util::{now_rfc3339, relative_time};
use crate::web::page::{PageCtx, redirect, redirect_flash, render};

pub struct Opt {
    pub value: String,
    pub label: String,
    pub selected: bool,
}

pub struct TxView {
    pub id: String,
    pub file_name: String,
    pub model: String,
    pub language: String,
    pub task: &'static str,
    pub created: String,
    pub status: &'static str,
    pub status_class: &'static str,
    pub error: String,
    pub preview: String,
    pub duration: String,
}

pub struct RunningView {
    pub id: String,
    pub label: String,
    pub percent: u32,
}

#[derive(Template)]
#[template(path = "transcribe.html")]
pub struct TranscribePage {
    pub ctx: PageCtx,
    pub models: Vec<Opt>,
    pub has_models: bool,
    pub languages: Vec<Opt>,
    pub running: Option<RunningView>,
    pub history: Vec<TxView>,
    pub ffmpeg_available: bool,
}

fn view(t: &TranscriptionRow) -> TxView {
    let (status, class) = match t.status {
        JobStatus::Queued | JobStatus::Running => ("Working", "info"),
        JobStatus::Done => ("Done", "ok"),
        JobStatus::Failed => ("Failed", "danger"),
        JobStatus::Cancelled => ("Cancelled", "muted"),
    };
    TxView {
        id: t.id.clone(),
        file_name: t.file_name.clone(),
        model: t.model_name.clone(),
        language: t.language.as_deref().map(language_name).unwrap_or("Auto").to_string(),
        task: if t.task == "translate" { "Translated to English" } else { "Transcribed" },
        created: relative_time(&t.created_at),
        status,
        status_class: class,
        error: t.error.clone().unwrap_or_default(),
        preview: crate::util::truncate_chars(&t.text, 280),
        duration: t.duration_ms.map(|d| format!("{:.1} s", d as f64 / 1000.0)).unwrap_or_default(),
    }
}

pub async fn page(State(state): State<SharedState>, headers: HeaderMap) -> Response {
    let models: Vec<_> =
        super::models_of_kind(&state, "transcription").await.into_iter().filter(|m| m.analysis.runnable).collect();
    let rows = state.db.call(|c| crate::db::jobs::list_transcriptions(c, 30)).await.unwrap_or_default();
    let running = rows
        .iter()
        .find(|t| t.status.is_active())
        .and_then(|t| state.jobs.get(&t.id).map(|p| (t.id.clone(), p)))
        .map(|(id, p)| RunningView { id, label: p.label.clone(), percent: p.percent() });
    // No auto-refresh here: it would clear a file the user is choosing. The running job
    // links to its own page, which does refresh.
    let ctx = PageCtx::new(&state, &headers, "Speech to text", "transcribe").at("/transcribe");
    let mut languages = vec![Opt { value: "auto".into(), label: "Detect automatically".into(), selected: true }];
    languages.extend(LANGUAGES.iter().map(|(c, n)| Opt { value: (*c).into(), label: (*n).into(), selected: false }));
    let page = TranscribePage {
        ctx: ctx.clone(),
        has_models: !models.is_empty(),
        models: models
            .iter()
            .enumerate()
            .map(|(i, m)| Opt { value: m.id.clone(), label: m.display_name.clone(), selected: i == 0 })
            .collect(),
        languages,
        running,
        history: rows.iter().map(view).collect(),
        ffmpeg_available: crate::media::ffmpeg_path(&state.settings()).is_some(),
    };
    render(&ctx, &page)
}

pub async fn start(State(state): State<SharedState>, mut mp: Multipart) -> Response {
    let mut model_id = String::new();
    let mut language = String::new();
    let mut task = "transcribe".to_string();
    let mut upload: Option<(String, std::path::PathBuf)> = None;
    let uploads = state.paths.uploads_dir.clone();
    while let Ok(Some(mut field)) = mp.next_field().await {
        match field.name().unwrap_or("") {
            "model_id" => model_id = field.text().await.unwrap_or_default(),
            "language" => language = field.text().await.unwrap_or_default(),
            "task" => task = field.text().await.unwrap_or_default(),
            "file" => {
                let fname = field.file_name().unwrap_or("").to_string();
                if fname.is_empty() {
                    while let Ok(Some(_)) = field.chunk().await {}
                    continue;
                }
                match crate::web::pages::chat::save_upload(&mut field, &uploads, &fname).await {
                    Ok(a) if a.kind == AttachmentKind::Audio => {
                        upload = Some((a.name.clone(), uploads.join(&a.file)));
                    }
                    Ok(a) if a.kind == AttachmentKind::Video => {
                        let video = uploads.join(&a.file);
                        let ffmpeg = crate::media::ffmpeg_path(&state.settings());
                        let extracted = extract_audio(ffmpeg.as_deref(), &video).await;
                        // Only the sound is kept; the (often large) video isn't needed.
                        let _ = tokio::fs::remove_file(&video).await;
                        match extracted {
                            Ok(wav) => upload = Some((a.name.clone(), wav)),
                            Err(e) => return redirect_flash("/transcribe", "error", &e),
                        }
                    }
                    Ok(a) => {
                        let _ = tokio::fs::remove_file(uploads.join(&a.file)).await;
                        return redirect_flash(
                            "/transcribe",
                            "error",
                            "Please choose an audio file (WAV, MP3, FLAC, OGG, M4A) or a video.",
                        );
                    }
                    Err(e) => return redirect_flash("/transcribe", "error", &e.user_message()),
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    let Some((file_name, path)) = upload else {
        return redirect_flash("/transcribe", "error", "Choose an audio file first.");
    };
    match start_job(&state, &model_id, &language, &task, &file_name, path).await {
        Ok(id) => redirect(&format!("/transcribe/{id}")),
        Err(e) => redirect_flash("/transcribe", "error", &e.user_message()),
    }
}

/// Extracts the audio track of a video to 16 kHz mono WAV with ffmpeg.
async fn extract_audio(
    ffmpeg: Option<&std::path::Path>,
    video: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    let ffmpeg =
        ffmpeg.ok_or_else(|| format!("Transcribing video needs ffmpeg. {}", crate::media::ffmpeg_install_hint()))?;
    let out = video.with_extension("wav");
    let status = tokio::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(video)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-y"])
        .arg(&out)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .status()
        .await
        .map_err(|e| format!("Could not run ffmpeg: {e}"))?;
    if !status.success() || !out.exists() {
        return Err("This video has no audio track ffmpeg can read.".into());
    }
    Ok(out)
}

async fn start_job(
    state: &SharedState,
    model_id: &str,
    language: &str,
    task: &str,
    file_name: &str,
    path: std::path::PathBuf,
) -> AppResult<String> {
    let mid = model_id.to_string();
    let model = state
        .db
        .call(move |c| crate::db::models::get(c, &mid))
        .await?
        .filter(|m| m.kind == "transcription" && m.analysis.runnable)
        .ok_or_else(|| {
            AppError::bad_request("Choose a Whisper model (download one from Discover if the list is empty).")
        })?;
    let language = (!language.is_empty() && language != "auto")
        .then(|| language.to_string())
        .filter(|l| LANGUAGES.iter().any(|(c, _)| c == l));
    let translate = task == "translate";
    let stored =
        path.strip_prefix(&state.paths.uploads_dir).map(|p| p.to_string_lossy().replace('\\', "/")).unwrap_or_default();
    let row = TranscriptionRow {
        id: crate::util::new_id(),
        model_id: model.id.clone(),
        model_name: model.display_name.clone(),
        file_name: file_name.to_string(),
        stored_file: stored,
        language: language.clone(),
        task: if translate { "translate".into() } else { "transcribe".into() },
        text: String::new(),
        segments: Vec::new(),
        status: JobStatus::Running,
        error: None,
        duration_ms: None,
        created_at: now_rfc3339(),
    };
    let r = row.clone();
    state.db.call(move |c| crate::db::jobs::insert_transcription(c, &r)).await?;
    let cancel = state.jobs.start(&row.id, &format!("Transcribing {file_name}"));
    let st = state.clone();
    let id = row.id.clone();
    tokio::spawn(async move {
        let started = Instant::now();
        let mut row = row;
        let jid = row.id.clone();
        let st2 = st.clone();
        let progress = move |f: f32| {
            st2.jobs.update(&jid, |p| {
                p.step = (f * 100.0) as usize;
                p.total = 100;
            });
        };
        let res = st.inference.whisper.transcribe(&st, &model, &path, language, translate, cancel, progress).await;
        match res {
            Ok(out) => {
                row.text = out.text;
                row.segments = out.segments;
                row.language = out.language;
                row.status = JobStatus::Done;
            }
            Err(AppError::Conflict(_)) => row.status = JobStatus::Cancelled,
            Err(e) => {
                row.status = JobStatus::Failed;
                row.error = Some(e.user_message());
            }
        }
        row.duration_ms = Some(started.elapsed().as_millis() as i64);
        let r = row.clone();
        let _ = st.db.call(move |c| crate::db::jobs::update_transcription(c, &r)).await;
        st.jobs.finish(&row.id, row.error.clone());
        st.jobs.remove(&row.id);
    });
    Ok(id)
}

pub struct SegView {
    pub start: String,
    pub end: String,
    pub text: String,
}

#[derive(Template)]
#[template(path = "transcript.html")]
pub struct TranscriptPage {
    pub ctx: PageCtx,
    pub t: TxView,
    pub text: String,
    pub segments: Vec<SegView>,
    pub audio_url: String,
    pub running: bool,
    pub percent: u32,
}

fn clock(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

fn srt_time(secs: f64) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, (ms % 3_600_000) / 60_000, (ms % 60_000) / 1000, ms % 1000)
}

async fn load(state: &SharedState, id: &str) -> AppResult<TranscriptionRow> {
    let i = id.to_string();
    state
        .db
        .call(move |c| crate::db::jobs::get_transcription(c, &i))
        .await?
        .ok_or_else(|| AppError::not_found("That transcript doesn't exist."))
}

pub async fn detail(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> AppResult<Response> {
    let t = load(&state, &id).await?;
    let running = t.status.is_active();
    let percent = state.jobs.get(&id).map(|p| p.percent()).unwrap_or(0);
    let ctx = PageCtx::new(&state, &headers, &t.file_name, "transcribe")
        .at(&format!("/transcribe/{id}"))
        .with_refresh(running.then_some(2));
    let audio_url = if t.stored_file.is_empty() { String::new() } else { format!("/files/uploads/{}", t.stored_file) };
    let page = TranscriptPage {
        ctx: ctx.clone(),
        t: view(&t),
        text: t.text.clone(),
        segments: t
            .segments
            .iter()
            .map(|s| SegView { start: clock(s.start), end: clock(s.end), text: s.text.clone() })
            .collect(),
        audio_url,
        running,
        percent,
    };
    Ok(render(&ctx, &page))
}

pub async fn cancel(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    state.jobs.cancel(&id);
    redirect_flash(&format!("/transcribe/{id}"), "ok", "Cancelling…")
}

pub async fn delete(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    if let Ok(t) = load(&state, &id).await {
        if t.status.is_active() {
            return redirect_flash(&format!("/transcribe/{id}"), "error", "Cancel it first.");
        }
        if let Some(p) = crate::util::safe_join(&state.paths.uploads_dir, &t.stored_file) {
            let _ = tokio::fs::remove_file(p).await;
        }
        let _ = state.db.call(move |c| crate::db::jobs::delete_transcription(c, &id)).await;
    }
    redirect_flash("/transcribe", "ok", "Deleted.")
}

#[derive(Deserialize, Default)]
pub struct ExportQuery {
    #[serde(default)]
    format: Option<String>,
}

pub async fn export(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Query(q): Query<ExportQuery>,
) -> AppResult<Response> {
    let t = load(&state, &id).await?;
    let base = crate::util::sanitize_component(t.file_name.rsplit_once('.').map(|(a, _)| a).unwrap_or(&t.file_name));
    let (body, ext) = if q.format.as_deref() == Some("srt") {
        let mut s = String::new();
        for (i, seg) in t.segments.iter().enumerate() {
            s.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, srt_time(seg.start), srt_time(seg.end), seg.text));
        }
        (s, "srt")
    } else {
        (t.text.clone(), "txt")
    };
    let mut r = body.into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{base}.{ext}\"")) {
        r.headers_mut().insert(header::CONTENT_DISPOSITION, v);
    }
    Ok(r)
}

pub async fn to_chat(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    let t = match load(&state, &id).await {
        Ok(t) => t,
        Err(e) => return e.into_response(),
    };
    if t.text.trim().is_empty() {
        return redirect_flash(&format!("/transcribe/{id}"), "error", "There is no text to send yet.");
    }
    let text = format!("Here is a transcript of \"{}\":\n\n{}\n\nPlease summarise it.", t.file_name, t.text);
    match crate::web::pages::chat::new_conversation_with_text(&state, &format!("Transcript: {}", t.file_name), &text)
        .await
    {
        Ok(cid) => match state.chat.start(&state, &cid, crate::inference::chat::StartMode::Reply).await {
            Ok(_) => redirect(&format!("/chat/{cid}#bottom")),
            Err(e) => redirect_flash(&format!("/chat/{cid}"), "info", &e.user_message()),
        },
        Err(e) => redirect_flash(&format!("/transcribe/{id}"), "error", &e.user_message()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_formats() {
        assert_eq!(clock(3725.4), "01:02:05");
        assert_eq!(srt_time(62.5), "00:01:02,500");
    }
}
