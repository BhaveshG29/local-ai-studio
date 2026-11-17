//! The Speech models page: Whisper Base (the model behind voice input) with its status and
//! buttons, the live transcription panel, and speech to text for recordings with any Whisper
//! model (upload audio, pick a language, get a transcript as TXT/SRT).

use std::time::Instant;

use askama::Template;
use axum::extract::{Multipart, Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::SharedState;
use crate::db::conversations::AttachmentKind;
use crate::db::downloads::DownloadState;
use crate::db::jobs::{JobStatus, TranscriptionRow};
use crate::error::{AppError, AppResult};
use crate::inference::dictation::{WHISPER_BASE_REPO, WhisperBaseStatus};
use crate::inference::whisper::{LANGUAGES, language_name};
use crate::util::{human_bytes, now_rfc3339, relative_time};
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
    /// Whisper Base, the model behind voice input and the live panel.
    pub whisper: WhisperBaseStatus,
    /// What the Whisper Base card shows beyond `whisper`, and which buttons it has.
    pub card: WhisperCard,
}

/// Why Whisper Base can or can't be used, decided in the order `whisper_base_status` picks
/// its `problem`: valid files first, then a download, then a library entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Readiness {
    Ready,
    /// The files are fine but the built-in engine can't run (e.g. CUDA chosen but broken).
    EngineUnavailable,
    /// The files are fine but loading them failed last time.
    LoadFailed,
    Queued,
    Downloading,
    Paused,
    DownloadFailed,
    /// In the library, but its folder is gone.
    FilesMissing,
    FilesDamaged,
    NotDownloaded,
}

impl Readiness {
    fn of(w: &WhisperBaseStatus, engine_ok: bool) -> Self {
        if w.ready {
            return Self::Ready;
        }
        if w.compatible {
            return if engine_ok { Self::LoadFailed } else { Self::EngineUnavailable };
        }
        match &w.download {
            Some(d) if d.active && d.status == "Queued" => Self::Queued,
            Some(d) if d.active => Self::Downloading,
            Some(d) if d.status == "Paused" => Self::Paused,
            Some(_) => Self::DownloadFailed,
            None if w.model_id.is_some() && !w.downloaded => Self::FilesMissing,
            None if w.model_id.is_some() => Self::FilesDamaged,
            None => Self::NotDownloaded,
        }
    }

    /// Ends "No, …" in the card's "Ready for transcription" row.
    fn reason(self) -> &'static str {
        match self {
            Self::Ready => "",
            Self::EngineUnavailable => "the built-in engine can't run right now",
            Self::LoadFailed => "it couldn't be loaded",
            Self::Queued => "it is waiting to download",
            Self::Downloading => "it is still downloading",
            Self::Paused => "its download is paused",
            Self::DownloadFailed => "its download failed",
            Self::FilesMissing => "its files are missing",
            Self::FilesDamaged => "its files are damaged",
            Self::NotDownloaded => "it isn't downloaded yet",
        }
    }

    /// `ok`, `info` (nothing wrong, just not there yet) or `error`.
    fn tone(self) -> &'static str {
        match self {
            Self::Ready => "ok",
            Self::Queued | Self::Downloading | Self::Paused | Self::NotDownloaded => "info",
            _ => "error",
        }
    }
}

/// The Whisper Base card on the Speech models page.
pub struct WhisperCard {
    /// `ok`, `info` or `error`: how the Ready / Not ready line looks.
    pub tone: &'static str,
    /// Icon of that line.
    pub icon: &'static str,
    /// The sentence after Ready / Not ready.
    pub summary: String,
    /// "Ready for transcription": why not ("" when ready).
    pub reason: &'static str,
    /// A download is waiting or running: its progress shows in a frame that updates itself.
    pub downloading: bool,
    /// "Download status" when no download is running.
    pub download_status: String,
    /// Where a paused download stopped, for its bar.
    pub paused_at: Option<u32>,
    /// The files are (or, not downloaded yet, will be) usable by the built-in engine.
    pub files_ok: bool,
    /// Where the files will be saved, while it isn't downloaded.
    pub save_dir: String,
    /// "Download" or "Resume download"; `None` hides the button.
    pub download_button: Option<&'static str>,
    pub can_cancel: bool,
    pub can_load: bool,
    pub can_unload: bool,
    pub can_delete: bool,
    /// The engine choice in Settings is what stops it (Load can't help): link there instead.
    pub settings_link: bool,
    /// The library's page about the model.
    pub details_href: Option<String>,
    /// What deleting does, in the confirmation.
    pub delete_note: String,
}

impl WhisperCard {
    fn new(w: &WhisperBaseStatus, engine_ok: bool, save_dir: String) -> Self {
        let why = Readiness::of(w, engine_ok);
        let active = w.download.as_ref().is_some_and(|d| d.active);
        let (icon, summary) = match why {
            Readiness::Ready if w.loaded => {
                ("check", "Loaded and ready for voice input in chat and the live panel.".into())
            }
            Readiness::Ready => ("check", "Voice input in chat and the live panel can use it now.".into()),
            // The percentage in `problem` would go stale: the frame below shows it live.
            Readiness::Downloading => {
                ("download", "Whisper Base is downloading. Voice input works as soon as it is done.".into())
            }
            Readiness::Queued | Readiness::Paused => ("download", w.problem.clone().unwrap_or_default()),
            Readiness::NotDownloaded => ("info", w.problem.clone().unwrap_or_default()),
            _ => ("alert", w.problem.clone().unwrap_or_default()),
        };
        // A paused or failed download says more than the library entry it would replace: that
        // entry's folder may hold the download's first small files, which aren't Whisper Base.
        let stopped = w.download.as_ref().filter(|d| !d.active && !w.compatible);
        let download_status = match (&w.download, stopped) {
            (_, Some(d)) if d.status == "Paused" => format!("Paused at {}%", d.percent),
            (_, Some(_)) => "Failed".to_string(),
            _ if w.downloaded => "Downloaded".to_string(),
            (Some(d), _) => d.status.clone(),
            (None, _) if w.model_id.is_some() => "Files missing".to_string(),
            (None, _) => "Not downloaded".to_string(),
        };
        let has_model = w.model_id.is_some();
        WhisperCard {
            tone: why.tone(),
            icon,
            summary,
            reason: why.reason(),
            downloading: active,
            download_status,
            paused_at: stopped.filter(|d| d.status == "Paused").map(|d| d.percent),
            files_ok: w.compatible || !has_model,
            save_dir,
            // A stopped download carries on where it stopped, also when the library entry it
            // would replace has lost its files (the status line then says "Resume it").
            download_button: if stopped.is_some() {
                Some("Resume download")
            } else {
                (w.download.is_none() && !has_model).then_some("Download")
            },
            can_cancel: w.download.is_some(),
            can_load: w.downloaded && w.compatible && !w.loaded && why != Readiness::EngineUnavailable,
            can_unload: w.loaded,
            can_delete: has_model,
            settings_link: why == Readiness::EngineUnavailable,
            details_href: w.model_ref.as_ref().map(|r| format!("/library/model/{r}")),
            delete_note: if w.downloaded {
                format!(
                    "Its files ({}) are removed from this computer. Voice input stops working until you download it again.",
                    w.size_label
                )
            } else {
                "Its files are already gone. This removes it from the library, so you can download it again.".into()
            },
        }
    }
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
    // No auto-refresh here: it would clear a file the user is choosing and stop a recording
    // in the live panel. The running job links to its own page, which does refresh, and a
    // download of Whisper Base shows its progress in a frame that refreshes by itself.
    let ctx = PageCtx::new(&state, &headers, "Speech models", "transcribe").at("/transcribe");
    let mut languages = vec![Opt { value: "auto".into(), label: "Detect automatically".into(), selected: true }];
    languages.extend(LANGUAGES.iter().map(|(c, n)| Opt { value: (*c).into(), label: (*n).into(), selected: false }));
    let whisper = crate::inference::dictation::whisper_base_status(&state).await;
    let save_dir = state.downloads.repo_dir(WHISPER_BASE_REPO).to_string_lossy().into_owned();
    let card = WhisperCard::new(&whisper, state.inference.device_label().is_ok(), save_dir);
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
        whisper,
        card,
    };
    render(&ctx, &page)
}

/// A small self-refreshing page shown in a frame on the Speech models page while Whisper
/// Base downloads, so the page itself never reloads (a file being chosen or a recording in
/// the live panel would be lost). Once the download ends it stops refreshing and offers to
/// reload the page.
#[derive(Template)]
#[template(path = "speech_progress.html")]
pub struct WhisperProgressFrame {
    pub theme: &'static str,
    pub dark: bool,
    pub css_rev: &'static str,
    /// Still downloading (or waiting to): the frame reloads itself every 2 seconds.
    pub active: bool,
    /// Waiting for its turn: no percentage yet.
    pub waiting: bool,
    pub title: String,
    pub percent: u32,
    /// Bytes, speed and time left while active; once it ended, what reloading the page is
    /// for (it follows the "Reload the page" button: "… to resume it.").
    pub detail: String,
    /// `ok`, `info` or `error`: the outcome's icon.
    pub tone: &'static str,
}

/// `GET /speech/whisper-base/progress`: the download progress frame.
pub async fn whisper_progress(State(state): State<SharedState>) -> Response {
    let w = crate::inference::dictation::whisper_base_status(&state).await;
    let jobs = state.downloads.list().await.unwrap_or_default();
    let job = w.download.as_ref().and_then(|d| jobs.iter().find(|j| j.row.id == d.id));
    let last_state = jobs
        .iter()
        .filter(|j| j.row.repo_id == WHISPER_BASE_REPO)
        .max_by(|a, b| a.row.created_at.cmp(&b.row.created_at))
        .map(|j| j.row.state);
    let mut frame = WhisperProgressFrame {
        theme: "",
        dark: false,
        css_rev: crate::web::assets::stylesheet_rev(),
        active: false,
        waiting: false,
        title: String::new(),
        percent: w.download.as_ref().map(|d| d.percent).unwrap_or(0),
        detail: String::new(),
        tone: "info",
    };
    match (&w.download, job) {
        (Some(d), job) if d.active => {
            frame.active = true;
            frame.waiting = d.status == "Queued";
            let verifying = job.is_some_and(|j| j.row.state == DownloadState::Verifying);
            frame.title = if frame.waiting {
                "Waiting to download Whisper Base…".into()
            } else if verifying {
                "Checking the downloaded files…".into()
            } else {
                "Downloading Whisper Base…".into()
            };
            frame.detail = match job {
                Some(j) if j.total_bytes > 0 && !frame.waiting => {
                    let mut parts = vec![format!("{} of {}", human_bytes(j.done_bytes), human_bytes(j.total_bytes))];
                    parts.extend([j.speed_label(), j.eta_label()].into_iter().filter(|s| !s.is_empty()));
                    parts.join(" · ")
                }
                _ if frame.waiting => "It starts when another download finishes.".into(),
                _ => "Starting…".into(),
            };
        }
        _ if w.downloaded && w.compatible => {
            frame.tone = "ok";
            frame.title = "Whisper Base is downloaded.".into();
            frame.detail = "to start using voice input.".into();
        }
        // How the download ended comes before "downloaded but unusable": a download that
        // stopped may have left its first small files where an old library entry points.
        (Some(d), _) if d.status == "Paused" => {
            frame.title = format!("The download is paused at {}%.", d.percent);
            frame.detail = "to resume it.".into();
        }
        (Some(_), _) => {
            frame.tone = "error";
            frame.title = "The download failed.".into();
            frame.detail = "to see why and try again.".into();
        }
        _ if last_state == Some(DownloadState::Cancelled) => {
            frame.title = "The download was cancelled.".into();
            frame.detail = "to see where it stands.".into();
        }
        _ if w.downloaded => {
            frame.tone = "error";
            frame.title = "Downloaded, but the files can't be used.".into();
            frame.detail = "to see what is wrong.".into();
        }
        (None, _) => {
            frame.title = "Whisper Base isn't downloading.".into();
            frame.detail = "to see where it stands.".into();
        }
    }
    let theme = state.settings().ui.theme;
    frame.theme = theme.slug();
    frame.dark = theme.is_dark();
    match frame.render() {
        Ok(html) => {
            let mut r = axum::response::Html(html).into_response();
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
            r
        }
        Err(e) => AppError::Internal(anyhow::anyhow!("template error: {e}")).into_response(),
    }
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
    use crate::inference::dictation::DownloadView;

    #[test]
    fn time_formats() {
        assert_eq!(clock(3725.4), "01:02:05");
        assert_eq!(srt_time(62.5), "00:01:02,500");
    }

    // ---- the Whisper Base card in every state ----------------------------------------------------

    fn nothing() -> WhisperBaseStatus {
        WhisperBaseStatus {
            name: "Whisper Base".into(),
            repo: WHISPER_BASE_REPO.into(),
            size_label: "279 MB".into(),
            problem: Some("Whisper Base isn't downloaded yet.".into()),
            ..Default::default()
        }
    }

    /// In the library; `ok`: its files pass the check.
    fn in_library(ok: bool) -> WhisperBaseStatus {
        WhisperBaseStatus {
            model_id: Some("m1".into()),
            model_ref: Some("local:m1".into()),
            downloaded: true,
            path: Some("/models/openai/whisper-base".into()),
            compatible: ok,
            ready: ok,
            problem: (!ok).then(|| "Whisper Base's files are damaged: config.json is missing.".into()),
            ..nothing()
        }
    }

    fn download(status: &str, active: bool) -> Option<DownloadView> {
        Some(DownloadView { id: "d1".into(), percent: 40, status: status.into(), active })
    }

    /// The card's buttons and links, in the order the template shows them.
    fn buttons(c: &WhisperCard) -> Vec<&'static str> {
        let mut b: Vec<&'static str> = c.download_button.into_iter().collect();
        for (on, label) in [
            (c.can_cancel, "Cancel download"),
            (c.can_load, "Load"),
            (c.can_unload, "Unload"),
            (c.settings_link, "Open Settings"),
            (c.details_href.is_some(), "Model details"),
            (c.can_delete, "Delete…"),
        ] {
            if on {
                b.push(label);
            }
        }
        b
    }

    fn card(w: &WhisperBaseStatus, engine_ok: bool) -> WhisperCard {
        WhisperCard::new(w, engine_ok, "/models/openai/whisper-base".into())
    }

    #[test]
    fn whisper_card_before_and_during_a_download() {
        let c = card(&nothing(), true);
        assert_eq!(buttons(&c), ["Download"]);
        assert_eq!(
            (c.tone, c.reason, c.download_status.as_str()),
            ("info", "it isn't downloaded yet", "Not downloaded")
        );
        assert!(c.files_ok && !c.downloading && c.paused_at.is_none());

        let w = WhisperBaseStatus {
            download: download("Queued", true),
            problem: Some("Whisper Base is waiting to download.".into()),
            ..nothing()
        };
        let c = card(&w, true);
        assert_eq!(buttons(&c), ["Cancel download"]);
        assert!(c.downloading);
        assert_eq!(
            (c.tone, c.reason, c.summary.as_str()),
            ("info", "it is waiting to download", "Whisper Base is waiting to download.")
        );

        let w = WhisperBaseStatus {
            download: download("Downloading", true),
            problem: Some("Whisper Base is downloading (40%).".into()),
            ..nothing()
        };
        let c = card(&w, true);
        assert_eq!(buttons(&c), ["Cancel download"]);
        assert!(c.downloading);
        assert!(!c.summary.contains('%'), "the percentage would go stale: {}", c.summary);

        let w = WhisperBaseStatus { download: download("Paused", false), ..nothing() };
        let c = card(&w, true);
        assert_eq!(buttons(&c), ["Resume download", "Cancel download"]);
        assert_eq!((c.download_status.as_str(), c.paused_at), ("Paused at 40%", Some(40)));
        assert_eq!((c.tone, c.reason), ("info", "its download is paused"));

        let w = WhisperBaseStatus { download: download("Failed", false), ..nothing() };
        let c = card(&w, true);
        assert_eq!(buttons(&c), ["Resume download", "Cancel download"]);
        assert_eq!((c.tone, c.download_status.as_str(), c.paused_at), ("error", "Failed", None));
    }

    #[test]
    fn whisper_card_once_downloaded() {
        let c = card(&in_library(true), true);
        assert_eq!(buttons(&c), ["Load", "Model details", "Delete…"]);
        assert_eq!((c.tone, c.download_status.as_str()), ("ok", "Downloaded"));
        assert_eq!(c.details_href.as_deref(), Some("/library/model/local:m1"));
        assert!(c.delete_note.contains("279 MB"), "{}", c.delete_note);

        let c = card(&WhisperBaseStatus { loaded: true, ..in_library(true) }, true);
        assert_eq!(buttons(&c), ["Unload", "Model details", "Delete…"]);
        assert!(c.summary.starts_with("Loaded"), "{}", c.summary);

        // Valid files, but loading them failed: Load tries again.
        let w = WhisperBaseStatus {
            ready: false,
            problem: Some("Whisper Base couldn't be loaded: out of memory. Press Load to try again.".into()),
            ..in_library(true)
        };
        let c = card(&w, true);
        assert_eq!(buttons(&c), ["Load", "Model details", "Delete…"]);
        assert_eq!((c.tone, c.reason), ("error", "it couldn't be loaded"));

        // Valid files, but the engine chosen in Settings can't run: Load can't help, Settings can.
        let w = WhisperBaseStatus {
            ready: false,
            problem: Some("Voice input can't run right now. NVIDIA CUDA can't be used.".into()),
            ..in_library(true)
        };
        let c = card(&w, false);
        assert_eq!(buttons(&c), ["Open Settings", "Model details", "Delete…"]);
        assert_eq!((c.tone, c.reason), ("error", "the built-in engine can't run right now"));
    }

    #[test]
    fn whisper_card_with_damaged_or_missing_files() {
        let c = card(&in_library(false), true);
        assert_eq!(buttons(&c), ["Model details", "Delete…"]);
        assert_eq!((c.tone, c.reason, c.files_ok), ("error", "its files are damaged", false));
        assert_eq!(c.download_status, "Downloaded");

        let missing = WhisperBaseStatus { downloaded: false, ..in_library(false) };
        let c = card(&missing, true);
        assert_eq!(buttons(&c), ["Model details", "Delete…"]);
        assert_eq!((c.reason, c.download_status.as_str()), ("its files are missing", "Files missing"));
        assert!(c.delete_note.starts_with("Its files are already gone."), "{}", c.delete_note);

        // Downloaded again from the Marketplace after the folder went missing, then paused (as
        // every download is when the app closes): the status says "Resume it", so the button
        // must be there.
        let w = WhisperBaseStatus {
            download: download("Paused", false),
            problem: Some("Whisper Base's download is paused at 40%. Resume it to use voice input.".into()),
            ..missing
        };
        let c = card(&w, true);
        assert_eq!(buttons(&c), ["Resume download", "Cancel download", "Model details", "Delete…"]);
        assert_eq!((c.tone, c.reason), ("info", "its download is paused"));
        assert_eq!((c.download_status.as_str(), c.paused_at), ("Paused at 40%", Some(40)));

        // That download already put its small files back in the folder, so the library entry
        // counts as downloaded (but unusable): the download still tells where things stand.
        let w = WhisperBaseStatus { downloaded: true, ..w };
        let c = card(&w, true);
        assert_eq!((c.download_status.as_str(), c.paused_at), ("Paused at 40%", Some(40)));
        assert_eq!(buttons(&c), ["Resume download", "Cancel download", "Model details", "Delete…"]);
        let failed = WhisperBaseStatus { download: download("Failed", false), ..w };
        assert_eq!(card(&failed, true).download_status, "Failed");

        // A usable Whisper Base wins over a stale stopped download.
        let c = card(&WhisperBaseStatus { download: download("Failed", false), ..in_library(true) }, true);
        assert_eq!((c.download_status.as_str(), c.paused_at), ("Downloaded", None));
        assert_eq!(c.download_button, None);
    }
}
