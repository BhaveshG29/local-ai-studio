//! Live speech to text ("dictation") with Whisper Base, for the chat microphone and the
//! live panel on the Speech models page. Both use this one implementation:
//!
//! * the page records the microphone and sends raw audio chunks to the JSON API
//!   (`web::speech`), which hands them to a [`DictationHub`] session;
//! * each session's worker ([`session`]) feeds the shared Whisper slot
//!   (`InferenceManager::whisper`, also used by file transcription) through
//!   [`LoadedWhisper::transcribe_live`](super::whisper::LoadedWhisper::transcribe_live);
//! * [`stream`] decides what to transcribe and when text becomes final (voice detector,
//!   pauses, splits, hallucination guard);
//! * [`check`] verifies Whisper Base's files for [`whisper_base_status`].

pub mod check;
pub mod session;
pub mod stream;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use futures_util::future::BoxFuture;

pub use session::{DictationError, DictationHub, Limits, LiveEngine, SessionState, SessionStatus};
pub use stream::Tuning;

use crate::SharedState;
use crate::db::downloads::DownloadState;
use crate::db::models::ModelRow;
use crate::inference::whisper::{LANGUAGES, LivePass};
use crate::util::human_bytes;

/// Hugging Face repository of the model used for voice input.
pub const WHISPER_BASE_REPO: &str = "openai/whisper-base";
/// The English-only version, also accepted.
pub const WHISPER_BASE_EN_REPO: &str = "openai/whisper-base.en";
/// Download option id for Whisper models (see `hub::analyze_whisper`).
pub const WHISPER_VARIANT: &str = "whisper";
/// Size of Whisper Base's five files, shown before it is downloaded.
pub const WHISPER_BASE_BYTES: u64 = 293_075_182;

const FORMAT: &str = "Safetensors · runs on the built-in engine (Candle)";

/// What the chat composer's microphone button shows.
#[derive(Clone, Debug, Default)]
pub struct MicView {
    /// Whisper Base is downloaded, its files are valid and the engine can run it.
    pub ready: bool,
    /// Why voice input can't be used, as one sentence (empty when ready).
    pub reason: String,
    /// Whisper Base is being downloaded right now.
    pub downloading: bool,
    /// Its size, as the Speech models page shows it (e.g. "279 MB").
    pub size: String,
}

/// An unfinished download of Whisper Base.
#[derive(Clone, Debug, Default)]
pub struct DownloadView {
    pub id: String,
    pub percent: u32,
    /// "Downloading", "Queued", "Paused" or "Failed".
    pub status: String,
    /// Still running (not paused or failed).
    pub active: bool,
}

/// What the Speech models page shows about Whisper Base.
#[derive(Clone, Debug, Default)]
pub struct WhisperBaseStatus {
    /// "Whisper Base".
    pub name: String,
    /// Hugging Face repository it comes from.
    pub repo: String,
    /// Library model id once downloaded.
    pub model_id: Option<String>,
    /// Model reference for the details page: `/library/model/{model_ref}` (`local:<id>`).
    pub model_ref: Option<String>,
    /// The files are on disk (they may still be unusable; see `problem`).
    pub downloaded: bool,
    /// An unfinished download, if any.
    pub download: Option<DownloadView>,
    /// Size on disk, or the expected download size when not downloaded (e.g. "279 MB").
    pub size_label: String,
    /// E.g. "Safetensors · runs on the built-in engine (Candle)".
    pub format: String,
    /// Folder holding the files, when downloaded.
    pub path: Option<String>,
    /// The built-in engine can run these files (config, tokenizer and weights are valid).
    pub compatible: bool,
    /// One sentence about compatibility.
    pub compatibility: String,
    /// In memory right now.
    pub loaded: bool,
    /// Downloaded, valid and the engine is available: recording can start.
    pub ready: bool,
    /// Why it isn't ready (None when ready).
    pub problem: Option<String>,
    /// Where it runs, e.g. "CPU" or "NVIDIA GPU · CUDA".
    pub device: String,
    /// The model knows many languages (false for the English-only `whisper-base.en`).
    pub multilingual: bool,
}

/// The library model used as Whisper Base.
#[derive(Clone, Debug)]
pub struct WhisperBaseModel {
    pub row: ModelRow,
    pub dir: PathBuf,
    pub fingerprint: check::Fingerprint,
    pub check: check::FileCheck,
}

/// Finds Whisper Base in the library: a transcription model from `openai/whisper-base`
/// (preferred) or `openai/whisper-base.en`, or any other Whisper model whose `config.json`
/// has Base's dimensions. A model whose files pass the check wins over one whose don't.
pub async fn find_whisper_base(state: &SharedState) -> Option<WhisperBaseModel> {
    let rows: Vec<ModelRow> =
        state.db.call(crate::db::models::list).await.ok()?.into_iter().filter(|m| m.kind == "transcription").collect();
    if rows.is_empty() {
        return None;
    }
    let st = state.clone();
    tokio::task::spawn_blocking(move || {
        let mut found: Vec<(bool, u8, WhisperBaseModel)> = rows
            .into_iter()
            .filter_map(|row| {
                let rank = match row.repo_id.as_deref() {
                    Some(WHISPER_BASE_REPO) => 0,
                    Some(WHISPER_BASE_EN_REPO) => 1,
                    _ => 2,
                };
                let dir = PathBuf::from(&row.path);
                let (fingerprint, check) = st.dictation.checks.check(&dir);
                (rank < 2 || check.base_size)
                    .then(|| (!check.ok(), rank, WhisperBaseModel { row, dir, fingerprint, check }))
            })
            .collect();
        found.sort_by_key(|(broken, rank, _)| (*broken, *rank));
        found.into_iter().next().map(|(_, _, m)| m)
    })
    .await
    .ok()
    .flatten()
}

/// An unfinished download of Whisper Base.
struct Unfinished {
    view: DownloadView,
    error: Option<String>,
    total_bytes: u64,
}

/// Whisper Base's download that isn't finished (running, waiting or paused), or else its
/// latest download if that one failed.
async fn unfinished_download(state: &SharedState) -> Option<Unfinished> {
    let jobs: Vec<_> =
        state.downloads.list().await.ok()?.into_iter().filter(|j| j.row.repo_id == WHISPER_BASE_REPO).collect();
    let latest = jobs
        .iter()
        .filter(|j| !j.row.state.is_finished())
        .max_by(|a, b| a.row.created_at.cmp(&b.row.created_at))
        .or_else(|| jobs.iter().max_by(|a, b| a.row.created_at.cmp(&b.row.created_at)))?;
    let status = match latest.row.state {
        DownloadState::Queued => "Queued",
        DownloadState::Downloading | DownloadState::Verifying => "Downloading",
        DownloadState::Paused => "Paused",
        DownloadState::Failed => "Failed",
        DownloadState::Completed | DownloadState::Cancelled => return None,
    };
    let view = DownloadView {
        id: latest.row.id.clone(),
        percent: latest.percent.floor().clamp(0.0, 100.0) as u32,
        status: status.into(),
        active: latest.row.state.is_active(),
    };
    Some(Unfinished { view, error: latest.row.error.clone(), total_bytes: latest.total_bytes })
}

/// Status plus the model it describes.
pub async fn status_and_model(state: &SharedState) -> (WhisperBaseStatus, Option<WhisperBaseModel>) {
    let model = find_whisper_base(state).await;
    let download = unfinished_download(state).await;
    let device = state.inference.device_label();
    let mut s = WhisperBaseStatus {
        name: "Whisper Base".into(),
        repo: WHISPER_BASE_REPO.into(),
        format: FORMAT.into(),
        size_label: human_bytes(WHISPER_BASE_BYTES),
        multilingual: true,
        ..Default::default()
    };
    if let Some(d) = &download {
        if d.total_bytes > 0 {
            s.size_label = human_bytes(d.total_bytes);
        }
        s.download = Some(d.view.clone());
    }
    if let Some(m) = &model {
        s.model_id = Some(m.row.id.clone());
        s.model_ref = Some(format!("local:{}", m.row.id));
        if let Some(repo) = &m.row.repo_id {
            s.repo = repo.clone();
        }
        s.downloaded = !m.check.missing;
        s.path = Some(m.row.path.clone());
        if !m.check.missing {
            s.size_label =
                human_bytes(if m.check.size_bytes > 0 { m.check.size_bytes } else { m.row.size_bytes as u64 });
        }
        s.compatible = m.check.ok();
        // Unknown for damaged files: assume the usual multilingual model.
        s.multilingual = !m.check.ok() || m.check.multilingual;
        s.loaded = state.inference.whisper.loaded_id().as_deref() == Some(m.row.id.as_str());
    }
    s.device = match (s.loaded, &device) {
        (true, _) => state.inference.whisper.loaded_device().unwrap_or_default(),
        (false, Ok(label)) => label.clone(),
        (false, Err(_)) => "Not available".into(),
    };
    s.compatibility = match &model {
        Some(m) if m.check.ok() => "Works with the built-in engine (Candle); nothing else to install.".into(),
        Some(m) => format!("Can't be used: {}.", m.check.problem.as_deref().unwrap_or("the files are damaged")),
        None => "Runs on the built-in engine (Candle) that comes with Local AI Studio.".into(),
    };
    let load_failure = model.as_ref().and_then(|m| state.dictation.checks.load_failure(&m.dir, &m.fingerprint));
    let download = download.map(|d| (d.view, d.error));
    s.problem = match (&model, &download) {
        (Some(m), _) if m.check.ok() => match (&device, load_failure) {
            (Err(e), _) => Some(format!("Voice input can't run right now. {e}")),
            (Ok(_), Some(e)) => Some(format!(
                "Whisper Base couldn't be loaded: {e}. Press Load to try again, or delete it and download it again."
            )),
            (Ok(_), None) => None,
        },
        (_, Some((d, _))) if d.active && d.status == "Queued" => Some("Whisper Base is waiting to download.".into()),
        (_, Some((d, _))) if d.active => Some(format!("Whisper Base is downloading ({}%).", d.percent)),
        (_, Some((d, _))) if d.status == "Paused" => {
            Some(format!("Whisper Base's download is paused at {}%. Resume it to use voice input.", d.percent))
        }
        (_, Some((_, error))) => Some(match error {
            Some(e) => format!("Whisper Base's download stopped: {}. Try again.", e.trim().trim_end_matches('.')),
            None => "Whisper Base's download stopped. Try again.".into(),
        }),
        (Some(m), None) if m.check.missing => Some(
            "Whisper Base's files are missing (the folder was moved or deleted). Delete it and download it again."
                .into(),
        ),
        (Some(m), None) => Some(format!(
            "Whisper Base's files are damaged: {}. Delete it and download it again.",
            m.check.problem.as_deref().unwrap_or("they can't be read")
        )),
        (None, None) => Some("Whisper Base isn't downloaded yet.".into()),
    };
    s.ready = s.problem.is_none();
    (s, model)
}

/// Everything about Whisper Base the UI needs.
pub async fn whisper_base_status(state: &SharedState) -> WhisperBaseStatus {
    status_and_model(state).await.0
}

/// The chat microphone's state.
pub async fn mic_view(state: &SharedState) -> MicView {
    let s = whisper_base_status(state).await;
    MicView {
        ready: s.ready,
        // The chat has no Load button: say where it is.
        reason: s
            .problem
            .unwrap_or_default()
            .replace("Press Load to try again", "Press Load on the Speech models page to try again"),
        downloading: s.download.as_ref().is_some_and(|d| d.active),
        size: s.size_label,
    }
}

/// `"auto"` (or nothing) means "detect it"; anything else must be a Whisper language code.
pub fn parse_language(language: &str) -> Result<Option<String>, DictationError> {
    let l = language.trim().to_lowercase();
    if l.is_empty() || l == "auto" {
        return Ok(None);
    }
    if LANGUAGES.iter().any(|(code, _)| *code == l) {
        Ok(Some(l))
    } else {
        Err(DictationError::BadRequest(format!(
            "\"{language}\" isn't a language Whisper knows. Use \"auto\" or a language code such as \"en\"."
        )))
    }
}

/// Whisper Base in the shared slot of `InferenceManager::whisper`.
struct BaseEngine {
    state: SharedState,
    model_id: String,
    dir: PathBuf,
}

impl LiveEngine for BaseEngine {
    fn prepare(&self) -> BoxFuture<'static, Result<(), String>> {
        let (state, id, dir) = (self.state.clone(), self.model_id.clone(), self.dir.clone());
        Box::pin(async move {
            match state.inference.whisper.get_or_load(&state, &id, &dir).await {
                Ok(_) => {
                    state.dictation.checks.clear_load_failure(&dir);
                    Ok(())
                }
                Err(e) => {
                    state.dictation.checks.remember_load_failure(&dir, &e);
                    Err(format!("Whisper Base couldn't be loaded: {e}"))
                }
            }
        })
    }

    fn transcribe(
        &self,
        pcm: Vec<f32>,
        language: Option<String>,
        full_window: bool,
    ) -> BoxFuture<'static, Result<Option<LivePass>, String>> {
        let (state, id, dir) = (self.state.clone(), self.model_id.clone(), self.dir.clone());
        Box::pin(async move {
            // Another Whisper model may have taken the slot (a file transcription): swap back.
            let model = state
                .inference
                .whisper
                .get_or_load(&state, &id, &dir)
                .await
                .map_err(|e| format!("Whisper Base couldn't be loaded again: {e}"))?;
            tokio::task::spawn_blocking(move || {
                let mut m = match model.try_lock() {
                    Ok(m) => m,
                    // A file transcription is using it: keep the audio and try again soon.
                    Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                    Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                };
                m.transcribe_live(&pcm, language.as_deref(), full_window).map(Some).map_err(|e| format!("{e:#}"))
            })
            .await
            .map_err(|e| {
                tracing::error!("a voice input pass crashed: {e}");
                "the speech model crashed".to_string()
            })?
        })
    }

    fn keep_alive(&self) {
        self.state.inference.whisper.touch(&self.model_id);
    }
}

/// Opens a live session on Whisper Base. `language`: "auto" or a Whisper language code.
pub async fn start_session(state: &SharedState, language: &str) -> Result<SessionState, DictationError> {
    let language = parse_language(language)?;
    let (status, model) = status_and_model(state).await;
    let model = match model {
        Some(m) if status.ready => m,
        _ => {
            return Err(DictationError::Unavailable(
                status.problem.unwrap_or_else(|| "Whisper Base isn't ready.".into()),
            ));
        }
    };
    let model_ready = status.loaded;
    let engine = Arc::new(BaseEngine { state: state.clone(), model_id: model.row.id.clone(), dir: model.dir.clone() });
    state.dictation.start_with_engine(engine, language, model_ready)
}

/// Loads Whisper Base and transcribes a second of silence to prove it works (the Speech
/// page's Load button). Returns a sentence for people either way.
pub async fn load_whisper_base(state: &SharedState) -> Result<String, String> {
    let (status, model) = status_and_model(state).await;
    let model = match model {
        Some(m) if m.check.ok() => m,
        _ => return Err(status.problem.unwrap_or_else(|| "Whisper Base isn't downloaded yet.".into())),
    };
    if let Err(e) = state.inference.device_label() {
        return Err(format!("Voice input can't run right now. {e}"));
    }
    let started = Instant::now();
    let loaded = match state.inference.whisper.get_or_load(state, &model.row.id, &model.dir).await {
        Ok(m) => m,
        Err(e) => {
            state.dictation.checks.remember_load_failure(&model.dir, &e);
            return Err(format!("Whisper Base couldn't be loaded: {e}"));
        }
    };
    let tested = tokio::task::spawn_blocking(move || {
        let mut m = match loaded.try_lock() {
            Ok(m) => m,
            // Busy transcribing: that shows it works too.
            Err(std::sync::TryLockError::WouldBlock) => return Ok(()),
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
        };
        m.transcribe_live(&vec![0.0; 16_000], Some("en"), false).map(|_| ()).map_err(|e| format!("{e:#}"))
    })
    .await
    .unwrap_or_else(|e| {
        tracing::error!("testing Whisper Base crashed: {e}");
        Err("it crashed".into())
    });
    let device = state.inference.whisper.loaded_device().unwrap_or_default();
    match tested {
        Ok(()) => {
            state.dictation.checks.clear_load_failure(&model.dir);
            Ok(format!(
                "Whisper Base is loaded and working ({device}, ready in {:.1} s).",
                started.elapsed().as_secs_f32()
            ))
        }
        Err(e) => {
            state.dictation.checks.remember_load_failure(&model.dir, &e);
            state.inference.unload_whisper();
            Err(format!("Whisper Base loaded but couldn't transcribe: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_are_checked() {
        assert_eq!(parse_language("auto"), Ok(None));
        assert_eq!(parse_language(""), Ok(None));
        assert_eq!(parse_language(" EN "), Ok(Some("en".into())));
        assert_eq!(parse_language("yue"), Ok(Some("yue".into())));
        assert!(matches!(parse_language("klingon"), Err(DictationError::BadRequest(_))));
    }
}
