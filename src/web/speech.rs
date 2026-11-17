//! Voice input with Whisper Base: the Speech models page's buttons (HTML forms that come back
//! to the page with a message) and the live dictation JSON API the page script talks to.
//!
//! API (all answers are JSON; errors are `{"error": "<sentence>", "code": "<code>"}`):
//!
//! | Request | Answer |
//! |---|---|
//! | `POST /api/speech/sessions` `{"language": "auto"}` (64 KiB at most) | session state; 409 `whisper_unavailable`, 429 `too_many`, 400 `bad_request` |
//! | `POST /api/speech/sessions/{id}/audio?rate=48000` (a usual rate), body: little-endian f32 mono | session state; 400 `bad_rate`, `bad_audio`, `too_large` |
//! | `GET /api/speech/sessions/{id}` | session state |
//! | `POST /api/speech/sessions/{id}/pause` | session state, once the words so far are final (at most 20 s; `pending` says more are coming) |
//! | `POST /api/speech/sessions/{id}/resume` | session state; 409 `not_active` after stop |
//! | `POST /api/speech/sessions/{id}/stop` | session state with status `done` (`error` if it failed; `finishing` after 45 s) |
//! | `DELETE /api/speech/sessions/{id}` | 204 |
//!
//! Unknown or expired sessions answer 404 `not_found`.

use std::time::Duration;

use axum::Json;
use axum::body::Body;
use axum::extract::{Path, RawQuery, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::SharedState;
use crate::db::downloads::DownloadState;
use crate::error::AppError;
use crate::inference::dictation::{self, DictationError, SessionState, WHISPER_BASE_REPO, WHISPER_VARIANT};
use crate::inference::whisper::FileTrouble;
use crate::web::page::redirect_flash;

/// Where the model buttons return to.
const BACK: &str = "/transcribe#whisper-base";
/// Largest audio chunk accepted: 2 MB (over 10 seconds at 48 kHz).
pub const MAX_CHUNK_BYTES: usize = 2 * 1024 * 1024;
/// Largest body for starting a session (it holds one short JSON object).
const MAX_START_BYTES: usize = 64 * 1024;
/// Accepted sample rates (Hz).
pub const MIN_RATE: u32 = 8_000;
pub const MAX_RATE: u32 = 192_000;

// ---- Speech page buttons -----------------------------------------------------------------

/// Waits (at most `max`) until download `id` is in a state `reached` accepts, or gone. A
/// download's state is saved by its own task a moment after it is told to resume or stop, and
/// the page these buttons return to should show what the button did, not the moment before.
async fn settle_download(state: &SharedState, id: &str, max: Duration, reached: fn(DownloadState) -> bool) {
    let deadline = tokio::time::Instant::now() + max;
    loop {
        let now = state.downloads.list().await.ok().map(|l| l.into_iter().find(|j| j.row.id == id));
        match now {
            Some(Some(j)) if !reached(j.row.state) => {}
            _ => return,
        }
        if tokio::time::Instant::now() >= deadline {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Downloads Whisper Base (or resumes its download). Harmless when it is already there.
pub async fn download(State(state): State<SharedState>) -> Response {
    let (status, model) = dictation::status_and_model(&state).await;
    if status.downloaded && status.compatible {
        return redirect_flash(BACK, "info", "Whisper Base is already downloaded.");
    }
    if let Some(d) = &status.download {
        if d.active {
            return redirect_flash(BACK, "info", &format!("Whisper Base is already downloading ({}%).", d.percent));
        }
        // Paused or failed: carry on where it stopped.
        return match state.downloads.resume(&d.id).await {
            Ok(()) => {
                settle_download(&state, &d.id, Duration::from_secs(2), DownloadState::is_active).await;
                redirect_flash(BACK, "ok", "Resuming the download of Whisper Base.")
            }
            Err(e) => redirect_flash(BACK, "error", &e.user_message()),
        };
    }
    if model.is_some() && status.downloaded {
        return redirect_flash(
            BACK,
            "error",
            "Whisper Base is downloaded but its files are damaged. Delete it first, then download it again.",
        );
    }
    match crate::web::pages::discover::enqueue_hf_variant(&state, WHISPER_BASE_REPO, Some(WHISPER_VARIANT)).await {
        Ok(id) => {
            // It starts at once unless other downloads take every slot (Settings → Storage).
            settle_download(&state, &id, Duration::from_secs(1), |s| s != DownloadState::Queued).await;
            let queued = state
                .downloads
                .list()
                .await
                .ok()
                .and_then(|l| l.into_iter().find(|j| j.row.id == id))
                .is_some_and(|j| j.row.state == DownloadState::Queued);
            let message = if queued {
                format!(
                    "Whisper Base ({}) will download as soon as the downloads ahead of it finish. You can keep using the app meanwhile.",
                    status.size_label
                )
            } else {
                format!("Downloading Whisper Base ({}). You can keep using the app meanwhile.", status.size_label)
            };
            redirect_flash(BACK, "ok", &message)
        }
        Err(AppError::Conflict(msg)) => redirect_flash(BACK, "info", &msg),
        Err(e) => redirect_flash(BACK, "error", &e.user_message()),
    }
}

/// Cancels Whisper Base's unfinished download (partial files are removed).
pub async fn cancel(State(state): State<SharedState>) -> Response {
    let Some(d) = dictation::whisper_base_status(&state).await.download else {
        return redirect_flash(BACK, "info", "Whisper Base isn't downloading.");
    };
    let done =
        if d.status == "Failed" { state.downloads.remove(&d.id).await } else { state.downloads.cancel(&d.id).await };
    match done {
        Ok(()) => {
            settle_download(&state, &d.id, Duration::from_secs(3), DownloadState::is_finished).await;
            redirect_flash(BACK, "ok", "The download of Whisper Base was cancelled.")
        }
        Err(e) => redirect_flash(BACK, "error", &e.user_message()),
    }
}

/// Loads Whisper Base into memory and checks it really transcribes.
pub async fn load(State(state): State<SharedState>) -> Response {
    match dictation::load_whisper_base(&state).await {
        Ok(msg) => redirect_flash(BACK, "ok", &msg),
        Err(msg) => redirect_flash(BACK, "error", &msg),
    }
}

/// Frees the memory Whisper Base takes (it loads again when needed).
pub async fn unload(State(state): State<SharedState>) -> Response {
    if state.dictation.recording() {
        return redirect_flash(
            BACK,
            "error",
            "Whisper Base is being used for voice input right now. Stop the recording first.",
        );
    }
    if !dictation::whisper_base_status(&state).await.loaded {
        return redirect_flash(BACK, "info", "Whisper Base isn't loaded.");
    }
    state.inference.unload_whisper();
    redirect_flash(BACK, "ok", "Whisper Base was unloaded from memory.")
}

/// Deletes Whisper Base and its files (refused while it is in use).
pub async fn delete(State(state): State<SharedState>) -> Response {
    if state.dictation.in_use() {
        return redirect_flash(
            BACK,
            "error",
            "Whisper Base is being used for voice input. Stop the recording first, then delete it.",
        );
    }
    let (status, model) = dictation::status_and_model(&state).await;
    let Some(model) = model else {
        return redirect_flash(BACK, "info", "Whisper Base isn't downloaded.");
    };
    let id = model.row.id.clone();
    let transcribing = {
        let id = id.clone();
        state
            .db
            .call(|c| crate::db::jobs::list_transcriptions(c, 50))
            .await
            .unwrap_or_default()
            .iter()
            .any(|t| t.status.is_active() && t.model_id == id)
    };
    if transcribing {
        return redirect_flash(
            BACK,
            "error",
            "Whisper Base is transcribing a file right now. Wait until it finishes, then delete it.",
        );
    }
    if status.loaded {
        state.inference.unload_whisper();
    }
    let models_dir = state.settings().models_dir(&state.paths);
    match crate::library::delete_model(&state.db, &models_dir, &id, true).await {
        Ok(()) => {
            state.dictation.checks.forget(&model.dir);
            redirect_flash(BACK, "ok", "Whisper Base was deleted.")
        }
        Err(e) => {
            tracing::warn!("couldn't delete Whisper Base: {e:#}");
            let why = match crate::inference::whisper::file_trouble(&e) {
                FileTrouble::Denied => "Local AI Studio isn't allowed to remove its files (permission denied)",
                _ => "its files couldn't be removed",
            };
            redirect_flash(BACK, "error", &format!("Whisper Base couldn't be deleted: {why}. Try again."))
        }
    }
}

// ---- Live dictation API ------------------------------------------------------------------

/// An API error: `{"error": "<sentence for people>", "code": "<machine code>"}`.
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into() }
    }
}

impl From<DictationError> for ApiError {
    fn from(e: DictationError) -> Self {
        match e {
            DictationError::NotFound => ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found",
                "This recording has ended or expired. Start a new one.",
            ),
            DictationError::TooMany => ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "too_many",
                "Too many recordings are open at once. Stop one, or wait a minute and try again.",
            ),
            DictationError::Unavailable(msg) => ApiError::new(StatusCode::CONFLICT, "whisper_unavailable", msg),
            DictationError::BadRequest(msg) => ApiError::new(StatusCode::BAD_REQUEST, "bad_request", msg),
            DictationError::NotActive(msg) => ApiError::new(StatusCode::CONFLICT, "not_active", msg),
        }
    }
}

fn no_store(mut r: Response) -> Response {
    r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    r
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        no_store((self.status, Json(serde_json::json!({ "error": self.message, "code": self.code }))).into_response())
    }
}

fn respond(result: Result<SessionState, DictationError>) -> Response {
    match result {
        Ok(s) => no_store(Json(s).into_response()),
        Err(e) => ApiError::from(e).into_response(),
    }
}

#[derive(Deserialize, Default)]
struct StartRequest {
    #[serde(default)]
    language: Option<String>,
}

/// `POST /api/speech/sessions`: starts recording. Body (optional): `{"language": "auto"}`
/// or a Whisper language code.
pub async fn start(State(state): State<SharedState>, body: Body) -> Response {
    let bad = || {
        ApiError::new(StatusCode::BAD_REQUEST, "bad_request", "Send JSON such as {\"language\": \"auto\"}.")
            .into_response()
    };
    // Read here (not with an extractor), so even a huge body gets the usual JSON error.
    let Ok(body) = axum::body::to_bytes(body, MAX_START_BYTES).await else {
        return bad();
    };
    let req: StartRequest = if body.iter().all(u8::is_ascii_whitespace) {
        StartRequest::default()
    } else {
        match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(_) => return bad(),
        }
    };
    respond(dictation::start_session(&state, req.language.as_deref().unwrap_or("auto")).await)
}

/// The sample rate from `?rate=48000`.
fn parse_rate(query: Option<&str>) -> Result<u32, ApiError> {
    let rate = query
        .and_then(|q| url::form_urlencoded::parse(q.as_bytes()).find(|(k, _)| k == "rate"))
        .and_then(|(_, v)| v.trim().parse::<u32>().ok());
    // Odd rates (8001 Hz…) would make every chunk hundreds of times slower to resample, holding
    // up every other recording; no microphone records at one.
    match rate {
        Some(r) if (MIN_RATE..=MAX_RATE).contains(&r) && crate::media::has_weight_table(r, dictation::stream::RATE) => {
            Ok(r)
        }
        _ => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "bad_rate",
            format!(
                "Say the audio's sample rate, for example ?rate=48000: a usual rate from {MIN_RATE} to {MAX_RATE} Hz, such as 16000, 44100 or 48000."
            ),
        )),
    }
}

/// Raw little-endian 32-bit float samples.
fn parse_samples(bytes: &[u8]) -> Result<Vec<f32>, ApiError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "bad_audio",
            "The audio must be 32-bit float samples: its length must be a multiple of 4 bytes.",
        ));
    }
    let samples: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c).clamp(-1.0, 1.0)).collect();
    if samples.iter().any(|s| s.is_nan()) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "bad_audio",
            "The audio contains values that aren't numbers.",
        ));
    }
    Ok(samples)
}

/// `POST /api/speech/sessions/{id}/audio?rate=<Hz>`: adds a chunk of microphone audio (raw
/// little-endian f32 mono). Accepted but ignored while paused, finishing or done.
pub async fn audio(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    body: Body,
) -> Response {
    if let Err(e) = state.dictation.state(&id) {
        return ApiError::from(e).into_response();
    }
    let rate = match parse_rate(query.as_deref()) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    let too_large = || {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "too_large",
            "The audio chunk is too large (2 MB at most). Send smaller chunks.",
        )
        .into_response()
    };
    let declared =
        headers.get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|n| n > MAX_CHUNK_BYTES as u64) {
        return too_large();
    }
    let Ok(bytes) = axum::body::to_bytes(body, MAX_CHUNK_BYTES).await else {
        return too_large();
    };
    let samples = match parse_samples(&bytes) {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };
    respond(state.dictation.push_audio(&id, &samples, rate))
}

/// `GET /api/speech/sessions/{id}`.
pub async fn session(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    respond(state.dictation.state(&id))
}

/// `POST /api/speech/sessions/{id}/pause`.
pub async fn pause(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    respond(state.dictation.pause(&id).await)
}

/// `POST /api/speech/sessions/{id}/resume`.
pub async fn resume(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    respond(state.dictation.resume(&id))
}

/// `POST /api/speech/sessions/{id}/stop`.
pub async fn stop(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    respond(state.dictation.stop(&id).await)
}

/// `DELETE /api/speech/sessions/{id}`.
pub async fn discard(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    if state.dictation.discard(&id) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        ApiError::from(DictationError::NotFound).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_and_samples_are_checked() {
        assert_eq!(parse_rate(Some("rate=48000")).ok(), Some(48_000));
        assert_eq!(parse_rate(Some("x=1&rate=8000")).ok(), Some(8_000));
        assert_eq!(parse_rate(Some("rate=44100")).ok(), Some(44_100));
        assert_eq!(parse_rate(Some("rate=11025")).ok(), Some(11_025));
        for bad in [
            None,
            Some(""),
            Some("rate=7999"),
            Some("rate=192001"),
            Some("rate=fast"),
            Some("rate=8001"),
            Some("rate=191999"),
        ] {
            assert_eq!(parse_rate(bad).err().map(|e| e.code), Some("bad_rate"), "{bad:?}");
        }
        let bytes: Vec<u8> = [0.5f32, -2.0, 0.25].iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(parse_samples(&bytes).ok(), Some(vec![0.5, -1.0, 0.25]));
        assert_eq!(parse_samples(&bytes[..5]).err().map(|e| e.code), Some("bad_audio"));
        assert_eq!(parse_samples(&f32::NAN.to_le_bytes()).err().map(|e| e.code), Some("bad_audio"));
    }
}
