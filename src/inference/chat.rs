//! Chat orchestration: turns a stored conversation into an engine request, runs it in the
//! background, and exposes live progress to any number of viewers (the streaming chat page).
//!
//! Generation runs independently of the browser: closing or reloading the page doesn't stop
//! it; the Stop button (or app shutdown) does. One generation per conversation at a time.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::{EngineMessage, EngineRequest, GenEvent, GenStats, ModelRef};
use crate::db::conversations::{
    self as convdb, Attachment, AttachmentKind, Message, MessageCompletion, MessageStatus, Role,
};
use crate::error::{AppError, AppResult};
use crate::util::now_rfc3339;

#[derive(Clone, Debug)]
pub struct Finished {
    pub status: MessageStatus,
    pub error: Option<String>,
    pub stats: GenStats,
    pub engine: String,
}

#[derive(Default)]
struct LiveInner {
    text: String,
    phase: Option<String>,
    finished: Option<Finished>,
}

/// A generation in progress (or just finished) for one conversation.
pub struct LiveGeneration {
    pub conversation_id: String,
    pub message_id: String,
    inner: Mutex<LiveInner>,
    version: watch::Sender<u64>,
    pub cancel: CancellationToken,
}

/// A consistent view of a live generation.
#[derive(Clone, Debug)]
pub struct LiveSnapshot {
    pub text: String,
    pub phase: Option<String>,
    pub finished: Option<Finished>,
}

impl LiveGeneration {
    fn new(conversation_id: &str, message_id: &str, initial: &str) -> Self {
        let (tx, _rx) = watch::channel(0);
        Self {
            conversation_id: conversation_id.into(),
            message_id: message_id.into(),
            inner: Mutex::new(LiveInner { text: initial.into(), ..Default::default() }),
            version: tx,
            cancel: CancellationToken::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LiveInner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn snapshot(&self) -> LiveSnapshot {
        let g = self.lock();
        LiveSnapshot { text: g.text.clone(), phase: g.phase.clone(), finished: g.finished.clone() }
    }

    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.version.subscribe()
    }

    fn bump(&self) {
        self.version.send_modify(|v| *v += 1);
    }

    fn push_text(&self, t: &str) {
        self.lock().text.push_str(t);
        self.bump();
    }

    fn set_phase(&self, p: String) {
        self.lock().phase = Some(p);
        self.bump();
    }

    fn finish(&self, f: Finished) {
        self.lock().finished = Some(f);
        self.bump();
    }

    pub fn is_finished(&self) -> bool {
        self.lock().finished.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartMode {
    /// Answer the last user message.
    Reply,
    /// Continue the last assistant message.
    Continue,
}

#[derive(Default)]
pub struct ChatService {
    live: Mutex<HashMap<String, Arc<LiveGeneration>>>,
    /// Unsent message text kept when a send is rejected, so the user doesn't retype it.
    drafts: Mutex<HashMap<String, String>>,
}

impl ChatService {
    pub fn save_draft(&self, conversation_id: &str, text: &str) {
        if !text.trim().is_empty() {
            self.drafts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(conversation_id.to_string(), text.to_string());
        }
    }

    pub fn take_draft(&self, conversation_id: &str) -> Option<String> {
        self.drafts.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(conversation_id)
    }

    fn map(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<LiveGeneration>>> {
        self.live.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn get(&self, conversation_id: &str) -> Option<Arc<LiveGeneration>> {
        self.map().get(conversation_id).cloned()
    }

    pub fn is_generating(&self, conversation_id: &str) -> bool {
        self.get(conversation_id).map(|g| !g.is_finished()).unwrap_or(false)
    }

    pub fn active_count(&self) -> usize {
        self.map().values().filter(|g| !g.is_finished()).count()
    }

    pub fn stop(&self, conversation_id: &str) -> bool {
        match self.get(conversation_id) {
            Some(g) if !g.is_finished() => {
                g.cancel.cancel();
                true
            }
            _ => false,
        }
    }

    pub fn stop_all(&self) {
        for g in self.map().values() {
            g.cancel.cancel();
        }
    }

    /// Waits (bounded) until running generations have saved their results.
    pub async fn wait_idle(&self, max: Duration) {
        let start = std::time::Instant::now();
        while self.active_count() > 0 && start.elapsed() < max {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Starts generating for a conversation.
    pub async fn start(
        &self,
        state: &crate::SharedState,
        conversation_id: &str,
        mode: StartMode,
    ) -> AppResult<Arc<LiveGeneration>> {
        if self.is_generating(conversation_id) {
            return Err(AppError::conflict("An answer is still being written. Wait for it to finish or press Stop."));
        }
        let cid = conversation_id.to_string();
        let (conv, messages) =
            state.db.call(move |c| Ok((convdb::get_conversation(c, &cid)?, convdb::messages(c, &cid)?))).await?;
        let conv = conv.ok_or_else(|| AppError::not_found("This conversation no longer exists."))?;
        let model_ref_str =
            conv.model_ref.clone().ok_or_else(|| AppError::bad_request("Pick a model for this chat first."))?;
        let model_ref = ModelRef::parse(&model_ref_str)
            .ok_or_else(|| AppError::bad_request("The selected model is invalid; pick another one."))?;
        let last = messages.last().cloned();
        let (target, history, continue_from) = match mode {
            StartMode::Reply => {
                let last = last.ok_or_else(|| AppError::bad_request("Write a message first."))?;
                if last.role != Role::User {
                    return Err(AppError::bad_request("There is no new message to answer."));
                }
                let msg = Message {
                    id: crate::util::new_id(),
                    conversation_id: conv.id.clone(),
                    seq: 0,
                    role: Role::Assistant,
                    content: String::new(),
                    attachments: Vec::new(),
                    model_ref: Some(model_ref_str.clone()),
                    status: MessageStatus::Generating,
                    error: None,
                    prompt_tokens: None,
                    completion_tokens: None,
                    duration_ms: None,
                    tokens_per_sec: None,
                    created_at: now_rfc3339(),
                };
                let inserted = state.db.call(move |c| convdb::append_message(c, msg)).await?;
                (inserted, messages, None)
            }
            StartMode::Continue => {
                let last = last.ok_or_else(|| AppError::bad_request("Nothing to continue."))?;
                if last.role != Role::Assistant || last.content.trim().is_empty() {
                    return Err(AppError::bad_request("Only an assistant answer can be continued."));
                }
                let history: Vec<Message> = messages[..messages.len() - 1].to_vec();
                let id = last.id.clone();
                state
                    .db
                    .call(move |c| {
                        c.execute("UPDATE messages SET status = 'generating', error = NULL WHERE id = ?1", [&id])
                    })
                    .await?;
                let content = last.content.clone();
                (last, history, Some(content))
            }
        };
        let live = Arc::new(LiveGeneration::new(&conv.id, &target.id, continue_from.as_deref().unwrap_or("")));
        self.map().insert(conv.id.clone(), live.clone());
        let st = state.clone();
        let lv = live.clone();
        tokio::spawn(async move {
            run(st, lv, conv, model_ref, history, continue_from).await;
        });
        Ok(live)
    }
}

async fn run(
    state: crate::SharedState,
    live: Arc<LiveGeneration>,
    conv: convdb::Conversation,
    model_ref: ModelRef,
    history: Vec<Message>,
    continue_from: Option<String>,
) {
    let started = std::time::Instant::now();
    let initial_len = continue_from.as_ref().map(|s| s.len()).unwrap_or(0);
    let cancel = live.cancel.clone();
    let shutdown = state.shutdown.clone();
    // Stop generation when the app shuts down.
    let watcher = {
        let c = cancel.clone();
        tokio::spawn(async move {
            shutdown.cancelled().await;
            c.cancel();
        })
    };

    let outcome = generate(&state, &live, &conv, &model_ref, &history, continue_from, &cancel).await;
    watcher.abort();

    let text = live.snapshot().text;
    let (status, error, stats, engine) = match outcome {
        Ok((stats, engine)) => {
            let status = if stats.stop_reason == "cancelled" { MessageStatus::Stopped } else { MessageStatus::Done };
            (status, None, stats, engine)
        }
        Err(e) => (
            MessageStatus::Error,
            Some(e),
            GenStats { duration_ms: started.elapsed().as_millis() as u64, ..Default::default() },
            String::new(),
        ),
    };
    let content = if initial_len == 0 { text.trim_start().to_string() } else { text.clone() };
    let done = MessageCompletion {
        content,
        status: Some(status),
        error: error.clone(),
        prompt_tokens: stats.prompt_tokens.map(|v| v as i64),
        completion_tokens: stats.completion_tokens.map(|v| v as i64),
        duration_ms: Some(stats.duration_ms as i64),
        tokens_per_sec: stats.tokens_per_sec,
    };
    let mid = live.message_id.clone();
    let cid = conv.id.clone();
    let needs_title = conv.title == "New chat";
    let first_user = history.iter().find(|m| m.role == Role::User).map(|m| m.content.clone()).unwrap_or_default();
    let now = now_rfc3339();
    let mr = model_ref.to_string();
    let save = state
        .db
        .call(move |c| {
            convdb::complete_message(c, &mid, &done)?;
            convdb::touch_conversation(c, &cid, &now)?;
            // Chats started elsewhere (e.g. from a transcript) get their name here; it never
            // replaces a name given in the meantime.
            if needs_title && let Some(t) = crate::util::chat_title_from(&first_user) {
                convdb::title_if_untitled(c, &cid, &t)?;
            }
            if mr.starts_with("ollama:") {
                crate::db::models::touch_meta(c, &mr, &now)?;
            } else if let Some(id) = mr.strip_prefix("local:") {
                crate::db::models::touch(c, id, &now)?;
            }
            Ok(())
        })
        .await;
    if let Err(e) = save {
        tracing::error!("could not save the generated message: {e:#}");
    }
    live.finish(Finished { status, error, stats, engine });
    // Keep the finished entry briefly so a page that is still connected sees the end.
    let st = state.clone();
    let cid = conv.id.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(30)).await;
        let mut map = st.chat.live.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if map.get(&cid).map(|g| Arc::ptr_eq(g, &live)).unwrap_or(false) {
            map.remove(&cid);
        }
    });
}

async fn generate(
    state: &crate::SharedState,
    live: &Arc<LiveGeneration>,
    conv: &convdb::Conversation,
    model_ref: &ModelRef,
    history: &[Message],
    continue_from: Option<String>,
    cancel: &CancellationToken,
) -> Result<(GenStats, String), String> {
    live.set_phase(match model_ref {
        ModelRef::Ollama(_) => "Connecting to Ollama…".into(),
        ModelRef::Local(id) if !state.inference.loaded_refs().contains(&format!("local:{id}")) => {
            "Loading the model into memory (first time can take a while)…".into()
        }
        ModelRef::Local(_) => "Preparing…".into(),
    });
    let engine = tokio::select! {
        e = state.inference.engine_for(model_ref) => e.map_err(|e| e.user_message())?,
        _ = cancel.cancelled() => return Ok((GenStats { stop_reason: "cancelled".into(), ..Default::default() }, String::new())),
    };
    let support = engine.param_support();
    let mut messages = Vec::new();
    if !conv.system_prompt.trim().is_empty() {
        messages.push(EngineMessage { role: "system".into(), content: conv.system_prompt.clone(), images: Vec::new() });
    }
    for m in history {
        if m.role == Role::Assistant && (m.status == MessageStatus::Error || m.content.trim().is_empty()) {
            continue;
        }
        if m.role == Role::System {
            continue;
        }
        let (content, images) = prepare_attachments(state, live, m, support.images, cancel).await?;
        messages.push(EngineMessage { role: m.role.as_str().into(), content, images });
    }
    if messages.iter().all(|m| m.role == "system") {
        return Err("There is nothing to answer.".into());
    }
    let req = EngineRequest { messages, params: conv.params.clone().sanitized(), continue_from };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let fut = engine.chat(req, cancel.clone(), tx);
    let handle = tokio::spawn(fut);
    let mut first_text = true;
    let mut result = Err("The engine stopped without an answer.".to_string());
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Phase(p) => live.set_phase(p),
            GenEvent::Text(t) => {
                if first_text {
                    first_text = false;
                    live.set_phase(String::new());
                }
                live.push_text(&t);
            }
            GenEvent::Done(stats) => {
                result = Ok((stats, engine.name()));
                break;
            }
            GenEvent::Error(e) => {
                result = Err(e);
                break;
            }
        }
    }
    let _ = handle.await;
    result
}

/// Resolves attachments into prompt text (transcripts, video notes) and image files.
async fn prepare_attachments(
    state: &crate::SharedState,
    live: &Arc<LiveGeneration>,
    m: &Message,
    images_supported: bool,
    cancel: &CancellationToken,
) -> Result<(String, Vec<PathBuf>), String> {
    let uploads = state.paths.uploads_dir.clone();
    let mut content = m.content.clone();
    let mut images = Vec::new();
    let mut updated: Vec<Attachment> = Vec::new();
    let mut changed = false;
    for a in &m.attachments {
        let mut a = a.clone();
        let path = crate::util::safe_join(&uploads, &a.file).ok_or("invalid attachment path")?;
        match a.kind {
            AttachmentKind::Image => {
                if !images_supported {
                    return Err(format!(
                        "\"{}\" is an image, but this model can't see images. Choose a vision model (for example llama3.2-vision, gemma3 or qwen2.5vl in Ollama) or remove the image.",
                        a.name
                    ));
                }
                images.push(path);
            }
            AttachmentKind::Audio => {
                if a.transcript.is_none() {
                    live.set_phase(format!("Transcribing {}…", a.name));
                    let t = transcribe_for_chat(state, &path, cancel).await?;
                    a.transcript = Some(t);
                    changed = true;
                }
                content.push_str(&format!(
                    "\n\n[Transcript of the attached audio \"{}\"]\n{}",
                    a.name,
                    a.transcript.clone().unwrap_or_default()
                ));
            }
            AttachmentKind::Video => {
                if !images_supported {
                    return Err(format!(
                        "\"{}\" is a video. Video questions need a vision model (frames are sampled and shown to it). Choose a vision model or remove the video.",
                        a.name
                    ));
                }
                if a.frames.is_empty() {
                    live.set_phase(format!("Extracting frames from {}…", a.name));
                    let n = state.settings().media.video_frames as usize;
                    let ffmpeg = crate::media::ffmpeg_path(&state.settings());
                    let (frames, duration) = crate::media::extract_frames(ffmpeg.as_deref(), &path, &uploads, n)
                        .await
                        .map_err(|e| e.to_string())?;
                    a.frames = frames;
                    a.duration_secs = duration;
                    changed = true;
                }
                for f in &a.frames {
                    if let Some(p) = crate::util::safe_join(&uploads, f) {
                        images.push(p);
                    }
                }
                content.push_str(&format!(
                    "\n\n[The attached video \"{}\"{} is shown as {} frames sampled evenly from start to end, in order.]",
                    a.name,
                    a.duration_secs.map(|d| format!(" ({d:.0} seconds long)")).unwrap_or_default(),
                    a.frames.len()
                ));
            }
        }
        updated.push(a);
    }
    if changed {
        let id = m.id.clone();
        let json = serde_json::to_string(&updated).unwrap_or_else(|_| "[]".into());
        let _ = state
            .db
            .call(move |c| {
                c.execute("UPDATE messages SET attachments_json = ?2 WHERE id = ?1", rusqlite::params![id, json])
            })
            .await;
    }
    Ok((content, images))
}

/// Transcribes audio for a chat message with the most recently used Whisper model.
async fn transcribe_for_chat(
    state: &crate::SharedState,
    path: &Path,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let models = state.db.call(crate::db::models::list).await.map_err(|e| e.to_string())?;
    let whisper = models
        .into_iter()
        .find(|m| m.kind == "transcription" && m.analysis.runnable)
        .ok_or_else(|| "To use audio in chat, first download a Whisper speech-to-text model (for example openai/whisper-base) from Discover.".to_string())?;
    let out = state
        .inference
        .whisper
        .transcribe(state, &whisper, path, None, false, cancel.clone(), |_| {})
        .await
        .map_err(|e| e.user_message())?;
    Ok(out.text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn live_generation_snapshots_are_consistent() {
        let g = LiveGeneration::new("c", "m", "");
        let mut rx = g.subscribe();
        g.push_text("Hel");
        g.push_text("lo");
        rx.changed().await.unwrap();
        assert_eq!(g.snapshot().text, "Hello");
        g.finish(Finished { status: MessageStatus::Done, error: None, stats: GenStats::default(), engine: "x".into() });
        assert!(g.is_finished());
    }
}
