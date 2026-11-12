//! Chat: conversations, streaming answers, attachments, editing, regeneration, settings,
//! export and import.

use std::path::Path as FsPath;

use askama::Template;
use axum::Form;
use axum::extract::{Multipart, Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use super::ModelChoice;
use crate::SharedState;
use crate::db::conversations::{
    self as convdb, Attachment, AttachmentKind, Conversation, Message, MessageStatus, Role,
};
use crate::error::{AppError, AppResult};
use crate::inference::ModelRef;
use crate::inference::chat::{LiveSnapshot, StartMode};
use crate::inference::params::GenParams;
use crate::util::{html_escape, human_bytes, now_rfc3339, relative_time, short_datetime};
use crate::web::page::{PageCtx, redirect, redirect_flash, render};
use crate::web::streaming::{self, LIVE_MARKER};

pub struct ConvItem {
    pub id: String,
    pub title: String,
    pub when: String,
    pub active: bool,
}

pub struct AttachmentView {
    pub kind: &'static str,
    pub url: String,
    pub name: String,
    pub size: String,
    pub transcript: String,
    pub frames: Vec<String>,
}

pub struct MsgView {
    pub id: String,
    pub is_user: bool,
    pub html: String,
    pub raw: String,
    pub time: String,
    pub time_full: String,
    pub status: &'static str,
    pub error: String,
    pub stats: String,
    pub who: String,
    pub attachments: Vec<AttachmentView>,
    pub can_regenerate: bool,
    pub can_continue: bool,
    pub can_edit: bool,
}

/// Renders one message (used by the page and by the live stream when an answer finishes).
#[derive(Template)]
#[template(path = "partials/message.html")]
pub struct MessagePartial<'a> {
    pub m: &'a MsgView,
    pub conv_id: &'a str,
}

#[derive(Template)]
#[template(path = "chat.html")]
pub struct ChatPage {
    pub ctx: PageCtx,
    pub conversations: Vec<ConvItem>,
    pub has_conv: bool,
    pub conv_id: String,
    pub title: String,
    pub system_prompt: String,
    pub params: GenParams,
    pub seed_text: String,
    pub messages: Vec<MsgView>,
    pub models: Vec<ModelChoice>,
    pub model_label: String,
    pub model_missing: bool,
    pub model_vision: bool,
    pub engine_name: String,
    pub live: bool,
    pub ollama_error: Option<String>,
    pub usage: String,
    pub whisper_available: bool,
    pub ffmpeg_available: bool,
    pub draft: String,
    pub live_marker: &'static str,
}

fn attachment_view(a: &Attachment) -> AttachmentView {
    AttachmentView {
        kind: match a.kind {
            AttachmentKind::Image => "image",
            AttachmentKind::Audio => "audio",
            AttachmentKind::Video => "video",
        },
        url: format!("/files/uploads/{}", a.file),
        name: a.name.clone(),
        size: human_bytes(a.size),
        transcript: a.transcript.clone().unwrap_or_default(),
        frames: a.frames.iter().map(|f| format!("/files/uploads/{f}")).collect(),
    }
}

fn stats_label(m: &Message) -> String {
    let mut parts = Vec::new();
    if let Some(c) = m.completion_tokens {
        parts.push(format!("{c} tokens"));
        // The engine's own measurement (without loading the model); older messages only
        // have the total time.
        let speed =
            m.tokens_per_sec.or_else(|| m.duration_ms.filter(|d| *d > 0).map(|ms| c as f64 / (ms as f64 / 1000.0)));
        if let Some(v) = speed.filter(|v| v.is_finite() && *v > 0.0) {
            parts.push(format!("{v:.1} tok/s"));
        }
    }
    if let Some(p) = m.prompt_tokens {
        parts.push(format!("{p} prompt tokens"));
    }
    if let Some(ms) = m.duration_ms.filter(|d| *d > 0) {
        parts.push(format!("{:.1} s", ms as f64 / 1000.0));
    }
    parts.join(" · ")
}

fn user_html(text: &str) -> String {
    html_escape(text)
}

pub fn message_view(m: &Message, model_names: &std::collections::HashMap<String, String>, is_last: bool) -> MsgView {
    let is_user = m.role == Role::User;
    MsgView {
        id: m.id.clone(),
        is_user,
        html: if is_user { user_html(&m.content) } else { crate::render::markdown_to_html(&m.content) },
        raw: m.content.clone(),
        time: short_datetime(&m.created_at),
        time_full: m.created_at.clone(),
        status: m.status.as_str(),
        error: m.error.clone().unwrap_or_default(),
        stats: if is_user { String::new() } else { stats_label(m) },
        who: if is_user {
            "You".into()
        } else {
            m.model_ref
                .as_ref()
                .and_then(|r| model_names.get(r).cloned())
                .or_else(|| {
                    m.model_ref
                        .as_ref()
                        .map(|r| r.trim_start_matches("ollama:").trim_start_matches("local:").to_string())
                })
                .unwrap_or_else(|| "Assistant".into())
        },
        attachments: m.attachments.iter().map(attachment_view).collect(),
        can_regenerate: !is_user && is_last,
        can_continue: !is_user
            && is_last
            && !m.content.trim().is_empty()
            && matches!(m.status, MessageStatus::Stopped | MessageStatus::Done),
        can_edit: is_user,
    }
}

#[derive(Deserialize, Default)]
pub struct ChatIndexQuery {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    new: Option<String>,
}

pub async fn index(State(state): State<SharedState>, headers: HeaderMap, Query(q): Query<ChatIndexQuery>) -> Response {
    if q.model.is_none()
        && q.new.is_none()
        && state.settings().ui.restore_last_chat
        && let Ok(Some(id)) = state.db.call(convdb::latest_conversation_id).await
    {
        return redirect(&format!("/chat/{id}"));
    }
    match build_page(&state, &headers, None, q.model.as_deref()).await {
        Ok((ctx, page)) => render(&ctx, &page),
        Err(e) => e.into_response(),
    }
}

async fn build_page(
    state: &SharedState,
    headers: &HeaderMap,
    conv: Option<(&Conversation, &[Message])>,
    preselect: Option<&str>,
) -> AppResult<(PageCtx, ChatPage)> {
    let convs = state.db.call(|c| convdb::list_conversations(c, 200)).await?;
    let current_id = conv.map(|(c, _)| c.id.clone()).unwrap_or_default();
    let selected = conv.and_then(|(c, _)| c.model_ref.clone()).or_else(|| preselect.map(String::from));
    let (models, ollama_error) = super::chat_choices(state, selected.as_deref()).await;
    let names: std::collections::HashMap<String, String> =
        models.iter().map(|m| (m.value.clone(), m.label.trim_end_matches(" · sees images").to_string())).collect();
    let settings = state.settings();
    let (messages, live) = match conv {
        Some((c, msgs)) => {
            let live = state.chat.get(&c.id).filter(|g| !g.is_finished());
            let live_id = live.as_ref().map(|g| g.message_id.clone());
            let shown: Vec<&Message> = msgs.iter().filter(|m| Some(&m.id) != live_id.as_ref()).collect();
            let n = shown.len();
            let views =
                shown.iter().enumerate().map(|(i, m)| message_view(m, &names, i + 1 == n && live.is_none())).collect();
            (views, live.is_some())
        }
        None => (Vec::new(), false),
    };
    let selected_choice = models.iter().find(|m| m.selected);
    let model_missing = selected.is_some() && selected_choice.is_none();
    let engine_name = match selected.as_deref().and_then(ModelRef::parse) {
        Some(ModelRef::Ollama(_)) => "Ollama".to_string(),
        Some(ModelRef::Local(id)) => format!("Built-in engine ({})", state.inference.label_for(&id)),
        None => String::new(),
    };
    let (usage, title, system_prompt, params) = match conv {
        Some((c, msgs)) => {
            let p: i64 = msgs.iter().filter_map(|m| m.prompt_tokens).sum();
            let g: i64 = msgs.iter().filter_map(|m| m.completion_tokens).sum();
            (
                if p + g > 0 { format!("{g} tokens generated · {p} prompt tokens processed") } else { String::new() },
                c.title.clone(),
                c.system_prompt.clone(),
                c.params.clone(),
            )
        }
        None => (
            String::new(),
            "New chat".into(),
            settings.generation.system_prompt.clone(),
            GenParams::from_defaults(&settings.generation),
        ),
    };
    let whisper_available = super::models_of_kind(state, "transcription").await.iter().any(|m| m.analysis.runnable);
    let title_for_ctx = if conv.is_some() { title.clone() } else { "Chat".into() };
    let ctx = PageCtx::new(state, headers, &title_for_ctx, "chat").at(&if current_id.is_empty() {
        "/chat".to_string()
    } else {
        format!("/chat/{current_id}")
    });
    let draft = if current_id.is_empty() {
        state.chat.take_draft("new").unwrap_or_default()
    } else {
        state.chat.take_draft(&current_id).unwrap_or_default()
    };
    let page = ChatPage {
        ctx: ctx.clone(),
        conversations: convs
            .into_iter()
            .map(|c| ConvItem {
                active: c.id == current_id,
                when: relative_time(&c.updated_at),
                id: c.id,
                title: c.title,
            })
            .collect(),
        has_conv: conv.is_some(),
        conv_id: current_id,
        title,
        system_prompt,
        seed_text: params.seed.map(|s| s.to_string()).unwrap_or_default(),
        params,
        messages,
        model_label: selected_choice
            .map(|m| m.label.clone())
            .unwrap_or_else(|| if model_missing { "Model not available".into() } else { "No model selected".into() }),
        model_vision: selected_choice.map(|m| m.vision).unwrap_or(false),
        models,
        model_missing,
        engine_name,
        live,
        ollama_error,
        usage,
        whisper_available,
        ffmpeg_available: crate::media::ffmpeg_path(&settings).is_some(),
        draft,
        live_marker: LIVE_MARKER,
    };
    Ok((ctx, page))
}

pub async fn view(State(state): State<SharedState>, headers: HeaderMap, Path(id): Path<String>) -> AppResult<Response> {
    let cid = id.clone();
    let (conv, msgs) =
        state.db.call(move |c| Ok((convdb::get_conversation(c, &cid)?, convdb::messages(c, &cid)?))).await?;
    let conv =
        conv.ok_or_else(|| AppError::not_found("This conversation doesn't exist (it may have been deleted)."))?;
    let (ctx, page) = build_page(&state, &headers, Some((&conv, &msgs)), None).await?;
    let live = state.chat.get(&conv.id).filter(|g| !g.is_finished());
    match (page.live, live) {
        (true, Some(live)) => {
            let html = page.render().map_err(|e| AppError::Internal(anyhow::anyhow!("template error: {e}")))?;
            let names: std::collections::HashMap<String, String> = page
                .models
                .iter()
                .map(|m| (m.value.clone(), m.label.trim_end_matches(" · sees images").to_string()))
                .collect();
            let model_label = page.model_label.trim_end_matches(" · sees images").to_string();
            let conv_id = conv.id.clone();
            let mid = live.message_id.clone();
            let final_html: streaming::FinalHtml = Box::new(move |snap: &LiveSnapshot| {
                // Build the message from the final state; the database has the same content.
                let fin = snap.finished.clone();
                let msg = Message {
                    id: mid.clone(),
                    conversation_id: conv_id.clone(),
                    seq: 0,
                    role: Role::Assistant,
                    content: snap.text.trim_start().to_string(),
                    attachments: Vec::new(),
                    model_ref: conv.model_ref.clone(),
                    status: fin.as_ref().map(|f| f.status).unwrap_or(MessageStatus::Done),
                    error: fin.as_ref().and_then(|f| f.error.clone()),
                    prompt_tokens: fin.as_ref().and_then(|f| f.stats.prompt_tokens.map(|v| v as i64)),
                    completion_tokens: fin.as_ref().and_then(|f| f.stats.completion_tokens.map(|v| v as i64)),
                    duration_ms: fin.as_ref().map(|f| f.stats.duration_ms as i64),
                    tokens_per_sec: fin.as_ref().and_then(|f| f.stats.tokens_per_sec),
                    created_at: now_rfc3339(),
                };
                let mut view = message_view(&msg, &names, true);
                if let Some(f) = &fin {
                    if !f.engine.is_empty() && !view.stats.is_empty() {
                        view.stats = format!("{} · {}", view.stats, f.engine);
                    }
                    if let Some(note) = &f.stats.note {
                        view.stats = format!("{} · {note}", view.stats);
                    }
                }
                MessagePartial { m: &view, conv_id: &conv_id }.render().unwrap_or_default()
            });
            // Streamed pages bypass `render`, which would expire the flash message it shows.
            let mut r = streaming::stream_page(html, live, model_label, final_html, state.shutdown.clone());
            crate::web::page::clear_flash(&ctx, &mut r);
            Ok(r)
        }
        _ => Ok(render(&ctx, &page)),
    }
}

/// Fields of the chat composer (multipart because of attachments).
struct ChatForm {
    model_ref: Option<String>,
    text: String,
    attachments: Vec<Attachment>,
}

/// Streams one uploaded file to disk, checks its type by content and normalises images.
pub async fn save_upload(
    field: &mut axum::extract::multipart::Field<'_>,
    uploads: &FsPath,
    original_name: &str,
) -> AppResult<Attachment> {
    let month = chrono::Utc::now().format("%Y-%m").to_string();
    let dir = uploads.join(&month);
    tokio::fs::create_dir_all(&dir).await?;
    let id = crate::util::new_id();
    let tmp = dir.join(format!("{id}.upload"));
    let mut file = tokio::fs::File::create(&tmp).await?;
    let mut size = 0usize;
    let mut head: Vec<u8> = Vec::new();
    loop {
        let chunk = match field.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => {
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(AppError::bad_request(format!("The upload was interrupted: {e}")));
            }
        };
        size += chunk.len();
        if size > crate::media::MAX_UPLOAD_BYTES {
            drop(file);
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(AppError::TooLarge(format!(
                "\"{original_name}\" is larger than {}.",
                human_bytes(crate::media::MAX_UPLOAD_BYTES as u64)
            )));
        }
        if head.len() < 64 {
            head.extend_from_slice(&chunk[..chunk.len().min(64 - head.len())]);
        }
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    drop(file);
    let Some((kind, mime)) = crate::media::sniff(&head, original_name) else {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(AppError::bad_request(format!(
            "\"{original_name}\" isn't a supported file. You can attach images (PNG, JPEG, WebP, GIF), audio (WAV, MP3, FLAC, OGG, M4A) or video (MP4, MOV, WebM, MKV)."
        )));
    };
    let (final_name, mime, size) = if kind == AttachmentKind::Image {
        let bytes = tokio::fs::read(&tmp).await?;
        let _ = tokio::fs::remove_file(&tmp).await;
        let (out, mime, ext) = tokio::task::spawn_blocking(move || crate::media::normalize_image(&bytes))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("{e}")))?
            .map_err(|e| AppError::bad_request(format!("\"{original_name}\" couldn't be read as an image: {e}")))?;
        let name = format!("{month}/{id}.{ext}");
        tokio::fs::write(uploads.join(&name), &out).await?;
        (name, mime, out.len() as u64)
    } else {
        let ext = match mime {
            "audio/wav" => "wav",
            "audio/flac" => "flac",
            "audio/mpeg" => "mp3",
            "audio/ogg" => "ogg",
            "audio/mp4" => "m4a",
            "video/quicktime" => "mov",
            "video/webm" => "webm",
            "video/x-matroska" => "mkv",
            "video/x-msvideo" => "avi",
            _ => "mp4",
        };
        let name = format!("{month}/{id}.{ext}");
        tokio::fs::rename(&tmp, uploads.join(&name)).await?;
        (name, mime, size as u64)
    };
    Ok(Attachment {
        kind,
        file: final_name,
        name: crate::util::truncate_chars(original_name, 120),
        mime: mime.to_string(),
        size,
        frames: Vec::new(),
        transcript: None,
        duration_secs: None,
    })
}

async fn parse_chat_form(state: &SharedState, mut mp: Multipart) -> AppResult<ChatForm> {
    let mut form = ChatForm { model_ref: None, text: String::new(), attachments: Vec::new() };
    let uploads = state.paths.uploads_dir.clone();
    let result: AppResult<()> = async {
        while let Some(mut field) =
            mp.next_field().await.map_err(|e| AppError::bad_request(format!("Couldn't read the form: {e}")))?
        {
            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                "files" => {
                    let fname = field.file_name().unwrap_or("").to_string();
                    if fname.is_empty() {
                        // An empty file input still submits a part with no name.
                        while let Ok(Some(_)) = field.chunk().await {}
                        continue;
                    }
                    if form.attachments.len() >= 8 {
                        return Err(AppError::bad_request("Attach at most 8 files per message."));
                    }
                    let a = save_upload(&mut field, &uploads, &fname).await?;
                    form.attachments.push(a);
                }
                "text" => {
                    form.text = field
                        .text()
                        .await
                        .map_err(|e| AppError::bad_request(format!("Couldn't read your message: {e}")))?;
                }
                "model_ref" => {
                    let v = field.text().await.unwrap_or_default();
                    form.model_ref = (!v.trim().is_empty()).then(|| v.trim().to_string());
                }
                _ => {
                    let _ = field.bytes().await;
                }
            }
        }
        Ok(())
    }
    .await;
    if let Err(e) = result {
        remove_attachments(state, &form.attachments).await;
        return Err(e);
    }
    if form.text.chars().count() > 200_000 {
        remove_attachments(state, &form.attachments).await;
        return Err(AppError::bad_request("That message is too long (over 200,000 characters)."));
    }
    Ok(form)
}

async fn remove_attachments(state: &SharedState, atts: &[Attachment]) {
    for a in atts {
        if let Some(p) = crate::util::safe_join(&state.paths.uploads_dir, &a.file) {
            let _ = tokio::fs::remove_file(p).await;
        }
    }
}

/// Checks attachments against what the chosen model and the installed tools can handle.
async fn check_attachments(state: &SharedState, model_ref: &str, atts: &[Attachment]) -> AppResult<()> {
    if atts.is_empty() {
        return Ok(());
    }
    let needs_vision = atts.iter().any(|a| matches!(a.kind, AttachmentKind::Image | AttachmentKind::Video));
    if needs_vision {
        let vision = match ModelRef::parse(model_ref) {
            Some(ModelRef::Ollama(name)) => {
                state.ollama.show(&name).await.map(|s| s.has_capability("vision")).unwrap_or(false)
            }
            _ => false,
        };
        if !vision {
            return Err(AppError::bad_request(
                "The selected model can't see images or video. Pick a model marked \"sees images\" (for example gemma3, llama3.2-vision or qwen2.5vl from the Ollama library), or remove the attachment.",
            ));
        }
    }
    if atts.iter().any(|a| a.kind == AttachmentKind::Video) && crate::media::ffmpeg_path(&state.settings()).is_none() {
        return Err(AppError::unavailable(format!(
            "Video needs ffmpeg to extract frames. {}",
            crate::media::ffmpeg_install_hint()
        )));
    }
    if atts.iter().any(|a| a.kind == AttachmentKind::Audio)
        && !super::models_of_kind(state, "transcription").await.iter().any(|m| m.analysis.runnable)
    {
        return Err(AppError::bad_request(
            "Audio is turned into text by a Whisper model first. Download one (for example openai/whisper-base) from Discover, then attach the audio again.",
        ));
    }
    Ok(())
}

async fn add_user_message_and_start(
    state: &SharedState,
    conv_id: &str,
    model_ref: &str,
    form: ChatForm,
) -> AppResult<()> {
    if form.text.trim().is_empty() && form.attachments.is_empty() {
        return Err(AppError::bad_request("Type a message or attach a file first."));
    }
    if state.chat.is_generating(conv_id) {
        return Err(AppError::conflict(
            "An answer is still being written. Wait for it or press Stop, then send again.",
        ));
    }
    if let Err(e) = check_attachments(state, model_ref, &form.attachments).await {
        remove_attachments(state, &form.attachments).await;
        return Err(e);
    }
    let title = crate::util::chat_title_from(&form.text);
    let text = if form.text.trim().is_empty() {
        if form.attachments.len() == 1 {
            "Please describe the attached file.".to_string()
        } else {
            "Please describe the attached files.".to_string()
        }
    } else {
        form.text.clone()
    };
    let msg = Message {
        id: crate::util::new_id(),
        conversation_id: conv_id.to_string(),
        seq: 0,
        role: Role::User,
        content: text,
        attachments: form.attachments,
        model_ref: None,
        status: MessageStatus::Done,
        error: None,
        prompt_tokens: None,
        completion_tokens: None,
        duration_ms: None,
        tokens_per_sec: None,
        created_at: now_rfc3339(),
    };
    let cid = conv_id.to_string();
    let now = now_rfc3339();
    state
        .db
        .call(move |c| {
            convdb::append_message(c, msg)?;
            convdb::touch_conversation(c, &cid, &now)?;
            if let Some(t) = &title {
                convdb::title_if_untitled(c, &cid, t)?;
            }
            Ok(())
        })
        .await?;
    state.chat.start(state, conv_id, StartMode::Reply).await?;
    Ok(())
}

pub async fn create(State(state): State<SharedState>, mp: Multipart) -> Response {
    let form = match parse_chat_form(&state, mp).await {
        Ok(f) => f,
        Err(e) => return redirect_flash("/chat?new=1", "error", &e.user_message()),
    };
    let Some(model_ref) = form.model_ref.clone().filter(|m| ModelRef::parse(m).is_some()) else {
        remove_attachments(&state, &form.attachments).await;
        state.chat.save_draft("new", &form.text);
        return redirect_flash(
            "/chat?new=1",
            "error",
            "Choose a model first. If the list is empty, download one from Discover.",
        );
    };
    let settings = state.settings();
    let now = now_rfc3339();
    let conv = Conversation {
        id: crate::util::new_id(),
        title: "New chat".into(),
        model_ref: Some(model_ref.clone()),
        system_prompt: settings.generation.system_prompt.clone(),
        params: GenParams::from_defaults(&settings.generation),
        created_at: now.clone(),
        updated_at: now,
    };
    let c2 = conv.clone();
    if let Err(e) = state.db.call(move |c| convdb::insert_conversation(c, &c2)).await {
        return AppError::from(e).into_response();
    }
    let has_content = !form.text.trim().is_empty() || !form.attachments.is_empty();
    if has_content {
        let text = form.text.clone();
        if let Err(e) = add_user_message_and_start(&state, &conv.id, &model_ref, form).await {
            // Don't leave an empty "New chat" behind: go back to the new-chat form with the
            // text and the chosen model kept.
            let id = conv.id.clone();
            let _ = state.db.call(move |c| convdb::delete_conversation(c, &id)).await;
            state.chat.save_draft("new", &text);
            return redirect_flash(
                &format!("/chat?new=1&model={}", crate::util::url_encode(&model_ref)),
                "error",
                &e.user_message(),
            );
        }
    }
    redirect(&format!("/chat/{}#bottom", conv.id))
}

async fn load_conv(state: &SharedState, id: &str) -> AppResult<Conversation> {
    let i = id.to_string();
    state
        .db
        .call(move |c| convdb::get_conversation(c, &i))
        .await?
        .ok_or_else(|| AppError::not_found("This conversation doesn't exist (it may have been deleted)."))
}

pub async fn send(State(state): State<SharedState>, Path(id): Path<String>, mp: Multipart) -> Response {
    let back = format!("/chat/{id}");
    let conv = match load_conv(&state, &id).await {
        Ok(c) => c,
        Err(e) => return e.into_response(),
    };
    let form = match parse_chat_form(&state, mp).await {
        Ok(f) => f,
        Err(e) => return redirect_flash(&back, "error", &e.user_message()),
    };
    let Some(model_ref) = conv.model_ref.clone() else {
        state.chat.save_draft(&id, &form.text);
        remove_attachments(&state, &form.attachments).await;
        return redirect_flash(&back, "error", "Choose a model for this chat first (Chat settings).");
    };
    let text = form.text.clone();
    match add_user_message_and_start(&state, &id, &model_ref, form).await {
        Ok(()) => redirect(&format!("{back}#bottom")),
        Err(e) => {
            state.chat.save_draft(&id, &text);
            redirect_flash(&back, "error", &e.user_message())
        }
    }
}

pub async fn stop(State(state): State<SharedState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    let stopped = state.chat.stop(&id);
    let in_frame = headers.get("sec-fetch-dest").and_then(|v| v.to_str().ok()).map(|v| v == "iframe").unwrap_or(false);
    if in_frame {
        // The stop form targets a hidden frame so the streaming page keeps running.
        return StatusCode::NO_CONTENT.into_response();
    }
    if stopped {
        redirect(&format!("/chat/{id}#bottom"))
    } else {
        redirect_flash(&format!("/chat/{id}"), "info", "Nothing is being generated.")
    }
}

/// Checks what `ChatService::start` would refuse before it generates anything. Regenerate
/// and Edit delete messages first, so a refusal afterwards would lose them for nothing.
async fn check_can_answer(state: &SharedState, id: &str) -> AppResult<()> {
    let i = id.to_string();
    let conv = state
        .db
        .call(move |c| convdb::get_conversation(c, &i))
        .await?
        .ok_or_else(|| AppError::not_found("This conversation no longer exists."))?;
    match conv.model_ref.as_deref().map(ModelRef::parse) {
        None => Err(AppError::bad_request("Pick a model for this chat first, in Chat settings.")),
        Some(None) => Err(AppError::bad_request("The selected model is invalid; pick another one in Chat settings.")),
        Some(Some(_)) => Ok(()),
    }
}

pub async fn regenerate(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    let back = format!("/chat/{id}");
    if state.chat.is_generating(&id) {
        return redirect_flash(&back, "error", "Wait for the current answer to finish, or press Stop.");
    }
    if let Err(e) = check_can_answer(&state, &id).await {
        return redirect_flash(&back, "error", &e.user_message());
    }
    let i = id.clone();
    let res = state
        .db
        .call(move |c| {
            let msgs = convdb::messages(c, &i)?;
            if let Some(last) = msgs.last()
                && last.role == Role::Assistant
            {
                convdb::delete_messages_from(c, &i, last.seq)?;
            }
            Ok(())
        })
        .await;
    if let Err(e) = res {
        return AppError::from(e).into_response();
    }
    match state.chat.start(&state, &id, StartMode::Reply).await {
        Ok(_) => redirect(&format!("{back}#bottom")),
        Err(e) => redirect_flash(&back, "error", &e.user_message()),
    }
}

pub async fn continue_answer(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    let back = format!("/chat/{id}");
    match state.chat.start(&state, &id, StartMode::Continue).await {
        Ok(_) => redirect(&format!("{back}#bottom")),
        Err(e) => redirect_flash(&back, "error", &e.user_message()),
    }
}

#[derive(Deserialize, Default)]
pub struct EditForm {
    #[serde(default)]
    text: String,
}

pub async fn edit_message(
    State(state): State<SharedState>,
    Path((id, mid)): Path<(String, String)>,
    Form(f): Form<EditForm>,
) -> Response {
    let back = format!("/chat/{id}");
    if f.text.trim().is_empty() {
        return redirect_flash(&back, "error", "The message can't be empty.");
    }
    if state.chat.is_generating(&id) {
        return redirect_flash(&back, "error", "Wait for the current answer to finish, or press Stop.");
    }
    if let Err(e) = check_can_answer(&state, &id).await {
        return redirect_flash(&back, "error", &e.user_message());
    }
    let (i, m, text) = (id.clone(), mid.clone(), f.text.clone());
    let res = state
        .db
        .call(move |c| {
            let Some(msg) = convdb::get_message(c, &m)? else {
                return Ok(false);
            };
            if msg.conversation_id != i || msg.role != Role::User {
                return Ok(false);
            }
            convdb::update_message_content(c, &m, &text)?;
            convdb::delete_messages_from(c, &i, msg.seq + 1)?;
            Ok(true)
        })
        .await;
    match res {
        Ok(true) => match state.chat.start(&state, &id, StartMode::Reply).await {
            Ok(_) => redirect(&format!("{back}#bottom")),
            Err(e) => redirect_flash(&back, "error", &e.user_message()),
        },
        Ok(false) => redirect_flash(&back, "error", "Only your own messages can be edited."),
        Err(e) => AppError::from(e).into_response(),
    }
}

#[derive(Deserialize, Default)]
pub struct RenameForm {
    #[serde(default)]
    title: String,
}

pub async fn rename(State(state): State<SharedState>, Path(id): Path<String>, Form(f): Form<RenameForm>) -> Response {
    let title = f.title.trim().to_string();
    if title.is_empty() || title.chars().count() > 120 {
        return redirect_flash(&format!("/chat/{id}"), "error", "Titles must be 1–120 characters.");
    }
    let i = id.clone();
    let now = now_rfc3339();
    match state.db.call(move |c| convdb::rename_conversation(c, &i, &title, &now)).await {
        Ok(_) => redirect_flash(&format!("/chat/{id}"), "ok", "Renamed."),
        Err(e) => AppError::from(e).into_response(),
    }
}

pub async fn delete(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    state.chat.stop(&id);
    let i = id.clone();
    match state.db.call(move |c| convdb::delete_conversation(c, &i)).await {
        Ok(_) => redirect_flash("/chat?new=1", "ok", "Conversation deleted."),
        Err(e) => AppError::from(e).into_response(),
    }
}

#[derive(Deserialize, Default)]
pub struct SettingsForm {
    #[serde(default)]
    model_ref: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    temperature: Option<String>,
    #[serde(default)]
    top_p: Option<String>,
    #[serde(default)]
    top_k: Option<String>,
    #[serde(default)]
    max_tokens: Option<String>,
    #[serde(default)]
    repeat_penalty: Option<String>,
    #[serde(default)]
    context_length: Option<String>,
    #[serde(default)]
    seed: Option<String>,
    #[serde(default)]
    reset: Option<String>,
}

fn parse_num<T: std::str::FromStr>(field: &str, v: &Option<String>) -> AppResult<Option<T>> {
    match v.as_deref().map(str::trim) {
        None | Some("") => Ok(None),
        Some(s) => s.parse::<T>().map(Some).map_err(|_| AppError::bad_request(format!("{field} must be a number."))),
    }
}

pub async fn settings(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Form(f): Form<SettingsForm>,
) -> Response {
    let back = format!("/chat/{id}");
    let mut conv = match load_conv(&state, &id).await {
        Ok(c) => c,
        Err(e) => return e.into_response(),
    };
    let result: AppResult<&'static str> = (|| {
        if f.reset.is_some() {
            let s = state.settings();
            conv.params = GenParams::from_defaults(&s.generation);
            conv.system_prompt = s.generation.system_prompt.clone();
            return Ok("Settings reset to your defaults.");
        }
        if let Some(m) = f.model_ref.as_deref().filter(|m| !m.is_empty()) {
            if ModelRef::parse(m).is_none() {
                return Err(AppError::bad_request("Unknown model."));
            }
            conv.model_ref = Some(m.to_string());
        }
        if let Some(sp) = &f.system_prompt {
            if sp.chars().count() > 20_000 {
                return Err(AppError::bad_request("The system prompt is too long (max 20,000 characters)."));
            }
            conv.system_prompt = sp.trim().to_string();
        }
        let p = &mut conv.params;
        if let Some(v) = parse_num::<f32>("Temperature", &f.temperature)? {
            p.temperature = v;
        }
        if let Some(v) = parse_num::<f32>("Top-p", &f.top_p)? {
            p.top_p = v;
        }
        if let Some(v) = parse_num::<u32>("Top-k", &f.top_k)? {
            p.top_k = v;
        }
        // An empty field means no limit (a missing one leaves the setting alone).
        if f.max_tokens.is_some() {
            p.max_tokens = parse_num::<u32>("Maximum answer length", &f.max_tokens)?.unwrap_or(0);
        }
        if let Some(v) = parse_num::<f32>("Repetition penalty", &f.repeat_penalty)? {
            p.repeat_penalty = v;
        }
        if let Some(v) = parse_num::<u32>("Context length", &f.context_length)? {
            p.context_length = v;
        }
        if f.seed.is_some() {
            p.seed = parse_num::<u64>("Seed", &f.seed)?;
        }
        *p = p.clone().sanitized();
        Ok("Chat settings saved.")
    })();
    match result {
        Ok(msg) => {
            let c = conv.clone();
            let now = now_rfc3339();
            match state
                .db
                .call(move |db| {
                    convdb::update_conversation_settings(
                        db,
                        &c.id,
                        c.model_ref.as_deref(),
                        &c.system_prompt,
                        &c.params,
                        &now,
                    )
                })
                .await
            {
                Ok(_) => redirect_flash(&back, "ok", msg),
                Err(e) => AppError::from(e).into_response(),
            }
        }
        Err(e) => redirect_flash(&back, "error", &e.user_message()),
    }
}

#[derive(Deserialize, Default)]
pub struct ExportQuery {
    #[serde(default)]
    format: Option<String>,
}

#[derive(serde::Serialize, Deserialize)]
pub struct ExportedMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub model_ref: Option<String>,
    #[serde(default)]
    pub prompt_tokens: Option<i64>,
    #[serde(default)]
    pub completion_tokens: Option<i64>,
    #[serde(default)]
    pub attachments: Vec<String>,
}

#[derive(serde::Serialize, Deserialize)]
pub struct ExportedConversation {
    pub format: String,
    pub version: u32,
    #[serde(default)]
    pub exported_at: Option<String>,
    pub title: String,
    #[serde(default)]
    pub model_ref: Option<String>,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub params: GenParams,
    #[serde(default)]
    pub created_at: Option<String>,
    pub messages: Vec<ExportedMessage>,
}

pub fn export_json(conv: &Conversation, msgs: &[Message]) -> ExportedConversation {
    ExportedConversation {
        format: "local-ai-studio.conversation".into(),
        version: 1,
        exported_at: Some(now_rfc3339()),
        title: conv.title.clone(),
        model_ref: conv.model_ref.clone(),
        system_prompt: conv.system_prompt.clone(),
        params: conv.params.clone(),
        created_at: Some(conv.created_at.clone()),
        messages: msgs
            .iter()
            .map(|m| ExportedMessage {
                role: m.role.as_str().into(),
                content: m.content.clone(),
                created_at: Some(m.created_at.clone()),
                model_ref: m.model_ref.clone(),
                prompt_tokens: m.prompt_tokens,
                completion_tokens: m.completion_tokens,
                attachments: m.attachments.iter().map(|a| a.name.clone()).collect(),
            })
            .collect(),
    }
}

pub fn export_markdown(conv: &Conversation, msgs: &[Message]) -> String {
    let mut s = format!("# {}\n\n", conv.title);
    if let Some(m) = &conv.model_ref {
        s.push_str(&format!("*Model:* `{m}`  \n"));
    }
    s.push_str(&format!("*Created:* {}\n\n", short_datetime(&conv.created_at)));
    if !conv.system_prompt.is_empty() {
        s.push_str(&format!("> **System prompt:** {}\n\n", conv.system_prompt.replace('\n', "\n> ")));
    }
    for m in msgs {
        let who = match m.role {
            Role::User => "You",
            Role::Assistant => "Assistant",
            Role::System => "System",
        };
        s.push_str(&format!("### {who} · {}\n\n", short_datetime(&m.created_at)));
        for a in &m.attachments {
            s.push_str(&format!("*Attachment:* {}\n\n", a.name));
        }
        s.push_str(&m.content);
        s.push_str("\n\n");
    }
    s
}

fn safe_filename(title: &str) -> String {
    let base = crate::util::sanitize_component(title);
    crate::util::truncate_chars(&base, 60).replace('…', "")
}

pub async fn export(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Query(q): Query<ExportQuery>,
) -> AppResult<Response> {
    let i = id.clone();
    let (conv, msgs) = state.db.call(move |c| Ok((convdb::get_conversation(c, &i)?, convdb::messages(c, &i)?))).await?;
    let conv = conv.ok_or_else(|| AppError::not_found("Conversation not found."))?;
    let name = safe_filename(&conv.title);
    let (body, mime, ext) = if q.format.as_deref() == Some("md") {
        (export_markdown(&conv, &msgs), "text/markdown; charset=utf-8", "md")
    } else {
        (
            serde_json::to_string_pretty(&export_json(&conv, &msgs)).map_err(|e| AppError::Internal(e.into()))?,
            "application/json",
            "json",
        )
    };
    let mut r = body.into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(mime));
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{name}.{ext}\"")) {
        r.headers_mut().insert(header::CONTENT_DISPOSITION, v);
    }
    Ok(r)
}

/// Parses an exported conversation, or a plain `{"messages": [...]}` / `[...]` list.
pub fn parse_import(bytes: &[u8]) -> Result<ExportedConversation, String> {
    if let Ok(c) = serde_json::from_slice::<ExportedConversation>(bytes) {
        return Ok(c);
    }
    #[derive(Deserialize)]
    struct Plain {
        #[serde(default)]
        title: Option<String>,
        messages: Vec<ExportedMessage>,
    }
    let plain: Plain = match serde_json::from_slice::<Plain>(bytes) {
        Ok(p) => p,
        Err(_) => match serde_json::from_slice::<Vec<ExportedMessage>>(bytes) {
            Ok(m) => Plain { title: None, messages: m },
            Err(e) => return Err(format!("This file isn't a conversation export ({e}).")),
        },
    };
    Ok(ExportedConversation {
        format: "local-ai-studio.conversation".into(),
        version: 1,
        exported_at: None,
        title: plain.title.unwrap_or_else(|| "Imported chat".into()),
        model_ref: None,
        system_prompt: String::new(),
        params: GenParams::default(),
        created_at: None,
        messages: plain.messages,
    })
}

pub async fn import(State(state): State<SharedState>, mut mp: Multipart) -> Response {
    let mut bytes = Vec::new();
    while let Ok(Some(field)) = mp.next_field().await {
        if field.name() == Some("file") {
            match field.bytes().await {
                Ok(b) => bytes = b.to_vec(),
                Err(e) => return redirect_flash("/chat?new=1", "error", &format!("Upload failed: {e}")),
            }
        }
    }
    if bytes.is_empty() {
        return redirect_flash("/chat?new=1", "error", "Choose a .json file exported from Local AI Studio.");
    }
    let parsed = match parse_import(&bytes) {
        Ok(p) => p,
        Err(e) => return redirect_flash("/chat?new=1", "error", &e),
    };
    if parsed.messages.len() > 10_000 {
        return redirect_flash("/chat?new=1", "error", "That conversation has too many messages to import.");
    }
    let now = now_rfc3339();
    let conv = Conversation {
        id: crate::util::new_id(),
        title: crate::util::truncate_chars(&parsed.title, 120),
        model_ref: parsed.model_ref.clone().filter(|m| ModelRef::parse(m).is_some()),
        system_prompt: parsed.system_prompt.clone(),
        params: parsed.params.clone().sanitized(),
        created_at: parsed.created_at.clone().unwrap_or_else(|| now.clone()),
        updated_at: now.clone(),
    };
    let id = conv.id.clone();
    let msgs: Vec<Message> = parsed
        .messages
        .into_iter()
        .filter(|m| matches!(m.role.as_str(), "user" | "assistant" | "system"))
        .map(|m| Message {
            id: crate::util::new_id(),
            conversation_id: id.clone(),
            seq: 0,
            role: Role::parse(&m.role),
            content: if m.attachments.is_empty() {
                m.content
            } else {
                format!("{}\n\n[Attachments not included in the import: {}]", m.content, m.attachments.join(", "))
            },
            attachments: Vec::new(),
            model_ref: m.model_ref,
            status: MessageStatus::Done,
            error: None,
            prompt_tokens: m.prompt_tokens,
            completion_tokens: m.completion_tokens,
            duration_ms: None,
            tokens_per_sec: None,
            created_at: m.created_at.unwrap_or_else(|| now.clone()),
        })
        .collect();
    let n = msgs.len();
    let res = state
        .db
        .call_mut(move |c| {
            let tx = c.transaction()?;
            convdb::insert_conversation(&tx, &conv)?;
            for m in msgs {
                convdb::append_message(&tx, m)?;
            }
            tx.commit()
        })
        .await;
    match res {
        Ok(()) => redirect_flash(&format!("/chat/{id}"), "ok", &format!("Imported {n} messages.")),
        Err(e) => AppError::from(e).into_response(),
    }
}

pub async fn raw_message(
    State(state): State<SharedState>,
    Path((id, mid)): Path<(String, String)>,
) -> AppResult<Response> {
    let m = mid.clone();
    let msg = state
        .db
        .call(move |c| convdb::get_message(c, &m))
        .await?
        .filter(|m| m.conversation_id == id)
        .ok_or_else(|| AppError::not_found("Message not found."))?;
    let mut r = msg.content.into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/markdown; charset=utf-8"));
    r.headers_mut()
        .insert(header::CONTENT_DISPOSITION, HeaderValue::from_static("attachment; filename=\"message.md\""));
    Ok(r)
}

/// Used by the transcription page: start a chat whose first message is a transcript.
pub async fn new_conversation_with_text(state: &SharedState, title: &str, text: &str) -> AppResult<String> {
    let settings = state.settings();
    let (choices, _) = super::chat_choices(state, None).await;
    let now = now_rfc3339();
    let conv = Conversation {
        id: crate::util::new_id(),
        title: crate::util::truncate_chars(title, 120),
        model_ref: choices.first().map(|c| c.value.clone()),
        system_prompt: settings.generation.system_prompt.clone(),
        params: GenParams::from_defaults(&settings.generation),
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    let id = conv.id.clone();
    let msg = Message {
        id: crate::util::new_id(),
        conversation_id: id.clone(),
        seq: 0,
        role: Role::User,
        content: text.to_string(),
        attachments: Vec::new(),
        model_ref: None,
        status: MessageStatus::Done,
        error: None,
        prompt_tokens: None,
        completion_tokens: None,
        duration_ms: None,
        tokens_per_sec: None,
        created_at: now,
    };
    state
        .db
        .call(move |c| {
            convdb::insert_conversation(c, &conv)?;
            convdb::append_message(c, msg)?;
            Ok(())
        })
        .await?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_accept_plain_message_lists() {
        let c = parse_import(br#"[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]"#).unwrap();
        assert_eq!(c.messages.len(), 2);
        let c = parse_import(br#"{"title":"T","messages":[{"role":"user","content":"x"}]}"#).unwrap();
        assert_eq!(c.title, "T");
        assert!(parse_import(b"not json").is_err());
    }

    #[test]
    fn markdown_export_contains_messages() {
        let conv = Conversation {
            id: "c".into(),
            title: "Trip".into(),
            model_ref: Some("ollama:x".into()),
            system_prompt: "Be nice".into(),
            params: GenParams::default(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
        };
        let msgs = vec![Message {
            id: "m".into(),
            conversation_id: "c".into(),
            seq: 1,
            role: Role::User,
            content: "Plan a trip".into(),
            attachments: vec![],
            model_ref: None,
            status: MessageStatus::Done,
            error: None,
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: None,
            tokens_per_sec: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        }];
        let md = export_markdown(&conv, &msgs);
        assert!(md.starts_with("# Trip"));
        assert!(md.contains("Plan a trip"));
        let json = export_json(&conv, &msgs);
        let s = serde_json::to_string(&json).unwrap();
        let back = parse_import(s.as_bytes()).unwrap();
        assert_eq!(back.messages[0].content, "Plan a trip");
        assert_eq!(back.system_prompt, "Be nice");
    }
}
