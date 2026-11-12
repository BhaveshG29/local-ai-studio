//! JSON API for scripts and other tools: health, system info, models, downloads, chat, and an
//! OpenAI-compatible `/v1/chat/completions` endpoint (text only).

use std::convert::Infallible;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::SharedState;
use crate::error::{AppError, AppResult};
use crate::inference::params::GenParams;
use crate::inference::{EngineMessage, EngineRequest, GenEvent, GenStats, ModelRef};

pub async fn health(State(state): State<SharedState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "app": crate::APP_NAME,
        "version": crate::APP_VERSION,
        "uptime_secs": state.server.started_at.elapsed().as_secs(),
    }))
}

pub async fn system(State(state): State<SharedState>) -> Json<Value> {
    let s = state.system.snapshot().await;
    Json(serde_json::to_value(&*s).unwrap_or(Value::Null))
}

pub async fn models(State(state): State<SharedState>) -> Json<Value> {
    let (items, err) = crate::web::pages::library_items(&state).await;
    Json(json!({ "models": items, "ollama_error": err }))
}

pub async fn downloads(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let jobs = state.downloads.list().await?;
    Ok(Json(json!({ "downloads": jobs })))
}

#[derive(Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: Value,
}

fn content_text(v: &Value) -> AppResult<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => out.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                    Some(other) => {
                        return Err(AppError::bad_request(format!(
                            "content part type \"{other}\" is not supported by this API (text only)"
                        )));
                    }
                    None => {}
                }
            }
            Ok(out)
        }
        Value::Null => Ok(String::new()),
        _ => Err(AppError::bad_request("message content must be a string")),
    }
}

fn parse_model(model: &str) -> AppResult<ModelRef> {
    if let Some(m) = ModelRef::parse(model) {
        return Ok(m);
    }
    crate::ollama::validate_model_name(model)?;
    Ok(ModelRef::Ollama(model.to_string()))
}

fn to_engine_messages(msgs: &[ApiMessage]) -> AppResult<Vec<EngineMessage>> {
    if msgs.is_empty() {
        return Err(AppError::bad_request("messages must not be empty"));
    }
    msgs.iter()
        .map(|m| {
            if !matches!(m.role.as_str(), "system" | "user" | "assistant") {
                return Err(AppError::bad_request(format!("unsupported role \"{}\"", m.role)));
            }
            Ok(EngineMessage { role: m.role.clone(), content: content_text(&m.content)?, images: Vec::new() })
        })
        .collect()
}

/// Runs a request to completion, collecting all text.
async fn run_to_end(state: &SharedState, model: &ModelRef, req: EngineRequest) -> AppResult<(String, GenStats)> {
    let engine = state.inference.engine_for(model).await?;
    let (tx, mut rx) = mpsc::unbounded_channel();
    // Shutting the app down stops the answer instead of waiting for it.
    let cancel = state.shutdown.child_token();
    let handle = tokio::spawn(engine.chat(req, cancel.clone(), tx));
    let mut text = String::new();
    let mut out = Err(AppError::upstream("the engine stopped without an answer"));
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Text(t) => text.push_str(&t),
            GenEvent::Phase(_) => {}
            GenEvent::Done(stats) => {
                out = Ok(stats);
                break;
            }
            GenEvent::Error(e) => {
                out = Err(AppError::upstream(e));
                break;
            }
        }
    }
    let _ = handle.await;
    out.map(|s| (text.trim_start().to_string(), s))
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ApiMessage>,
    #[serde(default)]
    pub params: Option<GenParams>,
    #[serde(default)]
    pub stream: bool,
}

/// `POST /api/v1/chat` — native API. With `stream: true` the response is NDJSON:
/// `{"delta": "..."}` lines followed by `{"done": true, ...}`.
pub async fn chat(State(state): State<SharedState>, Json(req): Json<ChatRequest>) -> AppResult<Response> {
    let model = parse_model(&req.model)?;
    let params = req.params.unwrap_or_else(|| GenParams::from_defaults(&state.settings().generation)).sanitized();
    let ereq = EngineRequest { messages: to_engine_messages(&req.messages)?, params, continue_from: None };
    if !req.stream {
        let (text, stats) = run_to_end(&state, &model, ereq).await?;
        return Ok(Json(
            json!({ "model": req.model, "message": {"role": "assistant", "content": text}, "stats": stats }),
        )
        .into_response());
    }
    let engine = state.inference.engine_for(&model).await?;
    Ok(stream_lines(
        engine,
        ereq,
        state.shutdown.child_token(),
        |ev| match ev {
            GenEvent::Text(t) => Some(format!("{}\n", json!({ "delta": t }))),
            GenEvent::Done(s) => Some(format!("{}\n", json!({ "done": true, "stats": s }))),
            GenEvent::Error(e) => Some(format!("{}\n", json!({ "error": e }))),
            GenEvent::Phase(_) => None,
        },
        "application/x-ndjson",
        None,
    ))
}

/// Streams engine events through `fmt`; cancels generation when the client disconnects or
/// `cancel` (a child of the app's shutdown token) fires.
fn stream_lines(
    engine: std::sync::Arc<dyn crate::inference::ChatEngine>,
    req: EngineRequest,
    cancel: CancellationToken,
    fmt: impl Fn(GenEvent) -> Option<String> + Send + 'static,
    content_type: &'static str,
    trailer: Option<&'static str>,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(64);
    tokio::spawn(async move {
        let (etx, mut erx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(engine.chat(req, cancel.clone(), etx));
        while let Some(ev) = erx.recv().await {
            let end = matches!(ev, GenEvent::Done(_) | GenEvent::Error(_));
            if let Some(line) = fmt(ev)
                && tx.send(Ok(Bytes::from(line))).await.is_err()
            {
                cancel.cancel();
                break;
            }
            if end {
                break;
            }
        }
        if let Some(t) = trailer {
            let _ = tx.send(Ok(Bytes::from_static(t.as_bytes()))).await;
        }
        let _ = handle.await;
    });
    let mut r = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)).into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    r
}

pub async fn openai_models(State(state): State<SharedState>) -> Json<Value> {
    let (items, _) = crate::web::pages::library_items(&state).await;
    let data: Vec<Value> = items
        .iter()
        .filter(|i| i.is_chat() && i.runnable)
        .map(|i| json!({ "id": i.model_ref, "object": "model", "created": 0, "owned_by": if i.engine == "Ollama" { "ollama" } else { "local-ai-studio" } }))
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

#[derive(Deserialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<ApiMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
}

/// `POST /v1/chat/completions` — OpenAI-compatible (text only).
pub async fn openai_chat(State(state): State<SharedState>, Json(req): Json<OpenAiRequest>) -> AppResult<Response> {
    let model = parse_model(&req.model)?;
    let mut params = GenParams::from_defaults(&state.settings().generation);
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(p) = req.top_p {
        params.top_p = p;
    }
    if let Some(m) = req.max_completion_tokens.or(req.max_tokens) {
        params.max_tokens = m;
    }
    params.seed = req.seed.or(params.seed);
    if let Some(fp) = req.frequency_penalty {
        params.repeat_penalty = 1.0 + fp.clamp(0.0, 2.0) / 2.0;
    }
    let ereq =
        EngineRequest { messages: to_engine_messages(&req.messages)?, params: params.sanitized(), continue_from: None };
    let id = format!("chatcmpl-{}", crate::util::new_id());
    let created = chrono::Utc::now().timestamp();
    let model_name = req.model.clone();
    if !req.stream {
        let (text, stats) = run_to_end(&state, &model, ereq).await?;
        let p = stats.prompt_tokens.unwrap_or(0);
        let c = stats.completion_tokens.unwrap_or(0);
        return Ok(Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": if stats.stop_reason == "length" || stats.stop_reason == "context" { "length" } else { "stop" },
            }],
            "usage": { "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c },
        }))
        .into_response());
    }
    let engine = state.inference.engine_for(&model).await?;
    let first = std::sync::atomic::AtomicBool::new(true);
    Ok(stream_lines(
        engine,
        ereq,
        state.shutdown.child_token(),
        move |ev| {
            let chunk = |delta: Value, finish: Value| {
                format!(
                    "data: {}\n\n",
                    json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
                )
            };
            match ev {
                GenEvent::Text(t) => {
                    let delta = if first.swap(false, std::sync::atomic::Ordering::Relaxed) {
                        json!({ "role": "assistant", "content": t })
                    } else {
                        json!({ "content": t })
                    };
                    Some(chunk(delta, Value::Null))
                }
                GenEvent::Done(s) => Some(chunk(
                    json!({}),
                    json!(if s.stop_reason == "length" || s.stop_reason == "context" { "length" } else { "stop" }),
                )),
                GenEvent::Error(e) => Some(format!("data: {}\n\n", json!({ "error": { "message": e } }))),
                GenEvent::Phase(_) => None,
            }
        },
        "text/event-stream",
        Some("data: [DONE]\n\n"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_parts_are_flattened() {
        let v: Value = serde_json::from_str(r#"[{"type":"text","text":"a"},{"type":"text","text":"b"}]"#).unwrap();
        assert_eq!(content_text(&v).unwrap(), "ab");
        let img: Value = serde_json::from_str(r#"[{"type":"image_url","image_url":{"url":"x"}}]"#).unwrap();
        assert!(content_text(&img).is_err());
    }

    #[test]
    fn model_names_resolve() {
        assert_eq!(parse_model("llama3.2").unwrap(), ModelRef::Ollama("llama3.2".into()));
        assert_eq!(parse_model("local:abc").unwrap(), ModelRef::Local("abc".into()));
        assert!(parse_model("../x").is_err());
    }
}
