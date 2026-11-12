//! Inference: a small engine abstraction with two text engines (Ollama and the built-in
//! Candle engine), plus speech-to-text (Whisper) and image generation (Stable Diffusion) on
//! Candle. New engines implement [`ChatEngine`] and are selected in [`manager`].

pub mod candle_llm;
pub mod chat;
pub mod diffusion;
pub mod manager;
pub mod params;
pub mod template;
pub mod threads;
pub mod tokenizer;
pub mod whisper;

use std::path::PathBuf;

use futures_util::future::BoxFuture;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use params::{GenParams, ParamSupport};

/// Identifies a model across engines: `ollama:<name>` or `local:<library id>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelRef {
    Ollama(String),
    Local(String),
}

impl ModelRef {
    pub fn parse(s: &str) -> Option<Self> {
        if let Some(n) = s.strip_prefix("ollama:") {
            (!n.is_empty()).then(|| ModelRef::Ollama(n.to_string()))
        } else if let Some(id) = s.strip_prefix("local:") {
            (!id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric())).then(|| ModelRef::Local(id.to_string()))
        } else {
            None
        }
    }
}

impl std::fmt::Display for ModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelRef::Ollama(n) => write!(f, "ollama:{n}"),
            ModelRef::Local(id) => write!(f, "local:{id}"),
        }
    }
}

/// One message given to an engine.
#[derive(Clone, Debug)]
pub struct EngineMessage {
    pub role: String,
    pub content: String,
    /// Image files (already validated) attached to this message.
    pub images: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct EngineRequest {
    pub messages: Vec<EngineMessage>,
    pub params: GenParams,
    /// When set, the model continues this partial assistant answer instead of starting anew.
    pub continue_from: Option<String>,
}

/// Statistics reported when generation ends.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GenStats {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub duration_ms: u64,
    pub tokens_per_sec: Option<f64>,
    /// `stop`, `length`, `context`, `cancelled`.
    pub stop_reason: String,
    pub note: Option<String>,
}

#[derive(Clone, Debug)]
pub enum GenEvent {
    /// Status shown before text arrives ("Loading model…").
    Phase(String),
    Text(String),
    Done(GenStats),
    Error(String),
}

/// A text-generation engine.
pub trait ChatEngine: Send + Sync {
    /// Engine name shown in the UI ("Ollama", "Built-in (CPU)").
    fn name(&self) -> String;
    /// Which settings are honoured.
    fn param_support(&self) -> ParamSupport;
    /// Runs one chat completion, sending events until `Done` or `Error`. Must stop promptly
    /// when `cancel` fires (sending `Done` with `stop_reason = "cancelled"`).
    fn chat(
        &self,
        req: EngineRequest,
        cancel: CancellationToken,
        events: mpsc::UnboundedSender<GenEvent>,
    ) -> BoxFuture<'static, ()>;
}

/// The Ollama engine: Ollama applies the model's template and manages hardware itself.
pub struct OllamaEngine {
    pub client: crate::ollama::OllamaClient,
    pub model: String,
    pub vision: bool,
}

impl ChatEngine for OllamaEngine {
    fn name(&self) -> String {
        "Ollama".into()
    }

    fn param_support(&self) -> ParamSupport {
        ParamSupport { images: self.vision, ..ParamSupport::ALL_TEXT }
    }

    fn chat(
        &self,
        req: EngineRequest,
        cancel: CancellationToken,
        events: mpsc::UnboundedSender<GenEvent>,
    ) -> BoxFuture<'static, ()> {
        let client = self.client.clone();
        let model = self.model.clone();
        let vision = self.vision;
        Box::pin(async move {
            use base64::Engine as _;
            use futures_util::StreamExt;
            let started = std::time::Instant::now();
            let mut messages = Vec::new();
            for m in &req.messages {
                let mut images = Vec::new();
                if vision {
                    for p in &m.images {
                        match tokio::fs::read(p).await {
                            Ok(bytes) => images.push(base64::engine::general_purpose::STANDARD.encode(bytes)),
                            Err(e) => {
                                let _ = events.send(GenEvent::Error(format!("Could not read an attached image: {e}")));
                                return;
                            }
                        }
                    }
                }
                messages.push(crate::ollama::ChatMessage { role: m.role.clone(), content: m.content.clone(), images });
            }
            if let Some(partial) = &req.continue_from {
                messages.push(crate::ollama::ChatMessage {
                    role: "assistant".into(),
                    content: partial.clone(),
                    images: Vec::new(),
                });
            }
            let p = &req.params;
            let options = crate::ollama::ChatOptions {
                temperature: p.temperature,
                top_p: p.top_p,
                top_k: p.top_k,
                // No limit (0) becomes the context length: Ollama reads 0 as "write nothing",
                // and -1 (infinite) can run on forever once it starts shifting the context.
                num_predict: if p.max_tokens == 0 { i64::from(p.context_length) } else { i64::from(p.max_tokens) },
                repeat_penalty: p.repeat_penalty,
                num_ctx: p.context_length,
                seed: p.seed,
            };
            let _ = events.send(GenEvent::Phase("Waiting for Ollama…".into()));
            let resp = tokio::select! {
                r = client.chat_request(&model, &messages, &options) => r,
                _ = cancel.cancelled() => {
                    let _ = events.send(GenEvent::Done(GenStats { stop_reason: "cancelled".into(), duration_ms: started.elapsed().as_millis() as u64, ..Default::default() }));
                    return;
                }
            };
            let resp = match resp {
                Ok(r) => r,
                Err(e) => {
                    let _ = events.send(GenEvent::Error(e.user_message()));
                    return;
                }
            };
            let mut stream = resp.bytes_stream();
            let mut lines = crate::ollama::LineBuffer::default();
            let mut in_thinking = false;
            loop {
                let next = tokio::select! {
                    n = tokio::time::timeout(std::time::Duration::from_secs(600), stream.next()) => n,
                    _ = cancel.cancelled() => {
                        // Dropping the response stream tells Ollama to stop generating.
                        let _ = events.send(GenEvent::Done(GenStats { stop_reason: "cancelled".into(), duration_ms: started.elapsed().as_millis() as u64, ..Default::default() }));
                        return;
                    }
                };
                let chunk = match next {
                    Err(_) => {
                        let _ = events.send(GenEvent::Error("Ollama stopped responding for 10 minutes.".into()));
                        return;
                    }
                    Ok(None) => {
                        let _ = events.send(GenEvent::Error(
                            "The connection to Ollama closed before the answer finished.".into(),
                        ));
                        return;
                    }
                    Ok(Some(Err(e))) => {
                        let _ = events.send(GenEvent::Error(format!("Connection to Ollama failed: {e}")));
                        return;
                    }
                    Ok(Some(Ok(b))) => b,
                };
                for line in lines.push(&chunk) {
                    let c: crate::ollama::ChatChunk = match serde_json::from_str(&line) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    if let Some(err) = c.error {
                        let _ = events.send(GenEvent::Error(format!("Ollama: {err}")));
                        return;
                    }
                    if let Some(msg) = &c.message {
                        if let Some(t) = msg.thinking.as_deref().filter(|t| !t.is_empty()) {
                            if !in_thinking {
                                in_thinking = true;
                                let _ = events.send(GenEvent::Text("<think>".into()));
                            }
                            let _ = events.send(GenEvent::Text(t.to_string()));
                        }
                        if !msg.content.is_empty() {
                            if in_thinking {
                                in_thinking = false;
                                let _ = events.send(GenEvent::Text("</think>\n\n".into()));
                            }
                            let _ = events.send(GenEvent::Text(msg.content.clone()));
                        }
                    }
                    if c.done {
                        if in_thinking {
                            let _ = events.send(GenEvent::Text("</think>".into()));
                        }
                        let tps = match (c.eval_count, c.eval_duration) {
                            (Some(n), Some(d)) if d > 0 => Some(n as f64 / (d as f64 / 1e9)),
                            _ => None,
                        };
                        let reason = c.done_reason.clone().unwrap_or_else(|| "stop".into());
                        let _ = events.send(GenEvent::Done(GenStats {
                            prompt_tokens: c.prompt_eval_count,
                            completion_tokens: c.eval_count,
                            duration_ms: started.elapsed().as_millis() as u64,
                            tokens_per_sec: tps,
                            stop_reason: if reason == "length" { "length".into() } else { "stop".into() },
                            note: None,
                        }));
                        return;
                    }
                }
            }
        })
    }
}
