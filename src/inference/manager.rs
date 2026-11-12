//! Owns loaded models and hands out engines.
//!
//! * Built-in models are loaded on a blocking thread and kept in memory (LRU, configurable
//!   maximum) until unloaded, evicted or idle for too long.
//! * Concurrent requests for the same built-in model are serialised by its mutex; a load
//!   failure is reported to the caller and never takes the server down.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use candle_core::Device;
use futures_util::future::BoxFuture;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::candle_llm::{LoadedLlm, StopReason};
use super::params::ParamSupport;
use super::template::TemplateMessage;
use super::{ChatEngine, EngineRequest, GenEvent, GenStats, ModelRef, OllamaEngine};
use crate::config::{Acceleration, Settings};
use crate::db::{self, Db};
use crate::error::{AppError, AppResult};
use crate::ollama::OllamaClient;
use crate::system::{AccelInfo, SystemMonitor};

/// Device labels shown in the interface.
pub const CPU_LABEL: &str = "CPU";
pub const CUDA_LABEL: &str = "NVIDIA GPU · CUDA";
pub const METAL_LABEL: &str = "Apple GPU · Metal";
/// A model that the automatic setting couldn't load on the GPU (usually: not enough
/// graphics memory) and loaded on the processor instead.
pub const CPU_FALLBACK_LABEL: &str = "CPU · didn't fit on the GPU";

fn unusable(pref: Acceleration, accel: &AccelInfo) -> String {
    let (name, detail) = match pref {
        Acceleration::Metal => ("Apple Metal", &accel.metal_detail),
        _ => ("NVIDIA CUDA", &accel.cuda_detail),
    };
    format!("{name} was selected in Settings, but it can't be used: {detail}")
}

/// Picks the Candle device according to the user's preference and what actually works.
/// `Auto` prefers a GPU — CUDA on Linux and Windows, Metal on Apple Silicon — and falls
/// back to the CPU.
pub fn select_device(pref: Acceleration, accel: &AccelInfo) -> Result<(Device, String), String> {
    let cpu = || (Device::Cpu, CPU_LABEL.to_string());
    match pref {
        Acceleration::Cpu => Ok(cpu()),
        Acceleration::Cuda if !accel.cuda_usable => Err(unusable(pref, accel)),
        Acceleration::Cuda => {
            Device::new_cuda(0).map(|d| (d, CUDA_LABEL.to_string())).map_err(|e| format!("CUDA device error: {e}"))
        }
        Acceleration::Metal if !accel.metal_usable => Err(unusable(pref, accel)),
        Acceleration::Metal => {
            Device::new_metal(0).map(|d| (d, METAL_LABEL.to_string())).map_err(|e| format!("Metal device error: {e}"))
        }
        Acceleration::Auto => {
            if accel.cuda_usable
                && let Ok(d) = Device::new_cuda(0)
            {
                return Ok((d, CUDA_LABEL.into()));
            }
            if accel.metal_usable
                && let Ok(d) = Device::new_metal(0)
            {
                return Ok((d, METAL_LABEL.into()));
            }
            Ok(cpu())
        }
    }
}

/// The label of the device [`select_device`] picks, without creating one. Creating a GPU
/// device allocates a context and memory on the card, which pages showing a label must not
/// do on every request.
pub fn device_label(pref: Acceleration, accel: &AccelInfo) -> Result<String, String> {
    let label = match pref {
        Acceleration::Cpu => CPU_LABEL,
        Acceleration::Cuda if !accel.cuda_usable => return Err(unusable(pref, accel)),
        Acceleration::Cuda => CUDA_LABEL,
        Acceleration::Metal if !accel.metal_usable => return Err(unusable(pref, accel)),
        Acceleration::Metal => METAL_LABEL,
        Acceleration::Auto if accel.cuda_usable => CUDA_LABEL,
        Acceleration::Auto if accel.metal_usable => METAL_LABEL,
        Acceleration::Auto => CPU_LABEL,
    };
    Ok(label.to_string())
}

struct Slot {
    llm: Arc<Mutex<LoadedLlm>>,
    last_used: Instant,
    /// Approximate memory taken by the weights.
    weights_bytes: u64,
    /// Where the model runs (readable while the model is busy generating).
    device_label: String,
}

/// Information about a loaded built-in model.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LoadedInfo {
    pub model_ref: String,
    pub name: String,
    pub device: String,
    pub load_ms: u64,
    pub idle_secs: u64,
    /// Approximate memory taken by the weights (0 when unknown).
    pub weights_bytes: u64,
}

pub struct InferenceManager {
    db: Db,
    settings: Arc<RwLock<Settings>>,
    system: Arc<SystemMonitor>,
    ollama: OllamaClient,
    llms: Mutex<HashMap<String, Slot>>,
    load_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// The device models are loaded on, shared by all of them: every new CUDA or Metal
    /// device brings its own queue, BLAS handle and compiled kernels. Released when nothing
    /// is loaded, so an idle app holds no graphics memory.
    device: Mutex<Option<(Acceleration, Device, String)>>,
    pub whisper: super::whisper::WhisperCache,
    pub diffusion: super::diffusion::DiffusionCache,
}

impl InferenceManager {
    pub fn new(db: Db, settings: Arc<RwLock<Settings>>, system: Arc<SystemMonitor>, ollama: OllamaClient) -> Self {
        Self {
            db,
            settings,
            system,
            ollama,
            llms: Mutex::new(HashMap::new()),
            load_locks: Mutex::new(HashMap::new()),
            device: Mutex::new(None),
            whisper: Default::default(),
            diffusion: Default::default(),
        }
    }

    fn settings(&self) -> Settings {
        self.settings.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// The device to load a model on (created on first use, then shared).
    pub fn device(&self) -> Result<(Device, String), String> {
        let pref = self.settings().inference.acceleration;
        let mut slot = self.device.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((p, device, label)) = slot.as_ref()
            && *p == pref
        {
            return Ok((device.clone(), label.clone()));
        }
        let (device, label) = select_device(pref, self.system.accel())?;
        *slot = Some((pref, device.clone(), label.clone()));
        Ok((device, label))
    }

    /// What the next model will run on, for display. Never creates a device.
    pub fn device_label(&self) -> Result<String, String> {
        device_label(self.settings().inference.acceleration, self.system.accel())
    }

    /// What a built-in model runs on: its actual device when it is loaded, otherwise the
    /// device it would be loaded on.
    pub fn label_for(&self, id: &str) -> String {
        let loaded =
            self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(id).map(|s| s.device_label.clone());
        loaded.unwrap_or_else(|| self.device_label().unwrap_or_else(|_| CPU_LABEL.into()))
    }

    /// Drops the shared device once no model uses it, returning its memory to the GPU.
    fn release_device_if_idle(&self) {
        let idle = self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_empty()
            && !self.whisper.is_loaded()
            && !self.diffusion.is_loaded();
        if idle {
            *self.device.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        }
    }

    /// `local:<id>` refs of loaded built-in models.
    pub fn loaded_refs(&self) -> HashSet<String> {
        self.llms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .map(|k| format!("local:{k}"))
            .collect()
    }

    pub fn loaded_info(&self) -> Vec<LoadedInfo> {
        let llms = self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        llms.iter()
            .map(|(id, slot)| {
                let (name, load_ms) = match slot.llm.try_lock() {
                    Ok(l) => (l.name.clone(), l.load_ms),
                    Err(_) => ("(generating)".into(), 0),
                };
                LoadedInfo {
                    model_ref: format!("local:{id}"),
                    name,
                    device: slot.device_label.clone(),
                    load_ms,
                    idle_secs: slot.last_used.elapsed().as_secs(),
                    weights_bytes: slot.weights_bytes,
                }
            })
            .collect()
    }

    /// Loads a built-in model (no-op if already loaded). Evicts least recently used models
    /// beyond the configured maximum.
    pub async fn load_local(&self, id: &str) -> AppResult<Arc<Mutex<LoadedLlm>>> {
        if let Some(slot) = self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get_mut(id) {
            slot.last_used = Instant::now();
            return Ok(slot.llm.clone());
        }
        let lock = {
            let mut locks = self.load_locks.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            locks.entry(id.to_string()).or_default().clone()
        };
        let _guard = lock.lock().await;
        if let Some(slot) = self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(id) {
            return Ok(slot.llm.clone());
        }
        let i = id.to_string();
        let model = self
            .db
            .call(move |c| db::models::get(c, &i))
            .await?
            .ok_or_else(|| AppError::not_found("That model is no longer in your library."))?;
        if !model.analysis.runnable {
            return Err(AppError::bad_request(format!(
                "\"{}\" can't run in the built-in engine: {}",
                model.display_name,
                model.analysis.problems.join(" ")
            )));
        }
        if !matches!(model.analysis.capability, crate::hub::Capability::Chat | crate::hub::Capability::Vision) {
            return Err(AppError::bad_request(format!("\"{}\" is not a chat model.", model.display_name)));
        }
        // Memory check (advisory: we refuse only when the model can't possibly fit).
        let snap = self.system.snapshot().await;
        let est = crate::system::estimate_llm_memory(
            model.size_bytes.max(0) as u64 * if model.format == "safetensors" { 2 } else { 1 },
            model.analysis.n_layers,
            model.analysis.n_kv_heads,
            model.analysis.head_dim,
            self.settings().generation.context_length as u64,
            4,
        );
        if est.total > snap.total_ram + snap.total_swap {
            return Err(AppError::bad_request(format!(
                "\"{}\" needs about {} of memory, more than this computer has ({}). Choose a smaller or more quantised version.",
                model.display_name,
                crate::util::human_bytes(est.total),
                crate::util::human_bytes(snap.total_ram)
            )));
        }
        self.evict_for_new();
        let (device, label) = self.device().map_err(AppError::bad_request)?;
        // With the automatic setting a model that can't be loaded on the GPU (usually: too
        // big for its memory) runs on the processor instead of failing.
        let cpu_fallback = self.settings().inference.acceleration == Acceleration::Auto && !device.is_cpu();
        let m = model.clone();
        let loaded = tokio::task::spawn_blocking(move || match LoadedLlm::load(&m, device, label) {
            Err(e) if cpu_fallback => {
                tracing::warn!(model = %m.display_name, "couldn't load on the GPU, using the CPU instead: {e:#}");
                LoadedLlm::load(&m, Device::Cpu, CPU_FALLBACK_LABEL.into())
            }
            other => other,
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("model loading crashed: {e}")))?
        .map_err(|e| AppError::bad_request(format!("Could not load \"{}\": {e:#}", model.display_name)))?;
        tracing::info!(model = %model.display_name, device = %loaded.device_label, ms = loaded.load_ms, "loaded model");
        let device_label = loaded.device_label.clone();
        let arc = Arc::new(Mutex::new(loaded));
        // GGUF weights stay at their file size in memory; safetensors are widened (as in the
        // memory estimate above).
        let weights_bytes = model.size_bytes.max(0) as u64 * if model.format == "safetensors" { 2 } else { 1 };
        self.llms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.to_string(), Slot { llm: arc.clone(), last_used: Instant::now(), weights_bytes, device_label });
        let now = crate::util::now_rfc3339();
        let i = id.to_string();
        let _ = self.db.call(move |c| db::models::touch(c, &i, &now)).await;
        Ok(arc)
    }

    fn evict_for_new(&self) {
        let max = self.settings().inference.max_loaded_models.max(1) as usize;
        let mut llms = self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while llms.len() >= max {
            let Some(oldest) = llms.iter().min_by_key(|(_, s)| s.last_used).map(|(k, _)| k.clone()) else {
                break;
            };
            tracing::info!(model = %oldest, "unloading least recently used model");
            llms.remove(&oldest);
        }
    }

    pub fn unload_local(&self, id: &str) -> bool {
        let removed = self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(id).is_some();
        self.release_device_if_idle();
        removed
    }

    pub fn unload_whisper(&self) {
        self.whisper.clear();
        self.release_device_if_idle();
    }

    pub fn unload_image_model(&self) {
        self.diffusion.clear();
        self.release_device_if_idle();
    }

    pub fn unload_all(&self) {
        self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
        self.whisper.clear();
        self.diffusion.clear();
        self.release_device_if_idle();
    }

    /// Unloads built-in models idle for longer than the configured time.
    pub fn reap_idle(&self) {
        let minutes = self.settings().inference.unload_after_minutes;
        if minutes == 0 {
            return;
        }
        let max_idle = Duration::from_secs(minutes as u64 * 60);
        let mut llms = self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        llms.retain(|id, slot| {
            // Busy models (lock held by a generation) are never reaped.
            let busy = slot.llm.try_lock().is_err();
            let keep = busy || slot.last_used.elapsed() < max_idle;
            if !keep {
                tracing::info!(model = %id, "unloading idle model");
            }
            keep
        });
        drop(llms);
        self.whisper.reap(max_idle);
        self.diffusion.reap(max_idle);
        self.release_device_if_idle();
    }

    /// Resolves a model reference to an engine, loading built-in models as needed.
    pub async fn engine_for(&self, model: &ModelRef) -> AppResult<Arc<dyn ChatEngine>> {
        match model {
            ModelRef::Ollama(name) => {
                if !self.ollama.enabled() {
                    return Err(AppError::unavailable("Ollama integration is turned off in Settings."));
                }
                let show = self.ollama.show(name).await.map_err(|e| match e {
                    AppError::NotFound(_) => AppError::not_found(format!(
                        "\"{name}\" is not installed in Ollama any more. Pick another model or download it again."
                    )),
                    other => other,
                })?;
                if show.has_capability("embedding") && !show.has_capability("completion") {
                    return Err(AppError::bad_request(format!("\"{name}\" is an embedding model and can't chat.")));
                }
                Ok(Arc::new(OllamaEngine {
                    client: self.ollama.clone(),
                    model: name.clone(),
                    vision: show.has_capability("vision"),
                }))
            }
            ModelRef::Local(id) => {
                let llm = self.load_local(id).await?;
                // The label comes from the slot: locking the model itself would block this
                // async task for as long as another chat is using it.
                let label =
                    self.llms.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get_mut(id).map(|slot| {
                        slot.last_used = Instant::now();
                        slot.device_label.clone()
                    });
                let label = label.unwrap_or_else(|| self.label_for(id));
                Ok(Arc::new(CandleEngine { llm, device_label: label }))
            }
        }
    }
}

/// The built-in engine for one loaded model.
pub struct CandleEngine {
    llm: Arc<Mutex<LoadedLlm>>,
    device_label: String,
}

/// Renders messages with the model template, dropping the oldest turns to fit the context.
pub fn build_prompt(llm: &LoadedLlm, req: &EngineRequest) -> Result<(Vec<u32>, Option<String>), String> {
    let ctx = llm.context_limit(&req.params);
    let reserve = (req.params.max_tokens as usize).min(ctx / 2).max(64);
    let mut messages: Vec<TemplateMessage> =
        req.messages.iter().map(|m| TemplateMessage::new(&m.role, &m.content)).collect();
    let mut note = None;
    loop {
        let mut prompt = llm.template.render(&messages, true).map_err(|e| format!("{e:#}"))?;
        if let Some(partial) = &req.continue_from {
            prompt.push_str(partial);
        }
        let ids = llm.tokenize_prompt(&prompt).map_err(|e| format!("{e:#}"))?;
        if ids.len() + reserve <= ctx {
            return Ok((ids, note));
        }
        // Drop the oldest non-system message (keep at least the last user message).
        let first_droppable = messages.iter().position(|m| m.role != "system");
        let non_system = messages.iter().filter(|m| m.role != "system").count();
        match first_droppable {
            Some(i) if non_system > 1 => {
                messages.remove(i);
                note = Some(
                    "Older messages were left out so the conversation fits the model's context window.".to_string(),
                );
            }
            _ => {
                return Err(format!(
                    "Your message is too long for this model: it needs {} tokens but the context window is {ctx} (including {reserve} reserved for the answer). Shorten the message or increase the context length in chat settings.",
                    ids.len()
                ));
            }
        }
    }
}

impl ChatEngine for CandleEngine {
    fn name(&self) -> String {
        format!("Built-in ({})", self.device_label)
    }

    fn param_support(&self) -> ParamSupport {
        ParamSupport { images: false, ..ParamSupport::ALL_TEXT }
    }

    fn chat(
        &self,
        req: EngineRequest,
        cancel: CancellationToken,
        events: mpsc::UnboundedSender<GenEvent>,
    ) -> BoxFuture<'static, ()> {
        let llm = self.llm.clone();
        Box::pin(async move {
            let ev = events.clone();
            let result = tokio::task::spawn_blocking(move || {
                // Wait for the model if another conversation is using it.
                let mut guard = loop {
                    match llm.try_lock() {
                        Ok(g) => break g,
                        Err(std::sync::TryLockError::Poisoned(p)) => break p.into_inner(),
                        Err(std::sync::TryLockError::WouldBlock) => {
                            if cancel.is_cancelled() {
                                let _ = ev.send(GenEvent::Done(GenStats {
                                    stop_reason: "cancelled".into(),
                                    ..Default::default()
                                }));
                                return;
                            }
                            let _ = ev.send(GenEvent::Phase("Waiting for the model to finish another answer…".into()));
                            std::thread::sleep(Duration::from_millis(200));
                        }
                    }
                };
                let (ids, note) = match build_prompt(&guard, &req) {
                    Ok(x) => x,
                    Err(e) => {
                        let _ = ev.send(GenEvent::Error(e));
                        return;
                    }
                };
                let _ = ev.send(GenEvent::Phase(format!("Reading your message ({} tokens)…", ids.len())));
                let ev2 = ev.clone();
                let mut on_text = |t: &str| {
                    let _ = ev2.send(GenEvent::Text(t.to_string()));
                };
                match guard.generate(&ids, &req.params, &cancel, &mut on_text) {
                    Ok(out) => {
                        let gen_ms = out.elapsed_ms.saturating_sub(out.prompt_ms).max(1);
                        let _ = ev.send(GenEvent::Done(GenStats {
                            prompt_tokens: Some(out.prompt_tokens as u64),
                            completion_tokens: Some(out.completion_tokens as u64),
                            duration_ms: out.elapsed_ms,
                            tokens_per_sec: (out.completion_tokens > 0)
                                .then(|| out.completion_tokens as f64 / (gen_ms as f64 / 1000.0)),
                            stop_reason: match out.stop {
                                StopReason::EndOfTurn => "stop",
                                StopReason::MaxTokens => "length",
                                StopReason::ContextFull => "context",
                                StopReason::Cancelled => "cancelled",
                            }
                            .into(),
                            note,
                        }));
                    }
                    Err(e) => {
                        let _ = ev.send(GenEvent::Error(format!("The built-in engine failed: {e:#}")));
                    }
                }
            })
            .await;
            if let Err(e) = result {
                let _ = events.send(GenEvent::Error(format!("The built-in engine crashed: {e}")));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_refs_parse() {
        assert_eq!(ModelRef::parse("ollama:llama3.2:3b"), Some(ModelRef::Ollama("llama3.2:3b".into())));
        assert_eq!(ModelRef::parse("local:abc123"), Some(ModelRef::Local("abc123".into())));
        assert_eq!(ModelRef::parse("local:../x"), None);
        assert_eq!(ModelRef::parse("foo"), None);
        assert_eq!(ModelRef::parse("ollama:"), None);
        assert_eq!(ModelRef::Ollama("a".into()).to_string(), "ollama:a");
    }

    #[test]
    fn cpu_is_always_selectable() {
        let accel = crate::system::probe_acceleration();
        let (d, label) = select_device(Acceleration::Cpu, &accel).unwrap();
        assert!(d.is_cpu());
        assert_eq!(label, CPU_LABEL);
        if !accel.cuda_usable {
            assert!(select_device(Acceleration::Cuda, &accel).is_err());
        }
        assert!(select_device(Acceleration::Auto, &accel).is_ok());
    }

    #[test]
    fn labels_match_the_selected_device() {
        let accel = crate::system::probe_acceleration();
        for pref in [Acceleration::Auto, Acceleration::Cpu, Acceleration::Cuda, Acceleration::Metal] {
            match (select_device(pref, &accel), device_label(pref, &accel)) {
                (Ok((_, chosen)), Ok(label)) => assert_eq!(chosen, label, "{pref:?}"),
                (Err(a), Err(b)) => assert_eq!(a, b, "{pref:?}"),
                (a, b) => panic!("{pref:?}: device {:?}, label {b:?}", a.map(|(_, l)| l)),
            }
        }
    }

    #[test]
    fn automatic_prefers_a_gpu() {
        let accel = |cuda: bool, metal: bool| AccelInfo {
            cuda_compiled: cuda,
            cuda_usable: cuda,
            cuda_detail: String::new(),
            metal_compiled: metal,
            metal_usable: metal,
            metal_detail: "not here".into(),
            cpu_simd: Vec::new(),
        };
        assert_eq!(device_label(Acceleration::Auto, &accel(true, false)).unwrap(), CUDA_LABEL);
        assert_eq!(device_label(Acceleration::Auto, &accel(false, true)).unwrap(), METAL_LABEL);
        assert_eq!(device_label(Acceleration::Auto, &accel(false, false)).unwrap(), CPU_LABEL);
        // An explicit choice that can't work is an error that explains why.
        let err = device_label(Acceleration::Metal, &accel(true, false)).unwrap_err();
        assert!(err.starts_with("Apple Metal was selected") && err.ends_with("not here"), "{err}");
    }
}
