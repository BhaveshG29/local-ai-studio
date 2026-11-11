//! Built-in text generation with Candle: quantized GGUF models and full-precision
//! safetensors checkpoints, with KV-cache prefix reuse across chat turns.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models as m;
use tokio_util::sync::CancellationToken;

use super::params::GenParams;
use super::template::ChatTemplate;
use super::tokenizer::{StreamDecoder, TextTokenizer};
use crate::db::models::ModelRow;

/// Token strings that end an assistant turn in common chat formats.
const END_TOKENS: &[&str] = &[
    "<|eot_id|>",
    "<|eom_id|>",
    "<|im_end|>",
    "<end_of_turn>",
    "<|end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "</s>",
    "<|return|>",
    "<|user|>",
    "<|im_start|>",
];

enum Weights {
    QLlama(m::quantized_llama::ModelWeights),
    QQwen2(m::quantized_qwen2::ModelWeights),
    QQwen3(m::quantized_qwen3::ModelWeights),
    QPhi3(m::quantized_phi3::ModelWeights),
    QGemma3(m::quantized_gemma3::ModelWeights),
    QGlm4(m::quantized_glm4::ModelWeights),
    QLfm2(m::quantized_lfm2::ModelWeights),
    Llama { model: m::llama::Llama, cache: m::llama::Cache, config: m::llama::Config, dtype: DType },
    Qwen2(m::qwen2::ModelForCausalLM),
    Qwen3(m::qwen3::ModelForCausalLM),
    Mistral(m::mistral::Model),
    Phi3(m::phi3::Model),
}

impl Weights {
    fn forward(&mut self, input: &Tensor, pos: usize, device: &Device) -> candle_core::Result<Tensor> {
        match self {
            Weights::QLlama(w) => w.forward(input, pos),
            Weights::QQwen2(w) => w.forward(input, pos),
            Weights::QQwen3(w) => w.forward(input, pos),
            Weights::QPhi3(w) => w.forward(input, pos),
            Weights::QGemma3(w) => w.forward(input, pos),
            Weights::QGlm4(w) => w.forward(input, pos),
            Weights::QLfm2(w) => w.forward(input, pos),
            Weights::Llama { model, cache, config, dtype } => {
                if pos == 0 {
                    *cache = m::llama::Cache::new(true, *dtype, config, device)?;
                }
                model.forward(input, pos, cache)
            }
            Weights::Qwen2(w) => w.forward(input, pos),
            Weights::Qwen3(w) => w.forward(input, pos),
            Weights::Mistral(w) => w.forward(input, pos),
            Weights::Phi3(w) => w.forward(input, pos),
        }
    }

    /// Clears the KV cache before processing a prompt from position 0.
    fn reset(&mut self) {
        match self {
            Weights::QQwen3(w) => w.clear_kv_cache(),
            Weights::Qwen2(w) => w.clear_kv_cache(),
            Weights::Qwen3(w) => w.clear_kv_cache(),
            Weights::Mistral(w) => w.clear_kv_cache(),
            Weights::Phi3(w) => w.clear_kv_cache(),
            // The quantized llama/qwen2/phi3/gemma/glm4/lfm2 models and llama (via a fresh
            // cache) reset automatically when called with position 0.
            _ => {}
        }
    }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    EndOfTurn,
    MaxTokens,
    ContextFull,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct GenOutcome {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub stop: StopReason,
    pub elapsed_ms: u64,
    pub prompt_ms: u64,
}

/// A model loaded into memory, ready to generate.
pub struct LoadedLlm {
    pub model_id: String,
    pub name: String,
    weights: Weights,
    pub tokenizer: TextTokenizer,
    pub template: ChatTemplate,
    pub eos_ids: Vec<u32>,
    bos_id: Option<u32>,
    add_bos: bool,
    pub device: Device,
    pub device_label: String,
    pub max_context: usize,
    cached: Vec<u32>,
    pub load_ms: u64,
}

impl LoadedLlm {
    /// Loads a library model on the given device. Blocking; call from a blocking thread.
    pub fn load(model: &ModelRow, device: Device, device_label: String) -> Result<Self> {
        let started = Instant::now();
        if !model.analysis.runnable {
            bail!("This model can't run in the built-in engine: {}", model.analysis.problems.join(" "));
        }
        let dir = PathBuf::from(&model.path);
        let mut llm = match model.primary_file.as_deref() {
            Some(f) if f.to_lowercase().ends_with(".gguf") => Self::load_gguf(&dir.join(f), device, device_label)?,
            _ => Self::load_safetensors(&dir, device, device_label)?,
        };
        llm.model_id = model.id.clone();
        llm.name = model.display_name.clone();
        llm.load_ms = started.elapsed().as_millis() as u64;
        Ok(llm)
    }

    fn load_gguf(path: &Path, device: Device, device_label: String) -> Result<Self> {
        let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| anyhow!("reading GGUF header: {e}"))?;
        let md = &content.metadata;
        let arch = md.get("general.architecture").and_then(|v| v.to_string().ok()).cloned().unwrap_or_default();
        let ctx_len = md
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u64().ok().or_else(|| v.to_u32().ok().map(u64::from)))
            .unwrap_or(4096) as usize;
        let tokenizer = TextTokenizer::from_gguf(&content)?;
        let get_id = |k: &str| md.get(k).and_then(|v| v.to_u32().ok());
        let bos_id = get_id("tokenizer.ggml.bos_token_id");
        let eos_id = get_id("tokenizer.ggml.eos_token_id");
        let eot_id = get_id("tokenizer.ggml.eot_token_id");
        let add_bos = md
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(matches!(tokenizer, TextTokenizer::Spm(_)));
        let tok_str = |id: Option<u32>| id.and_then(|i| tokenizer.token_str(i)).unwrap_or_default();
        let template_src = md.get("tokenizer.chat_template").and_then(|v| v.to_string().ok()).cloned();
        let has_chatml = tokenizer.token_id("<|im_start|>").is_some();
        let template = ChatTemplate::new(template_src, &tok_str(bos_id), &tok_str(eos_id), has_chatml);
        let mut eos_ids: Vec<u32> = [eos_id, eot_id].into_iter().flatten().collect();
        add_end_tokens(&tokenizer, &mut eos_ids);

        let weights = match arch.as_str() {
            "llama" | "mistral" => {
                Weights::QLlama(m::quantized_llama::ModelWeights::from_gguf(content, &mut file, &device)?)
            }
            "qwen2" => Weights::QQwen2(m::quantized_qwen2::ModelWeights::from_gguf(content, &mut file, &device)?),
            "qwen3" => Weights::QQwen3(m::quantized_qwen3::ModelWeights::from_gguf(content, &mut file, &device)?),
            "phi3" => Weights::QPhi3(m::quantized_phi3::ModelWeights::from_gguf(false, content, &mut file, &device)?),
            "gemma" | "gemma2" | "gemma3" => {
                Weights::QGemma3(m::quantized_gemma3::ModelWeights::from_gguf(content, &mut file, &device)?)
            }
            "glm4" => {
                Weights::QGlm4(m::quantized_glm4::ModelWeights::from_gguf(content, &mut file, &device, DType::F32)?)
            }
            "lfm2" => Weights::QLfm2(m::quantized_lfm2::ModelWeights::from_gguf(content, &mut file, &device)?),
            other => bail!("the \"{other}\" architecture is not supported by the built-in engine"),
        };
        Ok(Self {
            model_id: String::new(),
            name: String::new(),
            weights,
            tokenizer,
            template,
            eos_ids,
            bos_id,
            add_bos,
            device,
            device_label,
            max_context: ctx_len,
            cached: Vec::new(),
            load_ms: 0,
        })
    }

    fn load_safetensors(dir: &Path, device: Device, device_label: String) -> Result<Self> {
        let read_json = |name: &str| -> Option<serde_json::Value> {
            std::fs::read_to_string(dir.join(name)).ok().and_then(|t| serde_json::from_str(&t).ok())
        };
        let config = read_json("config.json").context("config.json is missing or invalid")?;
        let model_type = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let files = safetensors_files(dir)?;
        let dtype = if device.is_cuda() {
            DType::BF16
        } else if device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };
        // SAFETY: the files are memory-mapped read-only; they must not be modified while the
        // model is loaded (the library never rewrites model files in place).
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
        let weights = match model_type.as_str() {
            "llama" => {
                let cfg: m::llama::LlamaConfig = serde_json::from_value(config.clone())?;
                let cfg = cfg.into_config(false);
                let model = m::llama::Llama::load(vb, &cfg)?;
                let cache = m::llama::Cache::new(true, dtype, &cfg, &device)?;
                Weights::Llama { model, cache, config: cfg, dtype }
            }
            "qwen2" => {
                let cfg: m::qwen2::Config = serde_json::from_value(config.clone())?;
                Weights::Qwen2(m::qwen2::ModelForCausalLM::new(&cfg, vb)?)
            }
            "qwen3" => {
                let cfg: m::qwen3::Config = serde_json::from_value(config.clone())?;
                Weights::Qwen3(m::qwen3::ModelForCausalLM::new(&cfg, vb)?)
            }
            "mistral" => {
                let cfg: m::mistral::Config = serde_json::from_value(config.clone())?;
                Weights::Mistral(m::mistral::Model::new(&cfg, vb)?)
            }
            "phi3" => {
                let cfg: m::phi3::Config = serde_json::from_value(config.clone())?;
                Weights::Phi3(m::phi3::Model::new(&cfg, vb)?)
            }
            other => bail!("model type \"{other}\" is not supported by the built-in engine"),
        };
        let tokenizer = TextTokenizer::from_file(&dir.join("tokenizer.json"))?;
        let tok_cfg = read_json("tokenizer_config.json").unwrap_or_default();
        let token_text = |v: Option<&serde_json::Value>| -> String {
            match v {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Object(o)) => {
                    o.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string()
                }
                _ => String::new(),
            }
        };
        let bos = token_text(tok_cfg.get("bos_token"));
        let eos = token_text(tok_cfg.get("eos_token"));
        let template_src = std::fs::read_to_string(dir.join("chat_template.jinja"))
            .ok()
            .or_else(|| match tok_cfg.get("chat_template") {
                Some(serde_json::Value::String(s)) => Some(s.clone()),
                Some(serde_json::Value::Array(a)) => a
                    .iter()
                    .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| a.first())
                    .and_then(|t| t.get("template"))
                    .and_then(|t| t.as_str())
                    .map(String::from),
                _ => None,
            })
            .or_else(|| {
                read_json("chat_template.json")
                    .and_then(|v| v.get("chat_template").and_then(|s| s.as_str()).map(String::from))
            });
        let has_chatml = tokenizer.token_id("<|im_start|>").is_some();
        let template = ChatTemplate::new(template_src, &bos, &eos, has_chatml);
        let mut eos_ids = Vec::new();
        let gen_cfg = read_json("generation_config.json").unwrap_or_default();
        for src in [gen_cfg.get("eos_token_id"), config.get("eos_token_id")] {
            match src {
                Some(serde_json::Value::Number(n)) => {
                    if let Some(i) = n.as_u64() {
                        eos_ids.push(i as u32);
                    }
                }
                Some(serde_json::Value::Array(a)) => {
                    eos_ids.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32));
                }
                _ => {}
            }
        }
        if let Some(id) = tokenizer.token_id(&eos) {
            eos_ids.push(id);
        }
        add_end_tokens(&tokenizer, &mut eos_ids);
        let bos_id = tokenizer.token_id(&bos);
        let add_bos = tok_cfg.get("add_bos_token").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_context = config.get("max_position_embeddings").and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
        Ok(Self {
            model_id: String::new(),
            name: String::new(),
            weights,
            tokenizer,
            template,
            eos_ids,
            bos_id,
            add_bos,
            device,
            device_label,
            max_context,
            cached: Vec::new(),
            load_ms: 0,
        })
    }

    /// Tokenizes a rendered prompt, adding BOS when the model expects one and the template
    /// did not already insert it.
    pub fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut ids = self.tokenizer.encode(prompt)?;
        if self.add_bos
            && let Some(b) = self.bos_id
            && ids.first() != Some(&b)
        {
            ids.insert(0, b);
        }
        Ok(ids)
    }

    /// Effective context window for a request.
    pub fn context_limit(&self, params: &GenParams) -> usize {
        (params.context_length as usize).min(self.max_context).max(256)
    }

    /// Generates a completion for `prompt_ids`, calling `on_text` with each new piece of text.
    /// Blocking: run on a blocking thread.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        params: &GenParams,
        cancel: &CancellationToken,
        on_text: &mut (dyn FnMut(&str) + Send),
    ) -> Result<GenOutcome> {
        let started = Instant::now();
        let ctx = self.context_limit(params);
        if prompt_ids.is_empty() {
            bail!("empty prompt");
        }
        if prompt_ids.len() >= ctx {
            bail!(
                "The conversation is {} tokens long but the context window is {ctx} tokens. Start a new chat or raise the context length.",
                prompt_ids.len()
            );
        }
        let result = self.generate_inner(prompt_ids, params, cancel, on_text, ctx, started);
        if result.is_err() {
            // The KV cache may be in an unknown state after an error.
            self.cached.clear();
        }
        result
    }

    fn generate_inner(
        &mut self,
        prompt_ids: &[u32],
        params: &GenParams,
        cancel: &CancellationToken,
        on_text: &mut (dyn FnMut(&str) + Send),
        ctx: usize,
        started: Instant,
    ) -> Result<GenOutcome> {
        // Reuse the KV cache when the new prompt extends what is already cached.
        let common = self.cached.iter().zip(prompt_ids.iter()).take_while(|(a, b)| a == b).count();
        let mut pos;
        let to_feed: &[u32];
        if common > 0 && common == self.cached.len() && common < prompt_ids.len() {
            pos = common;
            to_feed = &prompt_ids[common..];
        } else {
            self.weights.reset();
            self.cached.clear();
            pos = 0;
            to_feed = prompt_ids;
        }
        // Reading the prompt: large matrix products, on all cores (Rayon's global pool).
        let mut logits = None;
        for chunk in to_feed.chunks(512) {
            if cancel.is_cancelled() {
                self.cached.clear();
                return Ok(GenOutcome {
                    prompt_tokens: prompt_ids.len(),
                    completion_tokens: 0,
                    stop: StopReason::Cancelled,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    prompt_ms: started.elapsed().as_millis() as u64,
                });
            }
            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            logits = Some(self.weights.forward(&input, pos, &self.device)?);
            pos += chunk.len();
            self.cached.extend_from_slice(chunk);
        }
        let prompt_ms = started.elapsed().as_millis() as u64;
        let logits = logits.context("no logits produced")?;
        // Writing the answer: one small step per token, on the fastest cores.
        let (generated, stop) =
            super::threads::generating(|| self.decode(logits, prompt_ids, params, cancel, on_text, ctx, pos))?;
        Ok(GenOutcome {
            prompt_tokens: prompt_ids.len(),
            completion_tokens: generated,
            stop,
            elapsed_ms: started.elapsed().as_millis() as u64,
            prompt_ms,
        })
    }

    /// Samples and feeds back one token at a time, starting from the prompt's last logits.
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &mut self,
        mut logits: Tensor,
        prompt_ids: &[u32],
        params: &GenParams,
        cancel: &CancellationToken,
        on_text: &mut (dyn FnMut(&str) + Send),
        ctx: usize,
        mut pos: usize,
    ) -> Result<(usize, StopReason)> {
        let temperature = params.temperature as f64;
        let sampling = if temperature <= 1e-4 {
            Sampling::ArgMax
        } else {
            let k = params.top_k as usize;
            let p = params.top_p as f64;
            match (k > 0, p > 0.0 && p < 1.0) {
                (true, true) => Sampling::TopKThenTopP { k, p, temperature },
                (true, false) => Sampling::TopK { k, temperature },
                (false, true) => Sampling::TopP { p, temperature },
                (false, false) => Sampling::All { temperature },
            }
        };
        let seed = params.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });
        let mut processor = LogitsProcessor::from_sampling(seed, sampling);
        let mut decoder = StreamDecoder::new();
        let mut history: Vec<u32> = prompt_ids.to_vec();
        let max_new = (params.max_tokens as usize).min(ctx.saturating_sub(prompt_ids.len()));
        let mut generated = 0usize;
        let mut stop =
            if max_new < params.max_tokens as usize { StopReason::ContextFull } else { StopReason::MaxTokens };
        while generated < max_new {
            let mut l = logits.flatten_all()?.to_dtype(DType::F32)?;
            if (params.repeat_penalty - 1.0).abs() > 1e-3 {
                let start = history.len().saturating_sub(64);
                l = candle_transformers::utils::apply_repeat_penalty(&l, params.repeat_penalty, &history[start..])?;
            }
            let next = processor.sample(&l)?;
            if self.eos_ids.contains(&next) {
                stop = StopReason::EndOfTurn;
                break;
            }
            history.push(next);
            generated += 1;
            if let Some(text) = decoder.push(&self.tokenizer, next)? {
                on_text(&text);
            }
            if cancel.is_cancelled() {
                stop = StopReason::Cancelled;
                break;
            }
            if generated >= max_new {
                break;
            }
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = self.weights.forward(&input, pos, &self.device)?;
            pos += 1;
            self.cached.push(next);
        }
        if let Some(rest) = decoder.finish(&self.tokenizer)? {
            on_text(&rest);
        }
        Ok((generated, stop))
    }
}

fn add_end_tokens(tok: &TextTokenizer, ids: &mut Vec<u32>) {
    for t in END_TOKENS {
        if let Some(id) = tok.token_id(t) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
}

/// The weight files of a safetensors checkpoint (single file or sharded with an index).
pub fn safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let map = v.get("weight_map").and_then(|w| w.as_object()).context("invalid model.safetensors.index.json")?;
        let mut files: Vec<String> = map.values().filter_map(|f| f.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        let paths: Vec<PathBuf> = files.iter().map(|f| dir.join(f)).collect();
        for p in &paths {
            if !p.exists() {
                bail!("shard {} is missing; download the model again", p.display());
            }
        }
        return Ok(paths);
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
        .collect();
    found.sort();
    if found.is_empty() {
        bail!("no .safetensors files in {}", dir.display());
    }
    Ok(found)
}
