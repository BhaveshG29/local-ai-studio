//! The local model library: analysis of model files on disk, registration of finished
//! downloads, imports of existing GGUF files, and a unified listing that also includes the
//! models managed by Ollama.
//!
//! A successful download does not mean a model can run: every model is analysed (file
//! headers, configs, tokenizer presence, supported architectures and quantisations) and
//! only models that pass are offered for inference.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::db::downloads::DownloadRow;
use crate::db::models::{ModelRow, StoredFile};
use crate::db::{self, Db};
use crate::hub::{BUILTIN_GGUF_ARCHS, BUILTIN_SAFETENSORS_TYPES, Capability, SdVersion};
use crate::ollama::{LocalModel, OllamaClient};
use crate::util::{human_bytes, human_params, now_rfc3339};

/// Result of inspecting a model's files.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelAnalysis {
    pub capability: Capability,
    pub format: String,
    pub architecture: Option<String>,
    pub runnable: bool,
    pub problems: Vec<String>,
    pub notes: Vec<String>,
    pub parameters: Option<u64>,
    pub context_length: Option<u64>,
    pub quantization: Option<String>,
    pub n_layers: Option<u64>,
    pub n_kv_heads: Option<u64>,
    pub head_dim: Option<u64>,
    pub has_chat_template: bool,
    pub tokenizer_kind: Option<String>,
    pub sd_version: Option<SdVersion>,
    pub mel_bins: Option<u64>,
    pub weights_bytes: u64,
}

/// llama.cpp `general.file_type` → label.
pub fn ftype_label(ftype: u32) -> Option<&'static str> {
    Some(match ftype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4",
        _ => return None,
    })
}

/// Reads a GGUF header (metadata and tensor table only — not the weights).
pub fn analyze_gguf(path: &Path) -> ModelAnalysis {
    use candle_core::quantized::gguf_file;
    let mut a = ModelAnalysis {
        format: "GGUF".into(),
        capability: Capability::Chat,
        weights_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        ..Default::default()
    };
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    if crate::util::is_split_gguf(&name) {
        a.problems.push("This is one part of a split GGUF model. The built-in engine needs a single-file GGUF; run split models through Ollama.".into());
        return a;
    }
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            a.problems.push(format!("Cannot open the file: {e}"));
            return a;
        }
    };
    let mut reader = std::io::BufReader::new(file);
    let content = match gguf_file::Content::read(&mut reader) {
        Ok(c) => c,
        Err(e) => {
            let msg = e.to_string();
            a.problems.push(if msg.contains("unknown dtype") {
                format!(
                    "Uses a quantisation type the built-in engine can't run ({}). Use a Q4_K_M / Q5_K_M / Q8_0 file, or run this file with Ollama.",
                    crate::util::parse_quantization(&name).unwrap_or_else(|| "IQ/TQ family".into())
                )
            } else if msg.contains("magic") {
                "This file is not a valid GGUF file (it may be corrupted or incomplete).".into()
            } else {
                format!("Could not read the GGUF header: {msg}")
            });
            return a;
        }
    };
    let md = &content.metadata;
    let get_str = |k: &str| md.get(k).and_then(|v| v.to_string().ok()).cloned();
    let get_u64 = |k: &str| {
        md.get(k).and_then(|v| {
            v.to_u64()
                .ok()
                .or_else(|| v.to_u32().ok().map(u64::from))
                .or_else(|| v.to_i32().ok().map(|x| x.max(0) as u64))
        })
    };
    let arch = get_str("general.architecture").unwrap_or_default();
    a.architecture = (!arch.is_empty()).then(|| arch.clone());
    a.context_length = get_u64(&format!("{arch}.context_length"));
    a.n_layers = get_u64(&format!("{arch}.block_count"));
    let heads = get_u64(&format!("{arch}.attention.head_count"));
    a.n_kv_heads = get_u64(&format!("{arch}.attention.head_count_kv")).or(heads);
    let emb = get_u64(&format!("{arch}.embedding_length"));
    a.head_dim = get_u64(&format!("{arch}.attention.key_length")).or(match (emb, heads) {
        (Some(e), Some(h)) if h > 0 => Some(e / h),
        _ => None,
    });
    a.quantization = get_u64("general.file_type")
        .and_then(|t| ftype_label(t as u32))
        .map(String::from)
        .or_else(|| crate::util::parse_quantization(&name));
    a.has_chat_template = md.contains_key("tokenizer.chat_template");
    a.tokenizer_kind = get_str("tokenizer.ggml.model");
    let params: u64 = content.tensor_infos.values().map(|t| t.shape.elem_count() as u64).sum();
    a.parameters = (params > 0).then_some(params);

    if arch.is_empty() {
        a.problems.push("The GGUF file does not declare its architecture.".into());
    } else if !BUILTIN_GGUF_ARCHS.contains(&arch.as_str()) {
        a.problems.push(format!(
            "The \"{arch}\" architecture is not supported by the built-in engine. Ollama can run most GGUF models (search for it in the Ollama tab, or use \"hf.co/<repo>\" with Ollama)."
        ));
    }
    match a.tokenizer_kind.as_deref() {
        Some("gpt2") | Some("llama") => {}
        Some(other) => {
            a.problems.push(format!("The embedded tokenizer type \"{other}\" is not supported by the built-in engine."))
        }
        None => a.problems.push("The GGUF file has no embedded tokenizer.".into()),
    }
    if !a.has_chat_template {
        a.notes.push(
            "No chat template is embedded; a generic template will be used, which may reduce answer quality.".into(),
        );
    }
    if md.keys().any(|k| k.starts_with("clip.")) {
        a.capability = Capability::Other;
        a.problems.push("This is a vision projector (mmproj) file, not a language model.".into());
    }
    if md.get("general.type").and_then(|v| v.to_string().ok()).map(|s| s == "mmproj").unwrap_or(false) {
        a.capability = Capability::Other;
        a.problems.push("This is a vision projector (mmproj) file, not a language model.".into());
    }
    a.runnable = a.problems.is_empty();
    a
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Parameter count from safetensors headers (8-byte length + JSON header).
pub fn safetensors_params(path: &Path) -> Option<u64> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let n = u64::from_le_bytes(len);
    if n > 100 * 1024 * 1024 {
        return None;
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let mut total = 0u64;
    for (k, t) in v.as_object()? {
        if k == "__metadata__" {
            continue;
        }
        let shape = t.get("shape")?.as_array()?;
        total += shape.iter().filter_map(|d| d.as_u64()).product::<u64>();
    }
    Some(total)
}

fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(m) = e.metadata() {
                total += m.len();
            }
        }
    }
    total
}

/// Analyses a model directory (safetensors LLM, Whisper or Stable Diffusion).
pub fn analyze_dir(dir: &Path, repo_id: Option<&str>) -> ModelAnalysis {
    let mut a = ModelAnalysis { weights_bytes: dir_size(dir), ..Default::default() };
    if dir.join("model_index.json").exists() {
        a.format = "diffusers".into();
        a.capability = Capability::ImageGeneration;
        let unet_cfg = read_json(&dir.join("unet/config.json"));
        let xattn = unet_cfg.as_ref().and_then(|c| c.get("cross_attention_dim")).and_then(|v| v.as_u64());
        let turbo = repo_id.map(|r| r.to_lowercase().contains("turbo")).unwrap_or(false);
        a.sd_version = match xattn {
            Some(2048) => Some(if turbo { SdVersion::XlTurbo } else { SdVersion::Xl }),
            Some(1024) => Some(if turbo { SdVersion::Turbo } else { SdVersion::V2_1 }),
            Some(768) => Some(SdVersion::V1_5),
            _ => None,
        };
        a.architecture = a.sd_version.map(|v| v.label().to_string());
        let need = |p: &str| dir.join(p).exists();
        let weight =
            |d: &str, b: &str| need(&format!("{d}/{b}.fp16.safetensors")) || need(&format!("{d}/{b}.safetensors"));
        if a.sd_version.is_none() {
            a.problems.push("Could not recognise the Stable Diffusion version from unet/config.json.".into());
        }
        for (d, b) in
            [("unet", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model"), ("text_encoder", "model")]
        {
            if !weight(d, b) {
                a.problems.push(format!("Missing {d} weights."));
            }
        }
        if a.sd_version.map(|v| v.is_xl()).unwrap_or(false) && !weight("text_encoder_2", "model") {
            a.problems.push("Missing text_encoder_2 weights (needed by SDXL).".into());
        }
        if !need("clip-tokenizer/tokenizer.json") {
            a.problems.push("Missing the CLIP tokenizer (clip-tokenizer/tokenizer.json). Download the model again from the Discover page.".into());
        }
        if a.sd_version.map(|v| v.is_xl()).unwrap_or(false) && !need("clip-tokenizer-2/tokenizer.json") {
            a.problems.push("Missing the second CLIP tokenizer (clip-tokenizer-2/tokenizer.json).".into());
        }
        a.runnable = a.problems.is_empty();
        return a;
    }
    let Some(config) = read_json(&dir.join("config.json")) else {
        a.format = "unknown".into();
        a.problems.push("No config.json found, so the model type is unknown.".into());
        return a;
    };
    let model_type = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("").to_string();
    a.architecture = (!model_type.is_empty()).then(|| model_type.clone());
    let weights: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
                .collect()
        })
        .unwrap_or_default();
    a.format = "safetensors".into();
    let params: u64 = weights.iter().filter_map(|w| safetensors_params(w)).sum();
    a.parameters = (params > 0).then_some(params);
    if model_type == "whisper" {
        a.capability = Capability::Transcription;
        a.mel_bins = config.get("num_mel_bins").and_then(|v| v.as_u64());
        if !dir.join("model.safetensors").exists() {
            a.problems.push("Missing model.safetensors.".into());
        }
        if !dir.join("tokenizer.json").exists() {
            a.problems.push("Missing tokenizer.json.".into());
        }
        a.runnable = a.problems.is_empty();
        return a;
    }
    a.capability = Capability::Chat;
    let get = |k: &str| config.get(k).and_then(|v| v.as_u64());
    a.n_layers = get("num_hidden_layers");
    let heads = get("num_attention_heads");
    a.n_kv_heads = get("num_key_value_heads").or(heads);
    a.head_dim = get("head_dim").or(match (get("hidden_size"), heads) {
        (Some(h), Some(n)) if n > 0 => Some(h / n),
        _ => None,
    });
    a.context_length = get("max_position_embeddings");
    a.quantization =
        config.get("torch_dtype").or_else(|| config.get("dtype")).and_then(|v| v.as_str()).map(|s| s.to_uppercase());
    a.has_chat_template = dir.join("chat_template.jinja").exists()
        || read_json(&dir.join("tokenizer_config.json")).map(|t| t.get("chat_template").is_some()).unwrap_or(false);
    a.tokenizer_kind = dir.join("tokenizer.json").exists().then(|| "tokenizer.json".to_string());
    if !BUILTIN_SAFETENSORS_TYPES.contains(&model_type.as_str()) {
        a.problems.push(format!(
            "Model type \"{model_type}\" is not supported by the built-in engine (supported: {}).",
            BUILTIN_SAFETENSORS_TYPES.join(", ")
        ));
    }
    if weights.is_empty() {
        a.problems.push("No .safetensors weight files found.".into());
    }
    if a.tokenizer_kind.is_none() {
        a.problems.push("Missing tokenizer.json.".into());
    }
    if config.get("quantization_config").is_some() {
        a.problems.push(
            "This checkpoint is quantised with a method (GPTQ/AWQ/bitsandbytes…) the built-in engine can't load."
                .into(),
        );
    }
    if !a.has_chat_template {
        a.notes.push("No chat template found; a generic template will be used.".into());
    }
    a.runnable = a.problems.is_empty();
    a
}

fn kind_label(c: Capability) -> &'static str {
    match c {
        Capability::Chat | Capability::Vision => "chat",
        Capability::Transcription => "transcription",
        Capability::ImageGeneration => "image-generation",
        _ => "unknown",
    }
}

/// Registers (or refreshes) the library entry for a finished Hugging Face download.
pub async fn register_hf_download(db: &Db, row: &DownloadRow, model_dir: &Path) -> Result<String> {
    let dir = model_dir.to_path_buf();
    let primary = row.options.primary_file.clone();
    let repo = row.repo_id.clone();
    let analysis = {
        let dir = dir.clone();
        let primary = primary.clone();
        let repo = repo.clone();
        tokio::task::spawn_blocking(move || match primary.as_deref() {
            Some(f) if f.to_lowercase().ends_with(".gguf") => analyze_gguf(&dir.join(f)),
            _ => analyze_dir(&dir, Some(&repo)),
        })
        .await
        .context("analysis task failed")?
    };
    let existing = {
        let repo = repo.clone();
        let primary = primary.clone();
        db.call(move |c| db::models::find_by_repo(c, &repo, primary.as_deref())).await?
    };
    // All files of the download: one file for most GGUF models, every part of a split one.
    let size: u64 = row.files.iter().map(|f| f.size).sum();
    let display_name = row.options.display_name.clone().unwrap_or_else(|| {
        let base = repo.rsplit('/').next().unwrap_or(&repo).to_string();
        match &analysis.quantization {
            Some(q) if primary.is_some() => format!("{base} · {q}"),
            _ => base,
        }
    });
    let model = ModelRow {
        id: existing.as_ref().map(|m| m.id.clone()).unwrap_or_else(crate::util::new_id),
        source: "huggingface".into(),
        repo_id: Some(repo.clone()),
        revision: row.revision.clone(),
        display_name: existing.as_ref().map(|m| m.display_name.clone()).unwrap_or(display_name),
        path: dir.to_string_lossy().to_string(),
        primary_file: primary,
        format: analysis.format.clone(),
        architecture: analysis.architecture.clone(),
        kind: kind_label(analysis.capability).into(),
        size_bytes: size as i64,
        quantization: analysis.quantization.clone(),
        parameters: analysis.parameters.map(|p| p as i64),
        context_length: analysis.context_length.map(|c| c as i64),
        files: row
            .files
            .iter()
            .map(|f| StoredFile { path: f.local_path.clone(), size: f.size, sha256: f.sha256.clone() })
            .collect(),
        analysis,
        created_at: existing.as_ref().map(|m| m.created_at.clone()).unwrap_or_else(now_rfc3339),
        last_used_at: existing.and_then(|m| m.last_used_at),
    };
    let id = model.id.clone();
    db.call(move |c| db::models::upsert(c, &model)).await?;
    Ok(id)
}

/// Adds an existing GGUF file from disk to the library (the file is not copied or moved).
pub async fn import_gguf(db: &Db, path: &Path) -> Result<ModelRow, String> {
    let p = path.to_path_buf();
    if !p.is_absolute() {
        return Err("Please enter the full path to the .gguf file.".into());
    }
    if !p.is_file() {
        return Err(format!("{} does not exist or is not a file.", p.display()));
    }
    if p.extension().map(|e| !e.eq_ignore_ascii_case("gguf")).unwrap_or(true) {
        return Err("Only .gguf files can be imported.".into());
    }
    let pp = p.clone();
    let analysis = tokio::task::spawn_blocking(move || analyze_gguf(&pp)).await.map_err(|e| e.to_string())?;
    let file_name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
    let dir = p.parent().unwrap_or(Path::new("/")).to_string_lossy().to_string();
    let rows = db.call(db::models::list).await.map_err(|e| e.to_string())?;
    if let Some(m) = rows.iter().find(|m| m.path == dir && m.primary_file.as_deref() == Some(file_name.as_str())) {
        return Err(format!("This file is already in your library as \"{}\".", m.display_name));
    }
    let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
    let model = ModelRow {
        id: crate::util::new_id(),
        source: "local".into(),
        repo_id: None,
        revision: None,
        display_name: file_name.trim_end_matches(".gguf").to_string(),
        path: dir,
        primary_file: Some(file_name.clone()),
        format: "GGUF".into(),
        architecture: analysis.architecture.clone(),
        kind: kind_label(analysis.capability).into(),
        size_bytes: size as i64,
        quantization: analysis.quantization.clone(),
        parameters: analysis.parameters.map(|p| p as i64),
        context_length: analysis.context_length.map(|c| c as i64),
        files: vec![StoredFile { path: file_name, size, sha256: None }],
        analysis,
        created_at: now_rfc3339(),
        last_used_at: None,
    };
    let m = model.clone();
    db.call(move |c| db::models::upsert(c, &m)).await.map_err(|e| e.to_string())?;
    Ok(model)
}

/// Re-runs analysis for a stored model (after an app update or file changes).
pub async fn reanalyze(db: &Db, id: &str) -> Result<ModelRow> {
    let i = id.to_string();
    let mut m = db.call(move |c| db::models::get(c, &i)).await?.context("model not found")?;
    let dir = PathBuf::from(&m.path);
    let primary = m.primary_file.clone();
    let repo = m.repo_id.clone();
    m.analysis = tokio::task::spawn_blocking(move || match primary.as_deref() {
        Some(f) if f.to_lowercase().ends_with(".gguf") => analyze_gguf(&dir.join(f)),
        _ => analyze_dir(&dir, repo.as_deref()),
    })
    .await?;
    m.kind = kind_label(m.analysis.capability).into();
    m.format = m.analysis.format.clone();
    m.architecture = m.analysis.architecture.clone();
    m.quantization = m.analysis.quantization.clone();
    m.parameters = m.analysis.parameters.map(|p| p as i64);
    m.context_length = m.analysis.context_length.map(|c| c as i64);
    let mm = m.clone();
    db.call(move |c| db::models::upsert(c, &mm)).await?;
    Ok(m)
}

/// Deletes a stored model. Files inside the models folder are removed; imported files
/// outside it are only removed when `delete_files` is set.
pub async fn delete_model(db: &Db, models_dir: &Path, id: &str, delete_files: bool) -> Result<()> {
    let i = id.to_string();
    let m = db.call(move |c| db::models::get(c, &i)).await?.context("model not found")?;
    let path = PathBuf::from(&m.path);
    let inside = path
        .canonicalize()
        .ok()
        .zip(models_dir.canonicalize().ok())
        .map(|(p, base)| p.starts_with(&base) && p != base)
        .unwrap_or(false);
    if m.source == "huggingface" && inside {
        match &m.primary_file {
            Some(f) => {
                // Other versions of the same repository share the folder.
                let others = db
                    .call({
                        let p = m.path.clone();
                        let id = m.id.clone();
                        move |c| {
                            Ok(db::models::list(c)?
                                .into_iter()
                                .filter(|x| x.path == p && x.id != id)
                                .collect::<Vec<_>>())
                        }
                    })
                    .await?;
                if others.is_empty() {
                    let _ = tokio::fs::remove_dir_all(&path).await;
                } else {
                    // Every file of this model (all parts of a split GGUF), except files
                    // another model in the folder uses too.
                    let shared: std::collections::HashSet<&str> = others
                        .iter()
                        .flat_map(|o| o.files.iter().map(|x| x.path.as_str()).chain(o.primary_file.as_deref()))
                        .collect();
                    let own = m.files.iter().map(|x| x.path.as_str()).chain(std::iter::once(f.as_str()));
                    for rel in own.filter(|rel| !shared.contains(rel)) {
                        if let Some(file) = crate::util::safe_join(&path, rel) {
                            let _ = tokio::fs::remove_file(&file).await;
                        }
                    }
                }
            }
            None => {
                tokio::fs::remove_dir_all(&path).await.with_context(|| format!("removing {}", path.display()))?;
            }
        }
    } else if delete_files && let Some(f) = &m.primary_file {
        tokio::fs::remove_file(path.join(f)).await.with_context(|| format!("removing {}", path.join(f).display()))?;
    }
    let i = id.to_string();
    db.call(move |c| db::models::delete(c, &i)).await?;
    Ok(())
}

/// One row of the unified library view.
#[derive(Clone, Debug, Serialize)]
pub struct LibraryItem {
    /// `local:<id>` or `ollama:<name>`.
    pub model_ref: String,
    pub source: String,
    pub name: String,
    pub original_name: String,
    pub format: String,
    pub architecture: String,
    pub size_bytes: u64,
    pub quantization: String,
    pub parameters: String,
    pub capability: Capability,
    pub capabilities: Vec<String>,
    pub runnable: bool,
    pub problems: Vec<String>,
    pub last_used: Option<String>,
    pub location: String,
    pub loaded: bool,
    pub engine: &'static str,
}

impl LibraryItem {
    pub fn size_label(&self) -> String {
        if self.size_bytes > 0 { human_bytes(self.size_bytes) } else { "—".into() }
    }
    pub fn is_chat(&self) -> bool {
        matches!(self.capability, Capability::Chat | Capability::Vision)
    }
    pub fn supports_images(&self) -> bool {
        self.capability == Capability::Vision || self.capabilities.iter().any(|c| c == "vision")
    }
}

pub fn item_from_row(m: &ModelRow, loaded: &HashSet<String>) -> LibraryItem {
    let model_ref = format!("local:{}", m.id);
    LibraryItem {
        loaded: loaded.contains(&model_ref),
        model_ref,
        source: if m.source == "local" { "Imported".into() } else { "Hugging Face".into() },
        name: m.display_name.clone(),
        original_name: m
            .repo_id
            .clone()
            .map(|r| match &m.primary_file {
                Some(f) => format!("{r} / {f}"),
                None => r,
            })
            .unwrap_or_else(|| m.primary_file.clone().unwrap_or_default()),
        format: m.format.clone(),
        architecture: m.architecture.clone().unwrap_or_default(),
        size_bytes: m.size_bytes.max(0) as u64,
        quantization: m.quantization.clone().unwrap_or_default(),
        parameters: m.parameters.map(|p| human_params(p as u64)).unwrap_or_default(),
        capability: m.analysis.capability,
        capabilities: vec![m.analysis.capability.label().to_string()],
        runnable: m.analysis.runnable,
        problems: m.analysis.problems.clone(),
        last_used: m.last_used_at.clone(),
        location: match &m.primary_file {
            Some(f) => Path::new(&m.path).join(f).to_string_lossy().to_string(),
            None => m.path.clone(),
        },
        engine: "Built-in",
    }
}

pub fn item_from_ollama(
    m: &LocalModel,
    meta: Option<&db::models::ModelMeta>,
    loaded: &HashSet<String>,
    caps: Option<&Vec<String>>,
) -> LibraryItem {
    let model_ref = format!("ollama:{}", m.name);
    let capabilities = caps.cloned().unwrap_or_default();
    let vision = capabilities.iter().any(|c| c == "vision");
    let embedding_only =
        capabilities.iter().any(|c| c == "embedding") && !capabilities.iter().any(|c| c == "completion");
    LibraryItem {
        loaded: loaded.contains(&model_ref),
        model_ref,
        source: "Ollama".into(),
        name: meta.and_then(|x| x.display_name.clone()).unwrap_or_else(|| m.name.clone()),
        original_name: m.name.clone(),
        format: if m.details.format.is_empty() { "ollama".into() } else { m.details.format.to_uppercase() },
        architecture: m.details.family.clone(),
        size_bytes: m.size,
        quantization: m.details.quantization_level.clone(),
        parameters: m.details.parameter_size.clone(),
        capability: if embedding_only {
            Capability::Embedding
        } else if vision {
            Capability::Vision
        } else {
            Capability::Chat
        },
        capabilities,
        runnable: !embedding_only,
        problems: if embedding_only {
            vec!["Embedding model: it creates vectors, it doesn't chat.".into()]
        } else {
            Vec::new()
        },
        last_used: meta.and_then(|x| x.last_used_at.clone()),
        location: "Managed by Ollama".into(),
        engine: "Ollama",
    }
}

/// Unified listing. Ollama errors are returned separately so the page can explain them.
pub async fn list_all(
    db: &Db,
    ollama: &OllamaClient,
    loaded: &HashSet<String>,
    ollama_caps: &std::collections::HashMap<String, Vec<String>>,
) -> (Vec<LibraryItem>, Option<String>) {
    let rows = db.call(db::models::list).await.unwrap_or_default();
    let meta = db.call(db::models::all_meta).await.unwrap_or_default();
    let mut items: Vec<LibraryItem> = rows.iter().map(|m| item_from_row(m, loaded)).collect();
    let mut ollama_error = None;
    if ollama.enabled() {
        match ollama.list_local().await {
            Ok(models) => {
                for m in &models {
                    let mm = meta.iter().find(|x| x.model_ref == format!("ollama:{}", m.name));
                    items.push(item_from_ollama(m, mm, loaded, ollama_caps.get(&m.name)));
                }
            }
            Err(e) => ollama_error = Some(e.to_string()),
        }
    }
    items.sort_by(|a, b| b.last_used.cmp(&a.last_used).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    (items, ollama_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftypes_map_to_labels() {
        assert_eq!(ftype_label(15), Some("Q4_K_M"));
        assert_eq!(ftype_label(30), Some("IQ4_XS"));
        assert_eq!(ftype_label(999), None);
    }

    #[test]
    fn non_gguf_files_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.gguf");
        std::fs::write(&p, b"definitely not gguf").unwrap();
        let a = analyze_gguf(&p);
        assert!(!a.runnable);
        assert!(!a.problems.is_empty());
    }

    #[test]
    fn split_ggufs_are_not_runnable() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m-00001-of-00002.gguf");
        std::fs::write(&p, b"x").unwrap();
        let a = analyze_gguf(&p);
        assert!(!a.runnable);
        assert!(a.problems[0].contains("split"));
    }

    #[test]
    fn safetensors_header_param_count() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("model.safetensors");
        let header = br#"{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":{"format":"pt"}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 24]);
        std::fs::write(&p, bytes).unwrap();
        assert_eq!(safetensors_params(&p), Some(6));
    }

    #[test]
    fn unsupported_safetensors_dirs_are_flagged() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type":"bert"}"#).unwrap();
        let a = analyze_dir(dir.path(), None);
        assert!(!a.runnable);
        assert!(a.problems.iter().any(|p| p.contains("bert")));
        assert!(a.problems.iter().any(|p| p.contains("tokenizer")));
    }

    #[test]
    fn whisper_dirs_are_recognised() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type":"whisper","num_mel_bins":80}"#).unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), "x").unwrap();
        let a = analyze_dir(dir.path(), Some("openai/whisper-tiny"));
        assert_eq!(a.capability, Capability::Transcription);
        assert_eq!(a.mel_bins, Some(80));
        assert!(a.runnable);
    }
}
