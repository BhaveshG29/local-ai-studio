//! Checks that a folder really holds a usable Whisper Base. A folder that exists is not
//! enough: `config.json` must be a Whisper configuration with Base's dimensions,
//! `tokenizer.json` must load and know the special tokens the engine uses, and
//! `model.safetensors` must have a readable header listing every weight the engine loads,
//! with sizes that add up to the file's size (which catches files cut short). The weights
//! themselves are never read here.
//!
//! Results are cached until a file's size or modification time changes, so pages can ask on
//! every render.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::SystemTime;

use candle_transformers::models::whisper as w;

use crate::util::human_bytes;

/// How much of a file is there: "120 MB of 277 MB", or, when the two round to the same
/// figure, how much is missing ("4.00 KB short of 277 MB").
fn shortfall(have: u64, want: u64) -> String {
    let (got, all) = (human_bytes(have), human_bytes(want));
    if got == all { format!("{} short of {all}", human_bytes(want - have)) } else { format!("{got} of {all}") }
}

/// Whisper Base: 512-wide, 6 encoder and 6 decoder layers.
pub const BASE_D_MODEL: usize = 512;
pub const BASE_LAYERS: usize = 6;

const FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

/// Sizes and modification times of the files that matter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fingerprint(Vec<Option<(u64, Option<SystemTime>)>>);

pub fn fingerprint(dir: &Path) -> Fingerprint {
    Fingerprint(
        FILES
            .iter()
            .map(|f| std::fs::metadata(dir.join(f)).ok().filter(|m| m.is_file()).map(|m| (m.len(), m.modified().ok())))
            .collect(),
    )
}

/// What the check found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FileCheck {
    /// What is wrong, in words for people ("tokenizer.json is missing"); `None`: usable.
    pub problem: Option<String>,
    /// The folder itself is gone.
    pub missing: bool,
    /// `config.json` describes a model of Base's size.
    pub base_size: bool,
    /// The model knows many languages (not an English-only `.en` model).
    pub multilingual: bool,
    /// Bytes of all files in the folder.
    pub size_bytes: u64,
}

impl FileCheck {
    pub fn ok(&self) -> bool {
        self.problem.is_none()
    }

    fn fail(mut self, problem: impl Into<String>) -> Self {
        self.problem = Some(problem.into());
        self
    }
}

pub fn is_base_size(cfg: &w::Config) -> bool {
    cfg.d_model == BASE_D_MODEL && cfg.encoder_layers == BASE_LAYERS && cfg.decoder_layers == BASE_LAYERS
}

/// Reads `config.json` as a Whisper configuration.
pub fn read_config(dir: &Path) -> Option<w::Config> {
    serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).ok()?).ok()
}

fn folder_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum())
        .unwrap_or(0)
}

/// Checks a folder (reads the config, the tokenizer and the weights' header).
pub fn check_files(dir: &Path) -> FileCheck {
    let mut c = FileCheck::default();
    if !dir.is_dir() {
        c.missing = true;
        return c.fail("its folder is missing");
    }
    c.size_bytes = folder_size(dir);
    let config: w::Config = match std::fs::read_to_string(dir.join("config.json")) {
        Err(_) => return c.fail("config.json is missing"),
        Ok(text) => match serde_json::from_str(&text) {
            Ok(cfg) => cfg,
            Err(_) => return c.fail("config.json can't be read"),
        },
    };
    c.base_size = is_base_size(&config);
    if !c.base_size {
        return c.fail("config.json describes a different Whisper size, not Base");
    }
    if config.num_mel_bins != 80
        || config.max_source_positions != 1500
        || config.vocab_size < 51_864
        || config.max_target_positions < 64
        || config.encoder_attention_heads == 0
        || config.decoder_attention_heads == 0
        || !config.d_model.is_multiple_of(config.decoder_attention_heads)
    {
        return c.fail("config.json has settings the built-in engine can't use");
    }
    if !dir.join("tokenizer.json").is_file() {
        return c.fail("tokenizer.json is missing");
    }
    let tokenizer = match tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")) {
        Ok(t) => t,
        Err(_) => return c.fail("tokenizer.json can't be read"),
    };
    let needed = [w::SOT_TOKEN, w::EOT_TOKEN, w::TRANSCRIBE_TOKEN, w::NO_TIMESTAMPS_TOKEN];
    if needed.iter().any(|t| tokenizer.token_to_id(t).is_none_or(|id| id as usize >= config.vocab_size)) {
        return c.fail("tokenizer.json lacks special tokens the engine needs");
    }
    c.multilingual = tokenizer.token_to_id("<|en|>").is_some() && config.vocab_size >= 51_865;
    match check_safetensors(&dir.join("model.safetensors"), &config) {
        Ok(()) => c,
        Err(problem) => c.fail(problem),
    }
}

/// Bytes per element of a safetensors dtype.
fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    })
}

/// Every weight Candle's Whisper reads, with its shape.
fn expected_tensors(cfg: &w::Config) -> Vec<(String, Vec<u64>)> {
    let d = cfg.d_model as u64;
    let mlp = 4 * d;
    let mut t: Vec<(String, Vec<u64>)> = Vec::new();
    let mut add = |name: String, shape: Vec<u64>| t.push((name, shape));
    let norm = |add: &mut dyn FnMut(String, Vec<u64>), p: &str| {
        add(format!("{p}.weight"), vec![d]);
        add(format!("{p}.bias"), vec![d]);
    };
    let attn = |add: &mut dyn FnMut(String, Vec<u64>), p: &str| {
        for proj in ["q_proj", "v_proj", "out_proj"] {
            add(format!("{p}.{proj}.weight"), vec![d, d]);
            add(format!("{p}.{proj}.bias"), vec![d]);
        }
        add(format!("{p}.k_proj.weight"), vec![d, d]);
    };
    let block = |add: &mut dyn FnMut(String, Vec<u64>), p: &str, cross: bool| {
        attn(add, &format!("{p}.self_attn"));
        norm(add, &format!("{p}.self_attn_layer_norm"));
        if cross {
            attn(add, &format!("{p}.encoder_attn"));
            norm(add, &format!("{p}.encoder_attn_layer_norm"));
        }
        add(format!("{p}.fc1.weight"), vec![mlp, d]);
        add(format!("{p}.fc1.bias"), vec![mlp]);
        add(format!("{p}.fc2.weight"), vec![d, mlp]);
        add(format!("{p}.fc2.bias"), vec![d]);
        norm(add, &format!("{p}.final_layer_norm"));
    };
    add("model.encoder.conv1.weight".into(), vec![d, cfg.num_mel_bins as u64, 3]);
    add("model.encoder.conv1.bias".into(), vec![d]);
    add("model.encoder.conv2.weight".into(), vec![d, d, 3]);
    add("model.encoder.conv2.bias".into(), vec![d]);
    norm(&mut add, "model.encoder.layer_norm");
    for i in 0..cfg.encoder_layers {
        block(&mut add, &format!("model.encoder.layers.{i}"), false);
    }
    add("model.decoder.embed_tokens.weight".into(), vec![cfg.vocab_size as u64, d]);
    add("model.decoder.embed_positions.weight".into(), vec![cfg.max_target_positions as u64, d]);
    norm(&mut add, "model.decoder.layer_norm");
    for i in 0..cfg.decoder_layers {
        block(&mut add, &format!("model.decoder.layers.{i}"), true);
    }
    t
}

/// Reads a safetensors header (8-byte length, then JSON) and checks it against `cfg`.
fn check_safetensors(path: &Path, cfg: &w::Config) -> Result<(), String> {
    let damaged = || "model.safetensors is damaged (its header can't be read)".to_string();
    let mut f = std::fs::File::open(path).map_err(|_| "model.safetensors is missing".to_string())?;
    let file_len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|_| "model.safetensors is empty".to_string())?;
    let n = u64::from_le_bytes(len);
    if n == 0 || n > 100 * 1024 * 1024 || 8 + n > file_len {
        return Err(damaged());
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header).map_err(|_| damaged())?;
    let header: serde_json::Value = serde_json::from_slice(&header).map_err(|_| damaged())?;
    let tensors = header.as_object().ok_or_else(damaged)?;
    let mut data_end = 0u64;
    let mut shapes: HashMap<&str, (Vec<u64>, &str)> = HashMap::new();
    for (name, t) in tensors.iter().filter(|(k, _)| *k != "__metadata__") {
        let dtype = t.get("dtype").and_then(|v| v.as_str()).ok_or_else(damaged)?;
        let shape: Vec<u64> = t
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(damaged)?
            .iter()
            .map(|d| d.as_u64())
            .collect::<Option<_>>()
            .ok_or_else(damaged)?;
        let offsets: Vec<u64> = t
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(damaged)?
            .iter()
            .map(|d| d.as_u64())
            .collect::<Option<_>>()
            .ok_or_else(damaged)?;
        let [begin, end] = offsets[..] else {
            return Err(damaged());
        };
        let bytes = shape.iter().try_fold(dtype_size(dtype).ok_or_else(damaged)?, |a, b| a.checked_mul(*b));
        if end < begin || bytes != Some(end - begin) {
            return Err(damaged());
        }
        data_end = data_end.max(end);
        shapes.insert(name.as_str(), (shape, dtype));
    }
    let expected_len = 8 + n + data_end;
    if expected_len > file_len {
        return Err(format!("model.safetensors is incomplete ({})", shortfall(file_len, expected_len)));
    }
    if expected_len < file_len {
        return Err("model.safetensors is damaged (it is larger than its contents)".into());
    }
    for (name, shape) in expected_tensors(cfg) {
        match shapes.get(name.as_str()) {
            None => return Err("model.safetensors is missing some of the model's weights".into()),
            Some((s, dtype)) if *s != shape || !matches!(*dtype, "F32" | "F16" | "BF16") => {
                return Err("model.safetensors doesn't match config.json".into());
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// Cached file checks, and load failures remembered until the files change.
#[derive(Default)]
pub struct CheckCache {
    files: Mutex<HashMap<PathBuf, (Fingerprint, FileCheck)>>,
    load_failures: Mutex<HashMap<PathBuf, (Fingerprint, String)>>,
}

impl CheckCache {
    /// Checks `dir`, or answers from the cache when its files haven't changed. Reads files:
    /// call it on a blocking thread.
    pub fn check(&self, dir: &Path) -> (Fingerprint, FileCheck) {
        let fp = fingerprint(dir);
        if let Some((f, c)) = self.files.lock().unwrap_or_else(PoisonError::into_inner).get(dir)
            && *f == fp
        {
            return (fp, c.clone());
        }
        let c = check_files(dir);
        self.files.lock().unwrap_or_else(PoisonError::into_inner).insert(dir.to_path_buf(), (fp.clone(), c.clone()));
        (fp, c)
    }

    /// Loading the model in `dir` failed: say so until its files change.
    pub fn remember_load_failure(&self, dir: &Path, message: &str) {
        self.load_failures
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(dir.to_path_buf(), (fingerprint(dir), message.to_string()));
    }

    pub fn clear_load_failure(&self, dir: &Path) {
        self.load_failures.lock().unwrap_or_else(PoisonError::into_inner).remove(dir);
    }

    /// The load failure remembered for these very files, if any.
    pub fn load_failure(&self, dir: &Path, fp: &Fingerprint) -> Option<String> {
        let mut map = self.load_failures.lock().unwrap_or_else(PoisonError::into_inner);
        match map.get(dir) {
            Some((f, msg)) if f == fp => Some(msg.clone()),
            Some(_) => {
                map.remove(dir);
                None
            }
            None => None,
        }
    }

    /// Forgets everything about `dir` (after deleting it).
    pub fn forget(&self, dir: &Path) {
        self.files.lock().unwrap_or_else(PoisonError::into_inner).remove(dir);
        self.clear_load_failure(dir);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn base_config() -> serde_json::Value {
        serde_json::json!({
            "num_mel_bins": 80, "max_source_positions": 1500, "d_model": 512,
            "encoder_attention_heads": 8, "encoder_layers": 6, "vocab_size": 51865,
            "max_target_positions": 448, "decoder_attention_heads": 8, "decoder_layers": 6,
            "suppress_tokens": [1, 2]
        })
    }

    /// A tokenizer that knows Whisper's special tokens (and nothing else useful).
    pub fn tokenizer_json() -> String {
        let mut vocab = serde_json::Map::new();
        let specials = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|en|>",
            "<|translate|>",
            "<|transcribe|>",
            "<|nocaptions|>",
            "<|notimestamps|>",
            "a",
        ];
        for (i, t) in specials.iter().enumerate() {
            vocab.insert((*t).into(), (50_257 + i).into());
        }
        serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "a" }
        })
        .to_string()
    }

    /// A safetensors file with every weight Base needs. The data part is a sparse hole, so
    /// the 290 MB file takes no disk space.
    pub fn write_weights(path: &Path, cfg: &w::Config, cut: u64, skip: Option<&str>) {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, shape) in expected_tensors(cfg) {
            if Some(name.as_str()) == skip {
                continue;
            }
            let bytes = shape.iter().product::<u64>() * 4;
            header.insert(
                name,
                serde_json::json!({ "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes] }),
            );
            offset += bytes;
        }
        let text = serde_json::Value::Object(header).to_string();
        let mut f = std::fs::File::create(path).unwrap();
        std::io::Write::write_all(&mut f, &(text.len() as u64).to_le_bytes()).unwrap();
        std::io::Write::write_all(&mut f, text.as_bytes()).unwrap();
        f.set_len(8 + text.len() as u64 + offset - cut).unwrap();
    }

    /// A folder that passes the check.
    pub fn fake_base(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), base_config().to_string()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
        let cfg: w::Config = serde_json::from_value(base_config()).unwrap();
        write_weights(&dir.join("model.safetensors"), &cfg, 0, None);
    }

    #[test]
    fn a_complete_folder_passes_and_the_real_one_matches() {
        let tmp = tempfile::tempdir().unwrap();
        fake_base(tmp.path());
        let c = check_files(tmp.path());
        assert_eq!(c.problem, None);
        assert!(c.base_size && c.multilingual);
        assert!(c.size_bytes > 280_000_000, "{}", c.size_bytes);
        // The expected weights are exactly the real Whisper Base's (when it is at hand).
        if let Ok(real) = std::env::var("LAS_WHISPER_BASE_DIR") {
            let c = check_files(Path::new(&real));
            assert_eq!(c.problem, None, "{real}");
        }
    }

    #[test]
    fn broken_folders_are_explained() {
        let cfg: w::Config = serde_json::from_value(base_config()).unwrap();
        let case = |prepare: &dyn Fn(&Path)| {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("whisper-base");
            fake_base(&dir);
            prepare(&dir);
            check_files(&dir).problem.unwrap_or_default()
        };
        assert_eq!(case(&|d| std::fs::remove_dir_all(d).unwrap()), "its folder is missing");
        assert_eq!(case(&|d| std::fs::remove_file(d.join("tokenizer.json")).unwrap()), "tokenizer.json is missing");
        assert_eq!(case(&|d| std::fs::remove_file(d.join("config.json")).unwrap()), "config.json is missing");
        assert_eq!(
            case(&|d| std::fs::write(d.join("config.json"), "{ not json").unwrap()),
            "config.json can't be read"
        );
        assert_eq!(
            case(&|d| {
                let mut tiny = base_config();
                tiny["d_model"] = 384.into();
                tiny["encoder_layers"] = 4.into();
                std::fs::write(d.join("config.json"), tiny.to_string()).unwrap();
            }),
            "config.json describes a different Whisper size, not Base"
        );
        assert_eq!(case(&|d| std::fs::write(d.join("tokenizer.json"), "[]").unwrap()), "tokenizer.json can't be read");
        assert_eq!(
            case(&|d| std::fs::write(d.join("tokenizer.json"), tokenizer_json().replace("notimestamps", "x")).unwrap()),
            "tokenizer.json lacks special tokens the engine needs"
        );
        assert_eq!(
            case(&|d| std::fs::remove_file(d.join("model.safetensors")).unwrap()),
            "model.safetensors is missing"
        );
        assert!(
            case(&|d| write_weights(&d.join("model.safetensors"), &cfg, 1_000_000, None))
                .starts_with("model.safetensors is incomplete"),
        );
        // A few bytes missing out of hundreds of megabytes still shows a difference.
        assert_eq!(shortfall(290_661_376 - 4_096, 290_661_376), "4.00 KB short of 277 MB");
        assert_eq!(shortfall(120 * 1024 * 1024, 290_661_376), "120 MB of 277 MB");
        assert_eq!(
            case(&|d| write_weights(&d.join("model.safetensors"), &cfg, 0, Some("model.decoder.layers.3.fc2.weight"))),
            "model.safetensors is missing some of the model's weights"
        );
        assert_eq!(
            case(&|d| std::fs::write(d.join("model.safetensors"), b"\x10\0\0\0\0\0\0\0{\"a\": broken}").unwrap()),
            "model.safetensors is damaged (its header can't be read)"
        );
    }

    #[test]
    fn checks_are_cached_until_files_change() {
        let tmp = tempfile::tempdir().unwrap();
        fake_base(tmp.path());
        let cache = CheckCache::default();
        let (fp, c) = cache.check(tmp.path());
        assert!(c.ok());
        // A failed load is remembered for these files…
        cache.remember_load_failure(tmp.path(), "out of memory");
        assert_eq!(cache.load_failure(tmp.path(), &fp).as_deref(), Some("out of memory"));
        // …until they change.
        std::fs::write(tmp.path().join("tokenizer.json"), "broken").unwrap();
        let (fp2, c2) = cache.check(tmp.path());
        assert_ne!(fp, fp2);
        assert_eq!(c2.problem.as_deref(), Some("tokenizer.json can't be read"));
        assert_eq!(cache.load_failure(tmp.path(), &fp2), None);
        // Unchanged files are answered from the cache.
        let started = std::time::Instant::now();
        for _ in 0..100 {
            assert_eq!(cache.check(tmp.path()).1, c2);
        }
        assert!(started.elapsed().as_millis() < 500);
    }
}
