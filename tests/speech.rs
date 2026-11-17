//! Voice input with Whisper Base: the Speech models page's buttons, Whisper Base detection,
//! and the live dictation API (sessions run on a stand-in model here). The tests marked
//! `#[ignore]` use the real model and need ffmpeg (with its flite voice) to make speech:
//!
//! ```text
//! LAS_WHISPER_BASE_DIR=/path/to/models/huggingface/openai/whisper-base \
//!   cargo test --test speech -- --ignored --nocapture
//! ```

mod common;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use common::*;
use futures_util::future::BoxFuture;
use local_ai_studio::db::downloads::DownloadState;
use local_ai_studio::db::models::ModelRow;
use local_ai_studio::inference::dictation::{self, LiveEngine};
use local_ai_studio::inference::whisper::LivePass;
use serde_json::{Value, json};

const BACK: &str = "/transcribe#whisper-base";

// ---- helpers --------------------------------------------------------------------------------

async fn post(app: &TestApp, path: &str) -> Resp {
    app.send(Request::post(path).header(header::ORIGIN, ORIGIN).body(Body::empty()).unwrap()).await
}

async fn post_raw(app: &TestApp, path: &str, bytes: Vec<u8>) -> Resp {
    app.send(
        Request::post(path)
            .header(header::ORIGIN, ORIGIN)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(Body::from(bytes))
            .unwrap(),
    )
    .await
}

fn f32_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

async fn post_audio(app: &TestApp, id: &str, rate: u32, samples: &[f32]) -> Resp {
    post_raw(app, &format!("/api/speech/sessions/{id}/audio?rate={rate}"), f32_bytes(samples)).await
}

async fn delete(app: &TestApp, path: &str) -> Resp {
    app.send(Request::delete(path).header(header::ORIGIN, ORIGIN).body(Body::empty()).unwrap()).await
}

fn assert_error(r: &Resp, status: StatusCode, code: &str) -> String {
    assert_eq!(r.status, status, "{}", r.body);
    let v = r.json();
    assert_eq!(v["code"], code, "{}", r.body);
    let msg = v["error"].as_str().unwrap_or_default().to_string();
    assert!(msg.ends_with('.') && msg.len() > 10, "not a sentence: {msg}");
    msg
}

/// A page button: answers 303 back to the Speech page with a message.
async fn button(app: &TestApp, action: &str) -> (String, String) {
    let r = post(app, &format!("/speech/whisper-base/{action}")).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{action}: {}", r.body);
    assert_eq!(r.location(), BACK, "{action}");
    r.flash().unwrap_or_else(|| panic!("{action}: no message"))
}

/// Quiet room noise, deterministic.
fn noise(secs: f32, db: f32, rate: u32, seed: u32) -> Vec<f32> {
    let amp = 10f32.powf(db / 20.0) * 3f32.sqrt();
    let mut x = 0x2545_F491u32 ^ seed;
    (0..(secs * rate as f32) as usize)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// Something the voice detector takes for speech: a wobbling tone over room noise.
fn voice(secs: f32, rate: u32) -> Vec<f32> {
    let bed = noise(secs, -60.0, rate, 9);
    (0..bed.len())
        .map(|i| {
            let t = i as f32 / rate as f32;
            let pitch = 180.0 + 40.0 * (t * 3.0).sin();
            0.1 * (2.0 * std::f32::consts::PI * pitch * t).sin() + bed[i]
        })
        .collect()
}

/// A stand-in for Whisper Base: "hears" one word per call.
#[derive(Default)]
struct FakeEngine {
    passes: AtomicUsize,
}

impl LiveEngine for FakeEngine {
    fn prepare(&self) -> BoxFuture<'static, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }

    fn transcribe(
        &self,
        pcm: Vec<f32>,
        language: Option<String>,
        _full_window: bool,
    ) -> BoxFuture<'static, Result<Option<LivePass>, String>> {
        self.passes.fetch_add(1, Ordering::SeqCst);
        let secs = pcm.len() as f32 / 16_000.0;
        Box::pin(async move {
            Ok(Some(LivePass {
                text: format!("Hello from a {secs:.0} second clip."),
                avg_logprob: -0.2,
                no_speech_prob: 0.01,
                language: Some(language.unwrap_or_else(|| "en".into())),
                tokens: 6,
            }))
        })
    }
}

fn fake_session(app: &TestApp) -> String {
    app.state.dictation.start_with_engine(Arc::new(FakeEngine::default()), None, true).unwrap().id
}

// ---- a Whisper Base that looks real -------------------------------------------------------

fn base_config() -> Value {
    json!({
        "model_type": "whisper", "num_mel_bins": 80, "max_source_positions": 1500, "d_model": 512,
        "encoder_attention_heads": 8, "encoder_layers": 6, "vocab_size": 51865, "max_target_positions": 448,
        "decoder_attention_heads": 8, "decoder_layers": 6, "suppress_tokens": [1, 2]
    })
}

fn tokenizer_json() -> String {
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
    let vocab: serde_json::Map<String, Value> =
        specials.iter().enumerate().map(|(i, t)| (t.to_string(), json!(50_257 + i))).collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "a" }
    })
    .to_string()
}

/// Every weight Whisper Base has, in a sparse file (290 MB that take no disk space).
fn write_weights(path: &Path) {
    let d = 512u64;
    let mut tensors: Vec<(String, Vec<u64>)> = vec![
        ("model.encoder.conv1.weight".into(), vec![d, 80, 3]),
        ("model.encoder.conv1.bias".into(), vec![d]),
        ("model.encoder.conv2.weight".into(), vec![d, d, 3]),
        ("model.encoder.conv2.bias".into(), vec![d]),
        ("model.encoder.layer_norm.weight".into(), vec![d]),
        ("model.encoder.layer_norm.bias".into(), vec![d]),
        ("model.decoder.embed_tokens.weight".into(), vec![51865, d]),
        ("model.decoder.embed_positions.weight".into(), vec![448, d]),
        ("model.decoder.layer_norm.weight".into(), vec![d]),
        ("model.decoder.layer_norm.bias".into(), vec![d]),
    ];
    for (part, cross) in [("encoder", false), ("decoder", true)] {
        for i in 0..6 {
            let p = format!("model.{part}.layers.{i}");
            let mut attns = vec!["self_attn"];
            if cross {
                attns.push("encoder_attn");
            }
            for a in attns {
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    tensors.push((format!("{p}.{a}.{proj}.weight"), vec![d, d]));
                    if proj != "k_proj" {
                        tensors.push((format!("{p}.{a}.{proj}.bias"), vec![d]));
                    }
                }
                tensors.push((format!("{p}.{a}_layer_norm.weight"), vec![d]));
                tensors.push((format!("{p}.{a}_layer_norm.bias"), vec![d]));
            }
            tensors.push((format!("{p}.fc1.weight"), vec![4 * d, d]));
            tensors.push((format!("{p}.fc1.bias"), vec![4 * d]));
            tensors.push((format!("{p}.fc2.weight"), vec![d, 4 * d]));
            tensors.push((format!("{p}.fc2.bias"), vec![d]));
            tensors.push((format!("{p}.final_layer_norm.weight"), vec![d]));
            tensors.push((format!("{p}.final_layer_norm.bias"), vec![d]));
        }
    }
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<u64>() * 4;
        header.insert(name, json!({ "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes] }));
        offset += bytes;
    }
    let text = Value::Object(header).to_string();
    let mut f = std::fs::File::create(path).unwrap();
    std::io::Write::write_all(&mut f, &(text.len() as u64).to_le_bytes()).unwrap();
    std::io::Write::write_all(&mut f, text.as_bytes()).unwrap();
    f.set_len(8 + text.len() as u64 + offset).unwrap();
}

/// Puts a Whisper model folder in the library the way a finished download does.
async fn register(app: &TestApp, repo: &str, dir: &Path) -> String {
    let analysis = local_ai_studio::library::analyze_dir(dir, Some(repo));
    let id = local_ai_studio::util::new_id();
    let row = ModelRow {
        id: id.clone(),
        source: "huggingface".into(),
        repo_id: Some(repo.into()),
        revision: Some("main".into()),
        display_name: repo.rsplit('/').next().unwrap().into(),
        path: dir.to_string_lossy().into_owned(),
        primary_file: None,
        format: "safetensors".into(),
        architecture: Some("whisper".into()),
        kind: "transcription".into(),
        size_bytes: analysis.weights_bytes as i64,
        quantization: None,
        parameters: analysis.parameters.map(|p| p as i64),
        context_length: None,
        analysis,
        files: Vec::new(),
        created_at: local_ai_studio::util::now_rfc3339(),
        last_used_at: None,
    };
    app.state.db.call(move |c| local_ai_studio::db::models::upsert(c, &row)).await.unwrap();
    id
}

/// A Whisper Base folder inside the app's models folder (as a download would leave it).
async fn fake_whisper_base(app: &TestApp) -> (String, std::path::PathBuf) {
    let dir = app.state.downloads.repo_dir(dictation::WHISPER_BASE_REPO);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), base_config().to_string()).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
    write_weights(&dir.join("model.safetensors"));
    let id = register(app, dictation::WHISPER_BASE_REPO, &dir).await;
    (id, dir)
}

// ---- detection --------------------------------------------------------------------------------

#[tokio::test]
async fn whisper_base_missing_is_explained_everywhere() {
    let app = TestApp::new();
    let s = dictation::whisper_base_status(&app.state).await;
    assert!(!s.ready && !s.downloaded && !s.compatible && !s.loaded);
    assert_eq!(s.problem.as_deref(), Some("Whisper Base isn't downloaded yet."));
    assert_eq!((s.name.as_str(), s.repo.as_str()), ("Whisper Base", "openai/whisper-base"));
    assert_eq!(s.size_label, local_ai_studio::util::human_bytes(dictation::WHISPER_BASE_BYTES));
    assert_eq!(s.device, app.state.inference.device_label().unwrap());
    assert!(s.model_id.is_none() && s.path.is_none() && s.download.is_none());
    let mic = dictation::mic_view(&app.state).await;
    assert!(!mic.ready && !mic.downloading);
    assert_eq!(mic.reason, "Whisper Base isn't downloaded yet.");
    let r = app.post_json("/api/speech/sessions", json!({ "language": "auto" })).await;
    assert_eq!(assert_error(&r, StatusCode::CONFLICT, "whisper_unavailable"), "Whisper Base isn't downloaded yet.");
}

#[tokio::test]
async fn whisper_base_files_are_verified() {
    let app = TestApp::new();
    let (id, dir) = fake_whisper_base(&app).await;
    let s = dictation::whisper_base_status(&app.state).await;
    assert!(s.ready && s.downloaded && s.compatible && !s.loaded, "{s:?}");
    assert_eq!(s.problem, None);
    assert_eq!(s.model_id.as_deref(), Some(id.as_str()));
    assert_eq!(s.model_ref, Some(format!("local:{id}")));
    assert_eq!(s.path.as_deref(), Some(dir.to_string_lossy().as_ref()));
    assert!(s.size_label.ends_with("MB"), "{}", s.size_label);
    assert!(s.format.contains("Candle") && s.compatibility.contains("Candle"));
    assert!(dictation::mic_view(&app.state).await.ready);

    // Damage is noticed (the check notices the change without restarting) and explained.
    std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
    let s = dictation::whisper_base_status(&app.state).await;
    assert!(s.downloaded && !s.compatible && !s.ready);
    let problem = "Whisper Base's files are damaged: tokenizer.json is missing. Delete it and download it again.";
    assert_eq!(s.problem.as_deref(), Some(problem));
    let r = app.post_json("/api/speech/sessions", json!({})).await;
    assert_eq!(assert_error(&r, StatusCode::CONFLICT, "whisper_unavailable"), problem);
    // A file cut short.
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
    let f = std::fs::OpenOptions::new().write(true).open(dir.join("model.safetensors")).unwrap();
    f.set_len(f.metadata().unwrap().len() - 4096).unwrap();
    let s = dictation::whisper_base_status(&app.state).await;
    assert!(s.problem.as_deref().unwrap().contains("model.safetensors is incomplete"), "{:?}", s.problem);
    write_weights(&dir.join("model.safetensors"));
    assert!(dictation::whisper_base_status(&app.state).await.ready);

    // The Delete button removes it, files and all.
    let (kind, msg) = button(&app, "delete").await;
    assert_eq!((kind.as_str(), msg.as_str()), ("ok", "Whisper Base was deleted."));
    assert!(!dir.exists());
    assert_eq!(
        dictation::whisper_base_status(&app.state).await.problem.as_deref(),
        Some("Whisper Base isn't downloaded yet.")
    );
}

#[tokio::test]
async fn other_whisper_models_count_only_when_base_sized() {
    let app = TestApp::new();
    let tmp = tempfile::tempdir().unwrap();
    // A fine-tune of Whisper Base from another repository counts…
    let tuned = tmp.path().join("whisper-base-tuned");
    std::fs::create_dir_all(&tuned).unwrap();
    std::fs::write(tuned.join("config.json"), base_config().to_string()).unwrap();
    std::fs::write(tuned.join("tokenizer.json"), tokenizer_json()).unwrap();
    write_weights(&tuned.join("model.safetensors"));
    // …Whisper Tiny doesn't.
    let tiny = tmp.path().join("whisper-tiny");
    std::fs::create_dir_all(&tiny).unwrap();
    let mut cfg = base_config();
    cfg["d_model"] = json!(384);
    cfg["encoder_layers"] = json!(4);
    cfg["decoder_layers"] = json!(4);
    std::fs::write(tiny.join("config.json"), cfg.to_string()).unwrap();
    register(&app, "openai/whisper-tiny", &tiny).await;
    assert!(!dictation::whisper_base_status(&app.state).await.downloaded);
    let id = register(&app, "someone/whisper-base-tuned", &tuned).await;
    let s = dictation::whisper_base_status(&app.state).await;
    assert!(s.ready, "{s:?}");
    assert_eq!(s.model_id.as_deref(), Some(id.as_str()));
    assert_eq!(s.repo, "someone/whisper-base-tuned");
}

// ---- the Speech page's buttons -----------------------------------------------------------------

#[tokio::test]
async fn buttons_explain_what_happened_when_nothing_is_downloaded() {
    let app = TestApp::new();
    let (kind, msg) = button(&app, "download").await;
    assert_eq!(kind, "error", "{msg}");
    assert!(msg.contains("Hugging Face"), "{msg}");
    assert_eq!(button(&app, "cancel").await, ("info".into(), "Whisper Base isn't downloading.".into()));
    assert_eq!(button(&app, "load").await, ("error".into(), "Whisper Base isn't downloaded yet.".into()));
    assert_eq!(button(&app, "unload").await, ("info".into(), "Whisper Base isn't loaded.".into()));
    assert_eq!(button(&app, "delete").await, ("info".into(), "Whisper Base isn't downloaded.".into()));
}

#[tokio::test]
async fn download_cancel_and_damaged_download() {
    let mut repo = FakeRepo::new("openai/whisper-base");
    repo.extra = json!({ "pipeline_tag": "automatic-speech-recognition", "tags": ["whisper"], "config": { "model_type": "whisper" } });
    // Small stand-in files: the download works, but the weights aren't a real Whisper Base.
    repo.files = vec![
        FakeFile::small("config.json", base_config().to_string().as_bytes()),
        FakeFile::small("tokenizer.json", tokenizer_json().as_bytes()),
        FakeFile::lfs("model.safetensors", bytes_of(2_000_000, 7)),
    ];
    let hf = FakeHf::start(vec![repo]).await;
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    let app = TestApp::with(up);

    // Slow transfers so there is time to look at (and cancel) a running download.
    *hf.state.chunk_delay.lock().unwrap() = Some(Duration::from_millis(50));
    let (kind, msg) = button(&app, "download").await;
    assert_eq!(kind, "ok", "{msg}");
    assert!(msg.starts_with("Downloading Whisper Base ("), "{msg}");
    let s = dictation::whisper_base_status(&app.state).await;
    let d = s.download.clone().expect("download shown");
    assert!(d.active && (d.status == "Queued" || d.status == "Downloading"), "{d:?}");
    assert!(s.problem.as_deref().unwrap().starts_with("Whisper Base is"), "{:?}", s.problem);
    assert!(dictation::mic_view(&app.state).await.downloading);
    // Pressing Download again doesn't start a second one.
    assert_eq!(button(&app, "download").await.0, "info");
    let (kind, msg) = button(&app, "cancel").await;
    assert_eq!((kind.as_str(), msg.as_str()), ("ok", "The download of Whisper Base was cancelled."));
    wait_until("cancelled", 30, || async {
        app.state.downloads.list().await.unwrap().iter().all(|j| j.row.state == DownloadState::Cancelled)
    })
    .await;
    assert_eq!(dictation::whisper_base_status(&app.state).await.download.map(|d| d.id), None);

    // Downloaded for real this time: the files are checked and found wanting.
    *hf.state.chunk_delay.lock().unwrap() = None;
    assert_eq!(button(&app, "download").await.0, "ok");
    wait_until("download finished", 60, || async {
        app.state.downloads.list().await.unwrap().iter().any(|j| j.row.state == DownloadState::Completed)
    })
    .await;
    let s = dictation::whisper_base_status(&app.state).await;
    assert!(s.downloaded && !s.compatible && !s.ready, "{s:?}");
    assert!(s.problem.as_deref().unwrap().starts_with("Whisper Base's files are damaged: model.safetensors"), "{s:?}");
    let (kind, msg) = button(&app, "download").await;
    assert_eq!(kind, "error");
    assert!(msg.contains("Delete it first"), "{msg}");
    assert_eq!(button(&app, "load").await.0, "error");
    assert_eq!(button(&app, "delete").await.0, "ok");
    assert!(!dictation::whisper_base_status(&app.state).await.downloaded);
}

/// When every download slot is taken, Download says Whisper Base waits for its turn rather
/// than that it is downloading.
#[tokio::test]
async fn a_download_that_waits_for_a_slot_says_so() {
    let mut whisper = FakeRepo::new("openai/whisper-base");
    whisper.extra = json!({ "pipeline_tag": "automatic-speech-recognition", "tags": ["whisper"], "config": { "model_type": "whisper" } });
    whisper.files = vec![
        FakeFile::small("config.json", base_config().to_string().as_bytes()),
        FakeFile::small("tokenizer.json", tokenizer_json().as_bytes()),
        FakeFile::lfs("model.safetensors", bytes_of(2_000_000, 7)),
    ];
    let mut other = FakeRepo::new("someone/Busy-Model-GGUF");
    other.files =
        vec![FakeFile::small("README.md", b"# Busy\n"), FakeFile::lfs("busy-Q4_K_M.gguf", bytes_of(3_000_000, 3))];
    other.extra = json!({ "tags": ["gguf"], "pipeline_tag": "text-generation", "gguf": { "total": 1_500_000_000u64, "architecture": "qwen2", "context_length": 32768 } });
    let hf = FakeHf::start(vec![whisper, other]).await;
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    // One download at a time (Settings → Storage), read when the app starts.
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("data")).unwrap();
    std::fs::write(
        dir.path().join("data/config.toml"),
        "config_version = 1\n\n[storage]\nmax_concurrent_downloads = 1\n",
    )
    .unwrap();
    let app = TestApp::in_dir(dir, up);
    *hf.state.chunk_delay.lock().unwrap() = Some(Duration::from_millis(50));
    let busy = local_ai_studio::web::pages::discover::enqueue_hf_variant(&app.state, "someone/Busy-Model-GGUF", None)
        .await
        .expect("the other download starts");
    wait_until("the other download to start", 20, || async {
        app.state
            .downloads
            .list()
            .await
            .unwrap()
            .iter()
            .any(|j| j.row.id == busy && j.row.state == DownloadState::Downloading)
    })
    .await;
    let (kind, msg) = button(&app, "download").await;
    assert_eq!(kind, "ok", "{msg}");
    assert!(
        msg.starts_with("Whisper Base (") && msg.contains("will download as soon as the downloads ahead of it finish"),
        "{msg}"
    );
    let d = dictation::whisper_base_status(&app.state).await.download.expect("download shown");
    assert_eq!((d.status.as_str(), d.active), ("Queued", true));
    // Cancel both, so nothing is left running.
    assert_eq!(button(&app, "cancel").await.0, "ok");
    app.state.downloads.cancel(&busy).await.unwrap();
}

#[tokio::test]
async fn whisper_base_in_use_is_not_deleted_or_unloaded() {
    let app = TestApp::new();
    let (_, dir) = fake_whisper_base(&app).await;
    let id = fake_session(&app);
    let (kind, msg) = button(&app, "delete").await;
    assert_eq!(kind, "error");
    assert!(msg.contains("being used for voice input"), "{msg}");
    assert!(dir.exists());
    assert_eq!(button(&app, "unload").await.0, "error");
    // The System page follows the same rule.
    let r = app.post_form("/system/unload", &[("model_ref", "whisper")]).await;
    assert_eq!(r.flash().map(|f| f.0).as_deref(), Some("error"), "{:?}", r.flash());
    assert!(r.flash_message().contains("being used for voice input"), "{:?}", r.flash());
    assert_eq!(delete(&app, &format!("/api/speech/sessions/{id}")).await.status, StatusCode::NO_CONTENT);
    assert_eq!(
        app.post_form("/system/unload", &[("model_ref", "whisper")]).await.flash().map(|f| f.0).as_deref(),
        Some("ok")
    );
    assert_eq!(button(&app, "delete").await.0, "ok");
}

// ---- the live API -------------------------------------------------------------------------------

#[tokio::test]
async fn unknown_sessions_are_not_found() {
    let app = TestApp::new();
    let id = "0123456789abcdef0123456789abcdef";
    assert_error(&app.get(&format!("/api/speech/sessions/{id}")).await, StatusCode::NOT_FOUND, "not_found");
    for action in ["pause", "resume", "stop"] {
        let r = post(&app, &format!("/api/speech/sessions/{id}/{action}")).await;
        assert_error(&r, StatusCode::NOT_FOUND, "not_found");
    }
    assert_error(&post_audio(&app, id, 48_000, &[0.0; 480]).await, StatusCode::NOT_FOUND, "not_found");
    assert_error(&delete(&app, &format!("/api/speech/sessions/{id}")).await, StatusCode::NOT_FOUND, "not_found");
}

#[tokio::test]
async fn bad_requests_are_explained() {
    let app = TestApp::new();
    let r = app.post_json("/api/speech/sessions", json!({ "language": "klingon" })).await;
    assert!(assert_error(&r, StatusCode::BAD_REQUEST, "bad_request").contains("klingon"));
    let r = post_raw(&app, "/api/speech/sessions", b"{not json".to_vec()).await;
    assert_error(&r, StatusCode::BAD_REQUEST, "bad_request");
    // A body far too large for its one setting gets the same kind of answer, not plain text.
    let huge = format!("{{\"language\": \"en\", \"x\": \"{}\"}}", "x".repeat(3 * 1024 * 1024));
    let r = post_raw(&app, "/api/speech/sessions", huge.into_bytes()).await;
    assert_error(&r, StatusCode::BAD_REQUEST, "bad_request");
}

#[tokio::test]
async fn audio_chunks_are_checked() {
    let app = TestApp::new();
    let id = fake_session(&app);
    let path = |q: &str| format!("/api/speech/sessions/{id}/audio{q}");
    let quarter = f32_bytes(&noise(0.25, -50.0, 48_000, 1));
    assert_error(&post_raw(&app, &path(""), quarter.clone()).await, StatusCode::BAD_REQUEST, "bad_rate");
    for rate in ["4000", "200000", "fast"] {
        let r = post_raw(&app, &path(&format!("?rate={rate}")), quarter.clone()).await;
        assert_error(&r, StatusCode::BAD_REQUEST, "bad_rate");
    }
    let r = post_raw(&app, &path("?rate=48000"), quarter[..quarter.len() - 2].to_vec()).await;
    assert_error(&r, StatusCode::BAD_REQUEST, "bad_audio");
    let r = post_raw(&app, &path("?rate=48000"), f32_bytes(&[0.1, f32::NAN])).await;
    assert_error(&r, StatusCode::BAD_REQUEST, "bad_audio");
    let r = post_raw(&app, &path("?rate=48000"), vec![0; 2 * 1024 * 1024 + 4]).await;
    assert_error(&r, StatusCode::BAD_REQUEST, "too_large");
    // Nothing above was taken; a good chunk is.
    let r = post_raw(&app, &path("?rate=48000"), quarter).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    assert_eq!(r.json()["seconds"], json!(0.25));
    let r = post_raw(&app, &path("?rate=48000"), Vec::new()).await;
    assert_eq!(r.status, StatusCode::OK);
    assert_eq!(r.headers.get(header::CACHE_CONTROL).unwrap(), "no-store");
    assert_eq!(r.headers.get(header::CONTENT_TYPE).unwrap(), "application/json");
}

#[tokio::test]
async fn a_live_session_over_http() {
    let app = TestApp::new();
    let id = fake_session(&app);
    let mut audio = noise(0.5, -60.0, 48_000, 1);
    audio.extend(voice(1.5, 48_000));
    audio.extend(noise(1.5, -60.0, 48_000, 2));
    let mut statuses = Vec::new();
    for chunk in audio.chunks(12_000) {
        let r = post_audio(&app, &id, 48_000, chunk).await;
        assert_eq!(r.status, StatusCode::OK, "{}", r.body);
        let v = r.json();
        statuses.push(v["status"].as_str().unwrap().to_string());
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(statuses.iter().all(|s| s == "listening" || s == "transcribing"), "{statuses:?}");
    // The pause after the voice made its text final.
    let v = app.get(&format!("/api/speech/sessions/{id}")).await.json();
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(keys, ["error", "id", "language", "partial", "pending", "seconds", "sound", "speech", "status", "text"]);
    assert_eq!(v["id"], json!(id));
    assert_eq!(v["speech"], json!(true));
    assert_eq!(v["sound"], json!(true));
    assert_eq!(v["seconds"], json!(3.5));
    assert_eq!(v["language"], json!("en"));
    assert_eq!(v["error"], Value::Null);

    let r = post(&app, &format!("/api/speech/sessions/{id}/pause")).await;
    assert_eq!(r.json()["status"], "paused");
    let text = r.json()["text"].as_str().unwrap().to_string();
    assert!(text.starts_with("Hello from a"), "{text}");
    assert_eq!(r.json()["partial"], "");
    assert_eq!(r.json()["pending"], json!(false), "every word is final");
    // Ignored while paused.
    assert_eq!(post_audio(&app, &id, 16_000, &voice(1.0, 16_000)).await.json()["seconds"], json!(3.5));
    assert_eq!(post(&app, &format!("/api/speech/sessions/{id}/resume")).await.json()["status"], "listening");
    post_audio(&app, &id, 16_000, &voice(1.0, 16_000)).await;
    let r = post(&app, &format!("/api/speech/sessions/{id}/stop")).await;
    let v = r.json();
    assert_eq!(v["status"], "done", "{v}");
    assert!(v["text"].as_str().unwrap().len() > text.len(), "{v}");
    assert_eq!(v["seconds"], json!(4.5));
    let r = post(&app, &format!("/api/speech/sessions/{id}/resume")).await;
    assert_error(&r, StatusCode::CONFLICT, "not_active");
    // Discarded for good.
    assert_eq!(delete(&app, &format!("/api/speech/sessions/{id}")).await.status, StatusCode::NO_CONTENT);
    assert_error(&app.get(&format!("/api/speech/sessions/{id}")).await, StatusCode::NOT_FOUND, "not_found");
}

#[tokio::test]
async fn the_number_of_sessions_is_limited() {
    let app = TestApp::new();
    fake_whisper_base(&app).await;
    let limit = app.state.dictation.limits().max_sessions;
    for _ in 0..limit {
        fake_session(&app);
    }
    let r = app.post_json("/api/speech/sessions", json!({ "language": "en" })).await;
    assert_error(&r, StatusCode::TOO_MANY_REQUESTS, "too_many");
}

#[tokio::test]
async fn cross_site_requests_are_blocked() {
    let app = TestApp::new();
    let id = fake_session(&app);
    let r = app
        .send(
            Request::post(format!("/api/speech/sessions/{id}/stop"))
                .header(header::ORIGIN, "http://evil.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(r.status, StatusCode::FORBIDDEN);
    assert_eq!(app.get(&format!("/api/speech/sessions/{id}")).await.json()["status"], "listening");
}

// ---- the real model ------------------------------------------------------------------------------

fn whisper_base_dir() -> Option<String> {
    std::env::var("LAS_WHISPER_BASE_DIR").ok().filter(|v| !v.trim().is_empty())
}

/// Speech from ffmpeg's flite voice, as 48 kHz mono samples.
fn flite(dir: &Path, name: &str, text: &str) -> Vec<f32> {
    let txt = dir.join(format!("{name}.txt"));
    std::fs::write(&txt, text).unwrap();
    let wav = dir.join(format!("{name}.wav"));
    let status = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
        .arg(format!("flite=textfile='{}':voice=slt", txt.display()))
        .args(["-ar", "48000", "-ac", "1"])
        .arg(&wav)
        .status()
        .expect("ffmpeg is needed for this test");
    assert!(status.success(), "ffmpeg (with flite) failed");
    let (pcm, rate) = local_ai_studio::media::decode_audio(&wav).unwrap();
    assert_eq!(rate, 48_000);
    pcm
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .map(|w| match w.as_str() {
            "18" => "eighteen".to_string(),
            _ => w,
        })
        .collect()
}

/// Words in common, in order (longest common subsequence).
fn common_words(a: &[String], b: &[String]) -> usize {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            dp[i + 1][j + 1] = if a[i] == b[j] { dp[i][j] + 1 } else { dp[i][j + 1].max(dp[i + 1][j]) };
        }
    }
    dp[a.len()][b.len()]
}

/// Sends 48 kHz audio to a session in 250 ms chunks, in real time.
async fn send_live(app: &TestApp, id: &str, audio: &[f32]) {
    let started = Instant::now();
    for (i, chunk) in audio.chunks(12_000).enumerate() {
        if let Some(wait) = Duration::from_secs_f32(i as f32 * 0.25).checked_sub(started.elapsed()) {
            tokio::time::sleep(wait).await;
        }
        let v = post_audio(app, id, 48_000, chunk).await.json();
        assert_ne!(v["status"], "error", "{v}");
    }
}

/// Streams 48 kHz audio in 250 ms chunks, `speed` times real time, and stops.
async fn dictate(app: &TestApp, audio: &[f32], speed: f32, language: &str) -> (Value, Vec<String>) {
    let r = app.post_json("/api/speech/sessions", json!({ "language": language })).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    let id = r.json()["id"].as_str().unwrap().to_string();
    let started = Instant::now();
    let mut shown = Vec::new();
    for (i, chunk) in audio.chunks(12_000).enumerate() {
        let due = Duration::from_secs_f32(i as f32 * 0.25 / speed);
        if let Some(wait) = due.checked_sub(started.elapsed()) {
            tokio::time::sleep(wait).await;
        }
        let v = post_audio(app, &id, 48_000, chunk).await.json();
        assert_ne!(v["status"], "error", "{v}");
        let partial = v["partial"].as_str().unwrap_or_default();
        let text = v["text"].as_str().unwrap_or_default();
        let transcript = if partial.is_empty() { text.to_string() } else { format!("{text} {partial}") };
        if shown.last() != Some(&transcript) {
            eprintln!("  {:5.1} s  [{}] {transcript}", started.elapsed().as_secs_f32(), v["status"]);
            shown.push(transcript);
        }
    }
    let sent = started.elapsed();
    let v = post(app, &format!("/api/speech/sessions/{id}/stop")).await.json();
    eprintln!("  stop answered {:.1} s after the last chunk", (started.elapsed() - sent).as_secs_f32());
    assert_eq!(v["status"], "done", "{v}");
    (v, shown)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Whisper Base and ffmpeg: LAS_WHISPER_BASE_DIR=/path/to/whisper-base"]
async fn real_whisper_base_dictation() {
    let Some(dir) = whisper_base_dir() else {
        eprintln!("skipped: LAS_WHISPER_BASE_DIR is not set");
        return;
    };
    let app = TestApp::new();
    register(&app, dictation::WHISPER_BASE_REPO, Path::new(&dir)).await;
    let s = dictation::whisper_base_status(&app.state).await;
    assert!(s.ready, "{s:?}");
    let tmp = tempfile::tempdir().unwrap();

    // The Load button really loads it and checks it transcribes.
    let (kind, msg) = button(&app, "load").await;
    assert_eq!(kind, "ok", "{msg}");
    eprintln!("{msg}");
    assert!(dictation::whisper_base_status(&app.state).await.loaded);

    // One sentence, in real time.
    let sentence = "Please send the quarterly report to the finance team by Friday.";
    let mut audio = noise(0.6, -55.0, 48_000, 1);
    audio.extend(flite(tmp.path(), "short", sentence));
    audio.extend(noise(1.0, -55.0, 48_000, 2));
    eprintln!("short sentence ({:.1} s):", audio.len() as f32 / 48_000.0);
    let (v, _) = dictate(&app, &audio, 1.0, "auto").await;
    eprintln!("final: {v}");
    let text = v["text"].as_str().unwrap().to_lowercase();
    for w in ["send", "quarterly", "report", "finance", "team", "friday"] {
        assert!(text.contains(w), "\"{w}\" missing from: {text}");
    }
    assert_eq!(v["language"], "en");
    assert_eq!(v["speech"], true);

    // Over 30 seconds without real pauses: split while speaking, nothing lost or doubled.
    let long = "Good morning everyone. Today we will review the plans for the new library building. The architects \
        finished the drawings last week, and the city council approved the budget on Monday. Construction should \
        start in early spring and take about eighteen months. The reading rooms will face the river, and the \
        children's section will have its own garden. We also plan to add a small cafe near the main entrance. The \
        old building will stay open until the new one is ready, so nobody loses access to books during the work. \
        Volunteers can help us move the collection next summer. Please send your questions to the planning office \
        before the end of the month.";
    let audio = flite(tmp.path(), "long", long);
    eprintln!("long text ({:.1} s):", audio.len() as f32 / 48_000.0);
    let (v, shown) = dictate(&app, &audio, 1.0, "en").await;
    eprintln!("final: {v}");
    let expected = words(long);
    let got = words(v["text"].as_str().unwrap());
    let same = common_words(&expected, &got);
    eprintln!("{same} of {} words found; {} words written", expected.len(), got.len());
    assert!(same as f32 >= 0.85 * expected.len() as f32, "too many words missed");
    assert!(same as f32 >= 0.85 * got.len() as f32, "too many extra words (doubled at a split?)");
    assert!(shown.len() > 10, "the text wasn't updated while speaking");

    // Paused in the middle of a word ("drawing|s"), then resumed: what is said after the
    // pause arrives too. Whisper Base answers the 14 s after this very pause with just
    // "The"; such a pass is now worked out again in two halves.
    let r = app.post_json("/api/speech/sessions", json!({ "language": "en" })).await;
    let id = r.json()["id"].as_str().unwrap().to_string();
    let (pause_at, end) = (6 * 48_000, (20 * 48_000).min(audio.len()));
    send_live(&app, &id, &audio[..pause_at]).await;
    let v = post(&app, &format!("/api/speech/sessions/{id}/pause")).await.json();
    assert_eq!(v["status"], "paused", "{v}");
    let before = v["text"].as_str().unwrap().to_string();
    eprintln!("paused after 6 s: {before}");
    assert_ne!(post(&app, &format!("/api/speech/sessions/{id}/resume")).await.json()["status"], "error");
    send_live(&app, &id, &audio[pause_at..end]).await;
    let v = post(&app, &format!("/api/speech/sessions/{id}/stop")).await.json();
    eprintln!("final: {v}");
    let text = v["text"].as_str().unwrap();
    assert!(text.starts_with(&before), "the text from before the pause changed: {text}");
    let after = text[before.len()..].to_lowercase();
    for w in ["council", "budget", "construction", "months"] {
        assert!(after.contains(w), "\"{w}\", said after the pause, is missing from: {after}");
    }

    // Silence, hum and hiss: no words, and no speech heard.
    let mut audio = vec![0.0; 48_000 * 3];
    let hiss = noise(8.0, -45.0, 48_000, 3);
    audio.extend(
        hiss.iter()
            .enumerate()
            .map(|(i, h)| h + 0.01 * (2.0 * std::f32::consts::PI * 50.0 * i as f32 / 48_000.0).sin()),
    );
    audio.extend(vec![0.0; 48_000 * 3]);
    eprintln!("silence and noise:");
    let (v, _) = dictate(&app, &audio, 4.0, "auto").await;
    eprintln!("final: {v}");
    assert_eq!(v["text"], "", "words were made up from noise");
    assert_eq!(v["speech"], false);
}
