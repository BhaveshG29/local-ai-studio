//! The Speech models page: the Whisper Base card (its status, its facts and only the buttons
//! that make sense right now), the download progress frame that updates while the page itself
//! never reloads, the library's guard for the model voice input uses, and the parts that were
//! already there (the file form, earlier transcripts, a transcript's breadcrumb).
//!
//! The test marked `#[ignore]` uploads real speech through the form and needs Whisper Base and
//! ffmpeg (with its flite voice):
//!
//! ```text
//! LAS_WHISPER_BASE_DIR=/path/to/models/huggingface/openai/whisper-base \
//!   cargo test --test speech_page -- --ignored --nocapture
//! ```

mod common;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use common::*;
use futures_util::future::BoxFuture;
use local_ai_studio::config::Acceleration;
use local_ai_studio::db::downloads::DownloadState;
use local_ai_studio::db::jobs::{JobStatus, TranscriptionRow};
use local_ai_studio::db::models::ModelRow;
use local_ai_studio::inference::dictation::{self, LiveEngine};
use local_ai_studio::inference::whisper::LivePass;
use scraper::{ElementRef, Html, Selector};
use serde_json::{Value, json};

const BACK: &str = "/transcribe#whisper-base";
const FRAME: &str = "/speech/whisper-base/progress";

fn sel(s: &str) -> Selector {
    Selector::parse(s).unwrap()
}

/// The text of an element, with whitespace collapsed.
fn text_of(e: ElementRef) -> String {
    e.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ")
}

/// What a sighted reader sees of an element: its text without the screen-reader-only parts.
fn visible_text_of(e: ElementRef) -> String {
    let mut parts: Vec<String> = Vec::new();
    for node in e.descendants() {
        if let Some(t) = node.value().as_text() {
            let hidden =
                node.ancestors().any(|a| a.value().as_element().is_some_and(|el| el.classes().any(|c| c == "sr-only")));
            if !hidden {
                parts.push((**t).to_string());
            }
        }
    }
    parts.join(" ").split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---- reading the card --------------------------------------------------------------------------

/// What the Whisper Base card shows.
#[derive(Debug)]
struct Card {
    /// The Ready / Not ready line.
    state: String,
    /// `wb-ok`, `wb-info` or `wb-error`.
    tone: String,
    /// Definition list: label → value.
    facts: HashMap<String, String>,
    /// Labels of the buttons and links under the facts.
    actions: Vec<String>,
    /// Where the card's forms (and its delete pop-up's) post.
    forms: Vec<String>,
    /// Icon of the compatibility row.
    compat_icon: String,
    /// The download progress frame: `(src, title)`.
    frame: Option<(String, String)>,
    /// The delete confirmation's text.
    delete_popup: Option<String>,
    links: Vec<String>,
}

impl Card {
    fn fact(&self, label: &str) -> &str {
        self.facts.get(label).map(String::as_str).unwrap_or_else(|| panic!("no {label} in {:?}", self.facts))
    }
}

fn read_card(body: &str) -> Card {
    let doc = Html::parse_document(body);
    let card = doc.select(&sel("section#whisper-base")).next().expect("the Whisper Base card");
    let state = card.select(&sel(".wb-state")).next().expect("status line");
    // The dot after Ready / Not ready is only seen; screen readers get a full stop instead.
    assert_eq!(
        state.select(&sel("b + .sr-only")).next().map(text_of).as_deref(),
        Some("."),
        "no pause for screen readers"
    );
    let tone = state.value().classes().find(|c| c.starts_with("wb-") && *c != "wb-state").unwrap_or("").to_string();
    let dts: Vec<String> = card.select(&sel("dl.wb-facts > dt")).map(text_of).collect();
    let dds: Vec<String> = card.select(&sel("dl.wb-facts > dd")).map(text_of).collect();
    assert_eq!(dts.len(), dds.len());
    let mut forms: Vec<String> =
        card.select(&sel("form")).map(|f| f.value().attr("action").unwrap_or("").to_string()).collect();
    let popup = doc.select(&sel("#wb-delete[popover]")).next();
    if let Some(p) = popup {
        forms.extend(p.select(&sel("form")).map(|f| f.value().attr("action").unwrap_or("").to_string()));
    }
    Card {
        state: visible_text_of(state),
        tone,
        facts: dts.into_iter().zip(dds).collect(),
        actions: card.select(&sel(".wb-actions button, .wb-actions a")).map(text_of).collect(),
        forms,
        compat_icon: card
            .select(&sel(".wb-compat use"))
            .next()
            .and_then(|u| u.value().attr("href"))
            .and_then(|h| h.rsplit('#').next())
            .unwrap_or("")
            .to_string(),
        frame: card.select(&sel("iframe")).next().map(|f| {
            (f.value().attr("src").unwrap_or("").to_string(), f.value().attr("title").unwrap_or("").to_string())
        }),
        delete_popup: popup.map(text_of),
        links: card.select(&sel("a[href]")).map(|a| a.value().attr("href").unwrap().to_string()).collect(),
    }
}

async fn card(app: &TestApp) -> Card {
    let r = app.get("/transcribe").await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    read_card(&r.body)
}

fn button_names(actions: &[String]) -> HashSet<&str> {
    actions.iter().map(String::as_str).collect()
}

/// The card works without a script: each of its buttons (and its pop-up's) submits a form to
/// a route that exists or opens a pop-up on the page, and its links lead somewhere. (The rest
/// of the page, such as the live panel's script-driven buttons, is checked elsewhere.)
async fn assert_card_is_wired(app: &TestApp) {
    let body = app.get("/transcribe").await.body;
    let doc = Html::parse_document(&body);
    let popovers: HashSet<&str> = doc.select(&sel("[popover][id]")).filter_map(|e| e.value().attr("id")).collect();
    for root in doc.select(&sel("#whisper-base, #wb-delete")) {
        for b in root.select(&sel("button")) {
            let label = text_of(b);
            if let Some(target) = b.value().attr("popovertarget") {
                assert!(popovers.contains(target), "\"{label}\" opens a missing pop-up #{target}");
                continue;
            }
            assert_eq!(b.value().attr("type").unwrap_or("submit"), "submit", "\"{label}\" does nothing");
            let form = b
                .ancestors()
                .filter_map(ElementRef::wrap)
                .find(|e| e.value().name() == "form")
                .unwrap_or_else(|| panic!("\"{label}\" is outside any form"));
            assert_eq!(form.value().attr("method"), Some("post"), "\"{label}\"");
            let action = form.value().attr("action").unwrap();
            assert!(action.starts_with("/speech/whisper-base/"), "\"{label}\" posts to {action}");
            let r = app
                .send(
                    axum::http::Request::builder()
                        .method(axum::http::Method::OPTIONS)
                        .uri(action)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await;
            assert_ne!(r.status, StatusCode::NOT_FOUND, "\"{label}\" posts to a missing route {action}");
        }
        for a in root.select(&sel("a[href]")) {
            let href = a.value().attr("href").unwrap().split('#').next().unwrap();
            let r = app.get(href).await;
            let outage = matches!(r.status, StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE);
            assert!(r.status.is_success() || r.status.is_redirection() || outage, "{href}: {}", r.status);
        }
        for f in root.select(&sel("iframe")) {
            let src = f.value().attr("src").unwrap();
            assert_eq!(app.get(src).await.status, StatusCode::OK, "{src}");
            assert!(f.value().attr("title").is_some_and(|t| !t.is_empty()), "the frame has no title");
        }
    }
}

/// The progress frame: the response and its document.
async fn frame(app: &TestApp) -> (Resp, Html) {
    let r = app.get(FRAME).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    let doc = Html::parse_document(&r.body);
    (r, doc)
}

fn refreshes(doc: &Html) -> Option<String> {
    doc.select(&sel("meta[http-equiv=refresh]")).next().and_then(|m| m.value().attr("content").map(String::from))
}

/// The frame's "Reload the page" link, which must open in the whole window. It must not end in
/// `#whisper-base`: the page is usually at that very address already (the buttons return
/// there), and a link to the same address with a fragment only scrolls, it doesn't reload.
fn assert_reload_link(doc: &Html) {
    let a = doc.select(&sel("a[target=_top]")).next().expect("a link out of the frame");
    assert_eq!(a.value().attr("href"), Some("/transcribe"));
    assert_eq!(text_of(a), "Reload the page");
}

/// A page button: answers 303 back to the card with a message.
async fn button(app: &TestApp, action: &str) -> (String, String) {
    let r = app.post_form(&format!("/speech/whisper-base/{action}"), &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{action}: {}", r.body);
    assert_eq!(r.location(), BACK, "{action}");
    r.flash().unwrap_or_else(|| panic!("{action}: no message"))
}

// ---- a Whisper Base that looks real (as in tests/speech.rs) ------------------------------------

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

/// A stand-in for Whisper Base in voice input sessions.
struct FakeEngine;

impl LiveEngine for FakeEngine {
    fn prepare(&self) -> BoxFuture<'static, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }

    fn transcribe(
        &self,
        _pcm: Vec<f32>,
        _language: Option<String>,
        _full_window: bool,
    ) -> BoxFuture<'static, Result<Option<LivePass>, String>> {
        Box::pin(async {
            Ok(Some(LivePass {
                text: "Hello.".into(),
                avg_logprob: -0.2,
                no_speech_prob: 0.01,
                language: Some("en".into()),
                tokens: 2,
            }))
        })
    }
}

/// An open voice input session (recording), as the chat microphone or the live panel makes.
fn recording(app: &TestApp) -> String {
    app.state.dictation.start_with_engine(Arc::new(FakeEngine), None, true).unwrap().id
}

/// A Hugging Face that serves a small stand-in for Whisper Base, slowly enough to watch.
async fn slow_hf() -> FakeHf {
    let mut repo = FakeRepo::new(dictation::WHISPER_BASE_REPO);
    repo.extra = json!({ "pipeline_tag": "automatic-speech-recognition", "tags": ["whisper"], "config": { "model_type": "whisper" } });
    repo.files = vec![
        FakeFile::small("config.json", base_config().to_string().as_bytes()),
        FakeFile::small("tokenizer.json", tokenizer_json().as_bytes()),
        FakeFile::lfs("model.safetensors", bytes_of(3_000_000, 7)),
    ];
    let hf = FakeHf::start(vec![repo]).await;
    *hf.state.chunk_delay.lock().unwrap() = Some(Duration::from_millis(200));
    hf
}

// ---- the page --------------------------------------------------------------------------------

#[tokio::test]
async fn the_page_is_called_speech_models() {
    let app = TestApp::new();
    let r = app.get("/transcribe").await;
    assert_eq!(r.status, StatusCode::OK);
    let doc = Html::parse_document(&r.body);
    assert_eq!(text_of(doc.select(&sel("title")).next().unwrap()), "Speech models · Local AI Studio");
    let h1: Vec<String> = doc.select(&sel("main h1")).map(text_of).collect();
    assert_eq!(h1, ["Speech models"]);
    let lead = text_of(doc.select(&sel(".page-head .lead")).next().unwrap());
    assert_eq!(
        lead,
        "Whisper Base powers voice input in chat and the live panel; any Whisper model can transcribe recordings."
    );
    // The sidebar marks the page.
    let current = doc.select(&sel(".sidebar a[aria-current=page]")).next().unwrap();
    assert_eq!(current.value().attr("href"), Some("/transcribe"));
    assert_eq!(text_of(current), "Speech models");
    // Header, Whisper Base, the live panel, the file form, earlier transcripts.
    let at = |needle: &str| r.body.find(needle).unwrap_or_else(|| panic!("no {needle}"));
    let order = [
        "<h1>Speech models</h1>",
        "id=\"whisper-base\"",
        "data-dictation-mode=\"panel\"",
        ">Transcribe a recording</h2>",
        ">Earlier transcripts</h2>",
    ];
    for pair in order.windows(2) {
        assert!(at(pair[0]) < at(pair[1]), "{} must come before {}", pair[0], pair[1]);
    }
    // Every heading level follows the one above it.
    let levels: Vec<u8> =
        doc.select(&sel("main h1, main h2, main h3")).map(|h| h.value().name()[1..].parse().unwrap()).collect();
    assert!(levels.windows(2).all(|w| w[1] <= w[0] + 1), "{levels:?}");
}

#[tokio::test]
async fn the_card_when_whisper_base_is_not_downloaded() {
    let app = TestApp::new();
    let c = card(&app).await;
    assert_eq!(c.state, "Not ready · Whisper Base isn't downloaded yet.");
    assert_eq!(c.tone, "wb-info", "not being downloaded yet isn't an error");
    let size = local_ai_studio::util::human_bytes(dictation::WHISPER_BASE_BYTES);
    assert_eq!(c.fact("Size"), format!("{size} to download"));
    assert_eq!(c.fact("Format"), "Safetensors · runs on the built-in engine (Candle)");
    assert_eq!(c.fact("Download status"), "Not downloaded");
    assert_eq!(c.fact("Compatibility"), "Runs on the built-in engine (Candle) that comes with Local AI Studio.");
    assert_eq!(c.compat_icon, "check");
    let save_dir = app.state.downloads.repo_dir(dictation::WHISPER_BASE_REPO);
    assert_eq!(c.fact("Location"), format!("Not downloaded yet. It will be saved in {}", save_dir.display()));
    assert_eq!(c.fact("Loaded"), "No");
    assert_eq!(c.fact("Ready for transcription"), "No, it isn't downloaded yet");
    assert_eq!(c.actions, ["Download"]);
    assert_eq!(c.forms, ["/speech/whisper-base/download"]);
    assert!(c.frame.is_none() && c.delete_popup.is_none());
    assert!(c.links.contains(&"/discover/hf/openai/whisper-base".to_string()), "{:?}", c.links);
    // The file form explains what to do without a model.
    let body = app.get("/transcribe").await.body;
    assert!(body.contains("No speech model yet"));
    assert_card_is_wired(&app).await;
}

#[tokio::test]
async fn the_card_when_whisper_base_is_downloaded() {
    let app = TestApp::new();
    let (id, dir) = fake_whisper_base(&app).await;
    let c = card(&app).await;
    assert_eq!(c.state, "Ready · Voice input in chat and the live panel can use it now.");
    assert_eq!(c.tone, "wb-ok");
    let s = dictation::whisper_base_status(&app.state).await;
    assert_eq!(c.fact("Size"), format!("{} on disk", s.size_label));
    assert_eq!(c.fact("Download status"), "Downloaded");
    assert_eq!(c.fact("Compatibility"), "Works with the built-in engine (Candle); nothing else to install.");
    assert_eq!(c.compat_icon, "check");
    assert_eq!(c.fact("Location"), dir.to_string_lossy());
    assert_eq!(c.fact("Loaded"), "No · it loads by itself when you start recording");
    assert_eq!(c.fact("Ready for transcription"), "Yes");
    // Load, the details and Delete; nothing to download, cancel or unload.
    assert_eq!(c.actions, ["Load", "Model details", "Delete…"]);
    assert_eq!(c.forms, ["/speech/whisper-base/load", "/speech/whisper-base/delete"]);
    let details = format!("/library/model/local:{id}");
    assert!(c.links.contains(&details), "{:?}", c.links);
    let page = app.get(&details).await;
    assert_eq!(page.status, StatusCode::OK);
    // A speech model's details leave out what only language models have.
    for llm_only in ["Chat template", "Context memory"] {
        assert!(!page.body.contains(llm_only), "the details of Whisper Base mention {llm_only}");
    }
    assert!(page.body.contains("Estimated total"));
    let popup = c.delete_popup.as_deref().unwrap();
    assert!(popup.starts_with("Delete Whisper Base?"), "{popup}");
    assert!(
        popup.contains(&format!(
            "Its files ({}) are removed from this computer. Voice input stops working until you download it again.",
            s.size_label
        )),
        "{popup}"
    );
    assert!(popup.ends_with("Delete Keep it"), "{popup}");
    assert_card_is_wired(&app).await;

    // Delete, from the pop-up: back to the start.
    let (kind, msg) = button(&app, "delete").await;
    assert_eq!((kind.as_str(), msg.as_str()), ("ok", "Whisper Base was deleted."));
    assert!(!dir.exists());
    let c = card(&app).await;
    assert_eq!(c.fact("Download status"), "Not downloaded");
    assert_eq!(c.actions, ["Download"]);
}

#[tokio::test]
async fn a_damaged_or_missing_whisper_base_is_explained() {
    let app = TestApp::new();
    let (_, dir) = fake_whisper_base(&app).await;
    let weights = dir.join("model.safetensors");
    let f = std::fs::OpenOptions::new().write(true).open(&weights).unwrap();
    f.set_len(f.metadata().unwrap().len() - 4096).unwrap();
    let c = card(&app).await;
    assert!(
        c.state.starts_with("Not ready · Whisper Base's files are damaged: model.safetensors is incomplete"),
        "{c:?}"
    );
    assert!(c.state.ends_with("Delete it and download it again."), "{}", c.state);
    assert_eq!(c.tone, "wb-error");
    assert_eq!(c.fact("Download status"), "Downloaded");
    assert!(c.fact("Compatibility").starts_with("Can't be used: model.safetensors is incomplete"), "{c:?}");
    assert_eq!(c.compat_icon, "x");
    assert_eq!(c.fact("Ready for transcription"), "No, its files are damaged");
    // Only what helps: the details and Delete (Download refuses damaged files, Load would fail).
    assert_eq!(c.actions, ["Model details", "Delete…"]);
    assert_eq!(c.forms, ["/speech/whisper-base/delete"]);

    // The folder is gone altogether.
    std::fs::remove_dir_all(&dir).unwrap();
    let c = card(&app).await;
    assert!(c.state.contains("files are missing"), "{}", c.state);
    assert_eq!(c.tone, "wb-error");
    assert_eq!(c.fact("Download status"), "Files missing");
    assert_eq!(c.fact("Ready for transcription"), "No, its files are missing");
    assert_eq!(c.fact("Location"), dir.to_string_lossy());
    assert_eq!(c.actions, ["Model details", "Delete…"]);
    assert!(c.delete_popup.unwrap().contains("Its files are already gone."));
    assert_eq!(button(&app, "delete").await.0, "ok");
    assert_eq!(card(&app).await.actions, ["Download"]);
}

#[tokio::test]
async fn an_engine_that_cant_run_points_to_settings_instead_of_load() {
    let app = TestApp::new();
    fake_whisper_base(&app).await;
    // An accelerator this computer can't have: Metal off a Mac, CUDA on a Mac.
    let unusable = if cfg!(target_os = "macos") { Acceleration::Cuda } else { Acceleration::Metal };
    app.state.update_settings(|s| s.inference.acceleration = unusable).unwrap();
    let c = card(&app).await;
    assert!(c.state.starts_with("Not ready · Voice input can't run right now."), "{}", c.state);
    assert_eq!(c.tone, "wb-error");
    assert_eq!(c.fact("Compatibility"), "Works with the built-in engine (Candle); nothing else to install.");
    assert_eq!(c.fact("Ready for transcription"), "No, the built-in engine can't run right now");
    // Load would only repeat the same error: the way out is the engine choice in Settings.
    assert_eq!(c.actions, ["Open Settings", "Model details", "Delete…"]);
    assert!(c.links.contains(&"/settings#inference".to_string()), "{:?}", c.links);
    assert!(app.get("/settings").await.body.contains("id=\"inference\""), "Settings has no #inference");
    assert_card_is_wired(&app).await;
    // The button itself still refuses with the reason.
    let (kind, msg) = button(&app, "load").await;
    assert_eq!(kind, "error");
    assert!(msg.starts_with("Voice input can't run right now."), "{msg}");
}

#[tokio::test]
async fn a_download_shows_its_progress_in_a_frame() {
    let hf = slow_hf().await;
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    let app = TestApp::with(up);

    let (kind, msg) = button(&app, "download").await;
    assert_eq!(kind, "ok", "{msg}");
    // Waiting for its turn or already running: the frame follows it either way.
    let c = card(&app).await;
    let (src, title) = c.frame.clone().expect("a progress frame");
    assert_eq!(src, FRAME);
    assert_eq!(title, "Download progress of Whisper Base");
    assert_eq!(c.actions, ["Cancel download"], "only Cancel makes sense while downloading");
    assert_eq!(c.forms, ["/speech/whisper-base/cancel"]);
    let job = app.state.downloads.list().await.unwrap()[0].row.id.clone();
    wait_until("downloading", 20, || async {
        app.state.downloads.list().await.unwrap()[0].row.state == DownloadState::Downloading
    })
    .await;
    let c = card(&app).await;
    assert_eq!(c.state, "Not ready · Whisper Base is downloading. Voice input works as soon as it is done.");
    assert_eq!(c.tone, "wb-info");
    assert_eq!(c.fact("Ready for transcription"), "No, it is still downloading");
    assert_card_is_wired(&app).await;
    // The file form doesn't ask for a download that is already running.
    let body = app.get("/transcribe").await.body;
    assert!(body.contains("Whisper Base is downloading (see above)."), "{body}");
    assert!(!body.contains("Download Whisper Base above"));

    // The frame refreshes itself, in the app's theme, with nothing around it.
    let (r, doc) = frame(&app).await;
    assert_eq!(refreshes(&doc).as_deref(), Some("2"));
    assert_eq!(r.headers.get("cache-control").unwrap(), "no-store");
    assert!(r.headers.get("content-type").unwrap().to_str().unwrap().starts_with("text/html"));
    let csp = r.headers.get("content-security-policy").unwrap().to_str().unwrap();
    assert!(csp.contains("frame-ancestors 'self'"), "{csp}");
    let html = doc.select(&sel("html")).next().unwrap();
    assert_eq!(html.value().attr("data-theme"), Some("white"));
    let sheets: Vec<&str> = doc.select(&sel("link[rel=stylesheet]")).filter_map(|l| l.value().attr("href")).collect();
    assert!(
        sheets.iter().any(|h| h.starts_with("/assets/app.css"))
            && sheets.iter().any(|h| h.starts_with("/assets/themes.css"))
    );
    assert!(doc.select(&sel("script, nav, .sidebar, .topbar, form")).next().is_none(), "{}", r.body);
    let status = text_of(doc.select(&sel("[role=status]")).next().unwrap());
    assert!(status.starts_with("Downloading Whisper Base…") && status.ends_with('%'), "{status}");
    let bar = doc.select(&sel("progress[max='100'][value]")).next().expect("a progress bar");
    assert_eq!(bar.value().attr("aria-label"), Some("Downloaded"));
    let detail = text_of(doc.select(&sel("body > p")).next().unwrap());
    assert!(detail.contains(" of "), "bytes so far: {detail}");

    // Paused (from the Downloads page, say): the page offers to resume, the frame stops.
    app.state.downloads.pause(&job).await.unwrap();
    wait_until("paused", 20, || async {
        app.state.downloads.list().await.unwrap()[0].row.state == DownloadState::Paused
    })
    .await;
    let (_, doc) = frame(&app).await;
    assert_eq!(refreshes(&doc), None);
    assert!(text_of(doc.select(&sel("body")).next().unwrap()).starts_with("The download is paused at"));
    assert_reload_link(&doc);
    let c = card(&app).await;
    assert!(c.frame.is_none());
    assert!(c.fact("Download status").starts_with("Paused at "), "{c:?}");
    assert_eq!(c.actions, ["Resume download", "Cancel download"]);
    assert_eq!(c.fact("Ready for transcription"), "No, its download is paused");
    let doc = Html::parse_document(&app.get("/transcribe").await.body);
    let bar = doc.select(&sel("#whisper-base progress")).next().expect("a bar where the download stopped");
    assert_eq!(bar.value().attr("aria-label"), Some("Downloaded"), "the bar needs a name");

    // Resumed, then cancelled from the card.
    let (kind, msg) = button(&app, "download").await;
    assert_eq!((kind.as_str(), msg.as_str()), ("ok", "Resuming the download of Whisper Base."));
    assert!(card(&app).await.frame.is_some());
    let (kind, msg) = button(&app, "cancel").await;
    assert_eq!((kind.as_str(), msg.as_str()), ("ok", "The download of Whisper Base was cancelled."));
    wait_until("cancelled", 20, || async {
        app.state.downloads.list().await.unwrap()[0].row.state == DownloadState::Cancelled
    })
    .await;
    let (_, doc) = frame(&app).await;
    assert_eq!(refreshes(&doc), None, "a finished download stops refreshing");
    assert_eq!(text_of(doc.select(&sel("[role=status]")).next().unwrap()), "The download was cancelled.");
    assert_reload_link(&doc);
    let c = card(&app).await;
    assert!(c.frame.is_none());
    assert_eq!(c.actions, ["Download"]);

    // Downloaded to the end: the frame says so (here the stand-in files can't be used).
    *hf.state.chunk_delay.lock().unwrap() = None;
    assert_eq!(button(&app, "download").await.0, "ok");
    wait_until("downloaded", 60, || async {
        app.state.downloads.list().await.unwrap().iter().any(|j| j.row.state == DownloadState::Completed)
    })
    .await;
    let (_, doc) = frame(&app).await;
    assert_eq!(refreshes(&doc), None);
    assert_eq!(text_of(doc.select(&sel("[role=status]")).next().unwrap()), "Downloaded, but the files can't be used.");
    assert_reload_link(&doc);
}

#[tokio::test]
async fn a_stopped_download_shows_over_the_library_entry_it_replaces() {
    let hf = slow_hf().await;
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    let app = TestApp::with(up);
    // Whisper Base was downloaded, then its folder went missing…
    let (_, dir) = fake_whisper_base(&app).await;
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(card(&app).await.fact("Download status"), "Files missing");
    // …so it is downloaded again from the Marketplace, which puts its small files back first,
    // and the download is paused (as every download is when the app closes).
    let r = app.post_form("/discover/hf/openai/whisper-base/download", &[("variant", "whisper")]).await;
    assert_eq!(r.location(), "/downloads", "{:?}", r.flash());
    wait_until("the small files", 20, || async { dir.join("tokenizer.json").exists() }).await;
    let job = app.state.downloads.list().await.unwrap()[0].row.id.clone();
    app.state.downloads.pause(&job).await.unwrap();
    wait_until("paused", 20, || async {
        app.state.downloads.list().await.unwrap()[0].row.state == DownloadState::Paused
    })
    .await;
    // The folder exists again, but what it holds isn't Whisper Base yet: the card and the
    // frame tell the download's story, and Resume is there as the status line asks.
    let c = card(&app).await;
    assert!(c.state.starts_with("Not ready · Whisper Base's download is paused at"), "{}", c.state);
    assert!(c.fact("Download status").starts_with("Paused at "), "{c:?}");
    assert_eq!(c.fact("Ready for transcription"), "No, its download is paused");
    assert_eq!(c.actions, ["Resume download", "Cancel download", "Model details", "Delete…"]);
    let (_, doc) = frame(&app).await;
    let status = text_of(doc.select(&sel("[role=status]")).next().unwrap());
    assert!(status.starts_with("The download is paused at"), "{status}");
    assert_card_is_wired(&app).await;

    let (kind, msg) = button(&app, "download").await;
    assert_eq!((kind.as_str(), msg.as_str()), ("ok", "Resuming the download of Whisper Base."));
    assert!(card(&app).await.frame.is_some());
    // Cancelled: the frame tells how the download ended, not what the leftover small files
    // make of the old library entry.
    assert_eq!(button(&app, "cancel").await.0, "ok");
    wait_until("cancelled", 20, || async {
        app.state.downloads.list().await.unwrap()[0].row.state == DownloadState::Cancelled
    })
    .await;
    let (_, doc) = frame(&app).await;
    assert_eq!(refreshes(&doc), None);
    assert_eq!(text_of(doc.select(&sel("[role=status]")).next().unwrap()), "The download was cancelled.");
    assert_reload_link(&doc);
}

#[tokio::test]
async fn the_progress_frame_without_a_download() {
    let app = TestApp::new();
    let (_, doc) = frame(&app).await;
    assert_eq!(refreshes(&doc), None, "nothing to follow: no refreshing");
    assert_eq!(text_of(doc.select(&sel("[role=status]")).next().unwrap()), "Whisper Base isn't downloading.");
    assert_reload_link(&doc);

    fake_whisper_base(&app).await;
    let (_, doc) = frame(&app).await;
    assert_eq!(text_of(doc.select(&sel("[role=status]")).next().unwrap()), "Whisper Base is downloaded.");
    assert!(doc.select(&sel("[role=status] use[href$='#check']")).next().is_some());

    // In the black theme, the frame is black too.
    let r = app.post_form("/settings/theme", &[("theme", "black")]).await;
    assert_eq!(r.flash().unwrap().0, "ok");
    let (r, doc) = frame(&app).await;
    assert_eq!(doc.select(&sel("html")).next().unwrap().value().attr("data-theme"), Some("black"));
    assert!(r.body.contains("<meta name=\"color-scheme\" content=\"dark\">"));
}

#[tokio::test]
async fn buttons_that_make_no_sense_are_not_shown() {
    let app = TestApp::new();
    // Not downloaded: Download only.
    assert_eq!(button_names(&card(&app).await.actions), HashSet::from(["Download"]));
    // Downloaded: Load and Delete, never Unload before it is loaded.
    fake_whisper_base(&app).await;
    let c = card(&app).await;
    assert!(!c.actions.iter().any(|a| a == "Unload" || a == "Download" || a == "Cancel download"), "{c:?}");
    // A recording keeps the card as it is; the buttons themselves refuse what can't be done.
    let id = recording(&app);
    let (kind, msg) = button(&app, "delete").await;
    assert_eq!(kind, "error");
    assert!(msg.contains("being used for voice input"), "{msg}");
    assert!(app.state.dictation.discard(&id));
}

// ---- the library ---------------------------------------------------------------------------------

#[tokio::test]
async fn the_library_keeps_whisper_base_while_voice_input_uses_it() {
    let app = TestApp::new();
    let (id, dir) = fake_whisper_base(&app).await;
    let model_ref = format!("local:{id}");
    let form = [("model_ref", model_ref.as_str()), ("action", "delete")];
    let delete = || app.post_form("/library/action", &form);
    let session = recording(&app);
    let r = delete().await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    assert_eq!(r.location(), "/library");
    let (kind, msg) = r.flash().unwrap();
    assert_eq!(kind, "error");
    assert_eq!(msg, "This model is being used for voice input right now. Stop the recording first, then delete it.");
    assert!(dir.join("model.safetensors").exists());
    assert!(dictation::whisper_base_status(&app.state).await.ready);

    // Paused still counts: resuming must find the model there.
    let paused = app.state.dictation.pause(&session).await.unwrap();
    assert_eq!(serde_json::to_value(&paused).unwrap()["status"], "paused");
    assert_eq!(delete().await.flash().unwrap().0, "error");

    // Other models can go meanwhile.
    let tmp = tempfile::tempdir().unwrap();
    let tiny = tmp.path().join("whisper-tiny");
    std::fs::create_dir_all(&tiny).unwrap();
    let mut cfg = base_config();
    cfg["d_model"] = json!(384);
    std::fs::write(tiny.join("config.json"), cfg.to_string()).unwrap();
    let tiny_id = register(&app, "openai/whisper-tiny", &tiny).await;
    let r = app.post_form("/library/action", &[("model_ref", &format!("local:{tiny_id}")), ("action", "delete")]).await;
    assert_eq!(r.flash().unwrap(), ("ok".into(), "Model deleted.".into()));

    // Once the recording has stopped, Whisper Base can be deleted.
    let done = app.state.dictation.stop(&session).await.unwrap();
    assert_eq!(serde_json::to_value(&done).unwrap()["status"], "done");
    let r = delete().await;
    assert_eq!(r.flash().unwrap(), ("ok".into(), "Model deleted.".into()));
    assert!(!dir.exists());
    assert!(!dictation::whisper_base_status(&app.state).await.downloaded);
}

// ---- what was already on the page ------------------------------------------------------------------

#[tokio::test]
async fn the_file_form_and_earlier_transcripts_still_work() {
    let app = TestApp::new();
    let (id, _) = fake_whisper_base(&app).await;
    let doc = Html::parse_document(&app.get("/transcribe").await.body);
    let form = doc.select(&sel("form.transcribe-form")).next().expect("the file form");
    assert_eq!(form.value().attr("action"), Some("/transcribe"));
    assert_eq!(form.value().attr("enctype"), Some("multipart/form-data"));
    let option = form.select(&sel("select[name=model_id] option[selected]")).next().unwrap();
    assert_eq!(option.value().attr("value"), Some(id.as_str()));
    assert!(form.select(&sel("input[type=file][name=file][required]")).next().is_some());
    assert!(form.select(&sel("select[name=language] option[value=auto][selected]")).next().is_some());
    assert!(form.select(&sel("input[name=task][value=translate]")).next().is_some());
    assert!(doc.select(&sel("h2#tx-file-title")).next().is_some_and(|h| text_of(h) == "Transcribe a recording"));

    // Wrong uploads are explained, as before.
    let r = app
        .post_multipart("/transcribe", &[("model_id", &id)], &[("file", "notes.txt", "text/plain", b"hi".to_vec())])
        .await;
    assert_eq!(r.location(), "/transcribe");
    assert_eq!(r.flash().unwrap().0, "error");
    assert!(r.flash_message().contains("audio (WAV, MP3"), "{:?}", r.flash());
    let r = app.post_multipart("/transcribe", &[("model_id", &id)], &[]).await;
    assert_eq!(r.flash().unwrap(), ("error".into(), "Choose an audio file first.".into()));

    // A finished transcript is listed and its page leads back to the Speech models page.
    let row = TranscriptionRow {
        id: local_ai_studio::util::new_id(),
        model_id: id.clone(),
        model_name: "whisper-base".into(),
        file_name: "meeting.wav".into(),
        stored_file: String::new(),
        language: Some("en".into()),
        task: "transcribe".into(),
        text: "Please send the quarterly report.".into(),
        segments: Vec::new(),
        status: JobStatus::Done,
        error: None,
        duration_ms: Some(1200),
        created_at: local_ai_studio::util::now_rfc3339(),
    };
    let tx = row.id.clone();
    app.state.db.call(move |c| local_ai_studio::db::jobs::insert_transcription(c, &row)).await.unwrap();
    let doc = Html::parse_document(&app.get("/transcribe").await.body);
    let item = doc.select(&sel(".tx-list a.tx-item")).next().expect("the transcript in the list");
    assert_eq!(item.value().attr("href"), Some(format!("/transcribe/{tx}").as_str()));
    assert!(text_of(item).contains("Please send the quarterly report."));
    let page = app.get(&format!("/transcribe/{tx}")).await;
    assert_eq!(page.status, StatusCode::OK);
    let doc = Html::parse_document(&page.body);
    let crumb = doc.select(&sel("nav.crumbs a")).next().unwrap();
    assert_eq!(crumb.value().attr("href"), Some("/transcribe"));
    assert_eq!(text_of(crumb), "Speech models");
    assert_eq!(text_of(doc.select(&sel("nav.crumbs > span:last-child")).next().unwrap()), "meeting.wav");
    assert_eq!(app.get(&format!("/transcribe/{tx}/export?format=txt")).await.body, "Please send the quarterly report.");
    assert_page_is_wired(&app, &format!("/transcribe/{tx}"), &mut HashSet::new()).await;
}

// ---- the real model ------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Whisper Base and ffmpeg: LAS_WHISPER_BASE_DIR=/path/to/whisper-base"]
async fn real_whisper_base_transcribes_an_uploaded_recording() {
    let Some(dir) = std::env::var("LAS_WHISPER_BASE_DIR").ok().filter(|v| !v.trim().is_empty()) else {
        eprintln!("skipped: LAS_WHISPER_BASE_DIR is not set");
        return;
    };
    let app = TestApp::new();
    let id = register(&app, dictation::WHISPER_BASE_REPO, Path::new(&dir)).await;
    let tmp = tempfile::tempdir().unwrap();
    let wav = tmp.path().join("short.wav");
    let status = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
        .arg("flite=text='Please send the quarterly report to the finance team by Friday.':voice=slt")
        .args(["-ar", "48000", "-ac", "1"])
        .arg(&wav)
        .status()
        .expect("ffmpeg is needed for this test");
    assert!(status.success(), "ffmpeg (with flite) failed");
    let r = app
        .post_multipart(
            "/transcribe",
            &[("model_id", &id), ("language", "auto"), ("task", "transcribe")],
            &[("file", "short.wav", "audio/wav", std::fs::read(&wav).unwrap())],
        )
        .await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    let page_url = r.location();
    let tx = page_url.trim_start_matches("/transcribe/").to_string();
    wait_until("transcript", 300, || async {
        let i = tx.clone();
        app.state
            .db
            .call(move |c| local_ai_studio::db::jobs::get_transcription(c, &i))
            .await
            .unwrap()
            .is_some_and(|t| !t.status.is_active())
    })
    .await;
    let page = app.get(&page_url).await.body;
    let doc = Html::parse_document(&page);
    let text = text_of(doc.select(&sel("#tx-text")).next().expect("the transcript")).to_lowercase();
    eprintln!("transcript: {text}");
    for w in ["send", "quarterly", "report", "finance", "team", "friday"] {
        assert!(text.contains(w), "\"{w}\" missing from: {text}");
    }
    assert_eq!(text_of(doc.select(&sel("nav.crumbs a")).next().unwrap()), "Speech models");
}
