//! Model discovery: Hugging Face search with filters and pagination, model pages with
//! compatibility verdicts, the Ollama library, the offline index, and behaviour without
//! internet access.

mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::json;

fn repo(id: &str, downloads: u64, extra: serde_json::Value, files: Vec<FakeFile>) -> FakeRepo {
    let mut r = FakeRepo::new(id);
    r.downloads = downloads;
    r.extra = extra;
    r.files = files;
    r
}

async fn hub_app() -> (TestApp, FakeHf) {
    let mut repos = vec![
        repo(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            900,
            json!({ "tags": ["gguf", "license:apache-2.0"], "pipeline_tag": "text-generation", "gguf": { "total": 494_000_000u64, "architecture": "qwen2", "context_length": 32768 } }),
            vec![
                FakeFile::lfs("qwen2.5-0.5b-instruct-q4_k_m.gguf", vec![0; 10]),
                FakeFile::lfs("qwen2.5-0.5b-instruct-q8_0.gguf", vec![0; 20]),
                FakeFile::lfs("qwen2.5-0.5b-instruct-iq2_xs.gguf", vec![0; 5]),
            ],
        ),
        repo(
            "acme/mamba-GGUF",
            500,
            json!({ "tags": ["gguf"], "pipeline_tag": "text-generation", "gguf": { "total": 1_000_000u64, "architecture": "mamba" } }),
            vec![FakeFile::lfs("mamba-Q4_K_M.gguf", vec![0; 10])],
        ),
        repo(
            "openai/whisper-base",
            800,
            json!({ "tags": ["transformers", "safetensors"], "pipeline_tag": "automatic-speech-recognition", "config": { "model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"] } }),
            vec![
                FakeFile::small("config.json", br#"{"model_type":"whisper"}"#),
                FakeFile::small("tokenizer.json", b"{}"),
                FakeFile::lfs("model.safetensors", vec![0; 30]),
            ],
        ),
        repo(
            "acme/onnx-only",
            100,
            json!({ "tags": ["onnx"], "pipeline_tag": "text-generation" }),
            vec![FakeFile::lfs("model.onnx", vec![0; 10])],
        ),
    ];
    for i in 0..45 {
        repos.push(repo(&format!("filler/qwen-filler-{i:02}"), i, json!({ "tags": ["safetensors"] }), vec![]));
    }
    let hf = FakeHf::start(repos).await;
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    (TestApp::with(up), hf)
}

#[tokio::test]
async fn search_lists_models_and_pages_through_results() {
    let (app, _hf) = hub_app().await;
    let r = app.get("/discover?tab=hf&q=qwen&sort=downloads").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("Qwen2.5-0.5B-Instruct-GGUF"), "{}", r.body);
    assert!(r.body.contains("apache-2.0"));
    assert!(r.body.contains("Next page"), "no pagination link");
    // Read the link the way a browser does (attribute entities decoded).
    let doc = scraper::Html::parse_document(&r.body);
    let next = doc
        .select(&scraper::Selector::parse("nav.pager a").unwrap())
        .find(|a| a.text().collect::<String>().contains("Next page"))
        .and_then(|a| a.value().attr("href"))
        .expect("next link")
        .to_string();
    assert!(next.contains("cursor=") && next.contains("q=qwen"), "{next}");
    let r2 = app.get(&next).await;
    assert_eq!(r2.status, StatusCode::OK);
    assert!(r2.body.contains("qwen-filler"), "{}", r2.body);
    assert!(r2.body.contains("First page"));
}

#[tokio::test]
async fn model_pages_explain_compatibility() {
    let (app, _hf) = hub_app().await;
    let r = app.get("/discover/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("Runs in Local AI Studio"));
    assert!(r.body.contains("Recommended"), "no recommended variant");
    // The IQ2 quantisation is offered with a note that it needs Ollama.
    assert!(r.body.contains("IQ2_XS"), "{}", r.body);
    let r = app.get("/discover/hf/acme/mamba-GGUF").await;
    assert!(r.body.contains("Runs through Ollama"), "{}", r.body);
    assert!(r.body.contains("mamba"));
    let r = app.get("/discover/hf/openai/whisper-base").await;
    assert!(r.body.contains("Runs in Local AI Studio"), "{}", r.body);
    let r = app.get("/discover/hf/acme/onnx-only").await;
    assert!(r.body.contains("Download only"), "{}", r.body);
}

#[tokio::test]
async fn searches_feed_the_offline_index() {
    let (app, hf) = hub_app().await;
    app.get("/discover?tab=hf&q=whisper").await;
    *hf.state.down.lock().unwrap() = true;
    let r = app.get("/discover?tab=index&q=whisper").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("whisper-base"), "{}", r.body);
    let r = app.get("/discover?tab=index&q=doesnotexist").await;
    assert!(r.body.contains("Nothing in the index matches"));
}

#[tokio::test]
async fn hub_unavailable_is_explained_and_cached_results_are_used() {
    let (app, hf) = hub_app().await;
    let first = app.get("/discover?tab=hf&q=qwen").await;
    assert!(first.body.contains("Qwen2.5-0.5B-Instruct-GGUF"));
    *hf.state.down.lock().unwrap() = true;
    let r = app.get("/discover?tab=hf&q=qwen").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("Qwen2.5-0.5B-Instruct-GGUF"), "cached results not shown");
    assert!(r.body.contains("unreachable"), "no notice about the outage");
    let r = app.get("/discover?tab=hf&q=never-searched-before").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("notice-error"), "{}", r.body);
}

#[tokio::test]
async fn no_internet_at_all() {
    let app = TestApp::new();
    for path in ["/discover?tab=hf&q=llama", "/discover?tab=ollama&q=llama"] {
        let r = app.get(path).await;
        assert_eq!(r.status, StatusCode::OK, "{path}");
        assert!(r.body.contains("notice-error"), "{path} shows no error");
    }
    let r = app.get("/discover/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF").await;
    assert!(matches!(r.status, StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE), "{}", r.status);
    assert!(r.body.contains("connection") || r.body.contains("reach"), "{}", r.body);
}

#[tokio::test]
async fn ollama_library_browsing() {
    let library = fake_library().await;
    let ollama = FakeOllama::start(&[]).await;
    let mut up = Upstreams::offline();
    up.library = library;
    up.ollama = ollama.url.clone();
    let app = TestApp::with(up);
    let r = app.get("/discover?tab=ollama&q=qwen").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("qwen3"), "{}", r.body);
    assert!(r.body.contains("gemma3"));
    assert!(r.body.contains("sees images"), "vision capability not shown");
    let r = app.get("/discover/ollama/qwen3").await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    assert!(r.body.contains("qwen3:0.6b") && r.body.contains("qwen3:8b"));
    assert!(r.body.contains("523MB") || r.body.contains("523 MB"), "{}", r.body);
    // Each version has a working download form.
    assert!(r.body.contains("action=\"/discover/ollama/pull\""));
    // Library pages that don't exist are reported.
    let r = app.get("/discover/ollama/no-such-model").await;
    assert!(r.status.is_client_error() || r.status.is_server_error());
}

#[tokio::test]
async fn an_unreadable_cached_library_page_is_fetched_again() {
    let library = fake_library().await;
    let mut up = Upstreams::offline();
    up.library = library.clone();
    let app = TestApp::with(up);
    // What an earlier version stored: ollama.com's bare JSON tag list (no sizes).
    let key = format!("ollama:{library}/library/qwen3/tags");
    let now = local_ai_studio::util::now_rfc3339();
    app.state
        .db
        .call(move |c| local_ai_studio::db::catalog::cache_put(c, &key, r#"{"tags":["latest","0.6b"]}"#, &now))
        .await
        .unwrap();
    let r = app.get("/discover/ollama/qwen3").await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    assert!(r.body.contains("523MB") || r.body.contains("523 MB"), "sizes missing: {}", r.body);
}

#[tokio::test]
async fn index_refresh_crawls_both_catalogs() {
    let (app_hf, hf) = hub_app().await;
    drop(app_hf);
    let library = fake_library().await;
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    up.library = library;
    let app = TestApp::with(up);
    let r = app.post_form("/discover/index/refresh", &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    wait_until("index refresh", 60, || async { !app.state.index.get().running }).await;
    let counts = app.state.db.call(local_ai_studio::db::catalog::count_by_source).await.unwrap();
    let total: i64 = counts.iter().map(|(_, n)| *n).sum();
    assert!(total >= 40, "only {total} entries indexed: {counts:?}");
    assert!(counts.iter().any(|(s, _)| s == "ollama"), "{counts:?}");
}
