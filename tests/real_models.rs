//! Tests that need real models, a real Ollama, internet access or a lot of disk space.
//!
//! They are ignored by default (and in CI). Run them on a real machine with, for example:
//!
//! ```text
//! LAS_TEST_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//!   cargo test --release --test real_models -- --ignored --nocapture builtin
//! LAS_TEST_OLLAMA_MODEL=qwen3:0.6b cargo test --release --test real_models -- --ignored ollama
//! LAS_TEST_NETWORK=1 cargo test --release --test real_models -- --ignored network
//! LAS_TEST_NETWORK=1 LAS_TEST_AUDIO=/path/to/speech.wav \
//!   cargo test --release --test real_models -- --ignored whisper
//! LAS_TEST_NETWORK=1 LAS_TEST_HEAVY=1 cargo test --release --test real_models -- --ignored image
//! ```
//!
//! Each test skips itself (and says so) when its variable is not set. The Ollama tests talk
//! to `http://127.0.0.1:11434` unless `LAS_TEST_OLLAMA_URL` says otherwise.

mod common;

use std::time::Duration;

use axum::http::StatusCode;
use common::*;
use local_ai_studio::db::conversations::{self as convdb, MessageStatus};
use local_ai_studio::db::downloads::DownloadState;
use serde_json::json;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn real_upstreams() -> Upstreams {
    Upstreams {
        hf: "https://huggingface.co".into(),
        // Not OLLAMA_HOST: that often holds a bind address such as 0.0.0.0 without a scheme.
        ollama: env("LAS_TEST_OLLAMA_URL").unwrap_or_else(|| "http://127.0.0.1:11434".into()),
        library: "https://ollama.com".into(),
        registry: "https://registry.ollama.ai".into(),
    }
}

async fn download_and_wait(app: &TestApp, repo: &str, variant: &str, minutes: u64) -> String {
    let r = app.post_form(&format!("/discover/hf/{repo}/download"), &[("variant", variant)]).await;
    assert_eq!(r.flash().map(|f| f.0).as_deref(), Some("ok"), "{:?}", r.flash());
    let id = app.state.downloads.list().await.unwrap().iter().find(|j| j.row.repo_id == repo).unwrap().row.id.clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(minutes * 60);
    loop {
        let jobs = app.state.downloads.list().await.unwrap();
        let j = jobs.iter().find(|j| j.row.id == id).unwrap();
        match j.row.state {
            DownloadState::Completed => return j.row.model_id.clone().expect("model registered"),
            DownloadState::Failed | DownloadState::Cancelled => panic!("download failed: {:?}", j.row.error),
            _ => {}
        }
        assert!(tokio::time::Instant::now() < deadline, "download too slow");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

#[tokio::test]
#[ignore = "needs a GGUF chat model: LAS_TEST_GGUF=/path/to/model.gguf"]
async fn builtin_engine_chats_with_a_real_gguf() {
    let Some(path) = env("LAS_TEST_GGUF") else {
        eprintln!("skipped: LAS_TEST_GGUF is not set");
        return;
    };
    let app = TestApp::new();
    let r = app.post_form("/library/import", &[("path", &path)]).await;
    let (kind, msg) = r.flash().unwrap();
    assert_eq!(kind, "ok", "{msg}");
    let model_ref = r.location().trim_start_matches("/library/model/").to_string();
    let r = app
        .post_multipart(
            "/chat/new",
            &[("model_ref", &model_ref), ("text", "What is the capital of France? Answer in one word.")],
            &[],
        )
        .await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    let id = r.location().trim_start_matches("/chat/").split('#').next().unwrap().to_string();
    // The page streams the answer as it is generated.
    let page = app.get(&format!("/chat/{id}")).await;
    assert!(page.body.contains("</html>"));
    wait_until("answer", 600, || async { !app.state.chat.is_generating(&id) }).await;
    let cid = id.clone();
    let msgs = app.state.db.call(move |c| convdb::messages(c, &cid)).await.unwrap();
    let answer = msgs.last().unwrap();
    assert_eq!(answer.status, MessageStatus::Done, "{:?}", answer.error);
    eprintln!("answer: {}", answer.content);
    assert!(answer.content.to_lowercase().contains("paris"), "{}", answer.content);
    assert!(answer.completion_tokens.unwrap_or(0) > 0);
    // Same seed, same answer.
    app.post_form(&format!("/chat/{id}/settings"), &[("seed", "1234"), ("temperature", "0.8")]).await;
    let mut answers = Vec::new();
    for _ in 0..2 {
        app.post_form(&format!("/chat/{id}/regenerate"), &[]).await;
        wait_until("answer", 600, || async { !app.state.chat.is_generating(&id) }).await;
        let cid = id.clone();
        let msgs = app.state.db.call(move |c| convdb::messages(c, &cid)).await.unwrap();
        answers.push(msgs.last().unwrap().content.clone());
    }
    assert_eq!(answers[0], answers[1], "a fixed seed must repeat the answer");
    // The OpenAI-compatible API uses the same engine.
    let r = app
        .post_json(
            "/v1/chat/completions",
            json!({ "model": model_ref, "max_tokens": 16, "messages": [{ "role": "user", "content": "Say hi." }] }),
        )
        .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    assert!(!r.json()["choices"][0]["message"]["content"].as_str().unwrap().is_empty());
    // Unloading frees the model.
    let r = app.post_form("/library/action", &[("model_ref", &model_ref), ("action", "unload")]).await;
    assert_eq!(r.flash().unwrap().0, "ok");
}

#[tokio::test]
#[ignore = "needs a running Ollama with a model: LAS_TEST_OLLAMA_MODEL=qwen3:0.6b"]
async fn ollama_chats_with_a_real_model() {
    let Some(model) = env("LAS_TEST_OLLAMA_MODEL") else {
        eprintln!("skipped: LAS_TEST_OLLAMA_MODEL is not set");
        return;
    };
    let app = TestApp::with(real_upstreams());
    let lib = app.get("/library").await;
    assert!(lib.body.contains(&model), "{model} is not listed; is Ollama running?");
    let r = app
        .post_json(
            "/v1/chat/completions",
            json!({ "model": format!("ollama:{model}"), "messages": [{ "role": "user", "content": "What is 2+2? Answer with a number." }] }),
        )
        .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    let text = r.json()["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    eprintln!("answer: {text}");
    assert!(text.contains('4'), "{text}");
}

#[tokio::test]
#[ignore = "needs internet access: LAS_TEST_NETWORK=1"]
async fn network_discovery_against_the_real_catalogs() {
    if env("LAS_TEST_NETWORK").is_none() {
        eprintln!("skipped: LAS_TEST_NETWORK is not set");
        return;
    }
    let app = TestApp::with(real_upstreams());
    let r = app.get("/discover?tab=hf&q=qwen2.5&format=gguf&sort=downloads").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("model-card"), "no results from huggingface.co");
    let r = app.get("/discover/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("Runs in Local AI Studio"));
    assert!(r.body.contains("Q4_K_M") || r.body.contains("q4_k_m"));
    let r = app.get("/discover?tab=ollama&q=qwen3").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("qwen3"), "ollama.com results could not be read; its page layout may have changed");
    let r = app.get("/discover/ollama/qwen3").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("qwen3:"), "no tags parsed");
    // Sizes come only from the HTML page (ollama.com's JSON answer has names only).
    assert!(r.body.contains("MB") || r.body.contains("GB"), "no sizes: the tags page itself was not read");
}

#[tokio::test]
#[ignore = "needs internet access: LAS_TEST_NETWORK=1 (downloads ~400 MB)"]
async fn network_download_of_a_small_gguf_then_chat() {
    if env("LAS_TEST_NETWORK").is_none() {
        eprintln!("skipped: LAS_TEST_NETWORK is not set");
        return;
    }
    let app = TestApp::with(real_upstreams());
    let model_id =
        download_and_wait(&app, "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf", 30).await;
    let model_ref = format!("local:{model_id}");
    let r = app
        .post_json("/v1/chat/completions", json!({ "model": model_ref, "max_tokens": 24, "messages": [{ "role": "user", "content": "Name a color." }] }))
        .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    eprintln!("answer: {}", r.json()["choices"][0]["message"]["content"]);
}

#[tokio::test]
#[ignore = "needs internet access and a speech recording: LAS_TEST_NETWORK=1 LAS_TEST_AUDIO=/path/speech.wav"]
async fn whisper_transcribes_real_speech() {
    let (Some(_), Some(audio)) = (env("LAS_TEST_NETWORK"), env("LAS_TEST_AUDIO")) else {
        eprintln!("skipped: LAS_TEST_NETWORK and LAS_TEST_AUDIO are needed");
        return;
    };
    let app = TestApp::with(real_upstreams());
    download_and_wait(&app, "openai/whisper-tiny", "", 20).await;
    let bytes = std::fs::read(&audio).unwrap();
    let name = std::path::Path::new(&audio).file_name().unwrap().to_string_lossy().to_string();
    let models = app.state.db.call(local_ai_studio::db::models::list).await.unwrap();
    let whisper = models.iter().find(|m| m.kind == "transcription").expect("whisper registered");
    assert!(whisper.analysis.runnable, "{:?}", whisper.analysis.problems);
    let r = app
        .post_multipart(
            "/transcribe",
            &[("model_id", &whisper.id), ("language", "auto"), ("task", "transcribe")],
            &[("file", &name, "audio/wav", bytes)],
        )
        .await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    let page_url = r.location();
    let id = page_url.trim_start_matches("/transcribe/").to_string();
    wait_until("transcript", 900, || async {
        let i = id.clone();
        app.state
            .db
            .call(move |c| local_ai_studio::db::jobs::get_transcription(c, &i))
            .await
            .unwrap()
            .map(|t| !t.status.is_active())
            .unwrap_or(false)
    })
    .await;
    let page = app.get(&page_url).await;
    let txt = app.get(&format!("{page_url}/export?format=txt")).await;
    eprintln!("transcript: {}", txt.body);
    assert!(!txt.body.trim().is_empty(), "{}", page.body);
    if let Some(expect) = env("LAS_TEST_AUDIO_EXPECT") {
        assert!(txt.body.to_lowercase().contains(&expect.to_lowercase()), "{}", txt.body);
    }
    let srt = app.get(&format!("{page_url}/export?format=srt")).await;
    assert!(srt.body.contains("-->"));
}

#[tokio::test]
#[ignore = "downloads Stable Diffusion Turbo (several GB): LAS_TEST_NETWORK=1 LAS_TEST_HEAVY=1"]
async fn image_generation_with_sd_turbo() {
    if env("LAS_TEST_NETWORK").is_none() || env("LAS_TEST_HEAVY").is_none() {
        eprintln!("skipped: LAS_TEST_NETWORK and LAS_TEST_HEAVY are needed");
        return;
    }
    let app = TestApp::with(real_upstreams());
    let model_id = download_and_wait(&app, "stabilityai/sd-turbo", "", 90).await;
    let r = app
        .post_form(
            "/create",
            &[
                ("model_id", &model_id),
                ("prompt", "a red apple on a wooden table"),
                ("size", "512x512"),
                ("steps", "1"),
                ("guidance", "0"),
                ("seed", "7"),
                ("count", "1"),
            ],
        )
        .await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    let rows = app.state.db.call(|c| local_ai_studio::db::jobs::list_generations(c, 5)).await.unwrap();
    let id = rows[0].id.clone();
    wait_until("image", 1800, || async {
        let i = id.clone();
        app.state
            .db
            .call(move |c| local_ai_studio::db::jobs::get_generation(c, &i))
            .await
            .unwrap()
            .map(|g| !g.status.is_active())
            .unwrap_or(false)
    })
    .await;
    let i = id.clone();
    let row = app.state.db.call(move |c| local_ai_studio::db::jobs::get_generation(c, &i)).await.unwrap().unwrap();
    assert!(row.error.is_none(), "{:?}", row.error);
    assert_eq!(row.outputs.len(), 1);
    let img = app.get(&format!("/files/outputs/{}", row.outputs[0].file)).await;
    assert_eq!(img.status, StatusCode::OK);
    assert_eq!(img.headers.get("content-type").unwrap(), "image/png");
}
