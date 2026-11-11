//! Chat tests through the real web routes, with a fake Ollama server as the model: streaming
//! pages, stop, regenerate, continue, edit, rename, delete, settings, export/import, attachments
//! the model can't handle, API endpoints, and shutdown in the middle of an answer.

mod common;

use std::collections::HashSet;
use std::time::Duration;

use axum::http::StatusCode;
use common::*;
use local_ai_studio::db::conversations::{self as convdb, MessageStatus, Role};
use serde_json::json;

const MODEL: &str = "ollama:qwen3:0.6b";

async fn setup(caps: &[&str]) -> (TestApp, FakeOllama) {
    let ollama = FakeOllama::start(&[("qwen3:0.6b", caps)]).await;
    let mut up = Upstreams::offline();
    up.ollama = ollama.url.clone();
    (TestApp::with(up), ollama)
}

async fn new_chat(app: &TestApp, text: &str) -> String {
    let r = app.post_multipart("/chat/new", &[("model_ref", MODEL), ("text", text)], &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    let loc = r.location();
    assert!(loc.starts_with("/chat/"), "{loc}");
    loc.trim_start_matches("/chat/").split('#').next().unwrap().to_string()
}

async fn wait_idle(app: &TestApp, id: &str) {
    wait_until("generation to finish", 30, || async { !app.state.chat.is_generating(id) }).await;
}

async fn messages(app: &TestApp, id: &str) -> Vec<convdb::Message> {
    let id = id.to_string();
    app.state.db.call(move |c| convdb::messages(c, &id)).await.unwrap()
}

#[tokio::test]
async fn a_chat_streams_an_answer_and_is_saved() {
    let (app, ollama) = setup(&["completion"]).await;
    let page = app.get("/chat?new=1").await;
    assert!(page.body.contains("qwen3:0.6b"), "model not offered: {}", page.body);
    ollama.set_reply(&["Hello", " there", ", friend."], Duration::from_millis(60));
    let id = new_chat(&app, "Say hello").await;
    // Opening the chat while the answer is written streams it into the page.
    let live = app.get(&format!("/chat/{id}")).await;
    assert_eq!(live.status, StatusCode::OK);
    assert!(live.body.contains("Hello there, friend."), "{}", live.body);
    assert!(live.body.contains(".composer.is-generating .btn-send{display:inline-flex}"), "composer not switched back");
    assert!(live.body.trim_end().ends_with("</html>"), "page not completed");
    wait_idle(&app, &id).await;
    let msgs = messages(&app, &id).await;
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, Role::User);
    assert_eq!(msgs[1].role, Role::Assistant);
    assert_eq!(msgs[1].content, "Hello there, friend.");
    assert_eq!(msgs[1].status, MessageStatus::Done);
    assert_eq!(msgs[1].completion_tokens, Some(3));
    // The speed is Ollama's own measurement (3 tokens in 250 ms), not tokens over the wall time
    // (which includes loading the model).
    assert_eq!(msgs[1].tokens_per_sec, Some(12.0));
    // What Ollama received: system prompt, the user message, and the default options.
    let sent = ollama.last_chat();
    assert_eq!(sent["model"], "qwen3:0.6b");
    let roles: Vec<&str> = sent["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["system", "user"]);
    assert!(sent["options"]["temperature"].is_number());
    assert!(sent["options"]["num_ctx"].is_number());
    // A finished chat renders statically with timestamps, stats and actions.
    let page = app.get(&format!("/chat/{id}")).await;
    assert!(page.body.contains("Regenerate"));
    assert!(page.body.contains("12.0 tok/s"), "speed not shown");
    assert!(page.body.contains("<time"));
    let mut seen = HashSet::new();
    assert_page_is_wired(&app, &format!("/chat/{id}"), &mut seen).await;
    // Restored after a restart.
    let app = app.restart().await;
    let page = app.get("/chat").await;
    assert_eq!(page.location(), format!("/chat/{id}"), "the last chat is not reopened");
    let page = app.get(&format!("/chat/{id}")).await;
    assert!(page.body.contains("Hello there, friend."));
}

#[tokio::test]
async fn stop_keeps_the_partial_answer() {
    let (app, ollama) = setup(&["completion"]).await;
    let words: Vec<String> = (0..200).map(|i| format!("word{i} ")).collect();
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    ollama.set_reply(&refs, Duration::from_millis(30));
    let id = new_chat(&app, "Talk for a long time").await;
    wait_until("some text", 20, || async {
        app.state.chat.get(&id).map(|g| !g.snapshot().text.is_empty()).unwrap_or(false)
    })
    .await;
    let r = app.post_form(&format!("/chat/{id}/stop"), &[]).await;
    assert!(r.status == StatusCode::SEE_OTHER || r.status == StatusCode::NO_CONTENT);
    wait_idle(&app, &id).await;
    let msgs = messages(&app, &id).await;
    let answer = &msgs[1];
    assert_eq!(answer.status, MessageStatus::Stopped);
    assert!(!answer.content.is_empty());
    assert!(answer.content.len() < words.concat().len(), "it did not stop");
    // Continue asks the model to keep writing into the same message.
    ollama.set_reply(&[" and more."], Duration::ZERO);
    let r = app.post_form(&format!("/chat/{id}/continue"), &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    wait_idle(&app, &id).await;
    let msgs = messages(&app, &id).await;
    assert_eq!(msgs.len(), 2, "continue must not add a new message");
    assert!(msgs[1].content.ends_with("and more."), "{}", msgs[1].content);
    // The model was given the partial answer to continue from.
    let sent = ollama.last_chat();
    let last = sent["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["role"], "assistant");
}

#[tokio::test]
async fn regenerate_edit_rename_settings_and_delete() {
    let (app, ollama) = setup(&["completion"]).await;
    ollama.set_reply(&["First answer."], Duration::ZERO);
    let id = new_chat(&app, "Question one").await;
    wait_idle(&app, &id).await;
    // The chat is named after its first message…
    let title = |app: &TestApp, id: &str| {
        let id = id.to_string();
        let db = app.state.db.clone();
        async move { db.call(move |c| convdb::get_conversation(c, &id)).await.unwrap().unwrap().title }
    };
    assert_eq!(title(&app, &id).await, "Question one");

    ollama.set_reply(&["Second answer."], Duration::ZERO);
    let r = app.post_form(&format!("/chat/{id}/regenerate"), &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    wait_idle(&app, &id).await;
    let msgs = messages(&app, &id).await;
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[1].content, "Second answer.");

    // Editing a user message drops everything after it and asks again.
    ollama.set_reply(&["Edited answer."], Duration::ZERO);
    let user_id = msgs[0].id.clone();
    let r = app.post_form(&format!("/chat/{id}/messages/{user_id}/edit"), &[("text", "Question one, rephrased")]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    wait_idle(&app, &id).await;
    let msgs = messages(&app, &id).await;
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].content, "Question one, rephrased");
    assert_eq!(msgs[1].content, "Edited answer.");

    // …later messages and edits don't rename it…
    assert_eq!(title(&app, &id).await, "Question one");
    let r = app.post_form(&format!("/chat/{id}/rename"), &[("title", "My renamed chat")]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    assert!(app.get(&format!("/chat/{id}")).await.body.contains("My renamed chat"));
    // …and neither does sending more after a rename.
    ollama.set_reply(&["Third answer."], Duration::ZERO);
    let r = app.post_multipart(&format!("/chat/{id}/send"), &[("text", "Another question")], &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    wait_idle(&app, &id).await;
    assert_eq!(title(&app, &id).await, "My renamed chat");
    let r = app.post_form(&format!("/chat/{id}/rename"), &[("title", "   ")]).await;
    assert_eq!(r.flash().unwrap().0, "error");

    // Per-chat settings are validated and used for the next answer.
    let r = app
        .post_form(
            &format!("/chat/{id}/settings"),
            &[
                ("model_ref", MODEL),
                ("system_prompt", "Answer like a pirate."),
                ("temperature", "0.2"),
                ("top_p", "0.5"),
                ("top_k", "10"),
                ("max_tokens", "64"),
                ("repeat_penalty", "1.2"),
                ("context_length", "2048"),
                ("seed", "42"),
            ],
        )
        .await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    let r = app.post_form(&format!("/chat/{id}/settings"), &[("temperature", "warm")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    ollama.set_reply(&["Arr."], Duration::ZERO);
    app.post_multipart(&format!("/chat/{id}/send"), &[("text", "Hi")], &[]).await;
    wait_idle(&app, &id).await;
    let sent = ollama.last_chat();
    assert_eq!(sent["messages"][0]["content"], "Answer like a pirate.");
    assert_eq!(sent["options"]["seed"], 42);
    assert_eq!(sent["options"]["num_predict"], 64);
    assert!((sent["options"]["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);

    let r = app.post_form(&format!("/chat/{id}/delete"), &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    assert_eq!(app.get(&format!("/chat/{id}")).await.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn export_and_import_round_trip() {
    let (app, ollama) = setup(&["completion"]).await;
    ollama.set_reply(&["The answer is **42**."], Duration::ZERO);
    let id = new_chat(&app, "What is the answer?").await;
    wait_idle(&app, &id).await;
    let md = app.get(&format!("/chat/{id}/export?format=md")).await;
    assert_eq!(md.status, StatusCode::OK);
    assert!(md.headers.get("content-disposition").unwrap().to_str().unwrap().contains("attachment"));
    assert!(md.body.contains("What is the answer?") && md.body.contains("The answer is **42**."));
    let js = app.get(&format!("/chat/{id}/export?format=json")).await;
    let exported = js.json();
    assert_eq!(exported["messages"].as_array().unwrap().len(), 2);
    let r = app
        .post_multipart("/chat/import", &[], &[("file", "chat.json", "application/json", js.body.clone().into_bytes())])
        .await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    let new_id = r.location().trim_start_matches("/chat/").to_string();
    assert_ne!(new_id, id);
    let msgs = messages(&app, &new_id).await;
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[1].content, "The answer is **42**.");
    // Broken files are refused with an explanation.
    let r = app
        .post_multipart("/chat/import", &[], &[("file", "chat.json", "application/json", b"{ not json".to_vec())])
        .await;
    assert_eq!(r.flash().unwrap().0, "error");
    // Export of everything from Settings.
    let all = app.post_form("/settings/data/export-all", &[]).await;
    assert_eq!(all.status, StatusCode::OK);
    assert_eq!(all.json().as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn invalid_input_is_refused_politely() {
    let (app, _ollama) = setup(&["completion"]).await;
    // Starting without a message opens an empty chat (to adjust its settings first)…
    let r = app.post_multipart("/chat/new", &[("model_ref", MODEL), ("text", "   ")], &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    let empty = r.location().trim_start_matches("/chat/").split('#').next().unwrap().to_string();
    assert!(messages(&app, &empty).await.is_empty());
    // …but sending an empty message is refused.
    let r = app.post_multipart(&format!("/chat/{empty}/send"), &[("text", "  ")], &[]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    assert!(messages(&app, &empty).await.is_empty());
    // No model.
    let r = app.post_multipart("/chat/new", &[("text", "hello")], &[]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    // A model that doesn't exist.
    let r = app.post_multipart("/chat/new", &[("model_ref", "ollama:ghost:1b"), ("text", "hello")], &[]).await;
    let flash = r.flash();
    let id = r.location().trim_start_matches("/chat/").split('#').next().unwrap_or("").to_string();
    if flash.as_ref().map(|f| f.0.as_str()) != Some("error") {
        // Accepted, then failed: the error must be stored on the answer.
        wait_idle(&app, &id).await;
        let msgs = messages(&app, &id).await;
        assert_eq!(msgs.last().unwrap().status, MessageStatus::Error);
        let err = msgs.last().unwrap().error.clone().unwrap_or_default();
        assert!(err.contains("is not installed"), "{err}");
    }
    // A message longer than the limit.
    let long = "x".repeat(200_001);
    let r = app.post_multipart("/chat/new", &[("model_ref", MODEL), ("text", &long)], &[]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    // A long but acceptable prompt is sent in full.
    let long_ok = "word ".repeat(30_000);
    let id = new_chat(&app, &long_ok).await;
    wait_idle(&app, &id).await;
    assert_eq!(messages(&app, &id).await[0].content.len(), long_ok.len());
}

#[tokio::test]
async fn attachments_the_model_cannot_handle_are_refused() {
    let (app, _ollama) = setup(&["completion"]).await;
    let png = tiny_png();
    let r = app
        .post_multipart(
            "/chat/new",
            &[("model_ref", MODEL), ("text", "What is this?")],
            &[("files", "dot.png", "image/png", png.clone())],
        )
        .await;
    let (kind, msg) = r.flash().unwrap();
    assert_eq!(kind, "error");
    assert!(msg.contains("can't see images"), "{msg}");
    // Back to the new-chat form with the model kept, and no empty chat left behind.
    assert!(r.location().starts_with("/chat?new=1&model="), "{}", r.location());
    let convs = app.state.db.call(|c| convdb::list_conversations(c, 10)).await.unwrap();
    assert!(convs.is_empty(), "an empty chat was left behind");
    let form = app.get(&r.location()).await;
    assert!(form.body.contains("What is this?"), "the typed text was lost");
    // Nothing is left behind in the uploads folder.
    let leftover = std::fs::read_dir(&app.state.paths.uploads_dir)
        .unwrap()
        .flatten()
        .flat_map(|d| std::fs::read_dir(d.path()).into_iter().flatten().flatten())
        .count();
    assert_eq!(leftover, 0, "rejected upload was kept");
    // Files that are not media are refused whatever their name says.
    let r = app
        .post_multipart(
            "/chat/new",
            &[("model_ref", MODEL), ("text", "hi")],
            &[("files", "notes.png", "image/png", b"just text".to_vec())],
        )
        .await;
    assert_eq!(r.flash().unwrap().0, "error");
    // Audio needs a Whisper model.
    let r = app
        .post_multipart(
            "/chat/new",
            &[("model_ref", MODEL), ("text", "hi")],
            &[("files", "a.wav", "audio/wav", tiny_wav())],
        )
        .await;
    let (kind, msg) = r.flash().unwrap();
    assert_eq!(kind, "error");
    assert!(msg.contains("Whisper"), "{msg}");
}

#[tokio::test]
async fn images_reach_a_vision_model() {
    let (app, ollama) = setup(&["completion", "vision"]).await;
    ollama.set_reply(&["A red dot."], Duration::ZERO);
    let page = app.get("/chat?new=1").await;
    assert!(page.body.contains("sees images"));
    let r = app
        .post_multipart(
            "/chat/new",
            &[("model_ref", MODEL), ("text", "What is this?")],
            &[("files", "dot.png", "image/png", tiny_png())],
        )
        .await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    let id = r.location().trim_start_matches("/chat/").split('#').next().unwrap().to_string();
    wait_idle(&app, &id).await;
    let sent = ollama.last_chat();
    let user = sent["messages"].as_array().unwrap().iter().find(|m| m["role"] == "user").unwrap().clone();
    let images = user["images"].as_array().expect("images sent");
    assert_eq!(images.len(), 1);
    assert!(!images[0].as_str().unwrap().is_empty());
    // The picture is shown in the conversation and served from the uploads folder.
    let page = app.get(&format!("/chat/{id}")).await;
    let src = page.body.split("<img src=\"").nth(1).unwrap().split('"').next().unwrap().to_string();
    let img = app.get(&src).await;
    assert_eq!(img.status, StatusCode::OK);
    assert!(img.headers.get("content-type").unwrap().to_str().unwrap().starts_with("image/"));
}

#[tokio::test]
async fn ollama_going_away_is_reported_not_hidden() {
    let ollama = FakeOllama::start(&[("qwen3:0.6b", &["completion"])]).await;
    let mut up = Upstreams::offline();
    up.ollama = ollama.url.clone();
    let app = TestApp::with(up);
    let id = new_chat(&app, "hello").await;
    wait_idle(&app, &id).await;
    // Point the app at a dead address (Ollama quit).
    app.state.update_settings(|s| s.ollama.base_url = closed_url()).unwrap();
    app.post_multipart(&format!("/chat/{id}/send"), &[("text", "still there?")], &[]).await;
    wait_idle(&app, &id).await;
    let msgs = messages(&app, &id).await;
    let last = msgs.last().unwrap();
    assert_eq!(last.status, MessageStatus::Error);
    assert!(last.error.as_deref().unwrap_or("").to_lowercase().contains("ollama"), "{:?}", last.error);
    let page = app.get(&format!("/chat/{id}")).await;
    assert!(page.body.contains("msg-error"));
}

#[tokio::test]
async fn shutdown_during_an_answer_saves_what_was_written() {
    let (app, ollama) = setup(&["completion"]).await;
    let words: Vec<String> = (0..300).map(|i| format!("w{i} ")).collect();
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    ollama.set_reply(&refs, Duration::from_millis(25));
    let id = new_chat(&app, "Long answer please").await;
    wait_until("some text", 20, || async {
        app.state.chat.get(&id).map(|g| !g.snapshot().text.is_empty()).unwrap_or(false)
    })
    .await;
    let app = app.restart().await;
    let msgs = messages(&app, &id).await;
    let answer = msgs.last().unwrap();
    assert_ne!(answer.status, MessageStatus::Generating, "left in a streaming state");
    assert!(!answer.content.is_empty(), "the partial answer was lost");
}

#[tokio::test]
async fn openai_compatible_and_native_apis() {
    let (app, ollama) = setup(&["completion"]).await;
    ollama.set_reply(&["Hi", " API"], Duration::ZERO);
    let models = app.get("/v1/models").await.json();
    assert!(models["data"].as_array().unwrap().iter().any(|m| m["id"] == MODEL), "{models}");
    let r = app
        .post_json(
            "/v1/chat/completions",
            json!({ "model": MODEL, "messages": [{ "role": "user", "content": "hello" }] }),
        )
        .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    let v = r.json();
    assert_eq!(v["choices"][0]["message"]["content"], "Hi API");
    assert_eq!(v["object"], "chat.completion");
    // Streaming uses server-sent events ending with [DONE].
    let r = app
        .post_json(
            "/v1/chat/completions",
            json!({ "model": MODEL, "stream": true, "messages": [{ "role": "user", "content": "hello" }] }),
        )
        .await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.headers.get("content-type").unwrap().to_str().unwrap().starts_with("text/event-stream"));
    assert!(r.body.contains("\"content\":\"Hi\""), "{}", r.body);
    assert!(r.body.trim_end().ends_with("data: [DONE]"));
    // Native API with NDJSON streaming.
    let r = app
        .post_json(
            "/api/v1/chat",
            json!({ "model": MODEL, "stream": true, "messages": [{ "role": "user", "content": "hello" }] }),
        )
        .await;
    assert_eq!(r.status, StatusCode::OK);
    let lines: Vec<serde_json::Value> = r.body.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.last().unwrap()["done"], true);
    // Bad requests get JSON errors.
    let r = app.post_json("/v1/chat/completions", json!({ "model": MODEL, "messages": [] })).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    assert!(r.json()["error"]["message"].is_string());
    let r = app
        .post_json("/v1/chat/completions", json!({ "model": MODEL, "messages": [{ "role": "user", "content": [{ "type": "image_url", "image_url": { "url": "x" } }] }] }))
        .await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST, "unsupported content must be refused, not ignored");
    let r = app
        .post_json(
            "/v1/chat/completions",
            json!({ "model": "../../bad", "messages": [{ "role": "user", "content": "x" }] }),
        )
        .await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn library_actions_for_ollama_models() {
    let (app, ollama) = setup(&["completion"]).await;
    let r = app.post_form("/library/action", &[("model_ref", MODEL), ("action", "load")]).await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    assert_eq!(ollama.state.running.lock().unwrap().as_slice(), ["qwen3:0.6b"]);
    let sys = app.get("/system").await;
    assert!(sys.body.contains("qwen3:0.6b"));
    let r = app.post_form("/system/unload", &[("model_ref", MODEL)]).await;
    assert_eq!(r.flash().unwrap().0, "ok");
    assert!(ollama.state.running.lock().unwrap().is_empty());
    let r = app.post_form("/library/action", &[("model_ref", MODEL), ("action", "delete")]).await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    assert!(ollama.state.models.lock().unwrap().is_empty());
}

/// A 1×1 red PNG.
fn tiny_png() -> Vec<u8> {
    let mut img = image::RgbImage::new(1, 1);
    img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// 0.1 s of silence as 16 kHz mono 16-bit WAV.
fn tiny_wav() -> Vec<u8> {
    let samples = 1600u32;
    let mut v = Vec::new();
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + samples * 2).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&16000u32.to_le_bytes());
    v.extend_from_slice(&32000u32.to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&(samples * 2).to_le_bytes());
    v.extend(std::iter::repeat_n(0u8, (samples * 2) as usize));
    v
}
