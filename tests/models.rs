//! Model compatibility checks on real files (crafted GGUF headers, broken files, model folders)
//! and the library flows that use them: import, inspect, rename, re-check, delete. Also checks
//! that chatting with a model that can't run fails cleanly instead of crashing.

mod common;

use axum::http::StatusCode;
use common::*;
use local_ai_studio::library::analyze_gguf;

fn llama_like(path: &std::path::Path, arch: &str, tokenizer: Option<&str>) {
    let mut kv = vec![
        ("general.architecture", GgufValue::Str(arch.into())),
        ("general.name", GgufValue::Str("Tiny test model".into())),
        ("general.file_type", GgufValue::U32(15)),
    ];
    kv.push((Box::leak(format!("{arch}.context_length").into_boxed_str()), GgufValue::U32(2048)));
    kv.push((Box::leak(format!("{arch}.block_count").into_boxed_str()), GgufValue::U32(2)));
    kv.push((Box::leak(format!("{arch}.embedding_length").into_boxed_str()), GgufValue::U32(64)));
    kv.push((Box::leak(format!("{arch}.attention.head_count").into_boxed_str()), GgufValue::U32(4)));
    if let Some(t) = tokenizer {
        kv.push(("tokenizer.ggml.model", GgufValue::Str(t.into())));
        kv.push((
            "tokenizer.ggml.tokens",
            GgufValue::StrArray(vec!["<unk>".into(), "<s>".into(), "</s>".into(), "a".into()]),
        ));
        kv.push(("tokenizer.ggml.scores", GgufValue::F32Array(vec![0.0, 0.0, 0.0, -1.0])));
    }
    write_gguf(path, &kv);
}

#[test]
fn gguf_headers_are_understood() {
    let dir = tempfile::tempdir().unwrap();
    let ok = dir.path().join("tiny-Q4_K_M.gguf");
    llama_like(&ok, "llama", Some("llama"));
    let a = analyze_gguf(&ok);
    assert_eq!(a.architecture.as_deref(), Some("llama"));
    assert_eq!(a.context_length, Some(2048));
    assert_eq!(a.n_layers, Some(2));
    assert_eq!(a.head_dim, Some(16));
    assert_eq!(a.quantization.as_deref(), Some("Q4_K_M"));
    assert!(a.runnable, "{:?}", a.problems);
    assert!(!a.notes.is_empty(), "missing chat template should be noted");
}

#[test]
fn unsupported_architecture_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("mamba.gguf");
    llama_like(&p, "mamba", Some("gpt2"));
    let a = analyze_gguf(&p);
    assert!(!a.runnable);
    assert!(a.problems.iter().any(|p| p.contains("mamba") && p.contains("Ollama")), "{:?}", a.problems);
}

#[test]
fn missing_tokenizer_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("no-tokenizer.gguf");
    llama_like(&p, "qwen2", None);
    let a = analyze_gguf(&p);
    assert!(!a.runnable);
    assert!(a.problems.iter().any(|p| p.contains("tokenizer")), "{:?}", a.problems);
}

#[test]
fn corrupted_and_truncated_files_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    let junk = dir.path().join("junk.gguf");
    std::fs::write(&junk, bytes_of(4096, 1)).unwrap();
    let a = analyze_gguf(&junk);
    assert!(!a.runnable);
    assert!(!a.problems.is_empty());
    let good = dir.path().join("good.gguf");
    llama_like(&good, "llama", Some("llama"));
    let bytes = std::fs::read(&good).unwrap();
    let truncated = dir.path().join("truncated.gguf");
    std::fs::write(&truncated, &bytes[..bytes.len() / 3]).unwrap();
    let a = analyze_gguf(&truncated);
    assert!(!a.runnable, "a truncated file was accepted");
    let empty = dir.path().join("empty.gguf");
    std::fs::write(&empty, b"").unwrap();
    assert!(!analyze_gguf(&empty).runnable);
}

#[test]
fn split_files_point_to_ollama() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("big-Q4_K_M-00001-of-00003.gguf");
    llama_like(&p, "llama", Some("llama"));
    let a = analyze_gguf(&p);
    assert!(!a.runnable);
    assert!(a.problems.iter().any(|p| p.contains("split")), "{:?}", a.problems);
}

#[tokio::test]
async fn importing_inspecting_renaming_and_deleting_a_gguf() {
    let app = TestApp::new();
    let file = app.dir.path().join("my-tiny-Q8_0.gguf");
    llama_like(&file, "llama", Some("llama"));
    // Validation of the path.
    let r = app.post_form("/library/import", &[("path", "relative.gguf")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    let r = app.post_form("/library/import", &[("path", "/definitely/not/here.gguf")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    let txt = app.dir.path().join("notes.txt");
    std::fs::write(&txt, "x").unwrap();
    let r = app.post_form("/library/import", &[("path", txt.to_str().unwrap())]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    // A real file is added in place.
    let r = app.post_form("/library/import", &[("path", file.to_str().unwrap())]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    let (kind, msg) = r.flash().unwrap();
    assert_eq!(kind, "ok", "{msg}");
    let detail = r.location();
    let page = app.get(&detail).await;
    assert_eq!(page.status, StatusCode::OK);
    assert!(page.body.contains("Memory needed"));
    assert!(page.body.contains("never a guarantee"), "the estimate must not promise anything");
    let model_ref = detail.trim_start_matches("/library/model/").to_string();
    // It is offered for chat.
    let chat = app.get("/chat?new=1").await;
    assert!(chat.body.contains(&model_ref));
    // Rename, re-check, delete (the imported file itself stays unless asked).
    let r = app
        .post_form("/library/action", &[("model_ref", &model_ref), ("action", "rename"), ("name", "My tiny model")])
        .await;
    assert_eq!(r.flash().unwrap().0, "ok");
    assert!(app.get("/library").await.body.contains("My tiny model"));
    let r = app.post_form("/library/action", &[("model_ref", &model_ref), ("action", "rename"), ("name", "")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    let r = app.post_form("/library/action", &[("model_ref", &model_ref), ("action", "reanalyze")]).await;
    assert_ne!(r.flash().unwrap().0, "error");
    let r = app.post_form("/library/action", &[("model_ref", &model_ref), ("action", "delete")]).await;
    assert_eq!(r.flash().unwrap().0, "ok");
    assert!(file.exists(), "an imported file was deleted without asking");
    assert!(app.get("/library").await.body.contains("Your library is empty"));
    // Importing the same file twice is detected.
    app.post_form("/library/import", &[("path", file.to_str().unwrap())]).await;
    let r = app.post_form("/library/import", &[("path", file.to_str().unwrap())]).await;
    assert_eq!(r.flash().unwrap().0, "error", "duplicate import accepted");
}

#[tokio::test]
async fn a_model_that_fails_to_load_does_not_crash_the_app() {
    let app = TestApp::new();
    // Valid header, but the tensors a llama model needs are missing.
    let file = app.dir.path().join("hollow-Q4_K_M.gguf");
    llama_like(&file, "llama", Some("llama"));
    let r = app.post_form("/library/import", &[("path", file.to_str().unwrap())]).await;
    let model_ref = r.location().trim_start_matches("/library/model/").to_string();
    let r = app.post_multipart("/chat/new", &[("model_ref", &model_ref), ("text", "hello")], &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    let id = r.location().trim_start_matches("/chat/").split('#').next().unwrap().to_string();
    wait_until("load failure", 60, || async { !app.state.chat.is_generating(&id) }).await;
    let page = app.get(&format!("/chat/{id}")).await;
    assert_eq!(page.status, StatusCode::OK);
    assert!(page.body.contains("msg-error"), "the failure is not shown");
    // The app keeps working.
    assert_eq!(app.get("/api/health").await.status, StatusCode::OK);
    let r = app.post_form("/library/action", &[("model_ref", &model_ref), ("action", "load")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
}

#[tokio::test]
async fn system_page_shows_hardware_and_never_promises() {
    let app = TestApp::new();
    let r = app.get("/system").await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.body.contains("Memory (RAM)"));
    assert!(r.body.contains("never a guarantee"));
    let v = app.get("/api/v1/system").await.json();
    assert!(v["total_ram"].as_u64().unwrap() > 0, "{v}");
    assert!(v["logical_cores"].as_u64().unwrap() > 0);
}
