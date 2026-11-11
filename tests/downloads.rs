//! Download manager tests against a fake Hugging Face server: complete downloads with checksum
//! verification, resuming after a dropped connection, corrupted files, missing files, gated and
//! private repositories, pause/resume/cancel, disk-space checks, duplicate detection,
//! concurrency, restart recovery and Ollama pulls.

mod common;

use std::time::Duration;

use axum::http::StatusCode;
use common::*;
use local_ai_studio::db::downloads::DownloadState;
use serde_json::json;

fn gguf_repo(id: &str, file: &str, data: Vec<u8>) -> FakeRepo {
    let mut r = FakeRepo::new(id);
    r.files = vec![FakeFile::small("README.md", b"# Test model\n"), FakeFile::lfs(file, data)];
    r.extra = json!({ "tags": ["gguf"], "pipeline_tag": "text-generation", "gguf": { "total": 1_500_000_000u64, "architecture": "qwen2", "context_length": 32768 } });
    r
}

async fn app_with_hf(repos: Vec<FakeRepo>) -> (TestApp, FakeHf) {
    let hf = FakeHf::start(repos).await;
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    (TestApp::with(up), hf)
}

async fn job_state(app: &TestApp, id: &str) -> (DownloadState, Option<String>) {
    let jobs = app.state.downloads.list().await.unwrap();
    let j = jobs.iter().find(|j| j.row.id == id).expect("job exists");
    (j.row.state, j.row.error.clone())
}

async fn wait_state(app: &TestApp, id: &str, want: DownloadState) -> Option<String> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let (s, err) = job_state(app, id).await;
        if s == want {
            return err;
        }
        if s.is_finished() && s != want {
            panic!("download ended as {s:?} (wanted {want:?}): {err:?}");
        }
        if tokio::time::Instant::now() > deadline {
            panic!("timed out waiting for {want:?}; state is {s:?}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_finished(app: &TestApp, id: &str) -> (DownloadState, Option<String>) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let (s, err) = job_state(app, id).await;
        if s.is_finished() {
            return (s, err);
        }
        if tokio::time::Instant::now() > deadline {
            panic!("timed out; state is {s:?}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Starts a download through the web form, as a person would.
async fn start_download(app: &TestApp, repo: &str, variant: &str) -> String {
    let r = app.post_form(&format!("/discover/hf/{repo}/download"), &[("variant", variant)]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    assert_eq!(r.flash().map(|f| f.0).as_deref(), Some("ok"), "{:?}", r.flash());
    let jobs = app.state.downloads.list().await.unwrap();
    jobs.iter().find(|j| j.row.repo_id == repo).expect("job created").row.id.clone()
}

#[tokio::test]
async fn gguf_download_completes_verifies_and_registers_the_model() {
    let data = bytes_of(3 * 1024 * 1024 + 123, 1);
    let (app, _hf) = app_with_hf(vec![gguf_repo("acme/tiny-GGUF", "tiny-Q4_K_M.gguf", data.clone())]).await;
    let id = start_download(&app, "acme/tiny-GGUF", "tiny-Q4_K_M.gguf").await;
    wait_state(&app, &id, DownloadState::Completed).await;
    let path = app.state.downloads.repo_dir("acme/tiny-GGUF").join("tiny-Q4_K_M.gguf");
    assert_eq!(std::fs::read(&path).unwrap(), data, "downloaded bytes differ");
    // The partial folder is cleaned up.
    let partial = app.state.settings().partial_dir(&app.state.paths).join(&id);
    assert!(!partial.exists(), "partial files left behind");
    // A library entry exists (the bytes are not a real GGUF, so it is marked as not runnable).
    let models = app.state.db.call(local_ai_studio::db::models::list).await.unwrap();
    let m = models.iter().find(|m| m.repo_id.as_deref() == Some("acme/tiny-GGUF")).expect("model registered");
    assert!(!m.analysis.runnable, "garbage bytes must not be considered runnable");
    assert!(!m.analysis.problems.is_empty());
    // Pages show it.
    let lib = app.get("/library").await;
    assert!(lib.body.contains("tiny"), "{}", lib.body);
    let dl = app.get("/downloads").await;
    assert!(dl.body.contains("Done"));
}

#[tokio::test]
async fn download_resumes_after_the_connection_drops() {
    let data = bytes_of(2 * 1024 * 1024, 2);
    let (app, hf) = app_with_hf(vec![gguf_repo("acme/flaky-GGUF", "flaky-Q4_K_M.gguf", data.clone())]).await;
    hf.state.cut_after.lock().unwrap().insert("flaky-Q4_K_M.gguf".into(), 700_000);
    let id = start_download(&app, "acme/flaky-GGUF", "flaky-Q4_K_M.gguf").await;
    wait_state(&app, &id, DownloadState::Completed).await;
    let path = app.state.downloads.repo_dir("acme/flaky-GGUF").join("flaky-Q4_K_M.gguf");
    assert_eq!(std::fs::read(&path).unwrap(), data);
    let reqs = hf.requests_for("flaky-Q4_K_M.gguf");
    assert!(reqs.len() >= 2, "expected a retry: {reqs:?}");
    assert!(
        reqs.iter().skip(1).any(|r| r.contains("range=bytes=") && !r.contains("range=bytes=0-")),
        "the retry did not resume with a Range request: {reqs:?}"
    );
}

#[tokio::test]
async fn corrupted_files_are_rejected() {
    let mut repo = gguf_repo("acme/corrupt-GGUF", "corrupt-Q4_K_M.gguf", bytes_of(512 * 1024, 3));
    repo.files[1].wrong_checksum = true;
    let (app, _hf) = app_with_hf(vec![repo]).await;
    let id = start_download(&app, "acme/corrupt-GGUF", "corrupt-Q4_K_M.gguf").await;
    let (state, err) = wait_finished(&app, &id).await;
    assert_eq!(state, DownloadState::Failed);
    let err = err.unwrap_or_default().to_lowercase();
    assert!(err.contains("checksum") || err.contains("corrupt") || err.contains("verif"), "{err}");
    let path = app.state.downloads.repo_dir("acme/corrupt-GGUF").join("corrupt-Q4_K_M.gguf");
    assert!(!path.exists(), "a file that failed verification was kept");
}

#[tokio::test]
async fn missing_files_fail_with_a_clear_message() {
    let mut repo = gguf_repo("acme/missing-GGUF", "missing-Q4_K_M.gguf", bytes_of(1000, 4));
    // Listed in the repository info but not downloadable.
    let (app, hf) = app_with_hf(vec![repo.clone()]).await;
    repo.files.retain(|f| f.name != "missing-Q4_K_M.gguf");
    let id = start_download(&app, "acme/missing-GGUF", "missing-Q4_K_M.gguf").await;
    hf.state.repos.lock().unwrap().insert(repo.id.clone(), repo);
    let (state, err) = wait_finished(&app, &id).await;
    assert_eq!(state, DownloadState::Failed);
    assert!(!err.unwrap_or_default().is_empty());
}

#[tokio::test]
async fn invalid_and_unknown_repositories_are_reported() {
    let (app, _hf) = app_with_hf(vec![]).await;
    let r = app.get("/discover/hf/nobody/nothing-here").await;
    assert_eq!(r.status, StatusCode::NOT_FOUND, "{}", r.body);
    let r = app.get("/discover/hf/bad%20name/x%3Cy").await;
    assert!(r.status.is_client_error());
    let r = app.post_form("/discover/hf/nobody/nothing-here/download", &[("variant", "x")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
}

#[tokio::test]
async fn gated_and_private_repositories_need_a_token() {
    let mut gated = gguf_repo("bigco/gated-GGUF", "gated-Q4_K_M.gguf", bytes_of(200_000, 5));
    gated.gated = true;
    let mut private = gguf_repo("someone/private-GGUF", "private-Q4_K_M.gguf", bytes_of(1000, 6));
    private.private = true;
    let (app, _hf) = app_with_hf(vec![gated, private]).await;
    // Gated without a token: explained up front, nothing downloaded.
    let page = app.get("/discover/hf/bigco/gated-GGUF").await;
    assert!(page.body.contains("gated"), "{}", page.body);
    let r = app.post_form("/discover/hf/bigco/gated-GGUF/download", &[("variant", "gated-Q4_K_M.gguf")]).await;
    let (kind, msg) = r.flash().unwrap();
    assert_eq!(kind, "error");
    assert!(msg.contains("token"), "{msg}");
    assert!(app.state.downloads.list().await.unwrap().is_empty());
    // Private repositories look like missing ones without authentication.
    let r = app.get("/discover/hf/someone/private-GGUF").await;
    assert!(r.status.is_client_error(), "{}", r.status);
    // With a valid token both work.
    app.state.secrets.set_hf_token(VALID_TOKEN).await.unwrap();
    let id = start_download(&app, "bigco/gated-GGUF", "gated-Q4_K_M.gguf").await;
    wait_state(&app, &id, DownloadState::Completed).await;
    let r = app.get("/discover/hf/someone/private-GGUF").await;
    assert_eq!(r.status, StatusCode::OK);
}

#[tokio::test]
async fn insufficient_disk_space_is_detected_before_downloading() {
    let mut repo = gguf_repo("acme/huge-GGUF", "huge-Q4_K_M.gguf", bytes_of(1000, 7));
    repo.files[1].declared_size = Some(900_000_000_000_000); // 900 TB
    let (app, hf) = app_with_hf(vec![repo]).await;
    let id = start_download(&app, "acme/huge-GGUF", "huge-Q4_K_M.gguf").await;
    let (state, err) = wait_finished(&app, &id).await;
    assert_eq!(state, DownloadState::Failed);
    assert!(err.unwrap_or_default().contains("disk space"));
    assert!(hf.requests_for("huge-Q4_K_M.gguf").is_empty(), "it started downloading anyway");
}

#[tokio::test]
async fn pause_resume_and_cancel() {
    let data = bytes_of(4 * 1024 * 1024, 8);
    let (app, hf) = app_with_hf(vec![
        gguf_repo("acme/slow-GGUF", "slow-Q4_K_M.gguf", data.clone()),
        // Different bytes: identical files would be reused instantly instead of downloaded.
        gguf_repo("acme/cancel-GGUF", "cancel-Q4_K_M.gguf", bytes_of(4 * 1024 * 1024, 18)),
    ])
    .await;
    *hf.state.chunk_delay.lock().unwrap() = Some(Duration::from_millis(40));
    let id = start_download(&app, "acme/slow-GGUF", "slow-Q4_K_M.gguf").await;
    wait_until("some progress", 30, || async {
        app.state.downloads.list().await.unwrap().iter().any(|j| j.row.id == id && j.done_bytes > 0)
    })
    .await;
    let r = app.post_form(&format!("/downloads/{id}/pause"), &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    wait_state(&app, &id, DownloadState::Paused).await;
    let part = app.state.settings().partial_dir(&app.state.paths).join(&id).join("slow-Q4_K_M.gguf.part");
    assert!(part.exists(), "the partial file was not kept while paused");
    let kept = std::fs::metadata(&part).unwrap().len();
    assert!(kept > 0);
    *hf.state.chunk_delay.lock().unwrap() = None;
    let r = app.post_form(&format!("/downloads/{id}/resume"), &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    wait_state(&app, &id, DownloadState::Completed).await;
    let path = app.state.downloads.repo_dir("acme/slow-GGUF").join("slow-Q4_K_M.gguf");
    assert_eq!(std::fs::read(&path).unwrap(), data);

    // Cancel removes partial data.
    *hf.state.chunk_delay.lock().unwrap() = Some(Duration::from_millis(40));
    let id2 = start_download(&app, "acme/cancel-GGUF", "cancel-Q4_K_M.gguf").await;
    wait_until("some progress", 30, || async {
        app.state.downloads.list().await.unwrap().iter().any(|j| j.row.id == id2 && j.done_bytes > 0)
    })
    .await;
    let r = app.post_form(&format!("/downloads/{id2}/cancel"), &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    wait_state(&app, &id2, DownloadState::Cancelled).await;
    let partial_dir = app.state.settings().partial_dir(&app.state.paths).join(&id2);
    wait_until("partial cleanup", 10, || async { !partial_dir.exists() }).await;
    // Finished jobs can be cleared from the list.
    let r = app.post_form("/downloads/clear", &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    assert!(app.state.downloads.list().await.unwrap().is_empty());
}

#[tokio::test]
async fn duplicates_are_detected() {
    let data = bytes_of(300_000, 9);
    let (app, _hf) = app_with_hf(vec![gguf_repo("acme/dup-GGUF", "dup-Q4_K_M.gguf", data)]).await;
    let id = start_download(&app, "acme/dup-GGUF", "dup-Q4_K_M.gguf").await;
    wait_state(&app, &id, DownloadState::Completed).await;
    let r = app.post_form("/discover/hf/acme/dup-GGUF/download", &[("variant", "dup-Q4_K_M.gguf")]).await;
    let (kind, msg) = r.flash().unwrap();
    assert_eq!(kind, "error");
    assert!(msg.contains("already"), "{msg}");
}

#[tokio::test]
async fn identical_files_in_different_repositories_are_stored_once() {
    let data = bytes_of(400_000, 10);
    let (app, _hf) = app_with_hf(vec![
        gguf_repo("acme/one-GGUF", "same-Q4_K_M.gguf", data.clone()),
        gguf_repo("mirror/two-GGUF", "same-Q4_K_M.gguf", data.clone()),
    ])
    .await;
    let a = start_download(&app, "acme/one-GGUF", "same-Q4_K_M.gguf").await;
    wait_state(&app, &a, DownloadState::Completed).await;
    let b = start_download(&app, "mirror/two-GGUF", "same-Q4_K_M.gguf").await;
    wait_state(&app, &b, DownloadState::Completed).await;
    let p1 = app.state.downloads.repo_dir("acme/one-GGUF").join("same-Q4_K_M.gguf");
    let p2 = app.state.downloads.repo_dir("mirror/two-GGUF").join("same-Q4_K_M.gguf");
    assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(std::fs::metadata(&p1).unwrap().ino(), std::fs::metadata(&p2).unwrap().ino(), "not hard-linked");
    }
}

#[tokio::test]
async fn several_downloads_run_at_once() {
    let repos: Vec<FakeRepo> = (0..4)
        .map(|i| {
            gguf_repo(
                &format!("acme/multi{i}-GGUF"),
                &format!("multi{i}-Q4_K_M.gguf"),
                bytes_of(600_000 + i * 1000, 20 + i as u8),
            )
        })
        .collect();
    let (app, _hf) = app_with_hf(repos).await;
    let mut ids = Vec::new();
    for i in 0..4 {
        ids.push(start_download(&app, &format!("acme/multi{i}-GGUF"), &format!("multi{i}-Q4_K_M.gguf")).await);
    }
    for id in &ids {
        wait_state(&app, id, DownloadState::Completed).await;
    }
}

#[tokio::test]
async fn interrupted_downloads_are_paused_after_a_restart_and_can_resume() {
    let data = bytes_of(3 * 1024 * 1024, 30);
    let (app, hf) = app_with_hf(vec![gguf_repo("acme/restart-GGUF", "restart-Q4_K_M.gguf", data.clone())]).await;
    *hf.state.chunk_delay.lock().unwrap() = Some(Duration::from_millis(40));
    let id = start_download(&app, "acme/restart-GGUF", "restart-Q4_K_M.gguf").await;
    wait_until("some progress", 30, || async {
        app.state.downloads.list().await.unwrap().iter().any(|j| j.row.id == id && j.done_bytes > 0)
    })
    .await;
    // Shutting down pauses running downloads; a new instance on the same data shows them as
    // paused and resumes from the partial file.
    let app = app.restart().await;
    let (state, _) = job_state(&app, &id).await;
    assert_eq!(state, DownloadState::Paused);
    *hf.state.chunk_delay.lock().unwrap() = None;
    app.state.downloads.resume(&id).await.unwrap();
    wait_state(&app, &id, DownloadState::Completed).await;
    let path = app.state.downloads.repo_dir("acme/restart-GGUF").join("restart-Q4_K_M.gguf");
    assert_eq!(std::fs::read(&path).unwrap(), data);
}

#[tokio::test]
async fn hub_outage_shows_cached_information_and_blocks_downloads() {
    let (app, hf) = app_with_hf(vec![gguf_repo("acme/cached-GGUF", "cached-Q4_K_M.gguf", bytes_of(1000, 40))]).await;
    // First visit caches the model information.
    let r = app.get("/discover/hf/acme/cached-GGUF").await;
    assert_eq!(r.status, StatusCode::OK);
    *hf.state.down.lock().unwrap() = true;
    let r = app.get("/discover/hf/acme/cached-GGUF").await;
    assert_eq!(r.status, StatusCode::OK, "cached page not shown: {}", r.status);
    assert!(r.body.contains("unreachable"), "no outage notice");
    let r = app.post_form("/discover/hf/acme/cached-GGUF/download", &[("variant", "cached-Q4_K_M.gguf")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
}

#[tokio::test]
async fn ollama_pull_reports_progress_and_completes() {
    let ollama = FakeOllama::start(&[]).await;
    let mut up = Upstreams::offline();
    up.ollama = ollama.url.clone();
    let app = TestApp::with(up);
    let r = app.post_form("/discover/ollama/pull", &[("name", "qwen3:0.6b"), ("back", "/discover/ollama/qwen3")]).await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    let jobs = app.state.downloads.list().await.unwrap();
    let id = jobs[0].row.id.clone();
    wait_state(&app, &id, DownloadState::Completed).await;
    assert_eq!(ollama.state.pulls.lock().unwrap().as_slice(), ["qwen3:0.6b"]);
    // The model now appears in the library and as a chat choice.
    let lib = app.get("/library").await;
    assert!(lib.body.contains("qwen3:0.6b"));
    // Unknown models fail with Ollama's explanation.
    *ollama.state.pull_error.lock().unwrap() = Some("pull model manifest: file does not exist".into());
    app.post_form("/discover/ollama/pull", &[("name", "nope:1b")]).await;
    let jobs = app.state.downloads.list().await.unwrap();
    let bad = jobs.iter().find(|j| j.row.repo_id == "nope:1b").unwrap().row.id.clone();
    let (state, err) = wait_finished(&app, &bad).await;
    assert_eq!(state, DownloadState::Failed);
    assert!(err.unwrap_or_default().contains("not found"));
    // Invalid names never reach Ollama.
    let r = app.post_form("/discover/ollama/pull", &[("name", "../../etc/passwd")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
}

#[tokio::test]
async fn ollama_not_running_is_explained() {
    let app = TestApp::new();
    let r = app.post_form("/discover/ollama/pull", &[("name", "qwen3:0.6b")]).await;
    let (kind, msg) = r.flash().unwrap();
    assert_eq!(kind, "error");
    assert!(msg.to_lowercase().contains("ollama"), "{msg}");
    let page = app.get("/settings").await;
    assert!(page.body.contains("Not running"));
    assert!(page.body.contains("How to install Ollama"));
}
