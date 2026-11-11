//! The real server: binding, port conflicts, detecting an already running copy, serving over
//! TCP, graceful shutdown, and the command-line program itself.

mod common;

use std::time::Duration;

use common::*;
use local_ai_studio::web::{self, BindOutcome};

#[tokio::test]
async fn an_explicit_busy_port_is_an_error() {
    let blocker = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = blocker.local_addr().unwrap().port();
    let err = match web::bind("127.0.0.1", port, true).await {
        Err(e) => e.to_string(),
        Ok(_) => panic!("bound a busy port"),
    };
    assert!(err.contains("already used"), "{err}");
}

#[tokio::test]
async fn a_busy_default_port_falls_back_to_the_next_free_one() {
    let blocker = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = blocker.local_addr().unwrap().port();
    match web::bind("127.0.0.1", port, false).await.unwrap() {
        BindOutcome::Bound(l) => assert_ne!(l.local_addr().unwrap().port(), port),
        BindOutcome::AlreadyRunning(_) => panic!("a foreign program was mistaken for Local AI Studio"),
    }
}

#[tokio::test]
async fn a_running_copy_is_detected_and_reused() {
    let app = TestApp::new();
    let listener = match web::bind("127.0.0.1", 0, true).await.unwrap() {
        BindOutcome::Bound(l) => l,
        BindOutcome::AlreadyRunning(_) => unreachable!(),
    };
    let port = listener.local_addr().unwrap().port();
    let state = app.state.clone();
    let server = tokio::spawn(web::serve(state.clone(), listener));
    // Talk to it over real TCP.
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let health: serde_json::Value =
        client.get(format!("http://127.0.0.1:{port}/api/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(health["app"], "Local AI Studio");
    let page = client.get(format!("http://127.0.0.1:{port}/")).send().await.unwrap();
    assert_eq!(page.status(), 200);
    // A second start on the same port finds the first one instead of failing.
    match web::bind("127.0.0.1", port, true).await.unwrap() {
        BindOutcome::AlreadyRunning(url) => assert!(url.ends_with(&format!(":{port}")), "{url}"),
        BindOutcome::Bound(_) => panic!("bound the same port twice"),
    }
    // Graceful shutdown from the Settings page.
    let r = client
        .post(format!("http://127.0.0.1:{port}/settings/shutdown"))
        .header("origin", format!("http://127.0.0.1:{port}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let done = tokio::time::timeout(Duration::from_secs(20), server).await;
    assert!(done.is_ok(), "the server did not stop");
    assert!(client.get(format!("http://127.0.0.1:{port}/api/health")).send().await.is_err());
}

#[test]
fn invalid_listen_addresses_are_refused() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    assert!(rt.block_on(web::bind("not-an-ip", 8765, true)).is_err());
}

#[test]
fn the_program_prints_system_information() {
    let dir = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_local-ai-studio"))
        .args(["--system-info", "--no-keychain", "--data-dir"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("CPU") || text.contains("cpu"), "{text}");
}

#[test]
fn the_program_refuses_network_addresses_without_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_local-ai-studio"))
        .args(["--host", "0.0.0.0", "--port", "0", "--no-browser", "--no-keychain", "--data-dir"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--allow-network"), "{err}");
}

#[test]
fn a_broken_config_file_is_reported_not_overwritten() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.toml"), "this is = = not toml").unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_local-ai-studio"))
        .args(["--no-browser", "--no-keychain", "--port", "0", "--data-dir"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("config.toml"), "{err}");
    assert_eq!(std::fs::read_to_string(dir.path().join("config.toml")).unwrap(), "this is = = not toml");
}
