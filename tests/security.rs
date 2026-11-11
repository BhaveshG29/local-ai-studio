//! Security tests: DNS-rebinding and cross-site request protection, network mode access
//! control, path traversal, HTML injection, and secret handling.

mod common;

use std::net::SocketAddr;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode, header};
use common::*;

#[tokio::test]
async fn unexpected_host_names_are_rejected() {
    let app = TestApp::new();
    for host in ["evil.example", "evil.example:8765", "192.168.1.20:8765", "attacker.localhost.evil:80"] {
        let r = app.send(Request::get("/").header(header::HOST, host).body(Body::empty()).unwrap()).await;
        assert_eq!(r.status, StatusCode::MISDIRECTED_REQUEST, "Host {host} was accepted");
    }
    for host in ["127.0.0.1:8765", "localhost:8765", "[::1]:8765"] {
        let r = app.send(Request::get("/").header(header::HOST, host).body(Body::empty()).unwrap()).await;
        assert_eq!(r.status, StatusCode::OK, "Host {host} was rejected");
    }
}

#[tokio::test]
async fn cross_site_posts_are_blocked() {
    let app = TestApp::new();
    let attempts = [
        ("origin", "http://evil.example"),
        ("origin", "null"),
        ("origin", "http://127.0.0.1:9999"),
        ("sec-fetch-site", "cross-site"),
        ("sec-fetch-site", "same-site"),
        ("referer", "http://evil.example/page"),
    ];
    for (h, v) in attempts {
        let r = app
            .send(
                Request::post("/settings/theme")
                    .header(h, v)
                    .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                    .body(Body::from("theme=neon"))
                    .unwrap(),
            )
            .await;
        assert_eq!(r.status, StatusCode::FORBIDDEN, "{h}: {v} was allowed");
    }
    assert_eq!(app.state.settings().ui.theme.slug(), "midnight", "a blocked request changed settings");
    // The same request from our own page works.
    let r = app.post_form("/settings/theme", &[("theme", "neon")]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    assert_eq!(app.state.settings().ui.theme.slug(), "neon");
}

#[tokio::test]
async fn security_headers_are_set() {
    let app = TestApp::new();
    let r = app.get("/").await;
    let h = |n: &str| r.headers.get(n).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    assert!(h("content-security-policy").contains("script-src 'none'"));
    assert!(h("content-security-policy").contains("frame-ancestors 'self'"));
    assert_eq!(h("x-content-type-options"), "nosniff");
    assert_eq!(h("x-frame-options"), "SAMEORIGIN");
    assert_eq!(h("referrer-policy"), "same-origin");
}

#[tokio::test]
async fn path_traversal_is_refused() {
    let app = TestApp::new();
    std::fs::write(app.state.paths.data_dir.join("secret.txt"), "top secret").unwrap();
    for path in [
        "/files/uploads/../secret.txt",
        "/files/uploads/..%2fsecret.txt",
        "/files/uploads/%2e%2e/secret.txt",
        "/files/outputs/..%5csecret.txt",
        "/files/models/x",
        "/assets/../Cargo.toml",
        "/assets/%2e%2e/Cargo.toml",
    ] {
        let r = app.get(path).await;
        assert!(!r.body.contains("top secret"), "{path} leaked a file");
        assert!(r.status.is_client_error(), "{path} answered {}", r.status);
    }
}

#[tokio::test]
async fn user_text_is_escaped_in_pages() {
    let ollama = FakeOllama::start(&[("qwen3:0.6b", &["completion"])]).await;
    // A raw HTML block followed by a Markdown paragraph.
    ollama.set_reply(
        &["<script>alert('answer')</script>\n\nSome **bold** text and maths $x^2 \\text{<script>alert(2)</script>}$"],
        std::time::Duration::ZERO,
    );
    let mut up = Upstreams::offline();
    up.ollama = ollama.url.clone();
    let app = TestApp::with(up);
    let r = app
        .post_multipart(
            "/chat/new",
            &[("model_ref", "ollama:qwen3:0.6b"), ("text", "<img src=x onerror=alert(1)> hello")],
            &[],
        )
        .await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{}", r.flash_message());
    let chat = r.location();
    let id = chat.trim_start_matches("/chat/").split('#').next().unwrap().to_string();
    wait_until("answer", 20, || async { !app.state.chat.is_generating(&id) }).await;
    let page = app.get(&format!("/chat/{id}")).await;
    assert!(!page.body.contains("<img src=x"), "user HTML was not escaped");
    assert!(page.body.contains("&lt;img src=x onerror=alert(1)&gt;"));
    assert!(!page.body.contains("<script>alert"), "model HTML was not escaped");
    assert!(page.body.contains("<strong>bold</strong>"), "markdown was not rendered");
    assert!(page.body.contains("<math>"), "maths was not rendered");
    assert!(page.body.contains("&lt;script&gt;alert(2)"), "text inside maths was not escaped");
    // Titles are escaped too.
    let r = app.post_form(&format!("/chat/{id}/rename"), &[("title", "<b>Title</b>\"'")]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    let page = app.get(&format!("/chat/{id}")).await;
    assert!(!page.body.contains("<b>Title</b>"));
}

#[tokio::test]
async fn network_mode_requires_the_access_token() {
    let app = TestApp::new();
    // Pretend the server listens on all interfaces (as after an explicit opt-in).
    *app.state.server.addr.write().unwrap() = Some("0.0.0.0:8765".parse().unwrap());
    let lan_client: SocketAddr = "192.168.1.50:51000".parse().unwrap();
    let lan = |path: &str| {
        let mut req = Request::get(path).header(header::HOST, "192.168.1.10:8765").body(Body::empty()).unwrap();
        req.extensions_mut().insert(ConnectInfo(lan_client));
        req
    };
    let r = app.send(lan("/")).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    assert_eq!(r.location(), "/unlock");
    let r = app.send(lan("/api/v1/models")).await;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);
    // Health and the unlock page stay reachable.
    assert_eq!(app.send(lan("/api/health")).await.status, StatusCode::OK);
    assert_eq!(app.send(lan("/unlock")).await.status, StatusCode::OK);
    // A wrong token is refused.
    let mut wrong = Request::post("/unlock")
        .header(header::HOST, "192.168.1.10:8765")
        .header(header::ORIGIN, "http://192.168.1.10:8765")
        .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(Body::from("token=nope"))
        .unwrap();
    wrong.extensions_mut().insert(ConnectInfo(lan_client));
    assert_eq!(app.send(wrong).await.status, StatusCode::UNAUTHORIZED);
    // The right token sets a cookie that grants access.
    let token = app.state.server.access_token.clone();
    let mut right = Request::post("/unlock")
        .header(header::HOST, "192.168.1.10:8765")
        .header(header::ORIGIN, "http://192.168.1.10:8765")
        .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(Body::from(format!("token={token}")))
        .unwrap();
    right.extensions_mut().insert(ConnectInfo(lan_client));
    let r = app.send(right).await;
    let cookie = r
        .headers
        .get(header::SET_COOKIE)
        .expect("access cookie")
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();
    let mut req = lan("/");
    req.headers_mut().insert(header::COOKIE, cookie.parse().unwrap());
    assert_eq!(app.send(req).await.status, StatusCode::OK);
    // Bearer tokens work for the API.
    let mut req = lan("/api/v1/models");
    req.headers_mut().insert(header::AUTHORIZATION, format!("Bearer {token}").parse().unwrap());
    assert_eq!(app.send(req).await.status, StatusCode::OK);
}

#[tokio::test]
async fn remote_clients_are_refused_when_network_mode_is_off() {
    let app = TestApp::new();
    *app.state.server.addr.write().unwrap() = Some("127.0.0.1:8765".parse().unwrap());
    let mut req = Request::get("/").header(header::HOST, "127.0.0.1:8765").body(Body::empty()).unwrap();
    req.extensions_mut().insert(ConnectInfo("10.0.0.7:4000".parse::<SocketAddr>().unwrap()));
    assert_eq!(app.send(req).await.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn hugging_face_token_never_reaches_config_files_or_pages() {
    let hf = FakeHf::start(vec![]).await;
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    let app = TestApp::with(up);
    // Malformed tokens are rejected before anything is stored.
    let r = app.post_form("/settings/hf-token", &[("action", "save"), ("token", "not a token")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    // A token the Hub rejects is not stored.
    let r = app
        .post_form("/settings/hf-token", &[("action", "save"), ("token", "hf_WrongTokenWrongTokenWrongToken00")])
        .await;
    assert_eq!(r.flash().unwrap().0, "error", "{:?}", r.flash());
    assert!(app.state.secrets.hf_token().await.is_none());
    // A valid token is verified and kept (in memory here: tests don't touch the keychain).
    let r = app.post_form("/settings/hf-token", &[("action", "save"), ("token", VALID_TOKEN)]).await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    assert!(r.flash_message().contains("test-user"));
    assert_eq!(app.state.secrets.hf_token().await.as_deref(), Some(VALID_TOKEN));
    let page = app.get("/settings").await;
    assert!(!page.body.contains(VALID_TOKEN), "the settings page shows the full token");
    assert!(!app.config_text().contains(VALID_TOKEN), "the token was written to config.toml");
    for entry in walk(&app.state.paths.data_dir) {
        if let Ok(text) = std::fs::read(&entry) {
            assert!(
                !String::from_utf8_lossy(&text).contains(VALID_TOKEN),
                "the token was written to {}",
                entry.display()
            );
        }
    }
    let r = app.post_form("/settings/hf-token", &[("action", "clear")]).await;
    assert_eq!(r.flash().unwrap().0, "ok");
    assert!(app.state.secrets.hf_token().await.is_none());
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}

#[tokio::test]
async fn oversized_requests_are_refused() {
    let app = TestApp::new();
    let big = "a".repeat(3 * 1024 * 1024);
    let r = app.post_form("/settings/save/generation", &[("system_prompt", &big)]).await;
    assert_eq!(r.status, StatusCode::PAYLOAD_TOO_LARGE);
}
