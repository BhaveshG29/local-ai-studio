//! Frontend tests without JavaScript: every page renders, contains no script, every link
//! leads somewhere, every form posts to a real route, and every button does something
//! (submits a form or opens an existing pop-up). Themes switch and persist.

mod common;

use std::collections::HashSet;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use scraper::{Html, Selector};

const PAGES: &[&str] = &[
    "/",
    "/chat?new=1",
    "/discover",
    "/discover?tab=ollama",
    "/discover?tab=index",
    "/downloads",
    "/library",
    "/create",
    "/transcribe",
    "/system",
    "/settings",
    "/about/licenses",
];

fn sel(s: &str) -> Selector {
    Selector::parse(s).unwrap()
}

#[tokio::test]
async fn every_page_renders_without_scripts() {
    let app = TestApp::new();
    for path in PAGES {
        let r = app.get(path).await;
        assert_eq!(r.status, StatusCode::OK, "{path}: {}", r.body);
        assert!(!r.body.to_lowercase().contains("<script"), "{path} contains a script");
        assert!(r.body.contains("data-theme=\"midnight\""), "{path} has no theme");
        let csp = r.headers.get("content-security-policy").and_then(|v| v.to_str().ok()).unwrap_or("");
        assert!(csp.contains("script-src 'none'"), "{path} CSP: {csp}");
        let doc = Html::parse_document(&r.body);
        assert!(doc.select(&sel("main#main")).next().is_some(), "{path} has no main landmark");
        assert!(doc.select(&sel("h1")).next().is_some(), "{path} has no heading");
    }
}

/// Every internal link must resolve, every form must post to an existing route, and every
/// button must either submit a form or open a pop-up that exists on the page.
#[tokio::test]
async fn no_dead_links_forms_or_buttons() {
    let app = TestApp::new();
    let mut checked_links: HashSet<String> = HashSet::new();
    for path in PAGES {
        assert_page_is_wired(&app, path, &mut checked_links).await;
    }
}

#[tokio::test]
async fn theme_switch_applies_everywhere_and_persists_after_restart() {
    let app = TestApp::new();
    for slug in ["graphite", "neon", "minimal", "warm", "glass", "midnight", "warm"] {
        let r = app.post_form("/settings/theme", &[("theme", slug), ("back", "/library")]).await;
        assert_eq!(r.status, StatusCode::SEE_OTHER);
        assert_eq!(r.location(), "/library");
        let page = app.get("/library").await;
        assert!(page.body.contains(&format!("data-theme=\"{slug}\"")), "theme {slug} not applied");
    }
    assert!(app.config_text().contains("theme = \"warm\""));
    let app = app.restart().await;
    let page = app.get("/").await;
    assert!(page.body.contains("data-theme=\"warm\""), "theme not restored after restart");
}

#[tokio::test]
async fn unknown_theme_is_rejected() {
    let app = TestApp::new();
    let r = app.post_form("/settings/theme", &[("theme", "pink")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    let r = app.post_form("/settings/theme", &[("theme", "neon"), ("back", "https://evil.example/")]).await;
    assert_eq!(r.location(), "/settings", "open redirect through the back parameter");
}

#[tokio::test]
async fn settings_sections_validate_and_persist() {
    let app = TestApp::new();
    let r = app.post_form("/settings/save/generation", &[("temperature", "hot")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    let r = app
        .post_form(
            "/settings/save/generation",
            &[
                ("temperature", "0.3"),
                ("top_p", "0.8"),
                ("top_k", "20"),
                ("max_tokens", "300"),
                ("repeat_penalty", "1.05"),
                ("context_length", "2048"),
                ("seed", "7"),
                ("system_prompt", "Be brief."),
            ],
        )
        .await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    let cfg = app.config_text();
    assert!(cfg.contains("temperature = 0.3"), "{cfg}");
    assert!(cfg.contains("seed = 7"));
    // Network exposure needs the explicit opt-in.
    let r = app.post_form("/settings/save/server", &[("host", "0.0.0.0"), ("port", "8800")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    assert!(!app.config_text().contains("0.0.0.0"));
    // Ports below 1024 are refused.
    let r = app.post_form("/settings/save/server", &[("host", "127.0.0.1"), ("port", "80")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    // Folder overrides must be absolute and writable.
    let r = app.post_form("/settings/save/storage", &[("models_dir", "relative/path")]).await;
    assert_eq!(r.flash().unwrap().0, "error");
    let custom = app.dir.path().join("my-models");
    let r = app
        .post_form(
            "/settings/save/storage",
            &[("models_dir", custom.to_str().unwrap()), ("max_concurrent_downloads", "3")],
        )
        .await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    assert!(custom.is_dir());
    let app = app.restart().await;
    assert_eq!(app.state.settings().storage.models_dir.as_deref(), Some(custom.as_path()));
    assert_eq!(app.state.settings().generation.system_prompt, "Be brief.");
}

#[tokio::test]
async fn empty_states_explain_what_to_do() {
    let app = TestApp::new();
    let lib = app.get("/library").await;
    assert!(lib.body.contains("Your library is empty"));
    let chat = app.get("/chat").await;
    assert!(chat.body.contains("No chat models yet"), "{}", chat.body);
    let create = app.get("/create").await;
    assert!(create.body.contains("No image model yet"));
    let tx = app.get("/transcribe").await;
    assert!(tx.body.contains("No speech model yet"));
    let dl = app.get("/downloads").await;
    assert!(dl.body.contains("No downloads yet"));
}

#[tokio::test]
async fn unknown_pages_get_a_themed_404_and_api_gets_json() {
    let app = TestApp::new();
    let r = app.get("/no/such/page").await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    assert!(r.body.contains("data-theme="), "404 page is not themed");
    let r = app.get("/chat/does-not-exist").await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    let r = app.get("/api/v1/nothing").await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn assets_are_served_with_caching_headers() {
    let app = TestApp::new();
    for (path, ty) in [
        ("/assets/app.css", "text/css"),
        ("/assets/themes.css", "text/css"),
        ("/assets/icons.svg", "image/svg+xml"),
        ("/assets/fonts/Inter.woff", "font/woff"),
    ] {
        let r = app.get(path).await;
        assert_eq!(r.status, StatusCode::OK, "{path}");
        let ct = r.headers.get("content-type").unwrap().to_str().unwrap();
        assert!(ct.starts_with(ty), "{path}: {ct}");
        let etag = r.headers.get("etag").expect("etag").to_str().unwrap().to_string();
        let again = app.send(Request::get(path).header("if-none-match", etag).body(Body::empty()).unwrap()).await;
        assert_eq!(again.status, StatusCode::NOT_MODIFIED, "{path}");
    }
    // Every icon referenced by the templates exists in the sprite.
    let sprite = app.get("/assets/icons.svg").await.body;
    for path in PAGES {
        let body = app.get(path).await.body;
        for part in body.split("/assets/icons.svg#").skip(1) {
            let id: String = part.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-').collect();
            assert!(sprite.contains(&format!("id=\"{id}\"")), "{path}: missing icon #{id}");
        }
    }
}

#[tokio::test]
async fn every_theme_defines_every_token() {
    // A theme that forgot a token would inherit it from the surrounding theme.
    let css = TestApp::new().get("/assets/themes.css").await.body;
    let blocks: Vec<&str> = css.split("[data-theme=\"").skip(1).filter(|b| b.contains("--font-body")).collect();
    assert_eq!(blocks.len(), 6, "expected six complete theme blocks");
    let tokens = |b: &str| -> HashSet<String> {
        let body = b.split('{').nth(1).unwrap_or("").split('}').next().unwrap_or("");
        body.split(';')
            .filter_map(|d| d.trim().strip_prefix("--").map(|r| r.split(':').next().unwrap().trim().to_string()))
            .collect()
    };
    let first = tokens(blocks[0]);
    assert!(first.len() > 80, "suspiciously few tokens: {}", first.len());
    for b in &blocks[1..] {
        let name: String = b.chars().take_while(|c| *c != '"').collect();
        let t = tokens(b);
        let missing: Vec<_> = first.difference(&t).collect();
        assert!(missing.is_empty(), "theme {name} misses {missing:?}");
    }
}

#[tokio::test]
async fn health_endpoint_reports_the_app() {
    let app = TestApp::new();
    let r = app.get("/api/health").await;
    assert_eq!(r.status, StatusCode::OK);
    let v = r.json();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["app"], "Local AI Studio");
}

#[tokio::test]
async fn many_simultaneous_requests_succeed() {
    let app = std::sync::Arc::new(TestApp::new());
    let mut tasks = Vec::new();
    for i in 0..40 {
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            let path = PAGES[i % PAGES.len()];
            app.get(path).await.status
        }));
    }
    for t in tasks {
        assert_eq!(t.await.unwrap(), StatusCode::OK);
    }
}
