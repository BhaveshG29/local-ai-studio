//! Frontend tests: every page renders with no script but the app's own file (and works
//! without it), every link leads somewhere, every form posts to a real route, and every
//! button does something (submits a form or opens an existing pop-up). The sidebar and the
//! chat composer have the structure the stylesheet, the streaming chat page and the voice
//! input rely on. The two themes switch, persist, and older theme names still load.

mod common;

use std::collections::HashSet;
use std::time::Duration;

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

/// The text of an element, with whitespace collapsed.
fn text_of(e: scraper::ElementRef) -> String {
    e.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ")
}

#[tokio::test]
async fn every_page_renders_with_only_the_app_script() {
    let app = TestApp::new();
    for path in PAGES {
        let r = app.get(path).await;
        assert_eq!(r.status, StatusCode::OK, "{path}: {}", r.body);
        assert!(r.body.contains("data-theme=\"white\""), "{path} is not in the default theme");
        let csp = r.headers.get("content-security-policy").and_then(|v| v.to_str().ok()).unwrap_or("");
        let script_src = csp.split(';').map(str::trim).find(|d| d.starts_with("script-src"));
        assert_eq!(script_src, Some("script-src 'self'"), "{path} CSP: {csp}");
        let doc = Html::parse_document(&r.body);
        // One script, the app's own file: no inline code, no event-handler attributes.
        let scripts: Vec<_> = doc.select(&sel("script")).collect();
        assert_eq!(scripts.len(), 1, "{path} has {} scripts", scripts.len());
        assert_eq!(r.body.to_lowercase().matches("<script").count(), 1, "{path}");
        let src = scripts[0].value().attr("src").unwrap_or("");
        assert!(src.starts_with("/assets/app.js?v="), "{path} loads {src}");
        // async, not defer: a chat page streaming an answer is parsed to its end only when the
        // answer is complete, and a deferred script would wait for that.
        assert!(scripts[0].value().attr("async").is_some(), "{path}: the script must not block the page");
        assert!(scripts[0].value().attr("defer").is_none(), "{path}: a deferred script waits for streamed answers");
        assert!(scripts[0].inner_html().trim().is_empty(), "{path} has inline script code");
        for e in doc.select(&sel("*")) {
            for (name, _) in e.value().attrs() {
                assert!(!name.starts_with("on"), "{path}: <{}> has an inline handler {name}", e.value().name());
            }
        }
        assert!(doc.select(&sel("main#main")).next().is_some(), "{path} has no main landmark");
        assert!(doc.select(&sel("h1")).next().is_some(), "{path} has no heading");
    }
}

/// The sidebar is the same on every page: it comes before the page content (a chat page
/// streams its answer, so nothing may follow the main column), it opens and closes with
/// focusable checkboxes and their labels (no script), and every destination keeps its key.
#[tokio::test]
async fn every_page_has_the_sidebar() {
    let app = TestApp::new();
    for path in PAGES {
        let r = app.get(path).await;
        let doc = Html::parse_document(&r.body);
        let sidebar_at = r.body.find("<aside class=\"sidebar\"").unwrap_or_else(|| panic!("{path} has no sidebar"));
        let main_at = r.body.find("<main").unwrap();
        assert!(sidebar_at < main_at, "{path}: the sidebar must come before <main>");
        // The narrow screens' top bar follows the sidebar (Tab reaches an open menu first),
        // and is also outside the main column, so it must come before it as well.
        let topbar_at = r.body.find("<header class=\"topbar\"").unwrap_or_else(|| panic!("{path} has no top bar"));
        assert!(sidebar_at < topbar_at && topbar_at < main_at, "{path}: the top bar is out of place");
        let input = doc
            .select(&sel("input#nav-toggle[type=checkbox]"))
            .next()
            .unwrap_or_else(|| panic!("{path}: no #nav-toggle"));
        assert_eq!(input.value().attr("aria-label"), Some("Collapse the sidebar"), "{path}: no accessible name");
        assert_eq!(input.value().attr("aria-controls"), Some("sidebar"));
        assert!(doc.select(&sel("label[for=nav-toggle]")).count() >= 2, "{path}: #nav-toggle has no button");
        // Narrow screens: the sidebar is a popover drawer (Esc and a click beside it close it),
        // opened by the top bar's Menu button and closed by its own button too.
        assert!(doc.select(&sel("aside#sidebar.sidebar[popover]")).next().is_some(), "{path}: no drawer");
        let menu = doc
            .select(&sel(".topbar button[type=button][popovertarget=sidebar]"))
            .next()
            .unwrap_or_else(|| panic!("{path}: no Menu button"));
        assert!(menu.value().attr("popovertargetaction").is_none(), "{path}: Menu must open the drawer");
        assert_eq!(text_of(menu), "Menu");
        let close = doc
            .select(&sel(".sidebar button[type=button][popovertarget=sidebar][popovertargetaction=hide]"))
            .next()
            .unwrap_or_else(|| panic!("{path}: no close button"));
        assert_eq!(text_of(close), "Close menu");
        let keys: HashSet<(String, String)> = doc
            .select(&sel(".sidebar a[accesskey]"))
            .map(|a| (a.value().attr("accesskey").unwrap().to_string(), a.value().attr("href").unwrap().to_string()))
            .collect();
        for (key, href) in [
            ("h", "/"),
            ("n", "/chat?new=1"),
            ("c", "/chat"),
            ("d", "/discover"),
            ("j", "/downloads"),
            ("l", "/library"),
            ("t", "/transcribe"),
            ("i", "/create"),
            ("g", "/settings"),
            ("y", "/system"),
        ] {
            assert!(keys.contains(&(key.to_string(), href.to_string())), "{path}: no link to {href} on key {key}");
        }
        let labels: Vec<String> = doc.select(&sel(".sidebar .nav-label")).map(text_of).collect();
        for label in ["New chat", "Chats", "Marketplace", "Downloads", "Library", "Speech models", "Images"] {
            assert!(labels.iter().any(|l| l == label), "{path}: no {label} in the sidebar: {labels:?}");
        }
        // Each sidebar link has a tooltip, which is all a collapsed sidebar shows.
        for a in doc.select(&sel(".sidebar .nav a")) {
            assert!(a.value().attr("title").is_some(), "{path}: {} has no title", text_of(a));
        }
        assert!(r.body.contains("Switch to Pure Black"), "{path}: no theme switch");
        assert!(doc.select(&sel(".sidebar button[popovertarget=keys-pop]")).next().is_some(), "{path}: no shortcuts");
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
    for (slug, label, other, scheme) in [
        ("black", "Pure Black", "white", "dark"),
        ("white", "Pure White", "black", "light"),
        ("black", "Pure Black", "white", "dark"),
    ] {
        let r = app.post_form("/settings/theme", &[("theme", slug), ("back", "/library")]).await;
        assert_eq!(r.status, StatusCode::SEE_OTHER);
        assert_eq!(r.location(), "/library");
        assert_eq!(r.flash().unwrap(), ("ok".to_string(), format!("Theme changed to {label}.")));
        let page = app.get("/library").await;
        assert!(page.body.contains(&format!("data-theme=\"{slug}\"")), "theme {slug} not applied");
        assert!(page.body.contains(&format!("<meta name=\"color-scheme\" content=\"{scheme}\">")), "{slug}");
        let doc = Html::parse_document(&page.body);
        // The sidebar's one-click switch offers the other theme and comes back to this page.
        let form = doc.select(&sel(".sidebar form.theme-switch")).next().expect("theme switch");
        assert_eq!(form.value().attr("action"), Some("/settings/theme"));
        let value =
            |name: &str| form.select(&sel(&format!("input[name={name}]"))).next().and_then(|i| i.value().attr("value"));
        assert_eq!(value("theme"), Some(other));
        assert_eq!(value("back"), Some("/library"));
        let switch = form.select(&sel("button[type=submit]")).next().unwrap();
        let other_label = if other == "white" { "Pure White" } else { "Pure Black" };
        assert_eq!(text_of(switch), format!("Switch to {other_label}"), "the switch has no accessible name");
        // Settings → Appearance shows both themes, the current one pressed.
        let settings = Html::parse_document(&app.get("/settings").await.body);
        let options: Vec<(String, String)> = settings
            .select(&sel("#appearance form[action='/settings/theme']"))
            .map(|f| {
                let theme = f.select(&sel("input[name=theme]")).next().unwrap().value().attr("value").unwrap();
                let pressed = f.select(&sel("button")).next().unwrap().value().attr("aria-pressed").unwrap();
                (theme.to_string(), pressed.to_string())
            })
            .collect();
        let expect: Vec<(String, String)> = ["white", "black"]
            .iter()
            .map(|t| (t.to_string(), if *t == slug { "true" } else { "false" }.to_string()))
            .collect();
        assert_eq!(options, expect);
    }
    assert!(app.config_text().contains("theme = \"black\""));
    let app = app.restart().await;
    let page = app.get("/").await;
    assert!(page.body.contains("data-theme=\"black\""), "theme not restored after restart");
}

#[tokio::test]
async fn unknown_theme_is_rejected() {
    let app = TestApp::new();
    for name in ["pink", "neon", ""] {
        let r = app.post_form("/settings/theme", &[("theme", name)]).await;
        assert_eq!(r.flash().unwrap().0, "error", "theme {name:?} was accepted");
    }
    assert_eq!(app.state.settings().ui.theme.slug(), "white");
    let r = app.post_form("/settings/theme", &[("theme", "black"), ("back", "https://evil.example/")]).await;
    assert_eq!(r.location(), "/settings", "open redirect through the back parameter");
}

/// A config.toml written by an earlier version names one of six themes: the app still starts,
/// in the closer of the two, and saves the new name the next time it writes the file.
#[tokio::test]
async fn older_theme_names_still_load() {
    let mut app = TestApp::new();
    for (old, new) in [("neon", "black"), ("warm", "white"), ("graphite", "black"), ("glass", "white"), ("x", "white")]
    {
        std::fs::write(&app.state.paths.config_file, format!("[ui]\ntheme = \"{old}\"\n")).unwrap();
        app = app.restart().await;
        let page = app.get("/").await;
        assert_eq!(page.status, StatusCode::OK);
        assert!(page.body.contains(&format!("data-theme=\"{new}\"")), "theme = \"{old}\" did not become {new}");
    }
    let r = app.post_form("/settings/save/ui", &[("restore_last_chat", "on")]).await;
    assert_eq!(r.flash().unwrap().0, "ok");
    assert!(app.config_text().contains("theme = \"white\""), "{}", app.config_text());
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
        ("/assets/app.js", "text/javascript"),
        ("/assets/icons.svg", "image/svg+xml"),
        ("/assets/fonts/Inter.woff", "font/woff"),
        ("/assets/fonts/JetBrainsMono.woff", "font/woff"),
    ] {
        let r = app.get(path).await;
        assert_eq!(r.status, StatusCode::OK, "{path}");
        let ct = r.headers.get("content-type").unwrap().to_str().unwrap();
        assert!(ct.starts_with(ty), "{path}: {ct}");
        let etag = r.headers.get("etag").expect("etag").to_str().unwrap().to_string();
        let again = app.send(Request::get(path).header("if-none-match", etag).body(Body::empty()).unwrap()).await;
        assert_eq!(again.status, StatusCode::NOT_MODIFIED, "{path}");
    }
    // Only the two fonts the themes use are bundled.
    for font in ["Manrope", "IBMPlexSans", "SpaceGrotesk", "Orbitron", "Fraunces", "NunitoSans", "PlusJakartaSans"] {
        assert_eq!(app.get(&format!("/assets/fonts/{font}.woff")).await.status, StatusCode::NOT_FOUND, "{font}");
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

/// The two themes define the same custom properties (a missing one would silently fall back
/// to the other theme's value), including the ones other stylesheets are written against, and
/// every property the stylesheet reads exists. Nothing of the six earlier themes is left.
#[tokio::test]
async fn both_themes_define_every_token() {
    let app = TestApp::new();
    let themes = app.get("/assets/themes.css").await.body;
    let css = app.get("/assets/app.css").await.body;
    assert_eq!(themes.matches("[data-theme=").count(), 2, "expected exactly two themes");
    let block = |selector: &str| -> HashSet<String> {
        let start = themes.find(selector).unwrap_or_else(|| panic!("no {selector} block"));
        let rest = &themes[start..];
        let body = &rest[rest.find('{').unwrap() + 1..];
        let body = &body[..body.find('}').unwrap()];
        body.split(';')
            .filter_map(|d| d.trim().strip_prefix("--").map(|r| r.split(':').next().unwrap().trim().to_string()))
            .collect()
    };
    assert!(themes.contains(":root, [data-theme=\"white\"]"), "Pure White must be the default");
    let white = block("[data-theme=\"white\"]");
    let black = block("[data-theme=\"black\"]");
    assert_eq!(white, black, "the themes define different properties");
    for t in [
        "bg",
        "surface",
        "surface-2",
        "surface-3",
        "text",
        "text-muted",
        "text-faint",
        "border",
        "border-strong",
        "accent",
        "accent-text",
        "danger",
        "danger-soft",
        "ok",
        "warn",
        "focus",
        "radius-sm",
        "radius",
        "radius-lg",
        "font-body",
        "font-mono",
    ] {
        assert!(white.contains(t), "the themes don't define --{t}");
    }
    let own: HashSet<String> = css
        .split("--")
        .skip(1)
        .filter_map(|p| p.split_once(':').map(|(name, _)| name.trim().to_string()))
        .filter(|n| n.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
        .collect();
    for part in css.split("var(--").skip(1) {
        let name: String = part.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-').collect();
        assert!(white.contains(&name) || own.contains(&name), "app.css uses --{name}, which nothing defines");
    }
    let both = format!("{themes}\n{css}").to_lowercase();
    for gone in [
        "midnight",
        "graphite",
        "neon",
        "minimal",
        "warm",
        "glass",
        "manrope",
        "plex",
        "grotesk",
        "orbitron",
        "fraunces",
        "nunito",
        "jakarta",
        "gradient",
        "backdrop-filter",
        "text-shadow",
    ] {
        assert!(!both.contains(gone), "the stylesheets still mention {gone}");
    }
}

/// The chat page keeps the order the streamed answer needs (everything but the messages comes
/// first), and the composer has the structure the voice input and Enter-to-send rely on.
#[tokio::test]
async fn the_chat_composer_and_streaming_layout() {
    let ollama = FakeOllama::start(&[("qwen3:0.6b", &["completion"])]).await;
    let mut up = Upstreams::offline();
    up.ollama = ollama.url.clone();
    let app = TestApp::with(up);
    ollama.set_reply(&["One", " two", " three."], Duration::from_millis(100));
    let r = app.post_multipart("/chat/new", &[("model_ref", "ollama:qwen3:0.6b"), ("text", "Count")], &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    let id = r.location().trim_start_matches("/chat/").split('#').next().unwrap().to_string();
    // Opened while the answer is written: the page streams up to the answer first.
    let live = app.get(&format!("/chat/{id}")).await.body;
    let at = |needle: &str| live.find(needle).unwrap_or_else(|| panic!("no {needle} in the streamed page"));
    let order = [
        "<aside class=\"sidebar\"",
        "class=\"conv-list\"",
        "<main",
        "name=\"las-control\"",
        "id=\"import-chat\"",
        "id=\"chat-settings\"",
        "<form class=\"composer is-generating\"",
        "id=\"stop-form\"",
        "class=\"thread\"",
        "class=\"live-wrap\"",
    ];
    for pair in order.windows(2) {
        assert!(at(pair[0]) < at(pair[1]), "{} must come before {}", pair[0], pair[1]);
    }
    wait_until("answer", 20, || async { !app.state.chat.is_generating(&id) }).await;
    let check_composer = |body: &str, action: &str| {
        let doc = Html::parse_document(body);
        let form = doc.select(&sel("form.composer")).next().expect("composer");
        let v = form.value();
        assert_eq!(v.attr("action"), Some(action));
        assert!(v.attr("data-enter-send").is_some() && v.attr("data-dictation").is_some(), "{action}");
        let field = form.select(&sel(".composer-field")).next().expect("text box");
        let text = field.select(&sel("textarea#composer-text[name=text][data-dictation-output]")).next();
        assert!(text.is_some(), "{action}: textarea");
        // The microphone follows the textarea inside the text box; the status line follows the box.
        let kids: Vec<_> = field.children().filter_map(scraper::ElementRef::wrap).collect();
        let tpos = kids.iter().position(|e| e.value().name() == "textarea").unwrap();
        assert!(kids[tpos + 1].value().classes().any(|c| c == "mic"), "{action}: the microphone isn't next");
        let next = field.next_siblings().filter_map(scraper::ElementRef::wrap).next().unwrap();
        assert!(next.value().classes().any(|c| c == "dictation-status"), "{action}: no status line");
        assert_eq!(next.value().attr("role"), Some("status"));
    };
    let page = app.get(&format!("/chat/{id}")).await.body;
    check_composer(&page, &format!("/chat/{id}/send"));
    let doc = Html::parse_document(&page);
    assert!(doc.select(&sel(".sidebar .conv-list a.conv.active[aria-current=page]")).next().is_some());
    assert!(doc.select(&sel(".sidebar button[popovertarget=import-chat]")).next().is_some(), "no Import a chat");
    assert!(doc.select(&sel(".conv-pane, #conv-toggle")).next().is_none(), "the old conversation pane is back");
    let send = doc.select(&sel("form.composer .btn-send")).next().unwrap();
    assert!(send.value().attr("disabled").is_none());
    // Edit forms send with Enter too.
    assert!(doc.select(&sel("form[action$=\"/edit\"][data-enter-send]")).next().is_some());
    let new_chat = app.get("/chat?new=1").await.body;
    check_composer(&new_chat, "/chat/new");
    // A chat whose model has gone can't send, and says why.
    let r = app.post_form(&format!("/chat/{id}/settings"), &[("model_ref", "local:gone")]).await;
    assert_eq!(r.flash().unwrap().0, "ok");
    let doc = Html::parse_document(&app.get(&format!("/chat/{id}")).await.body);
    let form = doc.select(&sel("form.composer")).next().unwrap();
    assert_eq!(form.value().attr("data-can-send"), Some("false"));
    let send = form.select(&sel(".btn-send")).next().unwrap();
    assert!(send.value().attr("disabled").is_some());
    assert!(send.value().attr("title").unwrap_or("").contains("Chat settings"));
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
