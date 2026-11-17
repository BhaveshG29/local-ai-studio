//! Voice input and Enter-to-send, the browser side: the one script (assets/app.js) and the
//! markup it works on. The script itself is exercised in a real browser (headless Chrome with
//! a fake microphone); these tests pin the contract between the templates, the stylesheet,
//! the icon sprite and the script, and that everything still works without the script:
//!
//! * every page loads one script, `async`, and nothing inline;
//! * the chat composer has the attributes the script looks for, and the microphone in each of
//!   its looks: crossed out when Whisper Base isn't ready (with its pop-up and a Download form
//!   outside the composer), a toggle when it is (with a "needs JavaScript" pop-up);
//! * the live panel on the Speech models page, ready and not.

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::http::StatusCode;
use common::*;
use local_ai_studio::db::models::ModelRow;
use local_ai_studio::inference::dictation;
use scraper::{ElementRef, Html, Selector};
use serde_json::{Value, json};

const PAGES: &[&str] = &[
    "/",
    "/chat?new=1",
    "/discover",
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
fn text_of(e: ElementRef) -> String {
    e.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ")
}

fn one<'a>(doc: &'a Html, selector: &str) -> ElementRef<'a> {
    doc.select(&sel(selector)).next().unwrap_or_else(|| panic!("nothing matches {selector}"))
}

fn none(doc: &Html, selector: &str) -> bool {
    doc.select(&sel(selector)).next().is_none()
}

fn inside_a_form(e: ElementRef) -> bool {
    e.ancestors().any(|n| n.value().as_element().is_some_and(|el| el.name() == "form"))
}

/// The `<use href>` of the first icon inside `e`.
fn icon_of(e: ElementRef) -> String {
    e.select(&sel("svg use")).next().and_then(|u| u.value().attr("href")).unwrap_or_default().to_string()
}

// ---- apps to look at ---------------------------------------------------------------------

/// An app with an Ollama chat model, so the new-chat page has a composer.
async fn app_with_chat_model(up: Upstreams) -> (TestApp, FakeOllama) {
    let ollama = FakeOllama::start(&[("qwen3:0.6b", &["completion"])]).await;
    let mut up = up;
    up.ollama = ollama.url.clone();
    (TestApp::with(up), ollama)
}

/// A conversation with an answer; returns its id.
async fn conversation(app: &TestApp, ollama: &FakeOllama) -> String {
    ollama.set_reply(&["Hi", " there."], Duration::from_millis(5));
    let r = app.post_multipart("/chat/new", &[("model_ref", "ollama:qwen3:0.6b"), ("text", "Hello")], &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER, "{:?}", r.flash());
    let id = r.location().trim_start_matches("/chat/").split('#').next().unwrap().to_string();
    wait_until("answer", 20, || async { !app.state.chat.is_generating(&id) }).await;
    id
}

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

/// A Whisper Base in the library, the way a finished download leaves it: valid files that
/// pass the app's checks (it can't really transcribe; these tests never ask it to).
async fn fake_whisper_base(app: &TestApp) -> PathBuf {
    let dir = app.state.downloads.repo_dir(dictation::WHISPER_BASE_REPO);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), base_config().to_string()).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
    write_weights(&dir.join("model.safetensors"));
    let analysis = local_ai_studio::library::analyze_dir(&dir, Some(dictation::WHISPER_BASE_REPO));
    let row = ModelRow {
        id: local_ai_studio::util::new_id(),
        source: "huggingface".into(),
        repo_id: Some(dictation::WHISPER_BASE_REPO.into()),
        revision: Some("main".into()),
        display_name: "whisper-base".into(),
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
    assert!(dictation::mic_view(&app.state).await.ready, "the stand-in Whisper Base isn't accepted");
    dir
}

// ---- the script ----------------------------------------------------------------------------

/// The one script loads `async` (a chat page streaming an answer is parsed to its end only
/// when the answer is complete; a deferred script would wait that long), and nothing on any
/// page runs code inline: no script elements with content, no event-handler attributes.
#[tokio::test]
async fn every_page_loads_only_the_app_script_async() {
    let (app, ollama) = app_with_chat_model(Upstreams::offline()).await;
    let conv = conversation(&app, &ollama).await;
    let chat_page = format!("/chat/{conv}");
    let mut pages: Vec<&str> = PAGES.to_vec();
    pages.push(&chat_page);
    for round in ["without Whisper Base", "with Whisper Base"] {
        if round == "with Whisper Base" {
            fake_whisper_base(&app).await;
        }
        for path in &pages {
            let r = app.get(path).await;
            assert_eq!(r.status, StatusCode::OK, "{path} ({round})");
            let doc = Html::parse_document(&r.body);
            let scripts: Vec<_> = doc.select(&sel("script")).collect();
            assert_eq!(scripts.len(), 1, "{path} ({round}): {} scripts", scripts.len());
            assert_eq!(r.body.to_lowercase().matches("<script").count(), 1, "{path} ({round})");
            let s = scripts[0].value();
            assert!(s.attr("src").unwrap_or("").starts_with("/assets/app.js?v="), "{path}: {:?}", s.attr("src"));
            assert!(s.attr("async").is_some() && s.attr("defer").is_none(), "{path}: the script must be async");
            assert!(scripts[0].inner_html().trim().is_empty(), "{path}: inline script code");
            assert!(doc.select(&sel("head script")).next().is_some(), "{path}: the script belongs in <head>");
            for e in doc.select(&sel("*")) {
                for (name, value) in e.value().attrs() {
                    assert!(!name.starts_with("on"), "{path}: <{}> has an inline handler {name}", e.value().name());
                    assert!(
                        !value.trim_start().to_lowercase().starts_with("javascript:"),
                        "{path}: a javascript: URL in {name}"
                    );
                }
            }
        }
    }
}

/// app.js is served as JavaScript and is everything the voice input needs: the page code
/// and the AudioWorklet that reads the microphone (registered under the name the page uses),
/// with no eval-like code, inline HTML, or dependencies.
#[tokio::test]
async fn the_script_is_served_and_self_contained() {
    let app = TestApp::new();
    let r = app.get("/assets/app.js").await;
    assert_eq!(r.status, StatusCode::OK);
    let ct = r.headers.get("content-type").unwrap().to_str().unwrap();
    assert!(ct.starts_with("text/javascript"), "{ct}");
    let js = r.body;
    assert!(js.contains("\"use strict\";"), "not strict");
    let registered = js.split("registerProcessor(").nth(1).unwrap_or_default().trim_start();
    assert!(registered.starts_with("\"las-mic\""), "the worklet isn't registered as las-mic");
    assert!(js.contains("new AudioWorkletNode(ctx, \"las-mic\""), "the page must use the worklet it registers");
    assert!(js.contains("audioWorklet.addModule(SCRIPT_URL)"), "the worklet is this same file");
    assert!(js.contains("document.currentScript"), "the worklet URL must be the script's own (with ?v=)");
    for banned in [
        "eval(",
        "new Function",
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "import(",
        "require(",
    ] {
        assert!(!js.contains(banned), "app.js uses {banned}");
    }
    for url in ["blob:", "data:text/javascript", "cdn.", "https://"] {
        assert!(!js.contains(url), "app.js mentions {url}");
    }
    // The endpoints it talks to exist: a GET on the POST route answers 405, and an unknown
    // session the API's own JSON 404.
    assert!(js.contains("\"/api/speech/sessions\""));
    assert_eq!(app.get("/api/speech/sessions").await.status, StatusCode::METHOD_NOT_ALLOWED);
    let r = app.get("/api/speech/sessions/0123456789abcdef0123456789abcdef").await;
    assert_eq!((r.status, r.json()["code"].as_str()), (StatusCode::NOT_FOUND, Some("not_found")));
    // A page's script URL carries the version, and the worklet loads that same URL.
    let page = app.get("/").await.body;
    let doc = Html::parse_document(&page);
    let src = one(&doc, "script").value().attr("src").unwrap().to_string();
    let r = app.get(&src).await;
    assert_eq!(r.status, StatusCode::OK, "{src}");
    assert_eq!(r.body, js);
}

/// Every icon the script swaps in exists in the sprite, and the sprite is well-formed XML
/// (a "--" inside a comment ends parsing early, and every icon after it disappears).
#[tokio::test]
async fn the_icons_the_script_uses_exist() {
    let app = TestApp::new();
    let sprite = app.get("/assets/icons.svg").await.body;
    let js = app.get("/assets/app.js").await.body;
    for id in ["mic", "mic-off", "mic-alert", "pause", "play"] {
        assert!(sprite.contains(&format!("<symbol id=\"{id}\"")), "no #{id} in the sprite");
        assert!(js.contains(&format!("\"{id}\"")), "app.js doesn't use #{id}");
    }
    for comment in sprite.split("<!--").skip(1) {
        let body = comment.split("-->").next().unwrap();
        assert!(!body.contains("--"), "a comment in icons.svg contains \"--\": {body}");
    }
    // The cross and the badge are drawn in the theme's danger colour, so they follow the theme.
    let symbol = |id: &str| {
        let start = sprite.find(&format!("<symbol id=\"{id}\"")).unwrap();
        sprite[start..start + sprite[start..].find("</symbol>").unwrap()].to_string()
    };
    assert!(symbol("mic-off").contains("var(--danger)"));
    // ... and is drawn bold enough to stand out at 20 px ("a prominent red cross").
    let cross = symbol("mic-off");
    let width: f32 = cross
        .split("stroke:var(--danger);stroke-width:")
        .nth(1)
        .and_then(|w| w.split('"').next())
        .unwrap()
        .parse()
        .unwrap();
    assert!(width >= 2.5, "the red cross is {width} wide");
    assert!(symbol("mic-alert").contains("var(--danger)") && symbol("mic-alert").contains("var(--danger-text)"));
}

// ---- the chat composer ------------------------------------------------------------------

/// What the script looks for in a composer, on the new-chat page and in a conversation.
fn check_composer(body: &str, path: &str) {
    let doc = Html::parse_document(body);
    let form = one(&doc, "form.composer");
    let v = form.value();
    assert!(v.attr("data-enter-send").is_some() && v.attr("data-dictation").is_some(), "{path}");
    assert!(v.attr("data-dictation-mode").is_none(), "{path}: the composer inserts at the caret");
    let text = form.select(&sel("textarea#composer-text[name=text][data-dictation-output]")).next();
    assert!(text.is_some(), "{path}: no text box for the words");
    let send = form.select(&sel("button.btn-send[type=submit]")).next().expect("Send");
    assert_eq!(send.value().attr("accesskey"), Some("s"), "{path}");
    let status = form.select(&sel(".dictation-status[data-dictation-status]")).next().expect("status line");
    assert_eq!(status.value().attr("role"), Some("status"), "{path}");
    assert_eq!(status.value().attr("aria-live"), Some("polite"), "{path}");
    assert!(status.select(&sel("[data-dictation-message]")).next().is_some(), "{path}: no message");
    // The timer and the "Transcribing" note change all the time: never announced.
    for part in ["[data-dictation-timer]", "[data-dictation-busy]"] {
        let e = status.select(&sel(part)).next().unwrap_or_else(|| panic!("{path}: no {part}"));
        let hidden = e.ancestors().filter_map(ElementRef::wrap).any(|a| a.value().attr("aria-hidden") == Some("true"));
        assert!(hidden, "{path}: {part} must be aria-hidden");
    }
    assert_eq!(doc.select(&sel("[data-dictation-status]")).count(), 1, "{path}: one live region per composer");
    // The microphone is at the right end of the text box (next to the text, inside its frame);
    // Pause, shown left of it while recording, comes just before it.
    let field = form.select(&sel(".composer-field")).next().unwrap();
    let kids: Vec<_> = field.children().filter_map(ElementRef::wrap).collect();
    let at = kids.iter().position(|e| e.value().name() == "textarea").unwrap();
    let after: Vec<_> = kids[at + 1..].iter().collect();
    assert!(
        after.iter().all(|e| e.value().classes().any(|c| c == "mic")),
        "{path}: the microphone isn't next to the text"
    );
    let last = after.last().expect("a microphone");
    assert_eq!(last.value().attr("accesskey"), Some("m"), "{path}: Alt+M");
    assert!(after[..after.len() - 1].iter().all(|e| e.value().classes().any(|c| c == "mic-pause")), "{path}");
}

#[tokio::test]
async fn the_composer_has_what_voice_input_and_enter_to_send_need() {
    let (app, ollama) = app_with_chat_model(Upstreams::offline()).await;
    let conv = conversation(&app, &ollama).await;
    for path in ["/chat?new=1".to_string(), format!("/chat/{conv}")] {
        check_composer(&app.get(&path).await.body, &path);
    }
    fake_whisper_base(&app).await;
    for path in ["/chat?new=1".to_string(), format!("/chat/{conv}")] {
        check_composer(&app.get(&path).await.body, &path);
    }
    // Edit forms send with Enter too (no microphone there).
    let doc = Html::parse_document(&app.get(&format!("/chat/{conv}")).await.body);
    let edit = one(&doc, "form[action$=\"/edit\"]");
    assert!(edit.value().attr("data-enter-send").is_some() && edit.value().attr("data-dictation").is_none());
    // Alt+M is listed with the other shortcuts, and no other control uses M.
    let keys = text_of(one(&doc, "#keys-pop"));
    assert!(keys.contains("M Voice input: start or stop (in a chat)"), "{keys}");
    assert_eq!(doc.select(&sel("[accesskey=m]")).count(), 1);
}

/// Case A: Whisper Base isn't there. The microphone is crossed out (in red, by the sprite),
/// says why on hover, and opens a pop-up that needs no script: the reason, what to do, a
/// link to the Speech models page, a Download button, and Close.
#[tokio::test]
async fn case_a_the_microphone_is_crossed_out_without_whisper_base() {
    let (app, _ollama) = app_with_chat_model(Upstreams::offline()).await;
    let r = app.get("/chat?new=1").await;
    let doc = Html::parse_document(&r.body);
    let reason = "Whisper Base isn't downloaded yet.";
    let mic = one(&doc, ".composer-field > button.mic");
    let v = mic.value();
    assert!(v.classes().any(|c| c == "mic-unavailable"), "{:?}", v.attr("class"));
    assert_eq!(v.attr("type"), Some("button"));
    assert_eq!(v.attr("popovertarget"), Some("mic-help"));
    // Hovering says why; a click opens the pop-up that says what to do.
    assert_eq!(v.attr("title"), Some(format!("{reason} Click to see what to do.").as_str()));
    assert!(v.attr("data-dictation-unavailable").is_some());
    assert!(v.attr("data-dictation-action").is_none() && v.attr("aria-pressed").is_none(), "it must never record");
    assert_eq!(icon_of(mic), "/assets/icons.svg#mic-off");
    assert_eq!(text_of(mic), "Voice input unavailable");
    assert!(none(&doc, ".mic-pause") && none(&doc, "#mic-nojs"));

    let pop = one(&doc, "#mic-help[popover]");
    assert_eq!(text_of(one(&doc, "#mic-help h2")), "Voice input needs Whisper Base");
    assert_eq!(text_of(one(&doc, "#mic-help [data-mic-reason]")), reason);
    assert!(text_of(pop).contains("Download it here or on the Speech models page"), "{}", text_of(pop));
    // The same size as on the Speech models page.
    let size = dictation::whisper_base_status(&app.state).await.size_label;
    assert!(text_of(pop).contains(&format!("({size} to download)")), "{}", text_of(pop));
    let link = one(&doc, "#mic-help a[href=\"/transcribe#whisper-base\"]");
    assert_eq!(text_of(link), "Go to Speech models");
    let close = one(&doc, "#mic-help button[popovertarget=mic-help][popovertargetaction=hide]");
    assert_eq!(text_of(close), "Close");
    // The Download button belongs to a form outside the composer (forms can't nest: inside,
    // it would send the chat message instead).
    let download = one(&doc, "#mic-help button[type=submit][form=mic-download]");
    assert_eq!(text_of(download), "Download Whisper Base");
    let form = one(&doc, "form#mic-download");
    assert_eq!(form.value().attr("method"), Some("post"));
    assert_eq!(form.value().attr("action"), Some("/speech/whisper-base/download"));
    assert!(!inside_a_form(form), "the download form is nested in another form");
    assert!(form.value().attr("hidden").is_some(), "the empty form must take no room");
    // The status line is empty (and takes no room) until the script has something to say.
    let status = one(&doc, ".dictation-status");
    assert!(status.value().classes().any(|c| c == "is-empty"));
    assert!(text_of(one(&doc, ".dictation-status [data-dictation-message]")).is_empty());

    // Download really starts it (here: Hugging Face can't be reached, and the page says so).
    let r = app.post_form("/speech/whisper-base/download", &[]).await;
    assert_eq!(r.status, StatusCode::SEE_OTHER);
    assert_eq!(r.location(), "/transcribe#whisper-base");
    assert_eq!(r.flash().unwrap().0, "error");
}

/// Case A while Whisper Base downloads: no Download button, a note instead.
#[tokio::test]
async fn case_a_says_when_whisper_base_is_downloading() {
    let mut repo = FakeRepo::new("openai/whisper-base");
    repo.extra = json!({ "pipeline_tag": "automatic-speech-recognition", "tags": ["whisper"], "config": { "model_type": "whisper" } });
    repo.files = vec![
        FakeFile::small("config.json", base_config().to_string().as_bytes()),
        FakeFile::small("tokenizer.json", tokenizer_json().as_bytes()),
        FakeFile::lfs("model.safetensors", bytes_of(4_000_000, 7)),
    ];
    let hf = FakeHf::start(vec![repo]).await;
    *hf.state.chunk_delay.lock().unwrap() = Some(Duration::from_millis(100));
    let mut up = Upstreams::offline();
    up.hf = hf.url.clone();
    let (app, _ollama) = app_with_chat_model(up).await;
    let r = app.post_form("/speech/whisper-base/download", &[]).await;
    assert_eq!(r.flash().unwrap().0, "ok", "{:?}", r.flash());
    assert!(dictation::mic_view(&app.state).await.downloading);
    let doc = Html::parse_document(&app.get("/chat?new=1").await.body);
    let mic = one(&doc, ".composer-field > button.mic");
    assert!(mic.value().attr("title").unwrap().starts_with("Whisper Base is"), "{:?}", mic.value().attr("title"));
    assert!(text_of(one(&doc, "#mic-help")).contains("Whisper Base is downloading…"));
    assert!(none(&doc, "#mic-help button[form=mic-download]") && none(&doc, "form#mic-download"));
    let _ = app.post_form("/speech/whisper-base/cancel", &[]).await;
}

/// Case B: Whisper Base is ready. The microphone starts and stops recording (the script
/// drives it; its name says which, so it has no aria-pressed: a toggle's name must not
/// change); without the script it opens a pop-up saying voice input needs JavaScript. The
/// crossed-out pop-up is there too (the script switches to it if Whisper Base goes away).
#[tokio::test]
async fn case_b_the_microphone_records_when_whisper_base_is_ready() {
    let (app, _ollama) = app_with_chat_model(Upstreams::offline()).await;
    fake_whisper_base(&app).await;
    let doc = Html::parse_document(&app.get("/chat?new=1").await.body);
    let mic = one(&doc, ".composer-field > button.mic:not(.mic-pause)");
    let v = mic.value();
    assert!(v.classes().any(|c| c == "mic-toggle") && !v.classes().any(|c| c == "mic-unavailable"));
    // Pause shows left of the microphone, so it comes first in the page (Tab follows the eye).
    let order: Vec<&str> = doc
        .select(&sel(".composer-field > button.mic"))
        .map(|b| b.value().attr("data-dictation-action").unwrap_or(""))
        .collect();
    assert_eq!(order, ["pause", "toggle"]);
    assert_eq!(v.attr("type"), Some("button"));
    assert_eq!(v.attr("data-dictation-action"), Some("toggle"));
    assert!(v.attr("aria-pressed").is_none(), "its name says what a click does");
    assert_eq!(v.attr("popovertarget"), Some("mic-nojs"));
    assert_eq!(icon_of(mic), "/assets/icons.svg#mic");
    assert_eq!(text_of(mic), "Start voice input");
    assert!(v.attr("data-dictation-unavailable").is_none());
    let pause = one(&doc, ".composer-field > button.mic-pause");
    assert_eq!(pause.value().attr("data-dictation-action"), Some("pause"));
    assert_eq!(pause.value().attr("popovertarget"), Some("mic-nojs"));
    let nojs = one(&doc, "#mic-nojs[popover]");
    assert!(text_of(nojs).contains("Voice input needs JavaScript"), "{}", text_of(nojs));
    assert!(text_of(nojs).contains("JavaScript is turned off in this browser, or the page's script couldn't load."));
    assert!(nojs.select(&sel("button[popovertarget=mic-nojs][popovertargetaction=hide]")).next().is_some());
    assert!(text_of(one(&doc, "#mic-help [data-mic-reason]")).is_empty());
    assert!(none(&doc, "form#mic-download"), "nothing to download");
    // Every page still has working buttons and links (the shared check from tests/pages.rs).
    let mut checked = std::collections::HashSet::new();
    assert_page_is_wired(&app, "/chat?new=1", &mut checked).await;
    assert_page_is_wired(&app, "/transcribe", &mut checked).await;
}

/// A chat whose model has gone can't send: the script reads data-can-send.
#[tokio::test]
async fn a_chat_without_a_model_says_so_to_the_script() {
    let (app, ollama) = app_with_chat_model(Upstreams::offline()).await;
    let conv = conversation(&app, &ollama).await;
    let r = app.post_form(&format!("/chat/{conv}/settings"), &[("model_ref", "local:gone")]).await;
    assert_eq!(r.flash().unwrap().0, "ok");
    let doc = Html::parse_document(&app.get(&format!("/chat/{conv}")).await.body);
    let form = one(&doc, "form.composer");
    assert_eq!(form.value().attr("data-can-send"), Some("false"));
    assert!(form.select(&sel(".btn-send[disabled]")).next().is_some());
    let js = app.get("/assets/app.js").await.body;
    assert!(js.contains("data-can-send") && js.contains("Pick a model in Chat settings first."));

    // A chat that never had a model (imported) can't send either, and says why.
    let chat = json!({ "title": "Imported", "messages": [{ "role": "user", "content": "hello" }] });
    let r = app
        .post_multipart(
            "/chat/import",
            &[],
            &[("file", "chat.json", "application/json", chat.to_string().into_bytes())],
        )
        .await;
    let id = r.location().trim_start_matches("/chat/").split('#').next().unwrap().to_string();
    assert_eq!(id.len(), 32, "{:?} {:?}", r.location(), r.flash());
    let doc = Html::parse_document(&app.get(&format!("/chat/{id}")).await.body);
    let form = one(&doc, "form.composer");
    assert_eq!(form.value().attr("data-can-send"), Some("false"));
    let send = one(&doc, ".btn-send");
    assert!(send.value().attr("disabled").is_some());
    assert_eq!(send.value().attr("title"), Some("Pick a model in Chat settings first"));
    assert!(text_of(one(&doc, ".chat > .notice")).contains("This chat has no model yet"));
    // The server refuses it too (a page without the script), keeping the text.
    let r = app.post_multipart(&format!("/chat/{id}/send"), &[("text", "still there?")], &[]).await;
    assert_eq!(r.flash().map(|f| f.0).as_deref(), Some("error"));
}

/// Nothing the recording heard is lost: a pause waits for words the model hasn't finished
/// (`pending`), words that come after it go in later, and forms or links that would replace
/// the page wait for the recording to end. (Exercised in a real browser; this pins the
/// contract with the server and the markup.)
#[tokio::test]
async fn the_script_keeps_what_a_recording_heard() {
    let app = TestApp::new();
    let js = app.get("/assets/app.js").await.body;
    for needle in [
        "state.pending",
        "state.sound",
        "beforeunload",
        "keepRecording",
        "SAY.leaveChat",
        "SAY.leavePanel",
        "this.shown",
        "cutPause",
        "onstatechange",
        "isSecureContext",
        "\"bad_rate\"",
    ] {
        assert!(js.contains(needle), "app.js lacks {needle}");
    }
}

// ---- the live panel ----------------------------------------------------------------------

fn check_panel_common(doc: &Html) {
    let panel = one(doc, "section#live.live-panel[data-dictation][data-dictation-mode=panel]");
    assert_eq!(text_of(one(doc, "#live h2")), "Live transcription");
    // Without JavaScript the controls are hidden and a note says so (no inline script). The
    // rule that hides them is in the head: <style> isn't allowed in the body.
    let noscript = panel.select(&sel("noscript")).next().expect("noscript");
    // Parsed as a page with scripting on, so its content is text here.
    let inner = noscript.text().collect::<String>();
    assert!(!inner.contains("<style"), "{inner}");
    let head = one(doc, "head noscript").text().collect::<String>();
    assert!(head.contains("<style>") && head.contains(".live-panel .live-js"), "{head}");
    assert!(inner.contains("Live transcription needs JavaScript"), "{inner}");
    assert!(inner.contains("The file form below still works."), "{inner}");
    let js_part = one(doc, "#live .live-js");
    for action in ["start", "pause", "stop", "copy", "clear", "download"] {
        let b = js_part
            .select(&sel(&format!("button[type=button][data-dictation-action={action}]")))
            .next()
            .unwrap_or_else(|| panic!("no {action} button"));
        // Should the script not load, the buttons explain instead of doing nothing.
        assert_eq!(b.value().attr("popovertarget"), Some("live-nojs"), "{action}");
        assert!(!text_of(b).is_empty(), "{action} has no name");
    }
    assert!(one(doc, "#live-nojs[popover]").select(&sel("button[popovertargetaction=hide]")).next().is_some());
    let status = one(doc, "#live [data-dictation-status]");
    assert_eq!(status.value().attr("role"), Some("status"));
    assert_eq!(status.value().attr("aria-live"), Some("polite"));
    let label = one(doc, "#live [data-dictation-label]");
    assert_eq!(label.value().attr("aria-hidden"), Some("true"));
    assert_eq!(text_of(label), "Inactive");
    assert_eq!(text_of(one(doc, "#live [data-dictation-timer]")), "0:00");
    let text = one(doc, "#live textarea#live-transcript[data-dictation-output]");
    assert!(text.value().attr("name").is_none(), "the transcript isn't part of any form");
    assert!(one(doc, "label[for=live-transcript]").value().classes().any(|c| c == "sr-only"));
}

#[tokio::test]
async fn the_live_panel_before_and_after_whisper_base() {
    let app = TestApp::new();
    let doc = Html::parse_document(&app.get("/transcribe").await.body);
    check_panel_common(&doc);
    // Not ready: the reason, a link to Whisper Base, and no Start.
    let problem = one(&doc, "#live [data-dictation-problem]");
    assert!(problem.value().attr("hidden").is_none());
    assert!(text_of(problem).contains("Whisper Base isn't downloaded yet."), "{}", text_of(problem));
    assert!(problem.select(&sel("a[href=\"#whisper-base\"]")).next().is_some());
    let start = one(&doc, "#live button[data-dictation-action=start]");
    assert!(start.value().attr("disabled").is_some() && start.value().attr("data-dictation-unavailable").is_some());
    assert!(text_of(one(&doc, "#live [data-dictation-message]")).contains("can start once Whisper Base is ready"));

    fake_whisper_base(&app).await;
    let doc = Html::parse_document(&app.get("/transcribe").await.body);
    check_panel_common(&doc);
    assert!(one(&doc, "#live [data-dictation-problem]").value().attr("hidden").is_some());
    let start = one(&doc, "#live button[data-dictation-action=start]");
    assert!(start.value().attr("disabled").is_none() && start.value().attr("data-dictation-unavailable").is_none());
    assert_eq!(
        text_of(one(&doc, "#live [data-dictation-message]")),
        "Inactive. Press Start and speak. The text appears below as you talk."
    );
    // The spoken language: detected, or any Whisper language.
    let options: Vec<_> = doc.select(&sel("#live select[data-dictation-language] option")).collect();
    assert!(options.len() > 50, "{} languages", options.len());
    assert_eq!(options[0].value().attr("value"), Some("auto"));
    assert!(options[0].value().attr("selected").is_some());
    assert!(options.iter().any(|o| o.value().attr("value") == Some("en")));
}

/// The data attributes the script works with exist in the pages it works on, and the
/// stylesheet has rules for the states it sets.
#[tokio::test]
async fn the_script_and_the_pages_agree() {
    let (app, _ollama) = app_with_chat_model(Upstreams::offline()).await;
    fake_whisper_base(&app).await;
    let js = app.get("/assets/app.js").await.body;
    let css = app.get("/assets/app.css").await.body;
    let chat = app.get("/chat?new=1").await.body;
    let speech = app.get("/transcribe").await.body;
    for (attr, pages) in [
        ("data-dictation-output", &[&chat, &speech][..]),
        ("data-dictation-status", &[&chat, &speech]),
        ("data-dictation-message", &[&chat, &speech]),
        ("data-dictation-timer", &[&chat, &speech]),
        ("data-dictation-busy", &[&chat]),
        ("data-dictation-label", &[&speech]),
        ("data-dictation-language", &[&speech]),
        ("data-dictation-problem", &[&speech]),
        ("data-dictation-problem-text", &[&speech]),
        ("data-mic-label", &[&chat]),
        ("data-pause-label", &[&chat, &speech]),
        ("data-mic-reason", &[&chat]),
        ("data-enter-send", &[&chat]),
    ] {
        assert!(js.contains(&format!("[{attr}]")) || js.contains(&format!("\"{attr}\"")), "app.js doesn't use {attr}");
        for page in pages {
            assert!(page.contains(attr), "a page lacks {attr}");
        }
    }
    for state in ["idle", "requesting", "loading", "recording", "paused", "finishing", "done", "error", "unavailable"] {
        assert!(js.contains(&format!("\"{state}\"")), "app.js never sets {state}");
    }
    for state in ["requesting", "loading", "recording", "paused", "finishing", "error"] {
        assert!(css.contains(&format!("[data-state=\"{state}\"]")), "no style for {state}");
    }
    for class in [
        ".mic-toggle",
        ".mic-pause",
        ".mic-dot",
        ".mic-unavailable",
        ".mic-error",
        ".is-empty",
        ".live-start",
        ".live-pause",
        ".live-stop",
        ".live-state",
    ] {
        assert!(css.contains(class), "no rule for {class}");
    }
    let start = css.find("/* ---------- Voice input: chat microphone and live transcription panel").unwrap();
    let end = css.find("/* end of Voice input */").unwrap();
    let section = &css[start..end];
    assert!(section.contains("prefers-reduced-motion") && section.contains("max-width: 640px"));
    // Red only through the theme's danger colour; no colours written out.
    for bad in ["#", "rgb(", "hsl(", "red;", "white;", "black;"] {
        let found = section.lines().filter(|l| !l.trim_start().starts_with("/*")).any(|l| l.contains(bad));
        assert!(!found, "the voice input styles write out a colour ({bad})");
    }
}
