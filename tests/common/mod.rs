//! Shared test harness: an in-process app (real router, real database in a temp folder) and
//! small fake upstream servers that speak the Hugging Face and Ollama HTTP APIs, so download,
//! discovery and chat flows run end to end without internet access.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use http_body_util::BodyExt;
use local_ai_studio::SharedState;
use local_ai_studio::app::{AppOptions, AppState};
use serde_json::{Value, json};
use sha1::Digest as _;
use tower::ServiceExt;

pub const HOST: &str = "127.0.0.1:8765";
pub const ORIGIN: &str = "http://127.0.0.1:8765";
pub const VALID_TOKEN: &str = "hf_TestTokenThatIsValid0123456789";

/// A URL where nothing listens (connection refused), standing in for "no internet" or
/// "Ollama not running".
pub fn closed_url() -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    drop(l);
    format!("http://127.0.0.1:{port}")
}

/// Where the app's outgoing requests go.
#[derive(Clone)]
pub struct Upstreams {
    pub hf: String,
    pub ollama: String,
    pub library: String,
    pub registry: String,
}

impl Upstreams {
    /// Everything unreachable.
    pub fn offline() -> Self {
        Self { hf: closed_url(), ollama: closed_url(), library: closed_url(), registry: closed_url() }
    }
}

pub struct Resp {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
}

impl Resp {
    pub fn location(&self) -> String {
        self.headers.get(header::LOCATION).and_then(|v| v.to_str().ok()).unwrap_or_default().to_string()
    }

    /// The flash message set by a redirect, as `(kind, message)`.
    pub fn flash(&self) -> Option<(String, String)> {
        for v in self.headers.get_all(header::SET_COOKIE) {
            let v = v.to_str().ok()?;
            if let Some(rest) = v.strip_prefix("las_flash=") {
                let value = rest.split(';').next().unwrap_or("");
                if value.is_empty() {
                    continue;
                }
                let decoded = percent_encoding::percent_decode_str(value).decode_utf8().ok()?.to_string();
                let (k, m) = decoded.split_once('|')?;
                return Some((k.to_string(), m.to_string()));
            }
        }
        None
    }

    pub fn flash_message(&self) -> String {
        self.flash().map(|f| f.1).unwrap_or_default()
    }

    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }
}

pub struct TestApp {
    pub dir: tempfile::TempDir,
    pub state: SharedState,
    pub up: Upstreams,
    router: Router,
}

impl TestApp {
    pub fn new() -> Self {
        Self::with(Upstreams::offline())
    }

    pub fn with(up: Upstreams) -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self::in_dir(dir, up)
    }

    /// Builds an app on an existing data folder (used to test restarts).
    pub fn in_dir(dir: tempfile::TempDir, up: Upstreams) -> Self {
        let state = AppState::build(AppOptions {
            data_dir: Some(dir.path().join("data")),
            use_keychain: false,
            hf_endpoint: Some(up.hf.clone()),
            ollama_url: Some(up.ollama.clone()),
            ollama_library: Some((up.library.clone(), up.registry.clone())),
            persist_settings: true,
            ignore_env_token: true,
        })
        .expect("building the app state");
        let router = local_ai_studio::web::router(state.clone());
        Self { dir, state, up, router }
    }

    /// Simulates a restart: drops this instance and builds a new one on the same data.
    pub async fn restart(self) -> Self {
        self.state.shutdown_gracefully().await;
        let TestApp { dir, up, .. } = self;
        Self::in_dir(dir, up)
    }

    pub async fn send(&self, mut req: Request<Body>) -> Resp {
        if !req.headers().contains_key(header::HOST) {
            req.headers_mut().insert(header::HOST, HeaderValue::from_static(HOST));
        }
        let resp = tokio::time::timeout(Duration::from_secs(60), self.router.clone().oneshot(req))
            .await
            .expect("request timed out")
            .unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = tokio::time::timeout(Duration::from_secs(60), resp.into_body().collect())
            .await
            .expect("response body timed out")
            .unwrap()
            .to_bytes();
        Resp { status, headers, body: String::from_utf8_lossy(&bytes).into_owned() }
    }

    pub async fn get(&self, path: &str) -> Resp {
        self.send(Request::get(path).body(Body::empty()).unwrap()).await
    }

    pub async fn post_form(&self, path: &str, form: &[(&str, &str)]) -> Resp {
        let body: String = form.iter().map(|(k, v)| format!("{}={}", enc(k), enc(v))).collect::<Vec<_>>().join("&");
        self.send(
            Request::post(path)
                .header(header::ORIGIN, ORIGIN)
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
    }

    /// Multipart form: text fields plus files `(field, file name, content type, bytes)`.
    pub async fn post_multipart(
        &self,
        path: &str,
        fields: &[(&str, &str)],
        files: &[(&str, &str, &str, Vec<u8>)],
    ) -> Resp {
        let boundary = "----LocalAIStudioTestBoundary7MA4YWxkTrZu0gW";
        let mut body: Vec<u8> = Vec::new();
        for (k, v) in fields {
            body.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes(),
            );
        }
        for (field, name, ctype, data) in files {
            body.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        self.send(
            Request::post(path)
                .header(header::ORIGIN, ORIGIN)
                .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
                .body(Body::from(body))
                .unwrap(),
        )
        .await
    }

    pub async fn post_json(&self, path: &str, body: Value) -> Resp {
        self.send(
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
    }

    pub fn config_text(&self) -> String {
        std::fs::read_to_string(&self.state.paths.config_file).unwrap_or_default()
    }
}

/// Checks one page: internal links resolve (or report an upstream outage, never 404), forms
/// post to routes that exist, every button submits a form or opens a pop-up on the page, and
/// every label points at a control.
pub async fn assert_page_is_wired(app: &TestApp, path: &str, checked_links: &mut HashSet<String>) {
    use scraper::{Html, Selector};
    let sel = |s: &str| Selector::parse(s).unwrap();
    let r = app.get(path).await;
    assert!(r.status.is_success(), "{path}: {}", r.status);
    let doc = Html::parse_document(&r.body);
    let ids: HashSet<String> =
        doc.select(&sel("[id]")).filter_map(|e| e.value().attr("id").map(String::from)).collect();
    let popovers: HashSet<String> =
        doc.select(&sel("[popover][id]")).filter_map(|e| e.value().attr("id").map(String::from)).collect();

    for a in doc.select(&sel("a[href]")) {
        let href = a.value().attr("href").unwrap();
        if !href.starts_with('/') || href.starts_with("//") {
            continue;
        }
        let target = href.split('#').next().unwrap().to_string();
        if target.is_empty() || !checked_links.insert(target.clone()) {
            continue;
        }
        let resp = app.get(&target).await;
        let upstream_outage = matches!(resp.status, StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            resp.status.is_success() || resp.status.is_redirection() || upstream_outage,
            "{path}: link {href} returned {}",
            resp.status
        );
    }

    // An OPTIONS request reaches a route without side effects: unknown routes answer 404,
    // known POST-only routes 405.
    for f in doc.select(&sel("form")) {
        let action = f.value().attr("action").unwrap_or(path);
        let method = f.value().attr("method").unwrap_or("get").to_lowercase();
        assert!(action.starts_with('/'), "{path}: form action {action} is not local");
        let target = action.split('?').next().unwrap();
        if method == "post" {
            let resp =
                app.send(Request::builder().method(Method::OPTIONS).uri(target).body(Body::empty()).unwrap()).await;
            assert_ne!(resp.status, StatusCode::NOT_FOUND, "{path}: form posts to a missing route {action}");
        } else {
            let resp = app.get(target).await;
            assert!(
                resp.status.is_success() || resp.status.is_redirection(),
                "{path}: GET form {action}: {}",
                resp.status
            );
        }
        for b in f.select(&sel("button[formaction]")) {
            let fa = b.value().attr("formaction").unwrap();
            let resp = app.send(Request::builder().method(Method::OPTIONS).uri(fa).body(Body::empty()).unwrap()).await;
            assert_ne!(resp.status, StatusCode::NOT_FOUND, "{path}: button posts to a missing route {fa}");
        }
    }

    for b in doc.select(&sel("button")) {
        let v = b.value();
        let label = b.text().collect::<String>().trim().to_string();
        if let Some(target) = v.attr("popovertarget") {
            assert!(popovers.contains(target), "{path}: button \"{label}\" opens missing pop-up #{target}");
            continue;
        }
        assert_eq!(v.attr("type").unwrap_or("submit"), "submit", "{path}: button \"{label}\" does nothing");
        let in_form = b.ancestors().any(|n| n.value().as_element().map(|e| e.name() == "form").unwrap_or(false));
        let form_attr = v.attr("form").map(|f| ids.contains(f)).unwrap_or(false);
        assert!(in_form || form_attr, "{path}: submit button \"{label}\" is outside any form");
    }

    for l in doc.select(&sel("label[for]")) {
        let id = l.value().attr("for").unwrap();
        assert!(ids.contains(id), "{path}: label for missing #{id}");
    }
}

pub fn enc(s: &str) -> String {
    percent_encoding::utf8_percent_encode(s, percent_encoding::NON_ALPHANUMERIC).to_string()
}

/// Polls `check` until it returns true or the time runs out.
pub async fn wait_until<F, Fut>(what: &str, secs: u64, mut check: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    loop {
        if check().await {
            return;
        }
        if tokio::time::Instant::now() > deadline {
            panic!("timed out waiting for: {what}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn serve(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// Deterministic pseudo-random bytes.
pub fn bytes_of(len: usize, seed: u8) -> Vec<u8> {
    let mut x: u32 = 0x9E37_79B9 ^ seed as u32;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x & 0xff) as u8
        })
        .collect()
}

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(data))
}

/// Git blob id of a small (non-LFS) file, as Hugging Face reports it.
pub fn git_blob_id(data: &[u8]) -> String {
    let mut h = sha1::Sha1::new();
    h.update(format!("blob {}\0", data.len()).as_bytes());
    h.update(data);
    hex::encode(h.finalize())
}

// ------------------------------------------------------------------------------------------
// Fake Hugging Face
// ------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct FakeFile {
    pub name: String,
    pub data: Arc<Vec<u8>>,
    /// Large (LFS) file: verified by SHA-256; otherwise by git blob id.
    pub lfs: bool,
    /// Report this checksum instead of the real one (simulates a corrupted upload).
    pub wrong_checksum: bool,
    /// Report this size instead of the real one.
    pub declared_size: Option<u64>,
}

impl FakeFile {
    pub fn lfs(name: &str, data: Vec<u8>) -> Self {
        Self { name: name.into(), data: Arc::new(data), lfs: true, wrong_checksum: false, declared_size: None }
    }
    pub fn small(name: &str, data: &[u8]) -> Self {
        Self {
            name: name.into(),
            data: Arc::new(data.to_vec()),
            lfs: false,
            wrong_checksum: false,
            declared_size: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct FakeRepo {
    pub id: String,
    pub gated: bool,
    pub private: bool,
    pub files: Vec<FakeFile>,
    /// Extra JSON fields merged into the model info (gguf, pipeline_tag, tags, config…).
    pub extra: Value,
    pub downloads: u64,
}

impl FakeRepo {
    pub fn new(id: &str) -> Self {
        Self { id: id.into(), extra: json!({}), ..Default::default() }
    }

    fn summary(&self) -> Value {
        let mut v = json!({
            "id": self.id,
            "author": self.id.split('/').next().unwrap_or(""),
            "downloads": self.downloads,
            "likes": 3,
            "lastModified": "2026-10-01T10:00:00.000Z",
            "createdAt": "2026-09-01T10:00:00.000Z",
            "tags": [],
            "gated": if self.gated { json!("manual") } else { json!(false) },
            "private": self.private,
        });
        if let (Some(obj), Some(extra)) = (v.as_object_mut(), self.extra.as_object()) {
            for (k, val) in extra {
                obj.insert(k.clone(), val.clone());
            }
        }
        v
    }

    fn info(&self) -> Value {
        let mut v = self.summary();
        let siblings: Vec<Value> = self
            .files
            .iter()
            .map(|f| {
                let size = f.declared_size.unwrap_or(f.data.len() as u64);
                if f.lfs {
                    let sha = if f.wrong_checksum { sha256_hex(b"something else") } else { sha256_hex(&f.data) };
                    json!({ "rfilename": f.name, "size": size, "blobId": git_blob_id(b"lfs pointer"), "lfs": { "sha256": sha, "size": size } })
                } else {
                    let oid = if f.wrong_checksum { git_blob_id(b"something else") } else { git_blob_id(&f.data) };
                    json!({ "rfilename": f.name, "size": size, "blobId": oid })
                }
            })
            .collect();
        let obj = v.as_object_mut().unwrap();
        obj.insert("sha".into(), json!("0123456789abcdef0123456789abcdef01234567"));
        obj.insert("siblings".into(), Value::Array(siblings));
        v
    }
}

#[derive(Default)]
pub struct HfState {
    pub repos: Mutex<HashMap<String, FakeRepo>>,
    /// File names whose next download is cut off after this many bytes.
    pub cut_after: Mutex<HashMap<String, usize>>,
    /// Pause between 64 KiB chunks (slow downloads, for pause/cancel tests).
    pub chunk_delay: Mutex<Option<Duration>>,
    /// Requests seen: `"<path> range=<range>"`.
    pub log: Mutex<Vec<String>>,
    /// When true every request fails with 503 (Hub outage).
    pub down: Mutex<bool>,
}

pub struct FakeHf {
    pub url: String,
    pub state: Arc<HfState>,
}

impl FakeHf {
    pub async fn start(repos: Vec<FakeRepo>) -> Self {
        let state = Arc::new(HfState::default());
        {
            let mut r = state.repos.lock().unwrap();
            for repo in repos {
                r.insert(repo.id.clone(), repo);
            }
        }
        let router = Router::new()
            .route("/api/models", get(hf_search))
            .route("/api/models/{owner}/{name}", get(hf_info))
            .route("/api/whoami-v2", get(hf_whoami))
            .route("/{owner}/{name}/resolve/{rev}/{*path}", get(hf_file))
            .with_state(state.clone());
        let url = serve(router).await;
        Self { url, state }
    }

    pub fn requests_for(&self, file: &str) -> Vec<String> {
        self.state.log.lock().unwrap().iter().filter(|l| l.contains(file)).cloned().collect()
    }
}

fn authorized(headers: &HeaderMap) -> bool {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|v| v == format!("Bearer {VALID_TOKEN}"))
        .unwrap_or(false)
}

fn hf_error(status: StatusCode, msg: &str) -> Response {
    (status, axum::Json(json!({ "error": msg }))).into_response()
}

async fn hf_search(
    State(st): State<Arc<HfState>>,
    Query(q): Query<HashMap<String, String>>,
    headers: HeaderMap,
) -> Response {
    if *st.down.lock().unwrap() {
        return hf_error(StatusCode::SERVICE_UNAVAILABLE, "maintenance");
    }
    let text = q.get("search").cloned().unwrap_or_default().to_lowercase();
    let limit: usize = q.get("limit").and_then(|l| l.parse().ok()).unwrap_or(30);
    let offset: usize = q.get("cursor").and_then(|c| c.parse().ok()).unwrap_or(0);
    let mut all: Vec<FakeRepo> = st
        .repos
        .lock()
        .unwrap()
        .values()
        .filter(|r| !r.private && r.id.to_lowercase().contains(&text))
        .cloned()
        .collect();
    all.sort_by(|a, b| b.downloads.cmp(&a.downloads).then(a.id.cmp(&b.id)));
    let page: Vec<Value> = all.iter().skip(offset).take(limit).map(|r| r.summary()).collect();
    let mut resp = axum::Json(Value::Array(page)).into_response();
    if offset + limit < all.len() {
        let host = headers.get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("localhost");
        let link =
            format!("<http://{host}/api/models?search={text}&limit={limit}&cursor={}>; rel=\"next\"", offset + limit);
        resp.headers_mut().insert(header::LINK, HeaderValue::from_str(&link).unwrap());
    }
    resp.headers_mut().insert("ratelimit", HeaderValue::from_static("\"api\";r=480;t=200"));
    resp
}

async fn hf_info(
    State(st): State<Arc<HfState>>,
    Path((owner, name)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    if *st.down.lock().unwrap() {
        return hf_error(StatusCode::SERVICE_UNAVAILABLE, "maintenance");
    }
    let id = format!("{owner}/{name}");
    let repo = st.repos.lock().unwrap().get(&id).cloned();
    match repo {
        None => hf_error(StatusCode::NOT_FOUND, "Repository not found"),
        Some(r) if r.private && !authorized(&headers) => {
            hf_error(StatusCode::UNAUTHORIZED, "Invalid username or password.")
        }
        Some(r) => axum::Json(r.info()).into_response(),
    }
}

async fn hf_whoami(headers: HeaderMap) -> Response {
    if authorized(&headers) {
        axum::Json(json!({ "name": "test-user", "type": "user" })).into_response()
    } else {
        hf_error(StatusCode::UNAUTHORIZED, "Invalid credentials in Authorization header")
    }
}

async fn hf_file(
    State(st): State<Arc<HfState>>,
    Path((owner, name, _rev, path)): Path<(String, String, String, String)>,
    headers: HeaderMap,
) -> Response {
    let range = headers.get(header::RANGE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    st.log.lock().unwrap().push(format!("{owner}/{name}/{path} range={range}"));
    if *st.down.lock().unwrap() {
        return hf_error(StatusCode::SERVICE_UNAVAILABLE, "maintenance");
    }
    let id = format!("{owner}/{name}");
    let Some(repo) = st.repos.lock().unwrap().get(&id).cloned() else {
        return hf_error(StatusCode::NOT_FOUND, "Repository not found");
    };
    if (repo.gated || repo.private) && !authorized(&headers) {
        return hf_error(
            StatusCode::FORBIDDEN,
            &format!(
                "Access to model {id} is restricted. You must have access to it and be authenticated to access it."
            ),
        );
    }
    let Some(file) = repo.files.iter().find(|f| f.name == path).cloned() else {
        return hf_error(StatusCode::NOT_FOUND, "Entry not found");
    };
    let total = file.data.len();
    let start = range
        .strip_prefix("bytes=")
        .and_then(|r| r.split('-').next())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    if start > total || (start == total && total > 0) {
        let mut r = StatusCode::RANGE_NOT_SATISFIABLE.into_response();
        r.headers_mut().insert(header::CONTENT_RANGE, HeaderValue::from_str(&format!("bytes */{total}")).unwrap());
        return r;
    }
    let cut = st.cut_after.lock().unwrap().remove(&file.name);
    let delay = *st.chunk_delay.lock().unwrap();
    let data = file.data.clone();
    let stream = async_stream(data, start, cut, delay);
    let mut resp = Response::new(Body::from_stream(stream));
    let h = resp.headers_mut();
    h.insert(header::CONTENT_LENGTH, HeaderValue::from(total - start));
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/octet-stream"));
    if start > 0 {
        *resp.status_mut() = StatusCode::PARTIAL_CONTENT;
        resp.headers_mut().insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&format!("bytes {start}-{}/{total}", total - 1)).unwrap(),
        );
    }
    resp
}

/// Streams `data[start..]` in 64 KiB chunks, optionally slowly, optionally failing after
/// `cut` bytes (the connection is then dropped mid-body).
fn async_stream(
    data: Arc<Vec<u8>>,
    start: usize,
    cut: Option<usize>,
    delay: Option<Duration>,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send {
    futures_util::stream::unfold((start, 0usize), move |(pos, sent)| {
        let data = data.clone();
        async move {
            if pos >= data.len() {
                return None;
            }
            if let Some(c) = cut
                && sent >= c
            {
                return Some((
                    Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "simulated network drop")),
                    (data.len(), sent),
                ));
            }
            if let Some(d) = delay {
                tokio::time::sleep(d).await;
            }
            let mut end = (pos + 64 * 1024).min(data.len());
            if let Some(c) = cut {
                end = end.min(pos + (c - sent).max(1));
            }
            let chunk = Bytes::copy_from_slice(&data[pos..end]);
            Some((Ok(chunk), (end, sent + (end - pos))))
        }
    })
}

// ------------------------------------------------------------------------------------------
// Fake Ollama
// ------------------------------------------------------------------------------------------

#[derive(Default)]
pub struct OllamaState {
    /// Installed models: name → capabilities.
    pub models: Mutex<Vec<(String, Vec<String>)>>,
    /// Text chunks streamed for every chat request.
    pub reply: Mutex<Vec<String>>,
    pub chunk_delay: Mutex<Duration>,
    /// Chat request bodies received.
    pub chats: Mutex<Vec<Value>>,
    /// Models pulled.
    pub pulls: Mutex<Vec<String>>,
    /// Error message for pulls (e.g. unknown model).
    pub pull_error: Mutex<Option<String>>,
    /// Loaded models (`/api/ps`).
    pub running: Mutex<Vec<String>>,
}

pub struct FakeOllama {
    pub url: String,
    pub state: Arc<OllamaState>,
}

impl FakeOllama {
    pub async fn start(models: &[(&str, &[&str])]) -> Self {
        let state = Arc::new(OllamaState::default());
        *state.models.lock().unwrap() =
            models.iter().map(|(n, caps)| (n.to_string(), caps.iter().map(|c| c.to_string()).collect())).collect();
        *state.reply.lock().unwrap() =
            vec!["Hello".into(), " from".into(), " the".into(), " fake".into(), " Ollama!".into()];
        let router = Router::new()
            .route("/api/version", get(|| async { axum::Json(json!({ "version": "0.12.6" })) }))
            .route("/api/tags", get(ol_tags))
            .route("/api/ps", get(ol_ps))
            .route("/api/show", post(ol_show))
            .route("/api/chat", post(ol_chat))
            .route("/api/pull", post(ol_pull))
            .route("/api/delete", delete(ol_delete))
            .route("/api/generate", post(ol_generate))
            .with_state(state.clone());
        let url = serve(router).await;
        Self { url, state }
    }

    pub fn set_reply(&self, chunks: &[&str], delay: Duration) {
        *self.state.reply.lock().unwrap() = chunks.iter().map(|c| c.to_string()).collect();
        *self.state.chunk_delay.lock().unwrap() = delay;
    }

    pub fn last_chat(&self) -> Value {
        self.state.chats.lock().unwrap().last().cloned().unwrap_or(Value::Null)
    }
}

async fn ol_tags(State(st): State<Arc<OllamaState>>) -> axum::Json<Value> {
    let models: Vec<Value> = st
        .models
        .lock()
        .unwrap()
        .iter()
        .map(|(n, _)| {
            json!({
                "name": n, "model": n, "modified_at": "2026-10-01T10:00:00Z", "size": 523_000_000u64,
                "digest": format!("{:0>64}", n.len()),
                "details": { "format": "gguf", "family": "qwen3", "parameter_size": "751.63M", "quantization_level": "Q4_K_M" }
            })
        })
        .collect();
    axum::Json(json!({ "models": models }))
}

async fn ol_ps(State(st): State<Arc<OllamaState>>) -> axum::Json<Value> {
    let models: Vec<Value> = st
        .running
        .lock()
        .unwrap()
        .iter()
        .map(|n| json!({ "name": n, "model": n, "size": 1_000_000u64, "size_vram": 500_000u64, "expires_at": "2026-11-10T12:00:00Z" }))
        .collect();
    axum::Json(json!({ "models": models }))
}

fn requested_model(body: &Value) -> String {
    body.get("model").or_else(|| body.get("name")).and_then(|v| v.as_str()).unwrap_or_default().to_string()
}

async fn ol_show(State(st): State<Arc<OllamaState>>, axum::Json(body): axum::Json<Value>) -> Response {
    let name = requested_model(&body);
    let caps = st.models.lock().unwrap().iter().find(|(n, _)| *n == name).map(|(_, c)| c.clone());
    match caps {
        None => (StatusCode::NOT_FOUND, axum::Json(json!({ "error": format!("model '{name}' not found") }))).into_response(),
        Some(caps) => axum::Json(json!({
            "license": "Apache License 2.0",
            "parameters": "temperature 0.6\nstop \"<|im_end|>\"",
            "template": "{{ .Prompt }}",
            "details": { "format": "gguf", "family": "qwen3", "parameter_size": "751.63M", "quantization_level": "Q4_K_M" },
            "model_info": { "general.architecture": "qwen3", "general.parameter_count": 751_632_384u64, "qwen3.context_length": 40960 },
            "capabilities": caps,
            "modified_at": "2026-10-01T10:00:00Z"
        }))
        .into_response(),
    }
}

async fn ol_chat(State(st): State<Arc<OllamaState>>, axum::Json(body): axum::Json<Value>) -> Response {
    let name = requested_model(&body);
    if !st.models.lock().unwrap().iter().any(|(n, _)| *n == name) {
        return (
            StatusCode::NOT_FOUND,
            axum::Json(json!({ "error": format!("model '{name}' not found, try pulling it first") })),
        )
            .into_response();
    }
    st.chats.lock().unwrap().push(body);
    let chunks = st.reply.lock().unwrap().clone();
    let delay = *st.chunk_delay.lock().unwrap();
    let n = chunks.len();
    let stream = futures_util::stream::unfold(0usize, move |i| {
        let chunks = chunks.clone();
        async move {
            if i > n {
                return None;
            }
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let line = if i < n {
                json!({ "model": "x", "message": { "role": "assistant", "content": chunks[i] }, "done": false })
            } else {
                json!({ "model": "x", "message": { "role": "assistant", "content": "" }, "done": true, "done_reason": "stop",
                        "prompt_eval_count": 12, "eval_count": n, "eval_duration": 250_000_000u64, "total_duration": 300_000_000u64 })
            };
            Some((Ok::<_, std::io::Error>(Bytes::from(format!("{line}\n"))), i + 1))
        }
    });
    let mut r = Response::new(Body::from_stream(stream));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/x-ndjson"));
    r
}

async fn ol_pull(State(st): State<Arc<OllamaState>>, axum::Json(body): axum::Json<Value>) -> Response {
    let name = requested_model(&body);
    st.pulls.lock().unwrap().push(name.clone());
    let mut lines = vec![json!({ "status": "pulling manifest" })];
    if let Some(err) = st.pull_error.lock().unwrap().clone() {
        lines.push(json!({ "error": err }));
    } else {
        for done in [0u64, 400, 1000] {
            lines.push(json!({ "status": "pulling 6a0746a1ec1a", "digest": "sha256:6a0746a1ec1a", "total": 1000, "completed": done }));
        }
        lines.push(json!({ "status": "verifying sha256 digest" }));
        lines.push(json!({ "status": "success" }));
        let mut models = st.models.lock().unwrap();
        if !models.iter().any(|(n, _)| *n == name) {
            models.push((name, vec!["completion".into()]));
        }
    }
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    ([(header::CONTENT_TYPE, "application/x-ndjson")], text).into_response()
}

async fn ol_delete(State(st): State<Arc<OllamaState>>, axum::Json(body): axum::Json<Value>) -> Response {
    let name = requested_model(&body);
    let mut models = st.models.lock().unwrap();
    let before = models.len();
    models.retain(|(n, _)| *n != name);
    if models.len() == before {
        (StatusCode::NOT_FOUND, axum::Json(json!({ "error": format!("model '{name}' not found") }))).into_response()
    } else {
        StatusCode::OK.into_response()
    }
}

async fn ol_generate(State(st): State<Arc<OllamaState>>, axum::Json(body): axum::Json<Value>) -> Response {
    let name = requested_model(&body);
    let unload = body.get("keep_alive").and_then(|v| v.as_i64()) == Some(0);
    let mut running = st.running.lock().unwrap();
    running.retain(|n| *n != name);
    if !unload {
        running.push(name.clone());
    }
    axum::Json(
        json!({ "model": name, "response": "", "done": true, "done_reason": if unload { "unload" } else { "load" } }),
    )
    .into_response()
}

// ------------------------------------------------------------------------------------------
// Fake ollama.com
// ------------------------------------------------------------------------------------------

pub const LIBRARY_SEARCH_HTML: &str = r#"<html><body><ul id="searchresults">
<li><a href="/library/qwen3"><div><h2><span>qwen3</span></h2><span title="Updated October 1, 2026">1 month ago</span></div>
<div><span class="rounded-md">tools</span><span class="rounded-md">thinking</span></div>
<p>Qwen3 is the latest generation of large language models in the Qwen series.</p>
<div><span title="9,876,543 downloads"><span>9.9M</span></span><span><span>35 tags</span></span><span class="rounded-md">0.6b</span><span class="rounded-md">8b</span></div></a></li>
<li><a href="/library/gemma3"><div><h2><span>gemma3</span></h2><span title="Updated September 2, 2026">2 months ago</span></div>
<div><span class="rounded-md">vision</span></div><p>The current, most capable model that runs on a single GPU.</p>
<div><span title="1,234,567 downloads"><span>1.2M</span></span><span class="rounded-md">1b</span><span class="rounded-md">4b</span></div></a></li>
</ul></body></html>"#;

pub const LIBRARY_TAGS_HTML: &str = r#"<html><head><meta name="description" content="Qwen3 is the latest generation of large language models."></head><body>
<div class="group px-4 py-3"><a href="/library/qwen3:0.6b"><span>qwen3:0.6b</span>
<span><span>7df6b6e09427</span> • 523MB • 40K context window • Text input • 1 month ago</span></a></div>
<div class="group px-4 py-3"><a href="/library/qwen3:8b"><span>qwen3:8b</span>
<span><span>500a1f067a9f</span> • 5.2GB • 40K context window • Text input • 1 month ago</span></a></div>
</body></html>"#;

pub async fn fake_library() -> String {
    // Like the real ollama.com, the tags page is a bare JSON list of names for clients that
    // accept JSON, and the full HTML page otherwise.
    async fn tags(headers: HeaderMap) -> Response {
        let accept = headers.get(header::ACCEPT).and_then(|v| v.to_str().ok()).unwrap_or("");
        if accept.contains("application/json") {
            axum::Json(json!({ "tags": ["latest", "0.6b", "8b"] })).into_response()
        } else {
            axum::response::Html(LIBRARY_TAGS_HTML).into_response()
        }
    }
    let router = Router::new()
        .route("/search", get(|| async { axum::response::Html(LIBRARY_SEARCH_HTML) }))
        .route("/library/qwen3/tags", get(tags));
    serve(router).await
}

// ------------------------------------------------------------------------------------------
// Tiny GGUF writer (for compatibility tests without real models)
// ------------------------------------------------------------------------------------------

pub enum GgufValue {
    U32(u32),
    Str(String),
    StrArray(Vec<String>),
    F32Array(Vec<f32>),
}

/// Writes a minimal, valid GGUF v3 file with the given metadata and one small F32 tensor.
pub fn write_gguf(path: &std::path::Path, kv: &[(&str, GgufValue)]) {
    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend_from_slice(&(v.len() as u64).to_le_bytes());
        out.extend_from_slice(v.as_bytes());
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes()); // tensors
    out.extend_from_slice(&(kv.len() as u64 + 1).to_le_bytes()); // + general.alignment
    s(&mut out, "general.alignment");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());
    for (k, v) in kv {
        s(&mut out, k);
        match v {
            GgufValue::U32(x) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            GgufValue::Str(x) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                s(&mut out, x);
            }
            GgufValue::StrArray(items) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&8u32.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for i in items {
                    s(&mut out, i);
                }
            }
            GgufValue::F32Array(items) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for i in items {
                    out.extend_from_slice(&i.to_le_bytes());
                }
            }
        }
    }
    // Tensor info: name, n_dims, dims, type (F32 = 0), offset.
    s(&mut out, "token_embd.weight");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&4u64.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    while out.len() % 32 != 0 {
        out.push(0);
    }
    for _ in 0..4 {
        out.extend_from_slice(&1.0f32.to_le_bytes());
    }
    std::fs::write(path, out).unwrap();
}
