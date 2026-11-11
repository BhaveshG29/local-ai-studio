//! Ollama integration: the local Ollama server API (list, show, pull, delete, chat, load/unload)
//! and the public ollama.com library catalog.
//!
//! ollama.com has no official catalog API, so the library pages are parsed defensively with
//! text patterns rather than styling classes, cached locally, and shown with a clear error
//! (plus cached results) when parsing or the network fails.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::config::Settings;
use crate::db::{self, Db};
use crate::error::{AppError, AppResult};

pub const LIBRARY_BASE: &str = "https://ollama.com";
pub const REGISTRY_BASE: &str = "https://registry.ollama.ai";

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub families: Option<Vec<String>>,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LocalModel {
    pub name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: ModelDetails,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunningModel {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub size_vram: u64,
    #[serde(default)]
    pub expires_at: String,
    #[serde(default)]
    pub context_length: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShowResponse {
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub modelfile: String,
    #[serde(default)]
    pub parameters: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub system: String,
    #[serde(default)]
    pub details: ModelDetails,
    #[serde(default)]
    pub model_info: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub modified_at: String,
}

impl ShowResponse {
    pub fn architecture(&self) -> Option<String> {
        self.model_info.get("general.architecture").and_then(|v| v.as_str()).map(String::from)
    }
    pub fn parameter_count(&self) -> Option<u64> {
        self.model_info.get("general.parameter_count").and_then(|v| v.as_u64())
    }
    pub fn context_length(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.model_info.get(&format!("{arch}.context_length")).and_then(|v| v.as_u64())
    }
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }
}

/// Whether Ollama is installed and/or running.
#[derive(Clone, Debug, Serialize)]
pub struct OllamaStatus {
    pub enabled: bool,
    pub base_url: String,
    pub running: bool,
    pub version: Option<String>,
    pub binary: Option<PathBuf>,
    pub error: Option<String>,
}

impl OllamaStatus {
    pub fn installed(&self) -> bool {
        self.running || self.binary.is_some()
    }
}

/// Platform-specific instructions shown when Ollama is missing or stopped.
pub fn install_instructions() -> (&'static str, Vec<&'static str>) {
    if cfg!(target_os = "macos") {
        (
            "macOS",
            vec![
                "Download Ollama from https://ollama.com/download/mac and move it to Applications.",
                "Open Ollama once; it keeps running in the menu bar.",
                "Or with Homebrew: brew install ollama && brew services start ollama",
            ],
        )
    } else if cfg!(windows) {
        (
            "Windows",
            vec![
                "Download the installer from https://ollama.com/download/windows and run it.",
                "Ollama starts automatically and appears in the system tray.",
            ],
        )
    } else {
        (
            "Linux",
            vec![
                "Install with: curl -fsSL https://ollama.com/install.sh | sh",
                "Start it with: sudo systemctl start ollama   (or run: ollama serve)",
            ],
        )
    }
}

fn known_binary_locations() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if cfg!(target_os = "macos") {
        v.push(PathBuf::from("/Applications/Ollama.app/Contents/Resources/ollama"));
        v.push(PathBuf::from("/usr/local/bin/ollama"));
        v.push(PathBuf::from("/opt/homebrew/bin/ollama"));
    } else if cfg!(windows) {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            v.push(PathBuf::from(local).join("Programs").join("Ollama").join("ollama.exe"));
        }
    } else {
        v.push(PathBuf::from("/usr/local/bin/ollama"));
        v.push(PathBuf::from("/usr/bin/ollama"));
        v.push(PathBuf::from("/snap/bin/ollama"));
    }
    v
}

pub fn find_binary() -> Option<PathBuf> {
    crate::util::find_executable("ollama").or_else(|| known_binary_locations().into_iter().find(|p| p.is_file()))
}

/// One line of the `/api/pull` progress stream.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct PullProgress {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default)]
    pub completed: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Messages for `/api/chat`.
#[derive(Clone, Debug, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct ChatOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub num_predict: i64,
    pub repeat_penalty: f32,
    pub num_ctx: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatChunkMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub thinking: Option<String>,
}

/// One line of the `/api/chat` stream.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub message: Option<ChatChunkMessage>,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Splits a byte stream into complete lines (NDJSON).
#[derive(Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
}

impl LineBuffer {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let s = String::from_utf8_lossy(&line).trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
        }
        out
    }
    pub fn finish(&mut self) -> Option<String> {
        let s = String::from_utf8_lossy(&self.buf).trim().to_string();
        self.buf.clear();
        (!s.is_empty()).then_some(s)
    }
}

#[derive(Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    settings: Arc<RwLock<Settings>>,
    db: Db,
    library_base: String,
    registry_base: String,
    /// Capabilities per model digest (from `/api/show`), so listings stay fast.
    caps: Arc<std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>>,
}

impl OllamaClient {
    pub fn new(http: reqwest::Client, settings: Arc<RwLock<Settings>>, db: Db) -> Self {
        Self {
            http,
            settings,
            db,
            library_base: LIBRARY_BASE.into(),
            registry_base: REGISTRY_BASE.into(),
            caps: Default::default(),
        }
    }

    /// Capabilities (`completion`, `vision`, `tools`, `thinking`, `embedding`…) of installed
    /// models, keyed by model name. Unreachable models are simply missing from the map.
    pub async fn capabilities(&self, models: &[LocalModel]) -> std::collections::HashMap<String, Vec<String>> {
        let mut out = std::collections::HashMap::new();
        let mut missing = Vec::new();
        {
            let cache = self.caps.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            for m in models {
                match cache.get(&m.digest) {
                    Some(c) => {
                        out.insert(m.name.clone(), c.clone());
                    }
                    None => missing.push((m.name.clone(), m.digest.clone())),
                }
            }
        }
        let fetched: Vec<_> = futures_util::stream::iter(missing)
            .map(|(name, digest)| async move {
                let r = self.show(&name).await;
                (name, digest, r)
            })
            .buffer_unordered(4)
            .collect()
            .await;
        let mut cache = self.caps.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for (name, digest, r) in fetched {
            if let Ok(s) = r {
                cache.insert(digest, s.capabilities.clone());
                out.insert(name, s.capabilities);
            }
        }
        out
    }

    /// Points library/registry lookups at another host (tests).
    pub fn with_library_base(mut self, library: &str, registry: &str) -> Self {
        self.library_base = library.trim_end_matches('/').to_string();
        self.registry_base = registry.trim_end_matches('/').to_string();
        self
    }

    fn settings(&self) -> crate::config::OllamaSettings {
        self.settings.read().unwrap_or_else(std::sync::PoisonError::into_inner).ollama.clone()
    }

    pub fn base_url(&self) -> String {
        self.settings().base_url
    }

    pub fn enabled(&self) -> bool {
        self.settings().enabled
    }

    fn not_running_error(&self) -> AppError {
        AppError::unavailable(format!(
            "Ollama isn't running at {}. Start Ollama (or install it from ollama.com), then try again.",
            self.base_url()
        ))
    }

    fn map_err(&self, e: reqwest::Error) -> AppError {
        if e.is_connect() {
            self.not_running_error()
        } else if e.is_timeout() {
            AppError::upstream("Ollama took too long to respond.")
        } else {
            AppError::upstream(format!("Error talking to Ollama: {e}"))
        }
    }

    async fn error_from(resp: reqwest::Response) -> AppError {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let msg = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
            .unwrap_or(body);
        if status == reqwest::StatusCode::NOT_FOUND {
            AppError::not_found(format!("Ollama: {msg}"))
        } else {
            AppError::upstream(format!("Ollama returned an error ({status}): {msg}"))
        }
    }

    /// Checks whether the server answers, without failing.
    pub async fn status(&self) -> OllamaStatus {
        let s = self.settings();
        let binary = tokio::task::spawn_blocking(find_binary).await.ok().flatten();
        if !s.enabled {
            return OllamaStatus {
                enabled: false,
                base_url: s.base_url,
                running: false,
                version: None,
                binary,
                error: None,
            };
        }
        match self.version().await {
            Ok(v) => OllamaStatus {
                enabled: true,
                base_url: s.base_url,
                running: true,
                version: Some(v),
                binary,
                error: None,
            },
            Err(e) => OllamaStatus {
                enabled: true,
                base_url: s.base_url,
                running: false,
                version: None,
                binary,
                error: Some(e.to_string()),
            },
        }
    }

    pub async fn version(&self) -> AppResult<String> {
        #[derive(Deserialize)]
        struct V {
            version: String,
        }
        let resp = self
            .http
            .get(format!("{}/api/version", self.base_url()))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp).await);
        }
        let v: V =
            resp.json().await.map_err(|_| AppError::upstream("The server at the Ollama address is not Ollama."))?;
        Ok(v.version)
    }

    pub async fn list_local(&self) -> AppResult<Vec<LocalModel>> {
        #[derive(Deserialize)]
        struct Tags {
            #[serde(default)]
            models: Vec<LocalModel>,
        }
        let resp = self
            .http
            .get(format!("{}/api/tags", self.base_url()))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp).await);
        }
        let mut tags: Tags =
            resp.json().await.map_err(|e| AppError::upstream(format!("Unexpected response from Ollama: {e}")))?;
        tags.models.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(tags.models)
    }

    pub async fn running(&self) -> AppResult<Vec<RunningModel>> {
        #[derive(Deserialize)]
        struct Ps {
            #[serde(default)]
            models: Vec<RunningModel>,
        }
        let resp = self
            .http
            .get(format!("{}/api/ps", self.base_url()))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp).await);
        }
        let ps: Ps =
            resp.json().await.map_err(|e| AppError::upstream(format!("Unexpected response from Ollama: {e}")))?;
        Ok(ps.models)
    }

    pub async fn show(&self, name: &str) -> AppResult<ShowResponse> {
        validate_model_name(name)?;
        let resp = self
            .http
            .post(format!("{}/api/show", self.base_url()))
            .json(&serde_json::json!({ "model": name }))
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp).await);
        }
        resp.json().await.map_err(|e| AppError::upstream(format!("Unexpected response from Ollama: {e}")))
    }

    pub async fn delete(&self, name: &str) -> AppResult<()> {
        validate_model_name(name)?;
        let resp = self
            .http
            .delete(format!("{}/api/delete", self.base_url()))
            .json(&serde_json::json!({ "model": name }))
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp).await);
        }
        Ok(())
    }

    /// Loads (keep_alive > 0) or unloads (keep_alive = 0) a model.
    pub async fn set_loaded(&self, name: &str, load: bool) -> AppResult<()> {
        validate_model_name(name)?;
        let keep_alive =
            if load { serde_json::Value::String(self.settings().keep_alive) } else { serde_json::Value::from(0) };
        let resp = self
            .http
            .post(format!("{}/api/generate", self.base_url()))
            .json(&serde_json::json!({ "model": name, "keep_alive": keep_alive }))
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp).await);
        }
        Ok(())
    }

    /// Pulls a model, reporting progress until done, failed or cancelled.
    pub async fn pull<F>(&self, name: &str, cancel: CancellationToken, mut on_progress: F) -> AppResult<()>
    where
        F: FnMut(&PullProgress) + Send,
    {
        validate_model_name(name)?;
        let resp = self
            .http
            .post(format!("{}/api/pull", self.base_url()))
            .json(&serde_json::json!({ "model": name, "stream": true }))
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp).await);
        }
        let mut stream = resp.bytes_stream();
        let mut lines = LineBuffer::default();
        let mut success = false;
        loop {
            let chunk = tokio::select! {
                _ = cancel.cancelled() => return Err(AppError::conflict("cancelled")),
                c = tokio::time::timeout(Duration::from_secs(300), stream.next()) => c,
            };
            let chunk = match chunk {
                Err(_) => {
                    return Err(AppError::upstream(
                        "Ollama stopped reporting progress for 5 minutes; the pull was abandoned.",
                    ));
                }
                Ok(None) => break,
                Ok(Some(Err(e))) => return Err(self.map_err(e)),
                Ok(Some(Ok(b))) => b,
            };
            for line in lines.push(&chunk) {
                let p: PullProgress = serde_json::from_str(&line).unwrap_or_default();
                if let Some(err) = &p.error {
                    return Err(AppError::upstream(friendly_pull_error(name, err)));
                }
                if p.status == "success" {
                    success = true;
                }
                on_progress(&p);
            }
        }
        if let Some(line) = lines.finish() {
            let p: PullProgress = serde_json::from_str(&line).unwrap_or_default();
            if let Some(err) = &p.error {
                return Err(AppError::upstream(friendly_pull_error(name, err)));
            }
            success |= p.status == "success";
        }
        if success {
            Ok(())
        } else {
            Err(AppError::upstream("The download ended before Ollama reported success. Try again to resume."))
        }
    }

    /// Starts a streaming chat request; returns the raw response for the backend to consume.
    pub async fn chat_request(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: &ChatOptions,
    ) -> AppResult<reqwest::Response> {
        validate_model_name(model)?;
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "options": options,
            "keep_alive": self.settings().keep_alive,
        });
        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url()))
            .json(&body)
            .send()
            .await
            .map_err(|e| self.map_err(e))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp).await);
        }
        Ok(resp)
    }

    // -----------------------------------------------------------------------------------------
    // ollama.com library
    // -----------------------------------------------------------------------------------------

    async fn fetch_cached(&self, url: &str, max_age: i64, force: bool) -> AppResult<(String, Option<String>)> {
        let key = format!("ollama:{url}");
        if !force {
            let k = key.clone();
            if let Ok(Some((body, _))) = self.db.call(move |c| db::catalog::cache_get(c, &k, Some(max_age))).await {
                return Ok((body, None));
            }
        }
        let live = async {
            let resp = self
                .http
                .get(url)
                .timeout(Duration::from_secs(20))
                // Ask for the HTML page only: ollama.com answers `/library/<name>/tags` with a
                // bare JSON list of tag names (no sizes) whenever JSON is acceptable.
                .header(reqwest::header::ACCEPT, "text/html,application/xhtml+xml;q=0.9")
                .send()
                .await
                .map_err(|e| {
                    if e.is_connect() || e.is_timeout() {
                        AppError::unavailable("Can't reach ollama.com. Check your internet connection.")
                    } else {
                        AppError::upstream(format!("Error loading the Ollama library: {e}"))
                    }
                })?;
            let status = resp.status();
            if status == reqwest::StatusCode::NOT_FOUND {
                return Err(AppError::not_found("That model is not in the Ollama library."));
            }
            if !status.is_success() {
                return Err(AppError::upstream(format!("ollama.com returned {status}")));
            }
            resp.text().await.map_err(|e| AppError::upstream(format!("Error reading ollama.com: {e}")))
        };
        match live.await {
            Ok(body) => {
                let k = key.clone();
                let b = body.clone();
                let now = crate::util::now_rfc3339();
                let _ = self.db.call(move |c| db::catalog::cache_put(c, &k, &b, &now)).await;
                Ok((body, None))
            }
            Err(e @ AppError::NotFound(_)) => Err(e),
            Err(e) => {
                let k = key.clone();
                match self.db.call(move |c| db::catalog::cache_get(c, &k, None)).await {
                    Ok(Some((body, at))) => Ok((body, Some(at))),
                    _ => Err(e),
                }
            }
        }
    }

    /// Searches the ollama.com library. `capability` is one of ollama.com's filters
    /// (`local`, `vision`, `tools`, `thinking`, `embedding`, `cloud`) or empty.
    pub async fn search_library(
        &self,
        query: &str,
        capability: &str,
        sort: &str,
        page: u32,
        force: bool,
    ) -> AppResult<LibraryPage> {
        let mut url =
            url::Url::parse(&format!("{}/search", self.library_base)).map_err(|e| AppError::Internal(e.into()))?;
        {
            let mut p = url.query_pairs_mut();
            if !query.trim().is_empty() {
                p.append_pair("q", query.trim());
            }
            if !capability.is_empty() {
                p.append_pair("c", capability);
            }
            if !sort.is_empty() && sort != "popular" {
                p.append_pair("o", sort);
            }
            if page > 1 {
                p.append_pair("page", &page.to_string());
            }
        }
        let (html, cached_at) = self.fetch_cached(url.as_str(), 3600, force).await?;
        let mut page_data = parse_search_page(&html);
        if page_data.models.is_empty() && html.len() > 2000 && !html.contains("No models found") {
            tracing::warn!("ollama.com search page parsed to zero results; the page layout may have changed");
        }
        page_data.cached_at = cached_at;
        let entries: Vec<_> = page_data.models.iter().map(crate::catalog::entry_from_ollama).collect();
        let _ = self.db.call_mut(move |c| db::catalog::upsert_many(c, &entries)).await;
        Ok(page_data)
    }

    /// Model description and all tags (variants) from ollama.com.
    pub async fn library_detail(&self, name: &str, force: bool) -> AppResult<LibraryDetail> {
        validate_library_name(name)?;
        let path = if name.contains('/') { format!("/{name}/tags") } else { format!("/library/{name}/tags") };
        let url = format!("{}{}", self.library_base, path);
        let (mut html, cached_at) = self.fetch_cached(&url, 6 * 3600, force).await?;
        let mut detail = parse_tags_page(name, &html);
        detail.cached_at = cached_at;
        if detail.tags.is_empty()
            && !force
            && let Ok((fresh, at)) = self.fetch_cached(&url, 6 * 3600, true).await
        {
            // The cached copy can't be read (e.g. stored by an older version): fetch it again.
            let d = parse_tags_page(name, &fresh);
            if !d.tags.is_empty() {
                detail = d;
                detail.cached_at = at;
            }
            html = fresh;
        }
        if detail.tags.is_empty() {
            // ollama.com may answer with its JSON tag list instead of the page (names only).
            detail.tags = parse_tags_json(name, &html);
        }
        if detail.tags.is_empty() {
            // Fall back to the registry's tag list (names only).
            if let Ok(tags) = self.registry_tags(name).await {
                detail.tags = tags
                    .into_iter()
                    .map(|t| LibraryTag { name: format!("{name}:{t}"), ..Default::default() })
                    .collect();
            }
        }
        if detail.tags.is_empty() {
            return Err(AppError::upstream("Couldn't read the list of versions for this model from ollama.com."));
        }
        Ok(detail)
    }

    async fn registry_tags(&self, name: &str) -> AppResult<Vec<String>> {
        #[derive(Deserialize)]
        struct TagList {
            #[serde(default)]
            tags: Vec<String>,
        }
        let repo = if name.contains('/') { name.to_string() } else { format!("library/{name}") };
        let resp = self
            .http
            .get(format!("{}/v2/{repo}/tags/list", self.registry_base))
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(|e| AppError::upstream(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(AppError::upstream(format!("registry returned {}", resp.status())));
        }
        let t: TagList = resp.json().await.map_err(|e| AppError::upstream(e.to_string()))?;
        Ok(t.tags)
    }
}

fn friendly_pull_error(name: &str, err: &str) -> String {
    if err.contains("file does not exist") || err.contains("not found") {
        format!("\"{name}\" was not found in the Ollama library. Check the name and tag.")
    } else if err.contains("no space left") {
        "Your disk is full. Free some space and try again.".to_string()
    } else {
        format!("Ollama could not download \"{name}\": {err}")
    }
}

/// Model names as accepted by Ollama (`name`, `name:tag`, `user/name:tag`, `hf.co/org/repo:Q4_K_M`).
pub fn validate_model_name(name: &str) -> AppResult<()> {
    let ok = !name.is_empty()
        && name.len() <= 300
        && !name.starts_with(['/', '.', '-'])
        && !name.contains("..")
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':' | '/'));
    if ok { Ok(()) } else { Err(AppError::bad_request(format!("\"{name}\" is not a valid Ollama model name."))) }
}

fn validate_library_name(name: &str) -> AppResult<()> {
    validate_model_name(name)?;
    if name.contains(':') || name.split('/').count() > 2 {
        return Err(AppError::bad_request("Invalid library model name."));
    }
    Ok(())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LibraryModel {
    /// `llama3.2` or `user/model` for community models.
    pub name: String,
    pub description: String,
    pub capabilities: Vec<String>,
    pub sizes: Vec<String>,
    pub pulls: u64,
    pub pulls_label: String,
    pub tag_count: Option<u32>,
    pub updated: String,
    pub updated_label: String,
}

impl LibraryModel {
    pub fn is_cloud_only(&self) -> bool {
        self.capabilities.iter().any(|c| c == "cloud") && self.sizes.is_empty()
    }
    pub fn has_capability(&self, c: &str) -> bool {
        self.capabilities.iter().any(|x| x == c)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LibraryPage {
    pub models: Vec<LibraryModel>,
    pub next_page: Option<u32>,
    pub cached_at: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LibraryTag {
    /// Full pull name, e.g. `llama3.2:3b-instruct-q4_K_M`.
    pub name: String,
    pub size_label: String,
    pub size_bytes: Option<u64>,
    pub context: String,
    pub inputs: String,
    pub digest: String,
    pub updated: String,
}

impl LibraryTag {
    pub fn tag(&self) -> &str {
        self.name.rsplit_once(':').map(|(_, t)| t).unwrap_or("latest")
    }
    pub fn has_image_input(&self) -> bool {
        self.inputs.to_lowercase().contains("image")
    }
    pub fn quantization(&self) -> Option<String> {
        crate::util::parse_quantization(self.tag())
    }
}

#[derive(Clone, Debug, Default)]
pub struct LibraryDetail {
    pub name: String,
    pub description: String,
    pub tags: Vec<LibraryTag>,
    pub cached_at: Option<String>,
}

fn sel(s: &str) -> scraper::Selector {
    scraper::Selector::parse(s).expect("static selector is valid")
}

fn element_text(el: scraper::ElementRef<'_>) -> String {
    el.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Parses a size label such as "7.8GB" or "815MB" into bytes (approximate, decimal units).
pub fn parse_size_label(s: &str) -> Option<u64> {
    static RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)").expect("valid regex"));
    let c = RE.captures(s)?;
    let v: f64 = c.get(1)?.as_str().parse().ok()?;
    let mult = match c.get(2)?.as_str().to_uppercase().as_str() {
        "KB" => 1e3,
        "MB" => 1e6,
        "GB" => 1e9,
        _ => 1e12,
    };
    Some((v * mult) as u64)
}

fn is_size_badge(s: &str) -> bool {
    static RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)^(?:\d+x)?e?\d+(?:\.\d+)?[kmbt]$").expect("valid regex"));
    RE.is_match(s)
}

/// Parses an ollama.com `/search` results page.
pub fn parse_search_page(html: &str) -> LibraryPage {
    let doc = scraper::Html::parse_document(html);
    let li_sel = sel("li");
    let a_sel = sel("a[href]");
    let h2_sel = sel("h2");
    let p_sel = sel("p");
    let span_sel = sel("span");
    let mut models = Vec::new();
    let mut next_page = None;
    static PAGE_RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"[?&]page=(\d+)").expect("valid regex"));
    static TAGS_RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"^(\d+)\s+tags?$").expect("valid regex"));

    for li in doc.select(&li_sel) {
        if let Some(hx) = li.value().attr("hx-get") {
            if let Some(c) = PAGE_RE.captures(hx) {
                next_page = c.get(1).and_then(|m| m.as_str().parse().ok());
            }
            continue;
        }
        let Some(a) = li.select(&a_sel).next() else { continue };
        let href = a.value().attr("href").unwrap_or("");
        let Some(name) = name_from_href(href) else { continue };
        let Some(h2) = a.select(&h2_sel).next() else { continue };
        let display = element_text(h2);
        let description = a.select(&p_sel).next().map(element_text).unwrap_or_default();
        let mut m = LibraryModel {
            name: if display.is_empty() { name.clone() } else { name },
            description,
            ..Default::default()
        };
        for span in a.select(&span_sel) {
            let title = span.value().attr("title").unwrap_or("");
            let text = element_text(span);
            if let Some(u) = title.strip_prefix("Updated ") {
                m.updated = u.to_string();
                m.updated_label = text.clone();
                continue;
            }
            if let Some(n) = title.strip_suffix(" downloads").or_else(|| title.strip_suffix(" pulls")) {
                m.pulls = n.replace(',', "").trim().parse().unwrap_or(0);
                continue;
            }
            if let Some(c) = TAGS_RE.captures(&text) {
                m.tag_count = c.get(1).and_then(|x| x.as_str().parse().ok());
                continue;
            }
            let is_badge = span.value().attr("class").map(|c| c.contains("rounded")).unwrap_or(false);
            if is_badge && !text.is_empty() && !text.contains(' ') {
                let t = text.to_lowercase();
                if is_size_badge(&t) {
                    if !m.sizes.contains(&t) {
                        m.sizes.push(t);
                    }
                } else if !m.capabilities.contains(&t) {
                    m.capabilities.push(t);
                }
            }
        }
        if m.pulls_label.is_empty() {
            m.pulls_label = crate::util::human_count(m.pulls);
        }
        if !models.iter().any(|x: &LibraryModel| x.name == m.name) {
            models.push(m);
        }
    }
    LibraryPage { models, next_page, cached_at: None }
}

/// `/library/llama3.2` → `llama3.2`; `/user/model` → `user/model`; other links → None.
fn name_from_href(href: &str) -> Option<String> {
    let path = href.split(['?', '#']).next()?;
    if let Some(n) = path.strip_prefix("/library/") {
        let n = n.trim_end_matches('/');
        return (!n.is_empty() && !n.contains('/') && !n.contains(':')).then(|| n.to_string());
    }
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    if parts.len() == 2
        && !parts[0].is_empty()
        && !parts[1].is_empty()
        && !matches!(parts[0], "blog" | "download" | "search" | "public" | "settings" | "models" | "docs")
        && !parts[1].contains(':')
    {
        return Some(format!("{}/{}", parts[0], parts[1]));
    }
    None
}

/// Parses an ollama.com `/library/<name>/tags` page.
pub fn parse_tags_page(name: &str, html: &str) -> LibraryDetail {
    let doc = scraper::Html::parse_document(html);
    let description = doc
        .select(&sel("meta[name=description]"))
        .next()
        .and_then(|m| m.value().attr("content"))
        .unwrap_or("")
        .to_string();
    static DIGEST_RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"\b([0-9a-f]{12})\b").expect("valid regex"));
    static CTX_RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)(\d+(?:\.\d+)?[KM]?)\s+context").expect("valid regex"));
    static INPUT_RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"([A-Za-z]+(?:,\s*[A-Za-z]+)*)\s+input").expect("valid regex"));
    static AGO_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)(\d+\s+\w+\s+ago|yesterday|just now|today)").expect("valid regex")
    });
    let prefix = if name.contains('/') { format!("/{name}:") } else { format!("/library/{name}:") };
    let mut tags: Vec<LibraryTag> = Vec::new();
    for a in doc.select(&sel("a[href]")) {
        let href = a.value().attr("href").unwrap_or("");
        let Some(tag) = href.strip_prefix(&prefix) else { continue };
        if tag.is_empty() || tag.contains('/') {
            continue;
        }
        let full = format!("{name}:{tag}");
        let text = element_text(a);
        // The compact (mobile) variant of each row carries all details as one line of text.
        let detailed = text.contains("context") || text.contains('•');
        if let Some(existing) = tags.iter_mut().find(|t| t.name == full) {
            if detailed && existing.size_label.is_empty() {
                fill_tag(existing, &text, &DIGEST_RE, &CTX_RE, &INPUT_RE, &AGO_RE);
            }
            continue;
        }
        let mut t = LibraryTag { name: full, ..Default::default() };
        if detailed {
            fill_tag(&mut t, &text, &DIGEST_RE, &CTX_RE, &INPUT_RE, &AGO_RE);
        }
        tags.push(t);
    }
    LibraryDetail { name: name.to_string(), description, tags, cached_at: None }
}

/// Parses ollama.com's JSON answer for `/library/<name>/tags` (`{"tags":["latest","0.6b",…]}`).
/// It only has names, so sizes and context lengths stay unknown.
pub fn parse_tags_json(name: &str, body: &str) -> Vec<LibraryTag> {
    #[derive(Deserialize)]
    struct Tags {
        tags: Vec<String>,
    }
    let Ok(t) = serde_json::from_str::<Tags>(body.trim()) else { return Vec::new() };
    t.tags
        .into_iter()
        .filter(|t| !t.is_empty() && !t.contains('/') && !t.contains(char::is_whitespace))
        .map(|t| LibraryTag { name: format!("{name}:{t}"), ..Default::default() })
        .collect()
}

fn fill_tag(
    t: &mut LibraryTag,
    text: &str,
    digest: &regex::Regex,
    ctx: &regex::Regex,
    input: &regex::Regex,
    ago: &regex::Regex,
) {
    if let Some(c) = digest.captures(text) {
        t.digest = c[1].to_string();
    }
    if let Some(bytes) = parse_size_label(text) {
        t.size_bytes = Some(bytes);
        static SZ: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
            regex::Regex::new(r"(?i)\d+(?:\.\d+)?\s*(?:KB|MB|GB|TB)").expect("valid regex")
        });
        t.size_label = SZ.find(text).map(|m| m.as_str().to_string()).unwrap_or_default();
    }
    if let Some(c) = ctx.captures(text) {
        t.context = c[1].to_string();
    }
    if let Some(c) = input.captures(text) {
        t.inputs = c[1].to_string();
    }
    if let Some(c) = ago.captures(text) {
        t.updated = c[1].to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEARCH_HTML: &str = r#"<html><body><ul id="searchresults">
<li class="border-t"><a href="/library/llama3.2-vision" class="group grid">
  <div><h2 title="llama3.2-vision"><span>llama3.2-vision</span></h2>
  <span class="text-muted" title="Updated May 22, 2025">1y</span></div>
  <div><span class="inline-flex rounded-md bg-indigo-50">vision</span></div>
  <p class="mt-1">Llama 3.2 Vision is a collection of instruction-tuned image reasoning generative models in 11B and 90B sizes.</p>
  <div><span class="inline-flex" title="5,616,244 downloads"><svg></svg><span>5.6M</span></span>
  <span class="inline-flex"><svg></svg><span>9 tags</span></span>
  <span class="inline-flex rounded-md bg-[#ddf4ff]">11b</span>
  <span class="inline-flex rounded-md bg-[#ddf4ff]">90b</span></div>
</a></li>
<li><a href="/srizon/pixie"><div><h2><span>srizon/pixie</span></h2><span title="Updated April 27, 2024">2y</span></div>
<div><span class="rounded-md">vision</span><span class="rounded-md">tools</span></div><p>Pixie model</p>
<div><span title="4,105 downloads"><span>4,105</span></span><span class="rounded-md">8x7b</span></div></a></li>
<li hx-get="/search?page=2&amp;q=llama" hx-trigger="revealed"></li>
</ul></body></html>"#;

    #[test]
    fn search_page_is_parsed() {
        let p = parse_search_page(SEARCH_HTML);
        assert_eq!(p.models.len(), 2);
        let m = &p.models[0];
        assert_eq!(m.name, "llama3.2-vision");
        assert_eq!(m.pulls, 5_616_244);
        assert_eq!(m.tag_count, Some(9));
        assert_eq!(m.capabilities, vec!["vision"]);
        assert_eq!(m.sizes, vec!["11b", "90b"]);
        assert_eq!(m.updated, "May 22, 2025");
        assert!(m.description.starts_with("Llama 3.2 Vision"));
        let c = &p.models[1];
        assert_eq!(c.name, "srizon/pixie");
        assert_eq!(c.sizes, vec!["8x7b"]);
        assert_eq!(c.capabilities, vec!["vision", "tools"]);
        assert_eq!(p.next_page, Some(2));
    }

    const TAGS_HTML: &str = r#"<html><head><meta name="description" content="Llama 3.2 Vision is a collection."></head><body>
<div class="group px-4 py-3">
  <a href="/library/llama3.2-vision:latest" class="md:hidden flex">
    <span>llama3.2-vision:latest</span>
    <span><span class="font-mono">6f2f9757ae97</span> • 7.8GB • 128K context window • <span>Text, Image input • 1 year ago</span></span>
  </a>
  <div class="hidden md:flex"><a href="/library/llama3.2-vision:latest">llama3.2-vision:latest</a><p>7.8GB</p></div>
</div>
<div class="group px-4 py-3">
  <a href="/library/llama3.2-vision:90b" class="md:hidden flex">
    <span>llama3.2-vision:90b</span>
    <span><span>d2a5e0c4b1f3</span> • 55GB • 128K context window • Text, Image input • 1 year ago</span>
  </a>
</div>
</body></html>"#;

    #[test]
    fn tags_page_is_parsed() {
        let d = parse_tags_page("llama3.2-vision", TAGS_HTML);
        assert_eq!(d.description, "Llama 3.2 Vision is a collection.");
        assert_eq!(d.tags.len(), 2);
        let t = &d.tags[0];
        assert_eq!(t.name, "llama3.2-vision:latest");
        assert_eq!(t.digest, "6f2f9757ae97");
        assert_eq!(t.size_label, "7.8GB");
        assert_eq!(t.size_bytes, Some(7_800_000_000));
        assert_eq!(t.context, "128K");
        assert_eq!(t.inputs, "Text, Image");
        assert!(t.has_image_input());
        assert_eq!(t.updated, "1 year ago");
        assert_eq!(d.tags[1].tag(), "90b");
    }

    #[test]
    fn json_tag_lists_are_understood() {
        let tags = parse_tags_json("qwen3", r#"{"tags":["latest","0.6b","8b-q4_K_M","bad/tag",""]}"#);
        let names: Vec<_> = tags.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["qwen3:latest", "qwen3:0.6b", "qwen3:8b-q4_K_M"]);
        assert!(tags[0].size_bytes.is_none());
        assert!(parse_tags_json("qwen3", "<!doctype html><html></html>").is_empty());
    }

    #[test]
    fn hrefs_map_to_names() {
        assert_eq!(name_from_href("/library/qwen3").as_deref(), Some("qwen3"));
        assert_eq!(name_from_href("/user/model").as_deref(), Some("user/model"));
        assert_eq!(name_from_href("/blog/post"), None);
        assert_eq!(name_from_href("/library/qwen3:8b"), None);
        assert_eq!(name_from_href("https://example.com/x"), None);
    }

    #[test]
    fn model_names_are_validated() {
        assert!(validate_model_name("llama3.2:3b").is_ok());
        assert!(validate_model_name("hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M").is_ok());
        assert!(validate_model_name("../../etc").is_err());
        assert!(validate_model_name("a b").is_err());
        assert!(validate_model_name("").is_err());
    }

    #[test]
    fn ndjson_lines_are_split_across_chunks() {
        let mut lb = LineBuffer::default();
        assert!(lb.push(b"{\"status\":\"pul").is_empty());
        let l = lb.push(b"ling\"}\n{\"status\":\"success\"}\n{\"x\"");
        assert_eq!(l, vec!["{\"status\":\"pulling\"}", "{\"status\":\"success\"}"]);
        assert_eq!(lb.finish().as_deref(), Some("{\"x\""));
    }

    #[test]
    fn size_labels_parse() {
        assert_eq!(parse_size_label("7.8GB"), Some(7_800_000_000));
        assert_eq!(parse_size_label("815MB"), Some(815_000_000));
        assert_eq!(parse_size_label("n/a"), None);
    }
}
