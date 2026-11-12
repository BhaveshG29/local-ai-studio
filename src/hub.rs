//! Hugging Face Hub API client and repository compatibility analysis.
//!
//! Only metadata is fetched here; file downloads are handled by [`crate::downloads`].
//! Responses are cached in SQLite so the marketplace keeps working offline with the last
//! results seen, and rate limits (`429`, `RateLimit` headers) are surfaced to the user.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Deserializer, Serialize};

use crate::config::Settings;
use crate::db::{self, Db};
use crate::error::{AppError, AppResult};
use crate::secrets::Secrets;
use crate::util::{human_bytes, is_split_gguf, parse_quantization};

/// Fields requested from the search API.
const SEARCH_EXPAND: &[&str] = &[
    "author",
    "downloads",
    "likes",
    "lastModified",
    "createdAt",
    "pipeline_tag",
    "library_name",
    "tags",
    "gated",
    "private",
    "safetensors",
    "gguf",
    "trendingScore",
];

fn de_gated<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Bool(true) => Some("auto".into()),
        _ => None,
    })
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SafetensorsInfo {
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub parameters: std::collections::BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GgufRepoInfo {
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub chat_template: Option<String>,
}

/// A model as returned by the search endpoint.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HubModelSummary {
    pub id: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default, rename = "createdAt")]
    pub created_at: Option<String>,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub library_name: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, deserialize_with = "de_gated")]
    pub gated: Option<String>,
    #[serde(default)]
    pub private: bool,
    #[serde(default)]
    pub safetensors: Option<SafetensorsInfo>,
    #[serde(default)]
    pub gguf: Option<GgufRepoInfo>,
    #[serde(default, rename = "trendingScore")]
    pub trending_score: Option<f64>,
}

impl HubModelSummary {
    pub fn author_name(&self) -> String {
        self.author.clone().unwrap_or_else(|| self.id.split('/').next().unwrap_or("").to_string())
    }
    pub fn short_name(&self) -> &str {
        self.id.rsplit('/').next().unwrap_or(&self.id)
    }
    pub fn parameter_count(&self) -> Option<u64> {
        self.safetensors
            .as_ref()
            .map(|s| s.total)
            .filter(|t| *t > 0)
            .or_else(|| self.gguf.as_ref().map(|g| g.total).filter(|t| *t > 0))
    }
    pub fn formats(&self) -> Vec<&'static str> {
        let mut f = Vec::new();
        if self.gguf.is_some() || self.tags.iter().any(|t| t == "gguf") {
            f.push("GGUF");
        }
        if self.safetensors.is_some() || self.tags.iter().any(|t| t == "safetensors") {
            f.push("safetensors");
        }
        if self.library_name.as_deref() == Some("diffusers") || self.tags.iter().any(|t| t == "diffusers") {
            f.push("diffusers");
        }
        if self.tags.iter().any(|t| t == "onnx") {
            f.push("ONNX");
        }
        if self.tags.iter().any(|t| t == "mlx") {
            f.push("MLX");
        }
        f
    }
    pub fn license(&self) -> Option<String> {
        self.tags.iter().find_map(|t| t.strip_prefix("license:").map(String::from))
    }
}

/// A file in a repository (`siblings` with `blobs=true`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HubSibling {
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default, rename = "blobId")]
    pub blob_id: Option<String>,
    #[serde(default)]
    pub lfs: Option<HubLfs>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HubLfs {
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
}

impl HubSibling {
    pub fn size(&self) -> u64 {
        self.size.or_else(|| self.lfs.as_ref().and_then(|l| l.size)).unwrap_or(0)
    }
    pub fn sha256(&self) -> Option<String> {
        self.lfs.as_ref().and_then(|l| l.sha256.clone())
    }
}

/// Full model information (`/api/models/{id}?blobs=true`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HubModelInfo {
    #[serde(flatten)]
    pub summary: HubModelSummary,
    #[serde(default)]
    pub sha: Option<String>,
    #[serde(default)]
    pub siblings: Vec<HubSibling>,
    #[serde(default, rename = "cardData")]
    pub card_data: Option<serde_json::Value>,
    #[serde(default)]
    pub config: Option<serde_json::Value>,
    #[serde(default, rename = "usedStorage")]
    pub used_storage: Option<u64>,
    #[serde(default)]
    pub disabled: Option<bool>,
}

impl HubModelInfo {
    pub fn model_type(&self) -> Option<String> {
        self.config.as_ref()?.get("model_type")?.as_str().map(String::from)
    }
    pub fn architectures(&self) -> Vec<String> {
        self.config
            .as_ref()
            .and_then(|c| c.get("architectures"))
            .and_then(|a| a.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default()
    }
    pub fn base_models(&self) -> Vec<String> {
        match self.card_data.as_ref().and_then(|c| c.get("base_model")) {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            _ => Vec::new(),
        }
    }
    pub fn total_size(&self) -> u64 {
        self.siblings.iter().map(|s| s.size()).sum()
    }
    pub fn sibling(&self, name: &str) -> Option<&HubSibling> {
        self.siblings.iter().find(|s| s.rfilename == name)
    }
    pub fn has_file(&self, name: &str) -> bool {
        self.sibling(name).is_some()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HubSort {
    #[default]
    Trending,
    Downloads,
    Likes,
    Updated,
    Created,
}

impl HubSort {
    pub fn parse(s: &str) -> Self {
        match s {
            "downloads" => HubSort::Downloads,
            "likes" => HubSort::Likes,
            "updated" => HubSort::Updated,
            "created" => HubSort::Created,
            _ => HubSort::Trending,
        }
    }
    pub fn slug(self) -> &'static str {
        match self {
            HubSort::Trending => "trending",
            HubSort::Downloads => "downloads",
            HubSort::Likes => "likes",
            HubSort::Updated => "updated",
            HubSort::Created => "created",
        }
    }
    fn api_value(self) -> &'static str {
        match self {
            HubSort::Trending => "trendingScore",
            HubSort::Downloads => "downloads",
            HubSort::Likes => "likes",
            HubSort::Updated => "lastModified",
            HubSort::Created => "createdAt",
        }
    }
}

/// Search filters accepted by the marketplace.
#[derive(Clone, Debug, Default)]
pub struct HubSearch {
    pub text: String,
    pub author: String,
    /// Pipeline tag, e.g. `text-generation`, `automatic-speech-recognition`, `text-to-image`.
    pub task: String,
    /// `gguf`, `safetensors`, `diffusers` or empty.
    pub format: String,
    /// Model architecture / family tag (e.g. `llama`, `qwen2`).
    pub architecture: String,
    pub sort: HubSort,
    pub cursor: Option<String>,
    pub limit: usize,
}

#[derive(Clone, Debug, Default)]
pub struct HubSearchPage {
    pub models: Vec<HubModelSummary>,
    pub next_cursor: Option<String>,
    /// Set when results come from the offline cache.
    pub cached_at: Option<String>,
    pub rate_limit: Option<RateLimit>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RateLimit {
    pub remaining: u64,
    pub reset_secs: u64,
}

/// Parses `ratelimit: "api";r=495;t=199`.
pub fn parse_rate_limit(header: &str) -> Option<RateLimit> {
    let mut remaining = None;
    let mut reset = None;
    for part in header.split(';') {
        let part = part.trim();
        if let Some(v) = part.strip_prefix("r=") {
            remaining = v.parse().ok();
        } else if let Some(v) = part.strip_prefix("t=") {
            reset = v.parse().ok();
        }
    }
    Some(RateLimit { remaining: remaining?, reset_secs: reset.unwrap_or(0) })
}

/// Extracts the `cursor` query value from a `Link: <...>; rel="next"` header.
pub fn parse_next_cursor(link: &str) -> Option<String> {
    for part in link.split(',') {
        let part = part.trim();
        if !part.contains("rel=\"next\"") {
            continue;
        }
        let start = part.find('<')? + 1;
        let end = part.find('>')?;
        let url = url::Url::parse(part.get(start..end)?).ok()?;
        return url.query_pairs().find(|(k, _)| k == "cursor").map(|(_, v)| v.to_string());
    }
    None
}

/// Validates a repository id (`owner/name`), rejecting anything that could alter URLs or paths.
pub fn validate_repo_id(id: &str) -> AppResult<()> {
    let ok = id.len() <= 200
        && id.split('/').count() == 2
        && id.split('/').all(|p| {
            !p.is_empty()
                && p != "."
                && p != ".."
                && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        });
    if ok {
        Ok(())
    } else {
        Err(AppError::bad_request(format!("\"{id}\" is not a valid Hugging Face model name (expected owner/name).")))
    }
}

/// Encodes each path segment of a repository file path for use in a URL.
pub fn encode_repo_path(path: &str) -> String {
    path.split('/').map(crate::util::url_encode).collect::<Vec<_>>().join("/")
}

#[derive(Clone)]
pub struct HubClient {
    http: reqwest::Client,
    settings: Arc<RwLock<Settings>>,
    secrets: Arc<Secrets>,
    db: Db,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WhoAmI {
    pub name: String,
    #[serde(default, rename = "fullname")]
    pub full_name: Option<String>,
}

impl HubClient {
    pub fn new(http: reqwest::Client, settings: Arc<RwLock<Settings>>, secrets: Arc<Secrets>, db: Db) -> Self {
        Self { http, settings, secrets, db }
    }

    pub fn endpoint(&self) -> String {
        self.settings.read().unwrap_or_else(std::sync::PoisonError::into_inner).huggingface.endpoint.clone()
    }

    pub async fn token(&self) -> Option<String> {
        self.secrets.hf_token().await
    }

    /// URL of a file at a revision (follows to the CDN / Xet bridge on download).
    pub fn file_url(&self, repo_id: &str, revision: &str, path: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint(),
            repo_id,
            crate::util::url_encode(revision),
            encode_repo_path(path)
        )
    }

    async fn get(&self, url: &str, auth: bool) -> AppResult<reqwest::Response> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let mut req =
                self.http.get(url).timeout(Duration::from_secs(30)).header(reqwest::header::ACCEPT, "application/json");
            if auth && let Some(t) = self.token().await {
                req = req.bearer_auth(t);
            }
            let resp = req.send().await.map_err(map_network_error)?;
            if resp.status() == StatusCode::TOO_MANY_REQUESTS {
                let wait = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .or_else(|| {
                        resp.headers()
                            .get("ratelimit")
                            .and_then(|v| v.to_str().ok())
                            .and_then(parse_rate_limit)
                            .map(|r| r.reset_secs)
                    })
                    .unwrap_or(30);
                if attempt == 1 && wait <= 5 {
                    tokio::time::sleep(Duration::from_secs(wait.max(1))).await;
                    continue;
                }
                return Err(AppError::upstream(format!(
                    "Hugging Face is limiting how many requests this computer can make. Please try again in about {}.",
                    crate::util::human_duration(wait)
                )));
            }
            if resp.status().is_server_error() && attempt < 3 {
                tokio::time::sleep(Duration::from_millis(500 * attempt)).await;
                continue;
            }
            return Ok(resp);
        }
    }

    fn search_url(&self, q: &HubSearch) -> String {
        // Settings only keep valid endpoints (see `Settings::normalize`); the fallback is never
        // expected, but a bad address must not take the page down.
        let mut url = url::Url::parse(&format!("{}/api/models", self.endpoint()))
            .unwrap_or_else(|_| url::Url::parse("https://huggingface.co/api/models").expect("constant URL"));
        {
            let mut p = url.query_pairs_mut();
            if !q.text.trim().is_empty() {
                p.append_pair("search", q.text.trim());
            }
            if !q.author.trim().is_empty() {
                p.append_pair("author", q.author.trim());
            }
            if !q.task.is_empty() {
                p.append_pair("pipeline_tag", &q.task);
            }
            let mut filters = Vec::new();
            if !q.format.is_empty() {
                filters.push(q.format.clone());
            }
            if !q.architecture.is_empty() {
                filters.push(q.architecture.clone());
            }
            for f in filters {
                p.append_pair("filter", &f);
            }
            p.append_pair("sort", q.sort.api_value());
            p.append_pair("direction", "-1");
            p.append_pair("limit", &q.limit.clamp(1, 100).to_string());
            for e in SEARCH_EXPAND {
                p.append_pair("expand[]", e);
            }
            if let Some(c) = &q.cursor {
                p.append_pair("cursor", c);
            }
        }
        url.to_string()
    }

    /// Searches the Hub. Falls back to cached results when offline.
    pub async fn search(&self, q: &HubSearch) -> AppResult<HubSearchPage> {
        let url = self.search_url(q);
        let cache_key = format!("hf:search:{url}");
        match self.search_live(&url).await {
            Ok((models, next_cursor, rate_limit)) => {
                let body = serde_json::to_string(&(&models, &next_cursor)).unwrap_or_default();
                let key = cache_key.clone();
                let now = crate::util::now_rfc3339();
                let _ = self.db.call(move |c| db::catalog::cache_put(c, &key, &body, &now)).await;
                // Feed the local index with what we just saw.
                let entries: Vec<_> = models.iter().map(crate::catalog::entry_from_hub).collect();
                let _ = self.db.call_mut(move |c| db::catalog::upsert_many(c, &entries)).await;
                Ok(HubSearchPage { models, next_cursor, cached_at: None, rate_limit })
            }
            Err(err) => {
                let key = cache_key.clone();
                let cached = self.db.call(move |c| db::catalog::cache_get(c, &key, None)).await.ok().flatten();
                match cached {
                    Some((body, fetched_at)) => {
                        let (models, next_cursor): (Vec<HubModelSummary>, Option<String>) =
                            serde_json::from_str(&body).unwrap_or_default();
                        tracing::warn!("Hugging Face search failed, serving cache: {err}");
                        Ok(HubSearchPage { models, next_cursor, cached_at: Some(fetched_at), rate_limit: None })
                    }
                    None => Err(err),
                }
            }
        }
    }

    async fn search_live(&self, url: &str) -> AppResult<(Vec<HubModelSummary>, Option<String>, Option<RateLimit>)> {
        let resp = self.get(url, true).await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::upstream(format!(
                "Hugging Face search failed ({status}): {}",
                crate::util::truncate_chars(&extract_hub_error(&body), 200)
            )));
        }
        let next = resp.headers().get(reqwest::header::LINK).and_then(|v| v.to_str().ok()).and_then(parse_next_cursor);
        let rl = resp.headers().get("ratelimit").and_then(|v| v.to_str().ok()).and_then(parse_rate_limit);
        let models: Vec<HubModelSummary> =
            resp.json().await.map_err(|e| AppError::upstream(format!("Unexpected response from Hugging Face: {e}")))?;
        Ok((models, next, rl))
    }

    /// Full information about one repository, including file list with sizes and hashes.
    pub async fn model_info(&self, repo_id: &str) -> AppResult<(HubModelInfo, Option<String>)> {
        validate_repo_id(repo_id)?;
        let url = format!("{}/api/models/{}?blobs=true", self.endpoint(), repo_id);
        let cache_key = format!("hf:info:{repo_id}");
        let live = async {
            let resp = self.get(&url, true).await?;
            let status = resp.status();
            match status {
                s if s.is_success() => {
                    let text = resp.text().await.map_err(map_network_error)?;
                    let info: HubModelInfo = serde_json::from_str(&text)
                        .map_err(|e| AppError::upstream(format!("Unexpected response from Hugging Face: {e}")))?;
                    Ok((info, text))
                }
                StatusCode::NOT_FOUND => Err(AppError::not_found(format!(
                    "The model \"{repo_id}\" does not exist on Hugging Face, or it is private. If it is private, add a Hugging Face token in Settings."
                ))),
                StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Err(AppError::Forbidden(format!(
                    "\"{repo_id}\" is private or restricted. Add a Hugging Face token with access in Settings."
                ))),
                s => {
                    let body = resp.text().await.unwrap_or_default();
                    Err(AppError::upstream(format!(
                        "Hugging Face returned an error ({s}): {}",
                        crate::util::truncate_chars(&extract_hub_error(&body), 200)
                    )))
                }
            }
        };
        match live.await {
            Ok((info, text)) => {
                let key = cache_key.clone();
                let now = crate::util::now_rfc3339();
                let _ = self.db.call(move |c| db::catalog::cache_put(c, &key, &text, &now)).await;
                Ok((info, None))
            }
            Err(e @ (AppError::NotFound(_) | AppError::Forbidden(_))) => Err(e),
            Err(e) => {
                let key = cache_key.clone();
                let cached = self.db.call(move |c| db::catalog::cache_get(c, &key, None)).await.ok().flatten();
                match cached.and_then(|(body, at)| serde_json::from_str::<HubModelInfo>(&body).ok().map(|i| (i, at))) {
                    Some((info, at)) => Ok((info, Some(at))),
                    None => Err(e),
                }
            }
        }
    }

    /// The model card (README.md) text, if any.
    pub async fn readme(&self, repo_id: &str, revision: &str) -> Option<String> {
        validate_repo_id(repo_id).ok()?;
        let url = self.file_url(repo_id, revision, "README.md");
        let resp = self.get(&url, true).await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let text = resp.text().await.ok()?;
        Some(strip_front_matter(&text).to_string())
    }

    /// Downloads a small JSON file from a repository (config.json, model_index.json…).
    pub async fn small_json(&self, repo_id: &str, revision: &str, path: &str) -> AppResult<serde_json::Value> {
        validate_repo_id(repo_id)?;
        let url = self.file_url(repo_id, revision, path);
        let resp = self.get(&url, true).await?;
        if !resp.status().is_success() {
            return Err(AppError::upstream(format!("Could not read {path} from {repo_id} ({})", resp.status())));
        }
        resp.json().await.map_err(|e| AppError::upstream(format!("{path} in {repo_id} is not valid JSON: {e}")))
    }

    /// Checks a token against the Hub and returns the account name.
    pub async fn whoami(&self, token: &str) -> AppResult<WhoAmI> {
        let url = format!("{}/api/whoami-v2", self.endpoint());
        let resp = self
            .http
            .get(&url)
            .bearer_auth(token)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(map_network_error)?;
        match resp.status() {
            s if s.is_success() => {
                resp.json().await.map_err(|e| AppError::upstream(format!("Unexpected response: {e}")))
            }
            StatusCode::UNAUTHORIZED => Err(AppError::bad_request(
                "Hugging Face did not accept this token. Check that you copied it completely.",
            )),
            s => Err(AppError::upstream(format!("Hugging Face could not verify the token ({s})."))),
        }
    }
}

/// Maps transport errors to messages people understand.
pub fn map_network_error(e: reqwest::Error) -> AppError {
    if e.is_timeout() {
        AppError::upstream("Hugging Face took too long to respond. Check your connection and try again.")
    } else if e.is_connect() {
        AppError::unavailable(
            "Can't reach Hugging Face. Check your internet connection. Downloaded models still work offline.",
        )
    } else {
        AppError::upstream(format!("Network error while talking to Hugging Face: {e}"))
    }
}

fn extract_hub_error(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
        .unwrap_or_else(|| body.to_string())
}

/// Removes a leading YAML front-matter block from a model card.
pub fn strip_front_matter(text: &str) -> &str {
    let t = text.trim_start_matches('\u{feff}');
    if let Some(rest) = t.strip_prefix("---")
        && let Some(end) = rest.find("\n---")
    {
        let after = &rest[end + 4..];
        return after.trim_start_matches(['\r', '\n']);
    }
    t
}

// ---------------------------------------------------------------------------------------------
// Compatibility analysis
// ---------------------------------------------------------------------------------------------

/// What a model does, as far as Local AI Studio is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Capability {
    Chat,
    Vision,
    Transcription,
    ImageGeneration,
    VideoGeneration,
    TextToSpeech,
    Embedding,
    #[default]
    Other,
}

impl Capability {
    pub fn label(self) -> &'static str {
        match self {
            Capability::Chat => "Text chat",
            Capability::Vision => "Image understanding",
            Capability::Transcription => "Speech to text",
            Capability::ImageGeneration => "Image generation",
            Capability::VideoGeneration => "Video generation",
            Capability::TextToSpeech => "Text to speech",
            Capability::Embedding => "Embeddings",
            Capability::Other => "Other",
        }
    }
}

/// Architectures the built-in engine can run from GGUF files.
pub const BUILTIN_GGUF_ARCHS: &[&str] =
    &["llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3", "glm4", "lfm2"];

/// `model_type` values the built-in engine can run from safetensors.
pub const BUILTIN_SAFETENSORS_TYPES: &[&str] = &["llama", "mistral", "qwen2", "qwen3", "phi3"];

/// Quantisation types supported by Candle's quantized kernels.
pub fn builtin_supports_quant(q: &str) -> bool {
    let q = q.to_uppercase();
    !(q.starts_with("IQ") || q.starts_with("TQ") || q == "MXFP4" || q.starts_with("Q4_0_") || q == "FP8")
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Variant {
    pub id: String,
    pub label: String,
    pub files: Vec<String>,
    pub total_size: u64,
    pub quant: Option<String>,
    /// Can the built-in engine run it?
    pub builtin: bool,
    /// Note shown next to the variant.
    pub note: Option<String>,
    pub recommended: bool,
    pub primary_file: Option<String>,
    /// Extra files from other repositories (e.g. CLIP tokenizers for Stable Diffusion).
    pub companions: Vec<(String, String)>,
    /// `hf.co/...` reference Ollama can pull directly (GGUF repos).
    pub ollama_ref: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Support {
    /// Runs with the built-in engine.
    Builtin,
    /// Can be pulled and run through Ollama (GGUF the built-in engine can't run).
    ViaOllama,
    /// Downloadable, but Local AI Studio cannot run it.
    DownloadOnly,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubCompat {
    pub capability: Capability,
    pub support: Support,
    pub format: String,
    pub summary: String,
    pub reasons: Vec<String>,
    pub variants: Vec<Variant>,
}

fn has_tag(info: &HubModelInfo, tag: &str) -> bool {
    info.summary.tags.iter().any(|t| t == tag)
}

/// Infers what a repository is and whether Local AI Studio can run it.
pub fn analyze_repo(info: &HubModelInfo) -> HubCompat {
    let pipeline = info.summary.pipeline_tag.clone().unwrap_or_default();
    let model_type = info.model_type().unwrap_or_default();
    let gguf_files: Vec<&HubSibling> =
        info.siblings.iter().filter(|s| s.rfilename.to_lowercase().ends_with(".gguf")).collect();

    // Stable Diffusion (diffusers layout).
    if info.has_file("model_index.json")
        && info.has_file("unet/config.json")
        && (has_tag(info, "diffusers:StableDiffusionPipeline") || has_tag(info, "diffusers:StableDiffusionXLPipeline"))
    {
        return analyze_diffusers(info);
    }
    if pipeline == "text-to-video" || pipeline == "image-to-video" {
        return unsupported(
            info,
            Capability::VideoGeneration,
            "Video generation models can be downloaded, but no Rust-native video generation engine is available yet, so Local AI Studio cannot run them.",
        );
    }
    if pipeline == "text-to-image" || pipeline == "image-to-image" {
        return unsupported(
            info,
            Capability::ImageGeneration,
            "Only Stable Diffusion 1.5, 2.x, SD-Turbo, SDXL and SDXL-Turbo repositories in the diffusers layout are supported for image generation.",
        );
    }
    if pipeline == "text-to-speech" || pipeline == "text-to-audio" {
        return unsupported(
            info,
            Capability::TextToSpeech,
            "Text-to-speech and audio generation models are not supported yet.",
        );
    }
    if model_type == "whisper" || (pipeline == "automatic-speech-recognition" && has_tag(info, "whisper")) {
        return analyze_whisper(info);
    }
    if pipeline == "automatic-speech-recognition" {
        return unsupported(info, Capability::Transcription, "Only Whisper models are supported for speech to text.");
    }
    // Before the GGUF check: embedding models are published as GGUF too, and their
    // architecture (qwen3, llama, …) would otherwise make them look like chat models.
    if pipeline == "feature-extraction" || pipeline == "sentence-similarity" {
        return unsupported(
            info,
            Capability::Embedding,
            "Embedding models are not used by Local AI Studio's chat yet.",
        );
    }
    if !gguf_files.is_empty() {
        return analyze_gguf(info, &gguf_files);
    }
    let has_safetensors = info.siblings.iter().any(|s| s.rfilename.ends_with(".safetensors"));
    if has_safetensors && matches!(pipeline.as_str(), "text-generation" | "conversational" | "") {
        return analyze_safetensors_llm(info);
    }
    let capability = match pipeline.as_str() {
        "image-text-to-text" | "visual-question-answering" => Capability::Vision,
        "text-generation" => Capability::Chat,
        _ => Capability::Other,
    };
    unsupported(
        info,
        capability,
        if capability == Capability::Vision {
            "Vision models run through Ollama. Look for this model (or a GGUF version with a vision projector) in the Ollama tab."
        } else if has_safetensors {
            "This model's format or task is not supported by Local AI Studio's engines."
        } else {
            "No supported weight files (GGUF or safetensors) were found in this repository."
        },
    )
}

fn unsupported(info: &HubModelInfo, capability: Capability, reason: &str) -> HubCompat {
    let all: Vec<String> = info.siblings.iter().map(|s| s.rfilename.clone()).collect();
    HubCompat {
        capability,
        support: Support::DownloadOnly,
        format: info.summary.formats().join(", "),
        summary: "Can be downloaded, but not run in Local AI Studio.".into(),
        reasons: vec![reason.to_string()],
        variants: vec![Variant {
            id: "all".into(),
            label: format!("All files · {}", human_bytes(info.total_size())),
            files: all,
            total_size: info.total_size(),
            builtin: false,
            note: Some("For use with other tools.".into()),
            ..Default::default()
        }],
    }
}

fn analyze_gguf(info: &HubModelInfo, files: &[&HubSibling]) -> HubCompat {
    let arch = info.summary.gguf.as_ref().and_then(|g| g.architecture.clone()).unwrap_or_default();
    let arch_ok = arch.is_empty() || BUILTIN_GGUF_ARCHS.contains(&arch.as_str());
    let has_mmproj = files.iter().any(|f| f.rfilename.to_lowercase().contains("mmproj"));
    let mut reasons = Vec::new();
    let mut variants = Vec::new();
    let mut split_groups: std::collections::BTreeMap<String, Vec<&HubSibling>> = Default::default();

    for f in files {
        let name = &f.rfilename;
        let lower = name.to_lowercase();
        if lower.contains("mmproj") {
            continue;
        }
        if is_split_gguf(name) {
            let key = crate::util::split_gguf_base(name);
            split_groups.entry(key).or_default().push(f);
            continue;
        }
        let quant = parse_quantization(name);
        let quant_ok = quant.as_deref().map(builtin_supports_quant).unwrap_or(true);
        let builtin = arch_ok && quant_ok;
        let note = if !arch_ok {
            Some(format!("The built-in engine doesn't support the \"{arch}\" architecture — use Ollama."))
        } else if !quant_ok {
            Some(format!(
                "{} quantisation isn't supported by the built-in engine — use Ollama.",
                quant.clone().unwrap_or_default()
            ))
        } else {
            None
        };
        let quant_tag = quant.clone().unwrap_or_else(|| "latest".into());
        variants.push(Variant {
            id: name.clone(),
            label: format!("{} · {}", quant.clone().unwrap_or_else(|| name.clone()), human_bytes(f.size())),
            files: vec![name.clone()],
            total_size: f.size(),
            quant,
            builtin,
            note,
            recommended: false,
            primary_file: Some(name.clone()),
            companions: Vec::new(),
            ollama_ref: Some(format!("hf.co/{}:{}", info.summary.id, quant_tag)),
        });
    }
    for (key, parts) in split_groups {
        let size: u64 = parts.iter().map(|p| p.size()).sum();
        let quant = parse_quantization(&key);
        variants.push(Variant {
            id: key.clone(),
            label: format!(
                "{} · {} ({} parts)",
                quant.clone().unwrap_or_else(|| key.clone()),
                human_bytes(size),
                parts.len()
            ),
            files: parts.iter().map(|p| p.rfilename.clone()).collect(),
            total_size: size,
            quant: quant.clone(),
            builtin: false,
            note: Some("Split into several files — run it with Ollama.".into()),
            recommended: false,
            primary_file: parts.first().map(|p| p.rfilename.clone()),
            companions: Vec::new(),
            ollama_ref: Some(format!("hf.co/{}:{}", info.summary.id, quant.unwrap_or_else(|| "latest".into()))),
        });
    }
    variants.sort_by_key(|v| v.total_size);
    // Recommend a balanced quantisation the built-in engine supports.
    let preference = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q8_0", "Q6_K", "Q3_K_M"];
    if let Some(idx) =
        preference.iter().find_map(|p| variants.iter().position(|v| v.builtin && v.quant.as_deref() == Some(*p)))
    {
        variants[idx].recommended = true;
    } else if let Some(v) = variants.iter_mut().find(|v| v.builtin) {
        v.recommended = true;
    }
    let any_builtin = variants.iter().any(|v| v.builtin);
    if !arch_ok {
        reasons.push(format!(
            "Architecture \"{arch}\" is not supported by the built-in engine. Ollama can run most GGUF models: use \"Get with Ollama\"."
        ));
    }
    if has_mmproj {
        reasons.push("This repository includes a vision projector (mmproj). Image input works when the model is run through Ollama; the built-in engine uses the text part only.".into());
    }
    let is_vision = has_mmproj || info.summary.pipeline_tag.as_deref() == Some("image-text-to-text");
    HubCompat {
        capability: if is_vision { Capability::Vision } else { Capability::Chat },
        support: if any_builtin { Support::Builtin } else { Support::ViaOllama },
        format: "GGUF".into(),
        summary: if any_builtin {
            "Ready to run with the built-in engine. Pick one file below — smaller files are faster but less accurate."
                .into()
        } else {
            "The built-in engine can't run these files, but Ollama can pull them directly from Hugging Face.".into()
        },
        reasons,
        variants,
    }
}

/// `model-00001-of-00004.safetensors`: one shard of a Transformers checkpoint.
fn is_weight_shard(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model-").and_then(|r| r.strip_suffix(".safetensors")) else {
        return false;
    };
    rest.split_once("-of-").is_some_and(|(a, b)| {
        !a.is_empty() && !b.is_empty() && (a.chars().chain(b.chars())).all(|c| c.is_ascii_digit())
    })
}

fn analyze_safetensors_llm(info: &HubModelInfo) -> HubCompat {
    let model_type = info.model_type().unwrap_or_default();
    let supported = BUILTIN_SAFETENSORS_TYPES.contains(&model_type.as_str());
    let has_tokenizer = info.has_file("tokenizer.json");
    let top_level: Vec<&HubSibling> =
        info.siblings.iter().filter(|s| s.rfilename.ends_with(".safetensors") && !s.rfilename.contains('/')).collect();
    // Only the Transformers-format weights, which the loader reads: some repositories also
    // carry the same weights in another layout (Mistral's `consolidated.safetensors`).
    let shards: Vec<&HubSibling> = top_level.iter().copied().filter(|s| is_weight_shard(&s.rfilename)).collect();
    let weight_files: Vec<&HubSibling> = if !shards.is_empty() {
        shards
    } else if let Some(single) = top_level.iter().copied().find(|s| s.rfilename == "model.safetensors") {
        vec![single]
    } else {
        top_level
    };
    let mut files: Vec<String> = weight_files.iter().map(|s| s.rfilename.clone()).collect();
    for extra in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "chat_template.jinja",
        "chat_template.json",
    ] {
        if info.has_file(extra) {
            files.push(extra.into());
        }
    }
    let size: u64 = files.iter().filter_map(|f| info.sibling(f)).map(|s| s.size()).sum();
    let mut reasons = Vec::new();
    if !supported {
        reasons.push(if model_type.is_empty() {
            "The model type could not be determined from config.json.".to_string()
        } else {
            format!(
                "The built-in engine runs safetensors models of type {} — this one is \"{model_type}\". Look for a GGUF version or use Ollama.",
                BUILTIN_SAFETENSORS_TYPES.join(", ")
            )
        });
    }
    if !has_tokenizer {
        reasons.push("tokenizer.json is missing, which the built-in engine needs.".into());
    }
    if weight_files.is_empty() {
        reasons.push("No .safetensors weights at the top level of the repository.".into());
    }
    let builtin = supported && has_tokenizer && !weight_files.is_empty();
    if builtin {
        reasons.push("Full-precision weights use much more memory than a GGUF version and run slower on CPU. If a GGUF version exists, prefer it.".into());
    }
    let is_vision = info.summary.pipeline_tag.as_deref() == Some("image-text-to-text");
    HubCompat {
        capability: if is_vision { Capability::Vision } else { Capability::Chat },
        support: if builtin { Support::Builtin } else { Support::DownloadOnly },
        format: "safetensors".into(),
        summary: if builtin {
            "Runs with the built-in engine (full precision).".into()
        } else {
            "Can be downloaded, but the built-in engine can't run it.".into()
        },
        reasons,
        variants: vec![Variant {
            id: "weights".into(),
            label: format!("Model weights + tokenizer · {}", human_bytes(size)),
            files,
            total_size: size,
            builtin,
            recommended: builtin,
            ..Default::default()
        }],
    }
}

fn analyze_whisper(info: &HubModelInfo) -> HubCompat {
    let mut files: Vec<String> = Vec::new();
    for f in
        ["config.json", "tokenizer.json", "model.safetensors", "generation_config.json", "preprocessor_config.json"]
    {
        if info.has_file(f) {
            files.push(f.into());
        }
    }
    let ok = info.has_file("model.safetensors") && info.has_file("tokenizer.json") && info.has_file("config.json");
    let size: u64 = files.iter().filter_map(|f| info.sibling(f)).map(|s| s.size()).sum();
    HubCompat {
        capability: Capability::Transcription,
        support: if ok { Support::Builtin } else { Support::DownloadOnly },
        format: "safetensors".into(),
        summary: if ok {
            "Speech-to-text with the built-in Whisper engine.".into()
        } else {
            "This Whisper repository is missing model.safetensors, tokenizer.json or config.json.".into()
        },
        reasons: Vec::new(),
        variants: vec![Variant {
            id: "whisper".into(),
            label: format!("Whisper model · {}", human_bytes(size)),
            files,
            total_size: size,
            builtin: ok,
            recommended: ok,
            ..Default::default()
        }],
    }
}

/// Stable Diffusion flavours the built-in engine supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SdVersion {
    V1_5,
    V2_1,
    Turbo,
    Xl,
    XlTurbo,
}

impl SdVersion {
    pub fn label(self) -> &'static str {
        match self {
            SdVersion::V1_5 => "Stable Diffusion 1.5",
            SdVersion::V2_1 => "Stable Diffusion 2.x",
            SdVersion::Turbo => "SD-Turbo",
            SdVersion::Xl => "Stable Diffusion XL",
            SdVersion::XlTurbo => "SDXL-Turbo",
        }
    }
    pub fn is_xl(self) -> bool {
        matches!(self, SdVersion::Xl | SdVersion::XlTurbo)
    }
    pub fn is_turbo(self) -> bool {
        matches!(self, SdVersion::Turbo | SdVersion::XlTurbo)
    }
    pub fn default_size(self) -> u32 {
        match self {
            SdVersion::V1_5 | SdVersion::Turbo | SdVersion::XlTurbo => 512,
            SdVersion::V2_1 => 768,
            SdVersion::Xl => 1024,
        }
    }
    pub fn default_steps(self) -> u32 {
        if self.is_turbo() { 1 } else { 30 }
    }
    pub fn default_guidance(self) -> f64 {
        if self.is_turbo() { 0.0 } else { 7.5 }
    }
    /// Repositories providing `tokenizer.json` for the text encoders.
    pub fn tokenizer_repos(self) -> Vec<&'static str> {
        match self {
            SdVersion::V1_5 | SdVersion::V2_1 | SdVersion::Turbo => vec!["openai/clip-vit-base-patch32"],
            SdVersion::Xl | SdVersion::XlTurbo => {
                vec!["openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"]
            }
        }
    }
}

/// Detects the Stable Diffusion flavour from the repository id and tags.
pub fn detect_sd_version(info: &HubModelInfo) -> Option<SdVersion> {
    let id = info.summary.id.to_lowercase();
    let xl = has_tag(info, "diffusers:StableDiffusionXLPipeline");
    if xl {
        return Some(if id.contains("turbo") { SdVersion::XlTurbo } else { SdVersion::Xl });
    }
    if id.contains("sd-turbo") {
        return Some(SdVersion::Turbo);
    }
    if id.contains("stable-diffusion-2") || id.contains("sd2") {
        return Some(SdVersion::V2_1);
    }
    if has_tag(info, "diffusers:StableDiffusionPipeline") {
        return Some(SdVersion::V1_5);
    }
    None
}

fn analyze_diffusers(info: &HubModelInfo) -> HubCompat {
    let Some(version) = detect_sd_version(info) else {
        return unsupported(info, Capability::ImageGeneration, "Unrecognised diffusers pipeline.");
    };
    let pick = |dir: &str, base: &str| -> Option<String> {
        let fp16 = format!("{dir}/{base}.fp16.safetensors");
        let full = format!("{dir}/{base}.safetensors");
        if info.has_file(&fp16) {
            Some(fp16)
        } else if info.has_file(&full) {
            Some(full)
        } else {
            None
        }
    };
    let mut files = vec!["model_index.json".to_string()];
    let mut missing = Vec::new();
    let mut parts =
        vec![("text_encoder", "model"), ("unet", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model")];
    if version.is_xl() {
        parts.push(("text_encoder_2", "model"));
    }
    for (dir, base) in parts {
        match pick(dir, base) {
            Some(f) => files.push(f),
            None => missing.push(dir),
        }
        let cfg = format!("{dir}/config.json");
        if info.has_file(&cfg) {
            files.push(cfg);
        }
    }
    if info.has_file("scheduler/scheduler_config.json") {
        files.push("scheduler/scheduler_config.json".into());
    }
    let size: u64 = files.iter().filter_map(|f| info.sibling(f)).map(|s| s.size()).sum();
    let ok = missing.is_empty();
    let companions: Vec<(String, String)> =
        version.tokenizer_repos().into_iter().map(|r| (r.to_string(), "tokenizer.json".to_string())).collect();
    HubCompat {
        capability: Capability::ImageGeneration,
        support: if ok { Support::Builtin } else { Support::DownloadOnly },
        format: "diffusers".into(),
        summary: if ok {
            format!(
                "{} — runs with the built-in image engine. CPU generation is slow (tens of seconds to minutes per image); a GPU build is much faster.",
                version.label()
            )
        } else {
            format!("Missing weights for: {}", missing.join(", "))
        },
        reasons: vec![format!(
            "Half-precision weights are downloaded when available. The CLIP tokenizer is fetched from {}.",
            version.tokenizer_repos().join(" and ")
        )],
        variants: vec![Variant {
            id: "pipeline".into(),
            label: format!("{} pipeline · {}", version.label(), human_bytes(size)),
            files,
            total_size: size,
            builtin: ok,
            recommended: ok,
            companions,
            note: None,
            ..Default::default()
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEARCH_JSON: &str = r#"[{"_id":"66e81c0ecfd2f50991047e92","id":"Qwen/Qwen2.5-0.5B-Instruct","author":"Qwen","gated":false,"lastModified":"2024-09-25T12:32:56.000Z","likes":740,"trendingScore":6,"private":false,"downloads":4509547,"safetensors":{"parameters":{"BF16":494032768},"total":494032768},"tags":["transformers","safetensors","qwen2","text-generation","license:apache-2.0"],"pipeline_tag":"text-generation","library_name":"transformers","createdAt":"2024-09-16T11:52:46.000Z"},
    {"id":"meta-llama/Llama-3.2-1B","gated":"manual","downloads":5,"likes":1,"tags":["gguf"]}]"#;

    #[test]
    fn search_results_parse() {
        let v: Vec<HubModelSummary> = serde_json::from_str(SEARCH_JSON).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].parameter_count(), Some(494032768));
        assert_eq!(v[0].license().as_deref(), Some("apache-2.0"));
        assert_eq!(v[0].gated, None);
        assert_eq!(v[1].gated.as_deref(), Some("manual"));
        assert_eq!(v[1].author_name(), "meta-llama");
        assert!(v[1].formats().contains(&"GGUF"));
    }

    #[test]
    fn link_header_cursor() {
        let link = "<https://huggingface.co/api/models?search=q&limit=3&cursor=eyJ0ZXN0IjoxfQ%3D%3D>; rel=\"next\"";
        assert_eq!(parse_next_cursor(link).as_deref(), Some("eyJ0ZXN0IjoxfQ=="));
        assert_eq!(parse_next_cursor("<https://x/y>; rel=\"prev\""), None);
        // Malformed headers are ignored, not a crash.
        assert_eq!(parse_next_cursor(">https://x/y?cursor=a<; rel=\"next\""), None);
    }

    #[test]
    fn embedding_gguf_repos_are_not_chat_models() {
        let mut info = gguf_info("qwen3", &[("Qwen3-Embedding-0.6B-Q8_0.gguf", 600)]);
        info.summary.pipeline_tag = Some("feature-extraction".into());
        let c = analyze_repo(&info);
        assert_eq!(c.capability, Capability::Embedding);
        assert_eq!(c.support, Support::DownloadOnly);
        assert!(c.variants.iter().all(|v| !v.builtin));
    }

    #[test]
    fn safetensors_variants_skip_duplicate_weight_layouts() {
        assert!(is_weight_shard("model-00001-of-00003.safetensors"));
        assert!(!is_weight_shard("consolidated.safetensors"));
        assert!(!is_weight_shard("model-x-of-00003.safetensors"));
        let names = [
            "consolidated.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
        ];
        let info = HubModelInfo {
            summary: HubModelSummary {
                id: "mistralai/Mistral-7B-Instruct-v0.3".into(),
                pipeline_tag: Some("text-generation".into()),
                ..Default::default()
            },
            siblings: names
                .iter()
                .map(|n| HubSibling { rfilename: (*n).into(), size: Some(1000), ..Default::default() })
                .collect(),
            config: Some(serde_json::json!({ "model_type": "mistral" })),
            ..Default::default()
        };
        let c = analyze_repo(&info);
        let files = &c.variants[0].files;
        assert!(files.iter().any(|f| f == "model-00002-of-00002.safetensors"), "{files:?}");
        assert!(!files.iter().any(|f| f == "consolidated.safetensors"), "{files:?}");
    }

    #[test]
    fn rate_limit_header() {
        assert_eq!(parse_rate_limit("\"api\";r=495;t=199"), Some(RateLimit { remaining: 495, reset_secs: 199 }));
        assert_eq!(parse_rate_limit("garbage"), None);
    }

    #[test]
    fn repo_ids_are_validated() {
        assert!(validate_repo_id("Qwen/Qwen2.5-0.5B-Instruct-GGUF").is_ok());
        assert!(validate_repo_id("../etc").is_err());
        assert!(validate_repo_id("a/b/c").is_err());
        assert!(validate_repo_id("a/..").is_err());
        assert!(validate_repo_id("a/b?x=1").is_err());
    }

    fn gguf_info(arch: &str, names: &[(&str, u64)]) -> HubModelInfo {
        HubModelInfo {
            summary: HubModelSummary {
                id: "org/model-GGUF".into(),
                tags: vec!["gguf".into()],
                pipeline_tag: Some("text-generation".into()),
                gguf: Some(GgufRepoInfo { architecture: Some(arch.into()), ..Default::default() }),
                ..Default::default()
            },
            siblings: names
                .iter()
                .map(|(n, s)| HubSibling { rfilename: (*n).into(), size: Some(*s), ..Default::default() })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn gguf_repos_are_analyzed() {
        let info = gguf_info(
            "qwen2",
            &[
                ("model-q2_k.gguf", 100),
                ("model-q4_k_m.gguf", 200),
                ("model-iq3_xs.gguf", 90),
                ("model-q8_0.gguf", 400),
                ("mmproj-model-f16.gguf", 50),
                ("big-Q4_K_M-00001-of-00002.gguf", 500),
                ("big-Q4_K_M-00002-of-00002.gguf", 500),
            ],
        );
        let c = analyze_repo(&info);
        assert_eq!(c.support, Support::Builtin);
        assert_eq!(c.capability, Capability::Vision);
        let rec: Vec<_> = c.variants.iter().filter(|v| v.recommended).collect();
        assert_eq!(rec.len(), 1);
        assert_eq!(rec[0].quant.as_deref(), Some("Q4_K_M"));
        let iq = c.variants.iter().find(|v| v.quant.as_deref() == Some("IQ3_XS")).unwrap();
        assert!(!iq.builtin);
        let split = c.variants.iter().find(|v| v.files.len() == 2).unwrap();
        assert!(!split.builtin);
        assert_eq!(split.total_size, 1000);
        assert!(c.variants.iter().all(|v| !v.files.iter().any(|f| f.contains("mmproj"))));
        assert_eq!(rec[0].ollama_ref.as_deref(), Some("hf.co/org/model-GGUF:Q4_K_M"));
    }

    #[test]
    fn unsupported_gguf_arch_points_to_ollama() {
        let info = gguf_info("deepseek2", &[("m-Q4_K_M.gguf", 10)]);
        let c = analyze_repo(&info);
        assert_eq!(c.support, Support::ViaOllama);
        assert!(c.variants.iter().all(|v| !v.builtin));
    }

    #[test]
    fn diffusers_turbo_is_detected() {
        let names = [
            "model_index.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "scheduler/scheduler_config.json",
        ];
        let info = HubModelInfo {
            summary: HubModelSummary {
                id: "stabilityai/sd-turbo".into(),
                tags: vec!["diffusers:StableDiffusionPipeline".into()],
                pipeline_tag: Some("text-to-image".into()),
                ..Default::default()
            },
            siblings: names
                .iter()
                .map(|n| HubSibling { rfilename: (*n).into(), size: Some(10), ..Default::default() })
                .collect(),
            ..Default::default()
        };
        assert_eq!(detect_sd_version(&info), Some(SdVersion::Turbo));
        let c = analyze_repo(&info);
        assert_eq!(c.capability, Capability::ImageGeneration);
        assert_eq!(c.support, Support::Builtin);
        let v = &c.variants[0];
        assert!(v.files.contains(&"unet/diffusion_pytorch_model.fp16.safetensors".to_string()));
        assert!(!v.files.contains(&"unet/diffusion_pytorch_model.safetensors".to_string()));
        assert_eq!(v.companions[0].0, "openai/clip-vit-base-patch32");
    }

    #[test]
    fn front_matter_is_stripped() {
        let t = "---\nlicense: mit\n---\n\n# Title\nBody";
        assert_eq!(strip_front_matter(t), "# Title\nBody");
        assert_eq!(strip_front_matter("# No front matter"), "# No front matter");
    }
}
