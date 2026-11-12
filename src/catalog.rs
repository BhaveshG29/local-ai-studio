//! The local model index.
//!
//! Metadata for models seen in Hugging Face and Ollama searches is stored in SQLite with a
//! full-text index, so the catalog can be searched offline. "Update index" crawls the most
//! popular models of each supported category page by page (respecting rate limits) — it never
//! downloads model files and never tries to mirror the whole Hub.

use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::db::{self, catalog::CatalogEntry};
use crate::hub::{HubClient, HubModelSummary, HubSearch, HubSort};
use crate::ollama::{LibraryModel, OllamaClient};

pub fn entry_from_hub(m: &HubModelSummary) -> CatalogEntry {
    CatalogEntry {
        source: "huggingface".into(),
        model_id: m.id.clone(),
        name: m.short_name().to_string(),
        author: m.author_name(),
        description: String::new(),
        task: m.pipeline_tag.clone().unwrap_or_default(),
        library: m.library_name.clone().unwrap_or_default(),
        tags: m.tags.iter().filter(|t| !t.contains(':')).cloned().collect::<Vec<_>>().join(" "),
        downloads: m.downloads as i64,
        likes: m.likes as i64,
        updated: m.last_modified.clone().unwrap_or_default(),
        params: m.parameter_count().map(crate::util::human_params).unwrap_or_default(),
        formats: m.formats().join(" ").to_lowercase(),
        indexed_at: crate::util::now_rfc3339(),
    }
}

pub fn entry_from_ollama(m: &LibraryModel) -> CatalogEntry {
    let task = if m.has_capability("embedding") {
        "feature-extraction"
    } else if m.has_capability("vision") {
        "image-text-to-text"
    } else {
        "text-generation"
    };
    CatalogEntry {
        source: "ollama".into(),
        model_id: m.name.clone(),
        name: m.name.clone(),
        author: m.name.split_once('/').map(|(u, _)| u.to_string()).unwrap_or_else(|| "ollama".into()),
        description: m.description.clone(),
        task: task.into(),
        library: "ollama".into(),
        tags: m.capabilities.join(" "),
        downloads: m.pulls as i64,
        likes: 0,
        updated: m.updated.clone(),
        params: m.sizes.join(" "),
        formats: "ollama".into(),
        indexed_at: crate::util::now_rfc3339(),
    }
}

/// Progress of an index refresh, shown on the Discover page.
#[derive(Clone, Debug, Default, Serialize)]
pub struct IndexProgress {
    pub running: bool,
    pub stage: String,
    pub indexed: usize,
    pub errors: Vec<String>,
    pub finished_at: Option<String>,
}

#[derive(Clone, Default)]
pub struct IndexStatus {
    inner: Arc<Mutex<IndexProgress>>,
}

impl IndexStatus {
    pub fn get(&self) -> IndexProgress {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
    fn update(&self, f: impl FnOnce(&mut IndexProgress)) {
        f(&mut self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
    }

    /// Marks a refresh as started. Returns false when one is already running, so two clicks
    /// can never start two crawls.
    pub fn try_start(&self) -> bool {
        let mut p = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if p.running {
            return false;
        }
        *p = IndexProgress { running: true, stage: "Starting".into(), ..Default::default() };
        true
    }
}

/// Categories crawled by "Update index": (label, pipeline tag, format filter).
const HUB_CATEGORIES: &[(&str, &str, &str)] = &[
    ("Hugging Face · GGUF chat models", "text-generation", "gguf"),
    ("Hugging Face · safetensors chat models", "text-generation", "safetensors"),
    ("Hugging Face · speech to text", "automatic-speech-recognition", ""),
    ("Hugging Face · image generation", "text-to-image", "diffusers"),
    ("Hugging Face · vision models", "image-text-to-text", ""),
];

/// Refreshes the local index. Runs in the background; progress goes to `status`.
pub async fn refresh_index(
    hub: HubClient,
    ollama: OllamaClient,
    db: db::Db,
    status: IndexStatus,
    pages_per_category: usize,
) {
    status.update(|p| *p = IndexProgress { running: true, stage: "Starting".into(), ..Default::default() });
    // If the crawl ends any other way than below (a panic in a page parser), the index must
    // not stay "being updated" until the app restarts.
    struct Finish(IndexStatus);
    impl Drop for Finish {
        fn drop(&mut self) {
            self.0.update(|p| {
                if p.running {
                    p.running = false;
                    p.stage = "Stopped by an error".into();
                }
            });
        }
    }
    let _finish = Finish(status.clone());
    let mut total = 0usize;
    for (label, task, format) in HUB_CATEGORIES {
        status.update(|p| p.stage = (*label).to_string());
        let mut cursor = None;
        for _ in 0..pages_per_category {
            let q = HubSearch {
                task: (*task).into(),
                format: (*format).into(),
                sort: HubSort::Downloads,
                cursor: cursor.clone(),
                limit: 100,
                ..Default::default()
            };
            match hub.search(&q).await {
                Ok(page) if page.cached_at.is_none() => {
                    total += page.models.len();
                    status.update(|p| p.indexed = total);
                    if page.rate_limit.as_ref().is_some_and(|r| r.remaining < 20) {
                        status
                            .update(|p| p.errors.push("Paused early to stay within Hugging Face's rate limit.".into()));
                        break;
                    }
                    match page.next_cursor {
                        Some(c) => cursor = Some(c),
                        None => break,
                    }
                }
                Ok(_) => {
                    status.update(|p| p.errors.push(format!("{label}: offline, kept cached data")));
                    break;
                }
                Err(e) => {
                    status.update(|p| p.errors.push(format!("{label}: {e}")));
                    break;
                }
            }
        }
    }
    status.update(|p| p.stage = "Ollama library".into());
    let mut page_no = 1;
    loop {
        match ollama.search_library("", "", "popular", page_no, true).await {
            Ok(page) => {
                total += page.models.len();
                status.update(|p| p.indexed = total);
                match page.next_page {
                    Some(n) if n > page_no && n <= 30 => page_no = n,
                    _ => break,
                }
            }
            Err(e) => {
                status.update(|p| p.errors.push(format!("Ollama library: {e}")));
                break;
            }
        }
    }
    let now = crate::util::now_rfc3339();
    let n2 = now.clone();
    let _ = db.call(move |c| db::kv::set(c, "index.last_refresh", &n2)).await;
    status.update(|p| {
        p.running = false;
        p.stage = "Done".into();
        p.finished_at = Some(now);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_entries_map_capabilities_to_tasks() {
        let m = LibraryModel {
            name: "llava".into(),
            capabilities: vec!["vision".into()],
            sizes: vec!["7b".into()],
            pulls: 10,
            ..Default::default()
        };
        let e = entry_from_ollama(&m);
        assert_eq!(e.task, "image-text-to-text");
        assert_eq!(e.author, "ollama");
        assert_eq!(e.params, "7b");
    }
}
