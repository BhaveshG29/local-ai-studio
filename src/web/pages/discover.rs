//! Model discovery: Hugging Face search, the Ollama library, the offline index, and model
//! detail pages with compatibility analysis and download buttons.

use std::collections::HashSet;

use askama::Template;
use axum::Form;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use serde::Deserialize;

use crate::SharedState;
use crate::db::catalog::{CatalogQuery, CatalogSort};
use crate::db::downloads::DownloadOptions;
use crate::error::{AppError, AppResult};
use crate::hub::{HubSearch, HubSort, Support, analyze_repo};
use crate::system::{Fit, assess_fit, estimate_llm_memory};
use crate::util::{human_bytes, human_count, human_params, relative_time, url_encode};
use crate::web::page::{PageCtx, redirect_flash, render};

#[derive(Clone, Debug)]
pub struct Opt {
    pub value: &'static str,
    pub label: &'static str,
    pub selected: bool,
}

fn opts(list: &[(&'static str, &'static str)], current: &str) -> Vec<Opt> {
    list.iter().map(|(v, l)| Opt { value: v, label: l, selected: *v == current }).collect()
}

const HF_TASKS: &[(&str, &str)] = &[
    ("", "Any task"),
    ("text-generation", "Chat / text generation"),
    ("image-text-to-text", "Vision (image + text)"),
    ("automatic-speech-recognition", "Speech to text"),
    ("text-to-image", "Image generation"),
    ("text-to-speech", "Text to speech"),
    ("text-to-video", "Video generation"),
    ("feature-extraction", "Embeddings"),
];
const HF_FORMATS: &[(&str, &str)] =
    &[("", "Any format"), ("gguf", "GGUF (quantised)"), ("safetensors", "safetensors"), ("diffusers", "diffusers")];
const HF_SORTS: &[(&str, &str)] = &[
    ("trending", "Trending"),
    ("downloads", "Most downloaded"),
    ("likes", "Most liked"),
    ("updated", "Recently updated"),
    ("created", "Newest"),
];
const HF_ARCHS: &[(&str, &str)] = &[
    ("", "Any family"),
    ("llama", "Llama"),
    ("qwen2", "Qwen 2 / 2.5"),
    ("qwen3", "Qwen 3"),
    ("mistral", "Mistral"),
    ("gemma3", "Gemma 3"),
    ("phi3", "Phi-3 / Phi-4"),
    ("whisper", "Whisper"),
    ("stable-diffusion", "Stable Diffusion"),
];
const OL_CAPS: &[(&str, &str)] = &[
    ("", "All models"),
    ("vision", "Sees images"),
    ("tools", "Tool use"),
    ("thinking", "Thinking"),
    ("embedding", "Embeddings"),
];
const OL_SORTS: &[(&str, &str)] = &[("popular", "Popular"), ("newest", "Newest")];
const IX_SOURCES: &[(&str, &str)] = &[("", "All sources"), ("huggingface", "Hugging Face"), ("ollama", "Ollama")];
const IX_SORTS: &[(&str, &str)] = &[
    ("relevance", "Best match"),
    ("downloads", "Most downloaded"),
    ("likes", "Most liked"),
    ("updated", "Recently updated"),
];

/// A plain-language name for a Hub `pipeline_tag`; empty when the publisher didn't set one
/// (not the filter's "Any task").
fn task_label(task: &str) -> &'static str {
    if task.is_empty() {
        return "";
    }
    HF_TASKS.iter().find(|(v, _)| *v == task).map(|(_, l)| *l).unwrap_or(match task {
        "image-to-text" => "Image to text",
        "conversational" => "Chat / text generation",
        _ => "Other task",
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn task_labels() {
        assert_eq!(super::task_label(""), "");
        assert_eq!(super::task_label("text-generation"), "Chat / text generation");
        assert_eq!(super::task_label("image-to-text"), "Image to text");
        assert_eq!(super::task_label("graph-ml"), "Other task");
    }
}

#[derive(Deserialize, Default)]
pub struct DiscoverQuery {
    #[serde(default)]
    tab: Option<String>,
    #[serde(default)]
    q: Option<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    sort: Option<String>,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    arch: Option<String>,
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default)]
    page: Option<u32>,
    #[serde(default)]
    cap: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    refresh: Option<String>,
}

pub struct HubCard {
    pub href: String,
    pub name: String,
    pub author: String,
    pub task: &'static str,
    pub formats: Vec<&'static str>,
    pub params: String,
    pub downloads: String,
    pub likes: String,
    pub updated: String,
    pub gated: bool,
    pub license: String,
}

pub struct OllamaCard {
    pub href: String,
    pub name: String,
    pub description: String,
    pub capabilities: Vec<String>,
    pub sizes: Vec<String>,
    pub pulls: String,
    pub tags: String,
    pub updated: String,
    pub cloud_only: bool,
}

pub struct IndexRow {
    pub href: String,
    pub source: &'static str,
    pub name: String,
    pub author: String,
    pub description: String,
    pub task: &'static str,
    pub params: String,
    pub downloads: String,
}

#[derive(Template)]
#[template(path = "discover.html")]
pub struct DiscoverPage {
    pub ctx: PageCtx,
    pub tab: &'static str,
    pub q: String,
    pub hf_tasks: Vec<Opt>,
    pub hf_formats: Vec<Opt>,
    pub hf_sorts: Vec<Opt>,
    pub hf_archs: Vec<Opt>,
    pub hf_author: String,
    pub hf_cards: Vec<HubCard>,
    pub hf_next: Option<String>,
    pub hf_first: Option<String>,
    pub hf_error: Option<String>,
    pub hf_cached_at: Option<String>,
    pub hf_rate: Option<String>,
    pub ol_caps: Vec<Opt>,
    pub ol_sorts: Vec<Opt>,
    pub ol_cards: Vec<OllamaCard>,
    pub ol_next: Option<String>,
    pub ol_error: Option<String>,
    pub ol_cached_at: Option<String>,
    pub ol_running: bool,
    pub ix_sources: Vec<Opt>,
    pub ix_tasks: Vec<Opt>,
    pub ix_sorts: Vec<Opt>,
    pub ix_rows: Vec<IndexRow>,
    pub ix_total: i64,
    pub ix_next: Option<String>,
    pub ix_prev: Option<String>,
    pub ix_counts: Vec<(String, String)>,
    pub ix_last: String,
    pub ix_running: bool,
    pub ix_stage: String,
    pub ix_indexed: usize,
    pub ix_errors: Vec<String>,
}

fn qs(pairs: &[(&str, &str)]) -> String {
    let parts: Vec<String> =
        pairs.iter().filter(|(_, v)| !v.is_empty()).map(|(k, v)| format!("{k}={}", url_encode(v))).collect();
    format!("/discover?{}", parts.join("&"))
}

pub async fn page(State(state): State<SharedState>, headers: HeaderMap, Query(q): Query<DiscoverQuery>) -> Response {
    let tab: &'static str = match q.tab.as_deref() {
        Some("ollama") => "ollama",
        Some("index") => "index",
        _ => "hf",
    };
    let text = q.q.clone().unwrap_or_default();
    let task = q.task.clone().unwrap_or_default();
    let format = q.format.clone().unwrap_or_default();
    let author = q.author.clone().unwrap_or_default();
    let arch = q.arch.clone().unwrap_or_default();
    let cap = q.cap.clone().unwrap_or_default();
    let source = q.source.clone().unwrap_or_default();
    let index_progress = state.index.get();
    let mut ctx = PageCtx::new(&state, &headers, "Discover models", "discover").at("/discover");
    if tab == "index" && index_progress.running {
        ctx = ctx.with_refresh(Some(2));
    }
    let mut p = DiscoverPage {
        ctx: ctx.clone(),
        tab,
        q: text.clone(),
        hf_tasks: opts(HF_TASKS, &task),
        hf_formats: opts(HF_FORMATS, &format),
        hf_sorts: opts(HF_SORTS, q.sort.as_deref().unwrap_or("trending")),
        hf_archs: opts(HF_ARCHS, &arch),
        hf_author: author.clone(),
        hf_cards: Vec::new(),
        hf_next: None,
        hf_first: None,
        hf_error: None,
        hf_cached_at: None,
        hf_rate: None,
        ol_caps: opts(OL_CAPS, &cap),
        ol_sorts: opts(OL_SORTS, q.sort.as_deref().unwrap_or("popular")),
        ol_cards: Vec::new(),
        ol_next: None,
        ol_error: None,
        ol_cached_at: None,
        ol_running: false,
        ix_sources: opts(IX_SOURCES, &source),
        ix_tasks: opts(HF_TASKS, &task),
        ix_sorts: opts(IX_SORTS, q.sort.as_deref().unwrap_or("relevance")),
        ix_rows: Vec::new(),
        ix_total: 0,
        ix_next: None,
        ix_prev: None,
        ix_counts: Vec::new(),
        ix_last: String::new(),
        ix_running: index_progress.running,
        ix_stage: index_progress.stage.clone(),
        ix_indexed: index_progress.indexed,
        ix_errors: index_progress.errors.clone(),
    };
    match tab {
        "hf" => {
            let sort = HubSort::parse(q.sort.as_deref().unwrap_or("trending"));
            // The architecture filter maps to a Hub tag; "stable-diffusion" is a tag too.
            let search = HubSearch {
                text: text.clone(),
                author: author.clone(),
                task: task.clone(),
                format: format.clone(),
                architecture: arch.clone(),
                sort,
                cursor: q.cursor.clone(),
                limit: 30,
            };
            match state.hub.search(&search).await {
                Ok(page) => {
                    let show_gated = state.settings().huggingface.show_gated;
                    p.hf_cards = page
                        .models
                        .iter()
                        .filter(|m| show_gated || m.gated.is_none())
                        .map(|m| HubCard {
                            href: format!("/discover/hf/{}", m.id),
                            name: m.short_name().to_string(),
                            author: m.author_name(),
                            task: task_label(m.pipeline_tag.as_deref().unwrap_or("")),
                            formats: m.formats(),
                            params: m.parameter_count().map(human_params).unwrap_or_default(),
                            downloads: human_count(m.downloads),
                            likes: human_count(m.likes),
                            updated: m.last_modified.as_deref().map(relative_time).unwrap_or_default(),
                            gated: m.gated.is_some(),
                            license: m.license().unwrap_or_default(),
                        })
                        .collect();
                    let base = [
                        ("tab", "hf"),
                        ("q", text.as_str()),
                        ("task", task.as_str()),
                        ("format", format.as_str()),
                        ("author", author.as_str()),
                        ("arch", arch.as_str()),
                        ("sort", sort.slug()),
                    ];
                    if let Some(c) = &page.next_cursor {
                        let mut v = base.to_vec();
                        v.push(("cursor", c.as_str()));
                        p.hf_next = Some(qs(&v));
                    }
                    if q.cursor.is_some() {
                        p.hf_first = Some(qs(&base));
                    }
                    p.hf_cached_at = page.cached_at.as_deref().map(relative_time);
                    p.hf_rate = page.rate_limit.filter(|r| r.remaining < 50).map(|r| {
                        format!(
                            "{} Hugging Face requests left in this {}-minute window.",
                            r.remaining,
                            (r.reset_secs / 60).max(1)
                        )
                    });
                }
                Err(e) => p.hf_error = Some(e.user_message()),
            }
        }
        "ollama" => {
            p.ol_running = state.ollama.version().await.is_ok();
            let sort = q.sort.clone().unwrap_or_else(|| "popular".into());
            let page_no = q.page.unwrap_or(1).max(1);
            match state.ollama.search_library(&text, &cap, &sort, page_no, q.refresh.is_some()).await {
                Ok(page) => {
                    p.ol_cards = page
                        .models
                        .iter()
                        .map(|m| OllamaCard {
                            href: format!("/discover/ollama/{}", m.name),
                            name: m.name.clone(),
                            description: m.description.clone(),
                            capabilities: m.capabilities.clone(),
                            sizes: m.sizes.clone(),
                            pulls: if m.pulls > 0 { human_count(m.pulls) } else { String::new() },
                            tags: m.tag_count.map(|t| t.to_string()).unwrap_or_default(),
                            updated: m.updated_label.clone(),
                            cloud_only: m.is_cloud_only(),
                        })
                        .collect();
                    if let Some(n) = page.next_page {
                        let n = n.to_string();
                        p.ol_next = Some(qs(&[
                            ("tab", "ollama"),
                            ("q", text.as_str()),
                            ("cap", cap.as_str()),
                            ("sort", sort.as_str()),
                            ("page", n.as_str()),
                        ]));
                    }
                    p.ol_cached_at = page.cached_at.as_deref().map(relative_time);
                }
                Err(e) => p.ol_error = Some(e.user_message()),
            }
        }
        _ => {
            let offset = q.offset.unwrap_or(0);
            let sort = match q.sort.as_deref() {
                Some("downloads") => CatalogSort::Downloads,
                Some("likes") => CatalogSort::Likes,
                Some("updated") => CatalogSort::Updated,
                _ => CatalogSort::Relevance,
            };
            let cq = CatalogQuery {
                text: text.clone(),
                source: (!source.is_empty()).then(|| source.clone()),
                task: (!task.is_empty()).then(|| task.clone()),
                format: None,
                sort,
                limit: 40,
                offset,
            };
            let (rows, total) = state.db.call(move |c| crate::db::catalog::search(c, &cq)).await.unwrap_or_default();
            p.ix_total = total;
            p.ix_rows = rows
                .into_iter()
                .map(|r| IndexRow {
                    href: if r.source == "ollama" {
                        format!("/discover/ollama/{}", r.model_id)
                    } else {
                        format!("/discover/hf/{}", r.model_id)
                    },
                    source: if r.source == "ollama" { "Ollama" } else { "Hugging Face" },
                    name: r.name,
                    author: r.author,
                    description: crate::util::truncate_chars(&r.description, 160),
                    task: task_label(&r.task),
                    params: r.params,
                    downloads: human_count(r.downloads.max(0) as u64),
                })
                .collect();
            let sort_s = q.sort.clone().unwrap_or_default();
            let mk = |off: usize| {
                let o = off.to_string();
                qs(&[
                    ("tab", "index"),
                    ("q", text.as_str()),
                    ("source", source.as_str()),
                    ("task", task.as_str()),
                    ("sort", sort_s.as_str()),
                    ("offset", o.as_str()),
                ])
            };
            if (offset + 40) < total as usize {
                p.ix_next = Some(mk(offset + 40));
            }
            if offset > 0 {
                p.ix_prev = Some(mk(offset.saturating_sub(40)));
            }
            p.ix_counts = state
                .db
                .call(crate::db::catalog::count_by_source)
                .await
                .unwrap_or_default()
                .into_iter()
                .map(|(s, n)| {
                    (if s == "ollama" { "Ollama".to_string() } else { "Hugging Face".to_string() }, n.to_string())
                })
                .collect();
            p.ix_last = state
                .db
                .call(|c| crate::db::kv::get(c, "index.last_refresh"))
                .await
                .ok()
                .flatten()
                .map(|t| relative_time(&t))
                .unwrap_or_else(|| "never".into());
        }
    }
    render(&ctx, &p)
}

pub async fn refresh_index(State(state): State<SharedState>) -> Response {
    if !state.index.try_start() {
        return redirect_flash("/discover?tab=index", "info", "The index is already being updated.");
    }
    let st = state.clone();
    tokio::spawn(async move {
        crate::catalog::refresh_index(st.hub.clone(), st.ollama.clone(), st.db.clone(), st.index.clone(), 3).await;
    });
    redirect_flash(
        "/discover?tab=index",
        "ok",
        "Updating the index in the background. Only model information is fetched — nothing is downloaded.",
    )
}

pub struct VariantView {
    pub id: String,
    pub label: String,
    pub size: String,
    pub quant: String,
    pub builtin: bool,
    pub note: String,
    pub recommended: bool,
    pub checked: bool,
    pub ollama_ref: String,
    pub fit_class: &'static str,
    pub fit_msg: String,
    pub downloaded: bool,
}

pub struct FileView {
    pub path: String,
    pub size: String,
    pub lfs: bool,
}

#[derive(Template)]
#[template(path = "hf_model.html")]
pub struct HfModelPage {
    pub ctx: PageCtx,
    pub id: String,
    pub name: String,
    pub author: String,
    pub task: &'static str,
    pub license: String,
    pub formats: Vec<&'static str>,
    pub params: String,
    pub downloads: String,
    pub likes: String,
    pub updated: String,
    pub gated: bool,
    pub has_token: bool,
    pub cached_at: Option<String>,
    pub support: &'static str,
    pub support_class: &'static str,
    pub capability: &'static str,
    pub format: String,
    pub summary: String,
    pub reasons: Vec<String>,
    pub variants: Vec<VariantView>,
    pub any_builtin: bool,
    pub files: Vec<FileView>,
    pub total_size: String,
    pub readme_html: Option<String>,
    pub in_library: Vec<(String, String)>,
    pub ollama_running: bool,
    /// Why the Ollama option is offered first (Settings → Inference → "When a model can run
    /// in both engines"); `None` keeps the built-in download first.
    pub prefer_ollama: Option<&'static str>,
    pub hf_url: String,
    pub base_models: Vec<String>,
}

fn fit_view(fit: Fit) -> &'static str {
    match fit {
        Fit::Comfortable => "ok",
        Fit::Tight => "warn",
        Fit::TooLarge => "danger",
        Fit::Unknown => "muted",
    }
}

pub async fn hf_model(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path((owner, name)): Path<(String, String)>,
) -> AppResult<Response> {
    let id = format!("{owner}/{name}");
    crate::hub::validate_repo_id(&id)?;
    let (info, cached_at) = state.hub.model_info(&id).await?;
    let compat = analyze_repo(&info);
    let snap = state.system.snapshot().await;
    let ctx_len = state.settings().generation.context_length as u64;
    let library = state.db.call(crate::db::models::list).await.unwrap_or_default();
    let mine: Vec<_> = library.iter().filter(|m| m.repo_id.as_deref() == Some(id.as_str())).collect();
    let downloaded_files: HashSet<String> = mine.iter().filter_map(|m| m.primary_file.clone()).collect();
    let any_builtin = compat.variants.iter().any(|v| v.builtin);
    let gguf_meta = info.summary.gguf.clone().unwrap_or_default();
    let variants = compat
        .variants
        .iter()
        .map(|v| {
            let weights = if compat.format == "safetensors" && compat.capability == crate::hub::Capability::Chat {
                v.total_size * 2
            } else {
                v.total_size
            };
            let est = estimate_llm_memory(
                weights,
                None,
                None,
                None,
                ctx_len.min(gguf_meta.context_length.unwrap_or(ctx_len)),
                4,
            );
            let (fit, msg) = assess_fit(est.total, snap.available_ram, snap.total_ram);
            VariantView {
                id: v.id.clone(),
                label: v.label.clone(),
                size: human_bytes(v.total_size),
                quant: v.quant.clone().unwrap_or_default(),
                builtin: v.builtin,
                note: v.note.clone().unwrap_or_default(),
                recommended: v.recommended,
                checked: v.recommended,
                ollama_ref: v.ollama_ref.clone().unwrap_or_default(),
                fit_class: fit_view(fit),
                fit_msg: msg,
                downloaded: v.primary_file.as_ref().map(|f| downloaded_files.contains(f)).unwrap_or(false)
                    || (v.primary_file.is_none() && mine.iter().any(|m| m.primary_file.is_none())),
            }
        })
        .collect::<Vec<_>>();
    let mut variants = variants;
    if !variants.iter().any(|v| v.checked)
        && let Some(v) = variants.first_mut()
    {
        v.checked = true;
    }
    let readme_html = state
        .hub
        .readme(&id, info.sha.as_deref().unwrap_or("main"))
        .await
        .filter(|r| !r.trim().is_empty())
        .map(|r| crate::render::model_card_html(&r, 8000));
    let mut files: Vec<FileView> = info
        .siblings
        .iter()
        .map(|s| FileView { path: s.rfilename.clone(), size: human_bytes(s.size()), lfs: s.lfs.is_some() })
        .collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let (support, support_class) = match compat.support {
        Support::Builtin => ("Runs in Local AI Studio", "ok"),
        Support::ViaOllama => ("Runs through Ollama", "info"),
        Support::DownloadOnly => ("Download only", "warn"),
    };
    let ollama_running = state.ollama.enabled() && state.ollama.version().await.is_ok();
    // Automatic: offer first the engine that runs on the graphics card. Ollama uses NVIDIA and
    // AMD cards by itself; the built-in engine only when built with CUDA (or on Apple Silicon).
    let accel = state.system.accel();
    let builtin_gpu = accel.cuda_usable || accel.metal_usable;
    let card_for_ollama = snap.gpus.iter().any(|g| !g.unified_memory && matches!(g.vendor.as_str(), "NVIDIA" | "AMD"));
    let prefer_ollama = if ollama_running && variants.iter().any(|v| !v.ollama_ref.is_empty()) {
        match state.settings().inference.preferred_backend {
            crate::config::BackendPreference::Ollama => Some("Shown first because Settings prefer Ollama."),
            crate::config::BackendPreference::Auto if !builtin_gpu && card_for_ollama => Some(
                "Recommended on this computer: Ollama runs the model on your graphics card, the built-in engine on the processor.",
            ),
            _ => None,
        }
    } else {
        None
    };
    let ctx = PageCtx::new(&state, &headers, &id, "discover").at(&format!("/discover/hf/{id}"));
    let page = HfModelPage {
        ctx: ctx.clone(),
        name: info.summary.short_name().to_string(),
        author: info.summary.author_name(),
        task: task_label(info.summary.pipeline_tag.as_deref().unwrap_or("")),
        license: info.summary.license().unwrap_or_default(),
        formats: info.summary.formats(),
        params: info.summary.parameter_count().map(human_params).unwrap_or_default(),
        downloads: human_count(info.summary.downloads),
        likes: human_count(info.summary.likes),
        updated: info.summary.last_modified.as_deref().map(relative_time).unwrap_or_default(),
        gated: info.summary.gated.is_some(),
        has_token: state.secrets.hf_token().await.is_some(),
        cached_at: cached_at.as_deref().map(relative_time),
        support,
        support_class,
        capability: compat.capability.label(),
        format: compat.format.clone(),
        summary: compat.summary.clone(),
        reasons: compat.reasons.clone(),
        variants,
        any_builtin,
        files,
        total_size: human_bytes(info.total_size()),
        readme_html,
        in_library: mine.iter().map(|m| (format!("/library/model/local:{}", m.id), m.display_name.clone())).collect(),
        ollama_running,
        prefer_ollama,
        hf_url: format!("{}/{}", state.hub.endpoint(), id),
        base_models: info.base_models(),
        id,
    };
    Ok(render(&ctx, &page))
}

#[derive(Deserialize, Default)]
pub struct HfDownloadForm {
    #[serde(default)]
    variant: Option<String>,
}

/// Queues the files of one variant (plus companion files) of a repository.
pub async fn enqueue_hf_variant(state: &SharedState, repo: &str, variant: Option<&str>) -> AppResult<String> {
    let (info, cached) = state.hub.model_info(repo).await?;
    if cached.is_some() {
        return Err(AppError::unavailable(
            "Hugging Face can't be reached right now, so the download can't start. Check your connection.",
        ));
    }
    if info.summary.gated.is_some() && state.secrets.hf_token().await.is_none() {
        return Err(AppError::Forbidden(format!(
            "\"{repo}\" is gated: open {}/{repo} in your browser, accept its terms, then add a Hugging Face token in Settings → Hugging Face.",
            state.hub.endpoint()
        )));
    }
    let compat = analyze_repo(&info);
    let v = match variant.filter(|v| !v.is_empty()) {
        Some(id) => compat.variants.iter().find(|v| v.id == id),
        None => compat.variants.iter().find(|v| v.recommended).or_else(|| compat.variants.first()),
    }
    .ok_or_else(|| AppError::bad_request("Choose which file to download."))?;
    let commit = info.sha.clone().unwrap_or_else(|| "main".into());
    let mut files = crate::downloads::files_from_siblings(&info, &commit, &v.files)?;
    for (i, (crepo, cpath)) in v.companions.iter().enumerate() {
        let (cinfo, ccached) = state.hub.model_info(crepo).await?;
        if ccached.is_some() {
            return Err(AppError::unavailable("Hugging Face can't be reached right now."));
        }
        let ccommit = cinfo.sha.clone().unwrap_or_else(|| "main".into());
        let mut extra = crate::downloads::files_from_siblings(&cinfo, &ccommit, std::slice::from_ref(cpath))?;
        let folder = if i == 0 { "clip-tokenizer" } else { "clip-tokenizer-2" };
        for f in &mut extra {
            f.local_path = format!("{folder}/{cpath}");
        }
        files.extend(extra);
    }
    let short = repo.rsplit('/').next().unwrap_or(repo);
    let display_name = match &v.quant {
        Some(q) => format!("{} · {q}", short.trim_end_matches("-GGUF").trim_end_matches("-gguf")),
        None => short.to_string(),
    };
    state
        .downloads
        .enqueue_hf(
            repo,
            &commit,
            files,
            DownloadOptions { display_name: Some(display_name), primary_file: v.primary_file.clone(), commit: None },
        )
        .await
}

pub async fn hf_download(
    State(state): State<SharedState>,
    Path((owner, name)): Path<(String, String)>,
    Form(f): Form<HfDownloadForm>,
) -> Response {
    let repo = format!("{owner}/{name}");
    match enqueue_hf_variant(&state, &repo, f.variant.as_deref()).await {
        Ok(_) => redirect_flash(
            "/downloads",
            "ok",
            "Download started. You can keep using the app; it continues in the background.",
        ),
        Err(e) => redirect_flash(&format!("/discover/hf/{repo}"), "error", &e.user_message()),
    }
}

pub struct TagView {
    pub name: String,
    pub tag: String,
    pub size: String,
    pub context: String,
    pub inputs: String,
    pub vision: bool,
    pub updated: String,
    pub installed: bool,
    pub fit_class: &'static str,
    pub fit_msg: String,
}

#[derive(Template)]
#[template(path = "ollama_model.html")]
pub struct OllamaModelPage {
    pub ctx: PageCtx,
    pub name: String,
    pub description: String,
    pub tags: Vec<TagView>,
    pub cached_at: Option<String>,
    pub running: bool,
    pub installed_anything: bool,
    pub library_url: String,
    pub install_os: &'static str,
    pub install_steps: Vec<&'static str>,
}

pub async fn ollama_model(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> AppResult<Response> {
    let name = name.trim_matches('/').to_string();
    let detail = state.ollama.library_detail(&name, false).await?;
    let running = state.ollama.enabled() && state.ollama.version().await.is_ok();
    let installed: HashSet<String> = if running {
        state.ollama.list_local().await.unwrap_or_default().into_iter().map(|m| m.name).collect()
    } else {
        HashSet::new()
    };
    let snap = state.system.snapshot().await;
    let tags: Vec<TagView> = detail
        .tags
        .iter()
        .map(|t| {
            let (fit, msg) = match t.size_bytes {
                Some(sz) => {
                    let est = estimate_llm_memory(sz, None, None, None, 4096, 2);
                    assess_fit(est.total, snap.available_ram, snap.total_ram)
                }
                None => (Fit::Unknown, "Size unknown.".to_string()),
            };
            let short = t.name.clone();
            let is_installed = installed.contains(&short)
                || (t.tag() == "latest" && installed.contains(&name))
                || installed.contains(&format!("{short}:latest"));
            TagView {
                tag: t.tag().to_string(),
                size: t.size_label.clone(),
                context: t.context.clone(),
                inputs: t.inputs.clone(),
                vision: t.has_image_input(),
                updated: t.updated.clone(),
                installed: is_installed,
                fit_class: fit_view(fit),
                fit_msg: msg,
                name: short,
            }
        })
        .collect();
    let (os, steps) = crate::ollama::install_instructions();
    let ctx = PageCtx::new(&state, &headers, &name, "discover").at(&format!("/discover/ollama/{name}"));
    let page = OllamaModelPage {
        ctx: ctx.clone(),
        description: detail.description,
        installed_anything: tags.iter().any(|t| t.installed),
        tags,
        cached_at: detail.cached_at.as_deref().map(relative_time),
        running,
        library_url: if name.contains('/') {
            format!("https://ollama.com/{name}")
        } else {
            format!("https://ollama.com/library/{name}")
        },
        install_os: os,
        install_steps: steps,
        name,
    };
    Ok(render(&ctx, &page))
}

#[derive(Deserialize, Default)]
pub struct PullForm {
    #[serde(default)]
    name: String,
    #[serde(default)]
    back: Option<String>,
}

pub async fn ollama_pull(State(state): State<SharedState>, Form(f): Form<PullForm>) -> Response {
    let back =
        f.back.filter(|b| b.starts_with('/') && !b.starts_with("//")).unwrap_or_else(|| "/discover?tab=ollama".into());
    match state.downloads.enqueue_ollama(f.name.trim()).await {
        Ok(_) => redirect_flash("/downloads", "ok", &format!("Ollama is downloading {}.", f.name.trim())),
        Err(e) => redirect_flash(&back, "error", &e.user_message()),
    }
}
