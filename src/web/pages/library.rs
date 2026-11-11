//! The local model library: everything downloaded (Hugging Face, imported GGUF files and
//! Ollama models) with load/unload, rename, delete, inspect and update actions.

use std::collections::HashMap;
use std::path::PathBuf;

use askama::Template;
use axum::Form;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use serde::Deserialize;

use crate::SharedState;
use crate::error::{AppError, AppResult};
use crate::hub::Capability;
use crate::inference::ModelRef;
use crate::library::LibraryItem;
use crate::system::{assess_fit, estimate_llm_memory};
use crate::util::{human_bytes, human_params, relative_time};
use crate::web::page::{PageCtx, redirect_flash, render};

pub struct LibRow {
    pub model_ref: String,
    pub href: String,
    pub name: String,
    pub original: String,
    pub source: String,
    pub format: String,
    pub arch: String,
    pub size: String,
    pub quant: String,
    pub params: String,
    pub capability: &'static str,
    pub cap_class: &'static str,
    pub runnable: bool,
    pub problem: String,
    pub loaded: bool,
    pub last_used: String,
    pub location: String,
    pub engine: &'static str,
    pub chat_href: String,
    pub use_href: String,
    pub use_label: &'static str,
    pub can_load: bool,
    pub can_open: bool,
    pub is_ollama: bool,
    pub duplicate_of: String,
    pub delete_note: &'static str,
}

fn cap_class(c: Capability) -> &'static str {
    match c {
        Capability::Chat => "cap-chat",
        Capability::Vision => "cap-vision",
        Capability::Transcription => "cap-audio",
        Capability::ImageGeneration => "cap-image",
        _ => "cap-other",
    }
}

fn row_from(item: &LibraryItem, dupes: &HashMap<String, String>) -> LibRow {
    let is_ollama = item.engine == "Ollama";
    let (use_href, use_label) = match item.capability {
        Capability::Chat | Capability::Vision if item.runnable => (format!("/chat?model={}", item.model_ref), "Chat"),
        Capability::Transcription if item.runnable => ("/transcribe".to_string(), "Transcribe"),
        Capability::ImageGeneration if item.runnable => ("/create".to_string(), "Create images"),
        _ => (String::new(), ""),
    };
    LibRow {
        href: format!("/library/model/{}", item.model_ref),
        model_ref: item.model_ref.clone(),
        name: item.name.clone(),
        original: item.original_name.clone(),
        source: item.source.clone(),
        format: item.format.clone(),
        arch: item.architecture.clone(),
        size: item.size_label(),
        quant: item.quantization.clone(),
        params: item.parameters.clone(),
        capability: item.capability.label(),
        cap_class: cap_class(item.capability),
        runnable: item.runnable,
        problem: item.problems.first().cloned().unwrap_or_default(),
        loaded: item.loaded,
        last_used: item.last_used.as_deref().map(relative_time).unwrap_or_else(|| "never".into()),
        location: item.location.clone(),
        engine: item.engine,
        chat_href: String::new(),
        use_href,
        use_label,
        can_load: item.runnable && matches!(item.capability, Capability::Chat | Capability::Vision),
        can_open: !is_ollama,
        is_ollama,
        duplicate_of: dupes.get(&item.model_ref).cloned().unwrap_or_default(),
        delete_note: if is_ollama {
            "Ollama will delete this model's files (shared layers used by other models are kept)."
        } else if item.source == "Imported" {
            "The model is removed from the library. Your original file is kept unless you tick the box."
        } else {
            "The downloaded files are deleted from disk."
        },
    }
}

#[derive(Deserialize, Default)]
pub struct LibraryQuery {
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    q: Option<String>,
}

pub struct KindTab {
    pub value: &'static str,
    pub label: &'static str,
    pub count: usize,
    pub active: bool,
}

#[derive(Template)]
#[template(path = "library.html")]
pub struct LibraryPage {
    pub ctx: PageCtx,
    pub rows: Vec<LibRow>,
    pub tabs: Vec<KindTab>,
    pub kind: String,
    pub q: String,
    pub ollama_error: Option<String>,
    pub total_size: String,
    pub total_count: usize,
    pub models_dir: String,
}

pub async fn page(State(state): State<SharedState>, headers: HeaderMap, Query(q): Query<LibraryQuery>) -> Response {
    let (items, ollama_error) = super::library_items(&state).await;
    let kind = q.kind.clone().unwrap_or_default();
    let text = q.q.clone().unwrap_or_default().to_lowercase();
    // Duplicate detection: same Ollama digest or same local file hash.
    let mut dupes: HashMap<String, String> = HashMap::new();
    if let Ok(models) = state.ollama.list_local().await {
        let mut by_digest: HashMap<String, String> = HashMap::new();
        for m in &models {
            match by_digest.get(&m.digest) {
                Some(first) => {
                    dupes.insert(format!("ollama:{}", m.name), first.clone());
                }
                None => {
                    by_digest.insert(m.digest.clone(), m.name.clone());
                }
            }
        }
    }
    let rows_db = state.db.call(crate::db::models::list).await.unwrap_or_default();
    let mut by_hash: HashMap<String, String> = HashMap::new();
    for m in &rows_db {
        if let Some(primary) = &m.primary_file
            && let Some(sha) = m.files.iter().find(|f| &f.path == primary).and_then(|f| f.sha256.clone())
        {
            match by_hash.get(&sha) {
                Some(first) => {
                    dupes.insert(format!("local:{}", m.id), first.clone());
                }
                None => {
                    by_hash.insert(sha, m.display_name.clone());
                }
            }
        }
    }
    let matches_kind = |i: &LibraryItem, k: &str| match k {
        "chat" => i.capability == Capability::Chat,
        "vision" => i.capability == Capability::Vision,
        "audio" => i.capability == Capability::Transcription,
        "image" => i.capability == Capability::ImageGeneration,
        "problems" => !i.runnable,
        _ => true,
    };
    let tabs = [
        ("", "All"),
        ("chat", "Chat"),
        ("vision", "Sees images"),
        ("audio", "Speech to text"),
        ("image", "Image generation"),
        ("problems", "Can't run"),
    ]
    .iter()
    .map(|(v, l)| KindTab {
        value: v,
        label: l,
        count: items.iter().filter(|i| matches_kind(i, v)).count(),
        active: kind == *v,
    })
    .collect();
    let total_size: u64 = items.iter().map(|i| i.size_bytes).sum();
    let total_count = items.len();
    let rows = items
        .iter()
        .filter(|i| matches_kind(i, &kind))
        .filter(|i| {
            text.is_empty() || i.name.to_lowercase().contains(&text) || i.original_name.to_lowercase().contains(&text)
        })
        .map(|i| row_from(i, &dupes))
        .collect();
    let ctx = PageCtx::new(&state, &headers, "Library", "library").at("/library");
    let page = LibraryPage {
        ctx: ctx.clone(),
        rows,
        tabs,
        kind,
        q: q.q.unwrap_or_default(),
        ollama_error,
        total_size: human_bytes(total_size),
        total_count,
        models_dir: state.settings().models_dir(&state.paths).to_string_lossy().to_string(),
    };
    render(&ctx, &page)
}

#[derive(Deserialize, Default)]
pub struct ImportForm {
    #[serde(default)]
    path: String,
}

pub async fn import(State(state): State<SharedState>, Form(f): Form<ImportForm>) -> Response {
    let path = PathBuf::from(f.path.trim().trim_matches('"'));
    match crate::library::import_gguf(&state.db, &path).await {
        Ok(m) if m.analysis.runnable => redirect_flash(
            &format!("/library/model/local:{}", m.id),
            "ok",
            &format!("Added \"{}\" to your library. The file stays where it is.", m.display_name),
        ),
        Ok(m) => redirect_flash(
            &format!("/library/model/local:{}", m.id),
            "info",
            &format!(
                "Added \"{}\", but it can't run in the built-in engine: {}",
                m.display_name,
                m.analysis.problems.join(" ")
            ),
        ),
        Err(e) => redirect_flash("/library", "error", &e),
    }
}

#[derive(Deserialize, Default)]
pub struct ActionForm {
    #[serde(default)]
    model_ref: String,
    #[serde(default)]
    action: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    delete_files: Option<String>,
    #[serde(default)]
    back: Option<String>,
}

pub async fn action(State(state): State<SharedState>, Form(f): Form<ActionForm>) -> Response {
    let back =
        f.back.clone().filter(|b| b.starts_with('/') && !b.starts_with("//")).unwrap_or_else(|| "/library".into());
    match run_action(&state, &f).await {
        Ok((to, msg)) => redirect_flash(to.as_deref().unwrap_or(&back), "ok", &msg),
        Err(e) => redirect_flash(&back, "error", &e.user_message()),
    }
}

async fn run_action(state: &SharedState, f: &ActionForm) -> AppResult<(Option<String>, String)> {
    let model = ModelRef::parse(&f.model_ref).ok_or_else(|| AppError::bad_request("Unknown model."))?;
    match (f.action.as_str(), &model) {
        ("load", ModelRef::Local(id)) => {
            state.inference.load_local(id).await?;
            Ok((None, "Model loaded into memory. Chats with it will start faster.".into()))
        }
        ("load", ModelRef::Ollama(name)) => {
            state.ollama.set_loaded(name, true).await?;
            Ok((None, format!("Ollama loaded {name}.")))
        }
        ("unload", ModelRef::Local(id)) => {
            state.inference.unload_local(id);
            Ok((None, "Model unloaded; its memory is free again.".into()))
        }
        ("unload", ModelRef::Ollama(name)) => {
            state.ollama.set_loaded(name, false).await?;
            Ok((None, format!("Ollama unloaded {name}.")))
        }
        ("rename", _) => {
            let name = f.name.clone().unwrap_or_default().trim().to_string();
            if name.is_empty() || name.chars().count() > 120 {
                return Err(AppError::bad_request("Names must be 1–120 characters."));
            }
            let r = f.model_ref.clone();
            let n = name.clone();
            match model {
                ModelRef::Local(id) => {
                    state.db.call(move |c| crate::db::models::rename(c, &id, &n)).await?;
                }
                ModelRef::Ollama(_) => {
                    state.db.call(move |c| crate::db::models::set_meta_name(c, &r, Some(&n))).await?;
                }
            }
            Ok((None, format!("Renamed to \"{name}\".")))
        }
        ("delete", ModelRef::Local(id)) => {
            state.inference.unload_local(id);
            let models_dir = state.settings().models_dir(&state.paths);
            crate::library::delete_model(&state.db, &models_dir, id, f.delete_files.is_some())
                .await
                .map_err(AppError::Internal)?;
            Ok((Some("/library".into()), "Model deleted.".into()))
        }
        ("delete", ModelRef::Ollama(name)) => {
            state.ollama.delete(name).await?;
            let r = f.model_ref.clone();
            let _ = state.db.call(move |c| crate::db::models::delete_meta(c, &r)).await;
            Ok((Some("/library".into()), format!("Deleted {name} from Ollama.")))
        }
        ("open", ModelRef::Local(id)) => {
            let i = id.clone();
            let m = state
                .db
                .call(move |c| crate::db::models::get(c, &i))
                .await?
                .ok_or_else(|| AppError::not_found("Model not found."))?;
            crate::util::open_in_file_manager(std::path::Path::new(&m.path))
                .map_err(|e| AppError::unavailable(format!("Couldn't open the folder: {e}")))?;
            Ok((None, format!("Opened {} in your file manager.", m.path)))
        }
        ("reanalyze", ModelRef::Local(id)) => {
            let m = crate::library::reanalyze(&state.db, id).await.map_err(AppError::Internal)?;
            Ok((
                None,
                if m.analysis.runnable {
                    "Checked again: the model can run.".into()
                } else {
                    format!("Checked again: {}", m.analysis.problems.join(" "))
                },
            ))
        }
        ("update", ModelRef::Ollama(name)) => {
            state.downloads.enqueue_ollama(name).await?;
            Ok((Some("/downloads".into()), format!("Checking {name} for updates; only changed parts are downloaded.")))
        }
        ("update", ModelRef::Local(id)) => {
            let i = id.clone();
            let m = state
                .db
                .call(move |c| crate::db::models::get(c, &i))
                .await?
                .ok_or_else(|| AppError::not_found("Model not found."))?;
            let Some(repo) = m.repo_id.clone() else {
                return Err(AppError::bad_request("Imported files can't be updated automatically."));
            };
            let (info, cached) = state.hub.model_info(&repo).await?;
            if cached.is_some() {
                return Err(AppError::unavailable("Hugging Face can't be reached right now."));
            }
            if info.sha.is_some() && info.sha == m.revision {
                Ok((None, "You already have the latest version.".into()))
            } else {
                Ok((
                    Some(format!("/discover/hf/{repo}")),
                    "A newer version is available on Hugging Face. Delete the old copy and download it again from this page to update.".into(),
                ))
            }
        }
        _ => Err(AppError::bad_request("That action isn't available for this model.")),
    }
}

pub struct KeyVal {
    pub key: &'static str,
    pub value: String,
}

#[derive(Template)]
#[template(path = "library_model.html")]
pub struct LibraryModelPage {
    pub ctx: PageCtx,
    pub row: LibRow,
    pub facts: Vec<KeyVal>,
    pub problems: Vec<String>,
    pub notes: Vec<String>,
    pub files: Vec<(String, String, String)>,
    pub memory: Vec<KeyVal>,
    pub fit_class: &'static str,
    pub fit_msg: String,
    pub gpu_msg: String,
    pub template_excerpt: String,
    pub license_excerpt: String,
    pub parameters_text: String,
    pub hf_href: String,
}

pub async fn detail(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Path(model_ref): Path<String>,
) -> AppResult<Response> {
    let (items, _) = super::library_items(&state).await;
    let item = items
        .iter()
        .find(|i| i.model_ref == model_ref)
        .ok_or_else(|| AppError::not_found("That model isn't in your library (it may have been deleted)."))?;
    let row = row_from(item, &HashMap::new());
    let snap = state.system.snapshot().await;
    let ctx_len = state.settings().generation.context_length as u64;
    let mut facts = vec![
        KeyVal { key: "Source", value: item.source.clone() },
        KeyVal { key: "Engine", value: item.engine.to_string() },
        KeyVal { key: "Format", value: item.format.clone() },
        KeyVal { key: "Architecture", value: item.architecture.clone() },
        KeyVal { key: "Parameters", value: item.parameters.clone() },
        KeyVal { key: "Quantisation", value: item.quantization.clone() },
        KeyVal { key: "Size on disk", value: item.size_label() },
        KeyVal { key: "Capabilities", value: item.capabilities.join(", ") },
        KeyVal { key: "Location", value: item.location.clone() },
    ];
    let mut problems = item.problems.clone();
    let mut notes = Vec::new();
    let mut files = Vec::new();
    let mut template_excerpt = String::new();
    let mut license_excerpt = String::new();
    let mut parameters_text = String::new();
    let mut hf_href = String::new();
    let (weights, layers, kv, hd, mctx) = match ModelRef::parse(&model_ref) {
        Some(ModelRef::Local(id)) => {
            let m = state
                .db
                .call(move |c| crate::db::models::get(c, &id))
                .await?
                .ok_or_else(|| AppError::not_found("Model not found."))?;
            let a = &m.analysis;
            if let Some(c) = a.context_length {
                facts.push(KeyVal { key: "Trained context", value: format!("{c} tokens") });
            }
            if let Some(t) = &a.tokenizer_kind {
                facts.push(KeyVal { key: "Tokenizer", value: t.clone() });
            }
            facts.push(KeyVal {
                key: "Chat template",
                value: if a.has_chat_template { "Included".into() } else { "Not included".into() },
            });
            if let Some(r) = &m.repo_id {
                facts.push(KeyVal { key: "Repository", value: r.clone() });
                hf_href = format!("/discover/hf/{r}");
            }
            if let Some(rev) = &m.revision {
                facts.push(KeyVal { key: "Revision", value: rev.chars().take(12).collect() });
            }
            facts.push(KeyVal { key: "Added", value: relative_time(&m.created_at) });
            problems = a.problems.clone();
            notes = a.notes.clone();
            files = m
                .files
                .iter()
                .map(|f| {
                    (
                        f.path.clone(),
                        human_bytes(f.size),
                        f.sha256.clone().map(|s| s.chars().take(16).collect()).unwrap_or_default(),
                    )
                })
                .collect();
            let w = if m.format == "safetensors" && a.capability == Capability::Chat {
                m.size_bytes.max(0) as u64 * 2
            } else {
                m.size_bytes.max(0) as u64
            };
            (w, a.n_layers, a.n_kv_heads, a.head_dim, a.context_length)
        }
        Some(ModelRef::Ollama(name)) => {
            let show = state.ollama.show(&name).await.ok();
            if let Some(s) = &show {
                if let Some(p) = s.parameter_count() {
                    facts.push(KeyVal { key: "Exact parameters", value: human_params(p) });
                }
                if let Some(c) = s.context_length() {
                    facts.push(KeyVal { key: "Trained context", value: format!("{c} tokens") });
                }
                template_excerpt = crate::util::truncate_chars(&s.template, 1200);
                license_excerpt = crate::util::truncate_chars(&s.license, 1200);
                parameters_text = s.parameters.clone();
            } else {
                notes.push("Ollama isn't reachable, so details are limited.".into());
            }
            (item.size_bytes, None, None, None, show.as_ref().and_then(|s| s.context_length()))
        }
        None => return Err(AppError::bad_request("Invalid model reference.")),
    };
    let ctx_used = mctx.map(|c| c.min(ctx_len)).unwrap_or(ctx_len);
    let est = estimate_llm_memory(weights, layers, kv, hd, ctx_used, 4);
    let (fit, fit_msg) = assess_fit(est.total, snap.available_ram, snap.total_ram);
    let gpu_msg = match snap.best_gpu_free() {
        Some(free) if est.total > free => format!(
            "Your GPU has about {} free, less than the {} needed, so a GPU engine would fall back to the CPU for part of the model.",
            human_bytes(free),
            human_bytes(est.total)
        ),
        Some(free) => {
            format!("Fits in the GPU's free memory ({} free) when run with a GPU-enabled engine.", human_bytes(free))
        }
        None => "No dedicated GPU memory detected.".into(),
    };
    let memory = vec![
        KeyVal { key: "Weights", value: human_bytes(est.weights) },
        KeyVal { key: "Context memory", value: format!("{} for {ctx_used} tokens", human_bytes(est.kv_cache)) },
        KeyVal { key: "Overhead", value: human_bytes(est.overhead) },
        KeyVal { key: "Estimated total", value: human_bytes(est.total) },
    ];
    let title = item.name.clone();
    let ctx = PageCtx::new(&state, &headers, &title, "library").at(&format!("/library/model/{model_ref}"));
    let fit_class = match fit {
        crate::system::Fit::Comfortable => "ok",
        crate::system::Fit::Tight => "warn",
        crate::system::Fit::TooLarge => "danger",
        crate::system::Fit::Unknown => "muted",
    };
    let page = LibraryModelPage {
        ctx: ctx.clone(),
        row,
        facts: facts.into_iter().filter(|f| !f.value.is_empty()).collect(),
        problems,
        notes,
        files,
        memory,
        fit_class,
        fit_msg,
        gpu_msg,
        template_excerpt,
        license_excerpt,
        parameters_text,
        hf_href,
    };
    Ok(render(&ctx, &page))
}
