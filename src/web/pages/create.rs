//! Image generation (Stable Diffusion via the built-in engine) with history.

use std::path::PathBuf;
use std::time::Instant;

use askama::Template;
use axum::Form;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::SharedState;
use crate::db::jobs::{GenerationRow, ImageOutput, ImageParams, JobStatus};
use crate::error::{AppError, AppResult};
use crate::hub::SdVersion;
use crate::inference::diffusion::ImageRequest;
use crate::util::{now_rfc3339, relative_time};
use crate::web::page::{PageCtx, redirect_flash, render};

pub struct ModelOpt {
    pub id: String,
    pub label: String,
    pub selected: bool,
}

pub struct SizeOpt {
    pub value: String,
    pub label: String,
    pub selected: bool,
}

pub struct ImgView {
    pub url: String,
    pub download: String,
    pub size: String,
    pub seed: u64,
}

pub struct GenView {
    pub id: String,
    pub prompt: String,
    pub negative: String,
    pub model: String,
    pub details: String,
    pub created: String,
    pub status: &'static str,
    pub status_class: &'static str,
    pub error: String,
    pub images: Vec<ImgView>,
    pub reuse_href: String,
}

pub struct RunningView {
    pub id: String,
    pub label: String,
    pub percent: u32,
    pub detail: String,
}

#[derive(Template)]
#[template(path = "create.html")]
pub struct CreatePage {
    pub ctx: PageCtx,
    pub models: Vec<ModelOpt>,
    pub has_models: bool,
    pub prompt: String,
    pub negative: String,
    pub sizes: Vec<SizeOpt>,
    pub steps: u32,
    pub guidance: String,
    pub seed: String,
    pub count: u32,
    pub version_label: String,
    pub turbo: bool,
    pub running: Option<RunningView>,
    pub history: Vec<GenView>,
    pub outputs_dir: String,
    /// Suggested folder for "Save to folder" (the user's Pictures folder when known).
    pub pictures_dir: String,
    pub engine_label: String,
}

fn round64(v: f64) -> u32 {
    ((v / 64.0).round() as u32).max(4) * 64
}

/// Size presets for a model's native resolution.
pub fn size_presets(base: u32) -> Vec<(u32, u32, &'static str)> {
    let s = base as f64;
    vec![
        (base, base, "Square"),
        (round64(s * 1.1547), round64(s * 0.866), "Landscape 4:3"),
        (round64(s * 0.866), round64(s * 1.1547), "Portrait 3:4"),
        (round64(s * 1.333), round64(s * 0.75), "Wide 16:9"),
        (round64(s * 0.75), round64(s * 1.333), "Tall 9:16"),
    ]
}

#[derive(Deserialize, Default)]
pub struct CreateQuery {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    negative: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    guidance: Option<f64>,
    #[serde(default)]
    seed: Option<String>,
    #[serde(default)]
    count: Option<u32>,
}

fn version_of(m: &crate::db::models::ModelRow) -> Option<SdVersion> {
    m.analysis.sd_version
}

pub async fn page(State(state): State<SharedState>, headers: HeaderMap, Query(q): Query<CreateQuery>) -> Response {
    let models = super::models_of_kind(&state, "image-generation").await;
    let runnable: Vec<_> = models.iter().filter(|m| m.analysis.runnable).collect();
    let selected = q
        .model
        .clone()
        .filter(|id| runnable.iter().any(|m| &m.id == id))
        .or_else(|| runnable.first().map(|m| m.id.clone()))
        .unwrap_or_default();
    let version = runnable.iter().find(|m| m.id == selected).and_then(|m| version_of(m)).unwrap_or(SdVersion::Turbo);
    let base = version.default_size();
    let size_q = q.size.clone().unwrap_or_else(|| format!("{base}x{base}"));
    let history_rows = state.db.call(|c| crate::db::jobs::list_generations(c, 40)).await.unwrap_or_default();
    let running = history_rows
        .iter()
        .find(|g| g.status.is_active())
        .and_then(|g| state.jobs.get(&g.id).map(|p| (g.id.clone(), p)))
        .map(|(id, p)| RunningView {
            id,
            label: p.label.clone(),
            percent: p.percent(),
            detail: if p.total > 0 { format!("Step {} of {}", p.step, p.total) } else { "Preparing…".into() },
        });
    let ctx = PageCtx::new(&state, &headers, "Create images", "create").at("/create");
    let history = history_rows.iter().map(gen_view).collect();
    let page = CreatePage {
        ctx: ctx.clone(),
        has_models: !runnable.is_empty(),
        models: runnable
            .iter()
            .map(|m| ModelOpt {
                id: m.id.clone(),
                label: format!(
                    "{} ({})",
                    m.display_name,
                    version_of(m).map(|v| v.label()).unwrap_or("Stable Diffusion")
                ),
                selected: m.id == selected,
            })
            .collect(),
        prompt: q.prompt.clone().unwrap_or_default(),
        negative: q.negative.clone().unwrap_or_default(),
        sizes: size_presets(base)
            .into_iter()
            .map(|(w, h, l)| {
                let v = format!("{w}x{h}");
                SizeOpt { selected: v == size_q, label: format!("{l} · {w}×{h}"), value: v }
            })
            .collect(),
        steps: q.steps.unwrap_or_else(|| version.default_steps()),
        guidance: q.guidance.unwrap_or_else(|| version.default_guidance()).to_string(),
        seed: q.seed.clone().unwrap_or_default(),
        count: q.count.unwrap_or(1).clamp(1, 4),
        version_label: version.label().into(),
        turbo: version.is_turbo(),
        running,
        history,
        outputs_dir: state.settings().outputs_dir(&state.paths).to_string_lossy().to_string(),
        pictures_dir: directories::UserDirs::new()
            .and_then(|u| u.picture_dir().map(|p| p.to_string_lossy().to_string()))
            .unwrap_or_default(),
        engine_label: state.inference.device_label().unwrap_or_else(|e| e),
    };
    render(&ctx, &page)
}

fn gen_view(g: &GenerationRow) -> GenView {
    let (status, class) = match g.status {
        JobStatus::Queued | JobStatus::Running => ("Generating", "info"),
        JobStatus::Done => ("Done", "ok"),
        JobStatus::Failed => ("Failed", "danger"),
        JobStatus::Cancelled => ("Cancelled", "muted"),
    };
    GenView {
        id: g.id.clone(),
        prompt: g.prompt.clone(),
        negative: g.negative_prompt.clone(),
        model: g.model_name.clone(),
        details: format!(
            "{}×{} · {} step{} · guidance {} · seed {}{}",
            g.params.width,
            g.params.height,
            g.params.steps,
            if g.params.steps == 1 { "" } else { "s" },
            g.params.guidance_scale,
            g.params.seed,
            g.duration_ms.map(|d| format!(" · {:.1} s", d as f64 / 1000.0)).unwrap_or_default()
        ),
        created: relative_time(&g.created_at),
        status,
        status_class: class,
        error: g.error.clone().unwrap_or_default(),
        images: g
            .outputs
            .iter()
            .map(|o| ImgView {
                url: format!("/files/outputs/{}", o.file),
                download: format!("/files/outputs/{}?download=1", o.file),
                size: format!("{}×{}", o.width, o.height),
                seed: o.seed,
            })
            .collect(),
        reuse_href: format!(
            "/create?model={}&prompt={}&negative={}&size={}x{}&steps={}&guidance={}&seed={}&count={}",
            crate::util::url_encode(&g.model_id),
            crate::util::url_encode(&g.prompt),
            crate::util::url_encode(&g.negative_prompt),
            g.params.width,
            g.params.height,
            g.params.steps,
            g.params.guidance_scale,
            g.params.seed,
            g.params.count
        ),
    }
}

/// A small self-refreshing document shown in an `<iframe>` on the Create page while an
/// image is generated, so the main page (and anything typed into it) never reloads.
#[derive(Template)]
#[template(path = "create_progress.html")]
pub struct ProgressFrame {
    pub theme: &'static str,
    pub dark: bool,
    pub css_rev: &'static str,
    pub running: bool,
    pub label: String,
    pub detail: String,
    pub percent: u32,
    pub g: GenView,
}

pub async fn progress(State(state): State<SharedState>, Path(id): Path<String>) -> AppResult<Response> {
    let i = id.clone();
    let row = state
        .db
        .call(move |c| crate::db::jobs::get_generation(c, &i))
        .await?
        .ok_or_else(|| AppError::not_found("That image job doesn't exist."))?;
    let running = row.status.is_active();
    let (label, detail, percent) = match state.jobs.get(&id) {
        Some(p) if running => (
            p.label.clone(),
            if p.total > 0 { format!("Step {} of {}", p.step, p.total) } else { "Preparing…".to_string() },
            p.percent(),
        ),
        _ if running => ("Working…".to_string(), String::new(), 0),
        _ => (String::new(), String::new(), 100),
    };
    let theme = state.settings().ui.theme;
    let frame = ProgressFrame {
        theme: theme.slug(),
        dark: theme.is_dark(),
        css_rev: crate::web::assets::stylesheet_rev(),
        running,
        label,
        detail,
        percent,
        g: gen_view(&row),
    };
    let html = frame.render().map_err(|e| AppError::Internal(anyhow::anyhow!("template error: {e}")))?;
    let mut r = axum::response::Html(html).into_response();
    r.headers_mut().insert(axum::http::header::CACHE_CONTROL, axum::http::HeaderValue::from_static("no-store"));
    Ok(r)
}

#[derive(Deserialize, Default)]
pub struct CreateForm {
    #[serde(default)]
    model_id: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    negative: String,
    #[serde(default)]
    size: String,
    #[serde(default)]
    steps: Option<String>,
    #[serde(default)]
    guidance: Option<String>,
    #[serde(default)]
    seed: Option<String>,
    #[serde(default)]
    count: Option<String>,
}

fn parse_request(f: &CreateForm, version: SdVersion) -> AppResult<ImageParams> {
    let (w, h) = f
        .size
        .split_once('x')
        .and_then(|(w, h)| Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?)))
        .unwrap_or((version.default_size(), version.default_size()));
    if !(256..=2048).contains(&w) || !(256..=2048).contains(&h) || w % 8 != 0 || h % 8 != 0 {
        return Err(AppError::bad_request("Width and height must be between 256 and 2048 and divisible by 8."));
    }
    let steps = match f.steps.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => s.parse::<u32>().map_err(|_| AppError::bad_request("Steps must be a whole number."))?,
        None => version.default_steps(),
    };
    if !(1..=100).contains(&steps) {
        return Err(AppError::bad_request("Steps must be between 1 and 100."));
    }
    let guidance = match f.guidance.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => s.parse::<f64>().map_err(|_| AppError::bad_request("Guidance must be a number."))?,
        None => version.default_guidance(),
    };
    if !(0.0..=20.0).contains(&guidance) || guidance.is_nan() {
        return Err(AppError::bad_request("Guidance must be between 0 and 20."));
    }
    let seed = match f.seed.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => s
            .parse::<u64>()
            .map_err(|_| AppError::bad_request("The seed must be a whole number (or empty for random)."))?,
        None => {
            let n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1);
            n % 4_294_967_295
        }
    };
    let count = f.count.as_deref().and_then(|c| c.trim().parse::<u32>().ok()).unwrap_or(1);
    if !(1..=4).contains(&count) {
        return Err(AppError::bad_request("You can create 1 to 4 images at a time."));
    }
    Ok(ImageParams { width: w, height: h, steps, guidance_scale: guidance, seed, count })
}

pub async fn start(State(state): State<SharedState>, Form(f): Form<CreateForm>) -> Response {
    match start_inner(&state, &f).await {
        Ok(_) => redirect_flash(
            "/create",
            "ok",
            "Started. You can follow the progress below and write your next prompt meanwhile.",
        ),
        Err(e) => redirect_flash(
            &format!(
                "/create?model={}&prompt={}",
                crate::util::url_encode(&f.model_id),
                crate::util::url_encode(&f.prompt)
            ),
            "error",
            &e.user_message(),
        ),
    }
}

async fn start_inner(state: &SharedState, f: &CreateForm) -> AppResult<String> {
    if f.prompt.trim().is_empty() {
        return Err(AppError::bad_request("Describe the image you want first."));
    }
    if f.prompt.chars().count() > 2000 || f.negative.chars().count() > 2000 {
        return Err(AppError::bad_request("Prompts are limited to 2,000 characters."));
    }
    let id = f.model_id.clone();
    let model = state
        .db
        .call(move |c| crate::db::models::get(c, &id))
        .await?
        .filter(|m| m.kind == "image-generation" && m.analysis.runnable)
        .ok_or_else(|| {
            AppError::bad_request("Choose an image model (download SD-Turbo from Discover if you have none).")
        })?;
    let version =
        model.analysis.sd_version.ok_or_else(|| AppError::bad_request("Unknown Stable Diffusion version."))?;
    let params = parse_request(f, version)?;
    let busy = state.db.call(|c| crate::db::jobs::list_generations(c, 10)).await?.iter().any(|g| g.status.is_active());
    if busy {
        return Err(AppError::conflict("An image is already being generated. Wait for it or cancel it."));
    }
    let row = GenerationRow {
        id: crate::util::new_id(),
        kind: "image".into(),
        model_id: model.id.clone(),
        model_name: model.display_name.clone(),
        prompt: f.prompt.trim().to_string(),
        negative_prompt: f.negative.trim().to_string(),
        params: params.clone(),
        outputs: Vec::new(),
        status: JobStatus::Running,
        error: None,
        duration_ms: None,
        created_at: now_rfc3339(),
    };
    let r = row.clone();
    state.db.call(move |c| crate::db::jobs::insert_generation(c, &r)).await?;
    let cancel = state.jobs.start(
        &row.id,
        &format!("Creating {} with {}", crate::util::count_noun(params.count, "image", "images"), model.display_name),
    );
    let st = state.clone();
    let id = row.id.clone();
    tokio::spawn(async move {
        run_generation(st, row, model, version, cancel).await;
    });
    Ok(id)
}

async fn run_generation(
    state: SharedState,
    mut row: GenerationRow,
    model: crate::db::models::ModelRow,
    version: SdVersion,
    cancel: tokio_util::sync::CancellationToken,
) {
    let started = Instant::now();
    let id = row.id.clone();
    let outputs_dir = state.settings().outputs_dir(&state.paths);
    let month = chrono::Utc::now().format("%Y-%m").to_string();
    let result: Result<Vec<ImageOutput>, String> = async {
        let (device, _) = state.inference.device()?;
        state.jobs.update(&id, |p| p.label = "Loading the image model (first time can take a minute)…".into());
        let dir = PathBuf::from(&model.path);
        let mid = model.id.clone();
        let st = state.clone();
        let pipeline =
            tokio::task::spawn_blocking(move || st.inference.diffusion.get_or_load(&dir, &mid, version, device))
                .await
                .map_err(|e| format!("loading crashed: {e}"))?
                .map_err(|e| format!("Could not load the image model: {e:#}"))?;
        tokio::fs::create_dir_all(outputs_dir.join("images").join(&month)).await.map_err(|e| e.to_string())?;
        let mut outs = Vec::new();
        let total_images = row.params.count as usize;
        for k in 0..total_images {
            let req = ImageRequest {
                prompt: row.prompt.clone(),
                negative_prompt: row.negative_prompt.clone(),
                width: row.params.width as usize,
                height: row.params.height as usize,
                steps: row.params.steps as usize,
                guidance_scale: row.params.guidance_scale,
                seed: row.params.seed + k as u64,
            };
            let p2 = pipeline.clone();
            let c2 = cancel.clone();
            let st = state.clone();
            let jid = id.clone();
            let label = format!("Image {} of {total_images}", k + 1);
            let img = tokio::task::spawn_blocking(move || {
                let mut guard = p2.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                guard.generate(&req, &c2, &mut |step, total| {
                    st.jobs.update(&jid, |p| {
                        p.label = label.clone();
                        p.step = step;
                        p.total = total;
                    });
                })
            })
            .await
            .map_err(|e| format!("generation crashed: {e}"))?
            .map_err(|e| if e.to_string() == "cancelled" { "cancelled".to_string() } else { format!("{e:#}") })?;
            let rel = format!("images/{month}/{}_{k}.png", id);
            let path = outputs_dir.join(&rel);
            let (w, h) = (img.width(), img.height());
            tokio::task::spawn_blocking(move || img.save(&path))
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| format!("Could not save the image: {e}"))?;
            outs.push(ImageOutput { file: rel, width: w, height: h, seed: req_seed(&row, k) });
            row.outputs = outs.clone();
            let r = row.clone();
            let _ = state.db.call(move |c| crate::db::jobs::update_generation(c, &r)).await;
        }
        Ok(outs)
    }
    .await;
    match result {
        Ok(outs) => {
            row.outputs = outs;
            row.status = JobStatus::Done;
        }
        Err(e) if e == "cancelled" => {
            row.status = JobStatus::Cancelled;
        }
        Err(e) => {
            tracing::warn!("image generation failed: {e}");
            row.status = JobStatus::Failed;
            row.error = Some(e);
        }
    }
    row.duration_ms = Some(started.elapsed().as_millis() as i64);
    let r = row.clone();
    let _ = state.db.call(move |c| crate::db::jobs::update_generation(c, &r)).await;
    state.jobs.finish(&id, row.error.clone());
    state.jobs.remove(&id);
}

fn req_seed(row: &GenerationRow, k: usize) -> u64 {
    row.params.seed + k as u64
}

pub async fn cancel(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    if state.jobs.cancel(&id) {
        redirect_flash("/create", "ok", "Cancelling after the current step…")
    } else {
        redirect_flash("/create", "info", "That generation isn't running.")
    }
}

pub async fn delete(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    let i = id.clone();
    let row = match state.db.call(move |c| crate::db::jobs::get_generation(c, &i)).await {
        Ok(Some(r)) => r,
        _ => return redirect_flash("/create", "error", "Not found."),
    };
    if row.status.is_active() {
        return redirect_flash("/create", "error", "Cancel the generation before deleting it.");
    }
    let dir = state.settings().outputs_dir(&state.paths);
    for o in &row.outputs {
        if let Some(p) = crate::util::safe_join(&dir, &o.file) {
            let _ = tokio::fs::remove_file(p).await;
        }
    }
    let _ = state.db.call(move |c| crate::db::jobs::delete_generation(c, &id)).await;
    redirect_flash("/create", "ok", "Deleted.")
}

#[derive(Deserialize, Default)]
pub struct SaveForm {
    #[serde(default)]
    folder: String,
}

pub async fn save_to_folder(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Form(f): Form<SaveForm>,
) -> Response {
    let dest = match crate::config::validate_writable_dir(&f.folder) {
        Ok(d) => d,
        Err(e) => return redirect_flash("/create", "error", &e),
    };
    let i = id.clone();
    let row = match state.db.call(move |c| crate::db::jobs::get_generation(c, &i)).await {
        Ok(Some(r)) => r,
        _ => return redirect_flash("/create", "error", "Not found."),
    };
    let src_dir = state.settings().outputs_dir(&state.paths);
    let mut saved = Vec::new();
    for o in &row.outputs {
        let Some(src) = crate::util::safe_join(&src_dir, &o.file) else { continue };
        let name = src.file_name().map(|n| n.to_owned()).unwrap_or_default();
        let target = dest.join(name);
        match tokio::fs::copy(&src, &target).await {
            Ok(_) => saved.push(target.to_string_lossy().to_string()),
            Err(e) => return redirect_flash("/create", "error", &format!("Could not save: {e}")),
        }
    }
    redirect_flash(
        "/create",
        "ok",
        &format!("Saved {} to {}", crate::util::count_noun(saved.len() as u64, "image", "images"), dest.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_multiples_of_64() {
        for base in [512, 768, 1024] {
            for (w, h, _) in size_presets(base) {
                assert_eq!(w % 64, 0);
                assert_eq!(h % 64, 0);
            }
        }
        assert_eq!(size_presets(512)[0], (512, 512, "Square"));
    }

    #[test]
    fn request_validation() {
        let mut f = CreateForm {
            size: "512x512".into(),
            steps: Some("4".into()),
            guidance: Some("0".into()),
            seed: Some("42".into()),
            count: Some("2".into()),
            ..Default::default()
        };
        let p = parse_request(&f, SdVersion::Turbo).unwrap();
        assert_eq!((p.width, p.height, p.steps, p.seed, p.count), (512, 512, 4, 42, 2));
        f.size = "500x500".into();
        assert!(parse_request(&f, SdVersion::Turbo).is_err());
        f.size = "512x512".into();
        f.steps = Some("0".into());
        assert!(parse_request(&f, SdVersion::Turbo).is_err());
        f.steps = None;
        f.count = Some("9".into());
        assert!(parse_request(&f, SdVersion::Turbo).is_err());
    }
}
