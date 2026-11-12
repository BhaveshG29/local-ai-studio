//! Hardware and system information dashboard.

use askama::Template;
use axum::Form;
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use serde::Deserialize;

use crate::SharedState;
use crate::util::{human_bytes, human_duration};
use crate::web::page::{PageCtx, redirect_flash, render};

pub struct Meter {
    pub label: String,
    pub detail: String,
    pub percent: u32,
    pub class: &'static str,
}

pub struct GpuView {
    pub name: String,
    pub vendor: String,
    pub vram: String,
    pub free: String,
    pub driver: String,
    pub unified: bool,
    pub source: String,
}

pub struct AccelRow {
    pub engine: &'static str,
    pub backend: &'static str,
    pub available: bool,
    pub detail: String,
}

/// One row of the "what size of model fits" guide.
pub struct GuideRow {
    pub size: &'static str,
    pub examples: &'static str,
    pub needs: String,
    pub fit_label: &'static str,
    pub fit_class: &'static str,
    pub gpu: &'static str,
}

/// Typical chat-model sizes at the common Q4_K_M quantisation (~4.85 bits per weight).
const GUIDE: &[(f64, &str, &str)] = &[
    (1.5, "1–2 B", "Qwen2.5 1.5B, Llama 3.2 1B, Gemma 3 1B"),
    (3.8, "3–4 B", "Llama 3.2 3B, Phi-3.5 mini, Qwen2.5 3B"),
    (8.0, "7–8 B", "Llama 3.1 8B, Qwen2.5 7B, Mistral 7B"),
    (14.0, "12–14 B", "Qwen2.5 14B, Phi-4, Gemma 3 12B"),
    (32.0, "27–32 B", "Qwen2.5 32B, Gemma 3 27B"),
    (70.0, "70 B", "Llama 3.3 70B"),
];

fn size_guide(total_ram: u64, available_ram: u64, best_vram: Option<u64>) -> Vec<GuideRow> {
    use crate::system::{Fit, assess_fit, estimate_llm_memory};
    GUIDE
        .iter()
        .map(|(billions, size, examples)| {
            let weights = (billions * 1e9 * 4.85 / 8.0) as u64;
            let est = estimate_llm_memory(weights, None, None, None, 4096, 2);
            let (fit, _) = assess_fit(est.total, available_ram, total_ram);
            let (fit_label, fit_class) = match fit {
                Fit::Comfortable => ("Should fit", "ok"),
                Fit::Tight => ("Tight right now", "warn"),
                Fit::TooLarge => ("Too large", "danger"),
                Fit::Unknown => ("Unknown", "muted"),
            };
            GuideRow {
                size,
                examples,
                needs: human_bytes(est.total),
                fit_label,
                fit_class,
                gpu: match best_vram {
                    Some(v) if est.total <= v => "Fits",
                    Some(_) => "Partly (rest on CPU)",
                    None => "",
                },
            }
        })
        .collect()
}

pub struct LoadedView {
    pub model_ref: String,
    pub name: String,
    pub device: String,
    pub idle: String,
    pub memory: String,
}

#[derive(Template)]
#[template(path = "system.html")]
pub struct SystemPage {
    pub ctx: PageCtx,
    pub live: bool,
    pub os: String,
    pub kernel: String,
    pub arch: String,
    pub hostname: String,
    pub cpu: String,
    pub cores: String,
    pub freq: String,
    pub features: Vec<String>,
    pub ram: Meter,
    pub swap: String,
    pub gpus: Vec<GpuView>,
    pub disks: Vec<Meter>,
    pub accel: Vec<AccelRow>,
    pub loaded: Vec<LoadedView>,
    pub ollama_error: Option<String>,
    pub server_url: String,
    pub uptime: String,
    pub data_dir: String,
    pub models_dir: String,
    pub whisper_loaded: String,
    pub image_loaded: String,
    pub guide: Vec<GuideRow>,
    pub has_gpu_memory: bool,
}

fn meter(label: String, used: u64, total: u64) -> Meter {
    let pct = if total > 0 { (used as f64 / total as f64 * 100.0).round() as u32 } else { 0 };
    Meter {
        label,
        detail: format!(
            "{} used of {} · {} free",
            human_bytes(used),
            human_bytes(total),
            human_bytes(total.saturating_sub(used))
        ),
        percent: pct.min(100),
        class: if pct >= 90 {
            "danger"
        } else if pct >= 75 {
            "warn"
        } else {
            "ok"
        },
    }
}

#[derive(Deserialize, Default)]
pub struct SystemQuery {
    #[serde(default)]
    live: Option<String>,
}

pub async fn page(State(state): State<SharedState>, headers: HeaderMap, Query(q): Query<SystemQuery>) -> Response {
    let s = state.system.snapshot().await;
    let live = q.live.is_some();
    let ctx = PageCtx::new(&state, &headers, "System", "system").at("/system").with_refresh(live.then_some(3));
    let settings = state.settings();
    let a = &s.accel;
    let mut accel = vec![
        AccelRow {
            engine: "Built-in engine",
            backend: "CPU",
            available: true,
            detail: {
                let p = crate::inference::threads::plan();
                let mut d = if p.prompt == p.generate {
                    format!(
                        "Always available. Uses {} processor thread{} (Settings → Inference).",
                        p.prompt,
                        if p.prompt == 1 { "" } else { "s" }
                    )
                } else {
                    format!(
                        "Always available. Reads prompts with {} threads and writes answers with {} (the fastest cores).",
                        p.prompt, p.generate
                    )
                };
                if !a.cpu_simd.is_empty() {
                    d.push_str(&format!(" Optimised kernels: {}.", a.cpu_simd.join(", ")));
                }
                d
            },
        },
        // Only the kind of GPU this platform can have: Metal on macOS, CUDA elsewhere.
        if cfg!(target_os = "macos") {
            AccelRow {
                engine: "Built-in engine",
                backend: "Apple Metal",
                available: a.metal_usable,
                detail: a.metal_detail.clone(),
            }
        } else {
            AccelRow {
                engine: "Built-in engine",
                backend: "NVIDIA CUDA",
                available: a.cuda_usable,
                detail: a.cuda_detail.clone(),
            }
        },
    ];
    let mut loaded: Vec<LoadedView> = state
        .inference
        .loaded_info()
        .into_iter()
        .map(|l| LoadedView {
            model_ref: l.model_ref,
            name: l.name,
            device: l.device,
            idle: format!("idle {}", human_duration(l.idle_secs)),
            memory: if l.weights_bytes > 0 { format!("≈ {}", human_bytes(l.weights_bytes)) } else { String::new() },
        })
        .collect();
    let mut ollama_error = None;
    match state.ollama.running().await {
        Ok(models) => {
            accel.push(AccelRow {
                engine: "Ollama",
                backend: "Managed by Ollama",
                available: true,
                detail: "Ollama picks CPU or GPU itself; the GPU share of each loaded model is shown below.".into(),
            });
            for m in models {
                let gpu = if m.size > 0 {
                    format!("{}% on GPU", (m.size_vram as f64 / m.size as f64 * 100.0).round())
                } else {
                    String::new()
                };
                loaded.push(LoadedView {
                    model_ref: format!("ollama:{}", m.name),
                    name: m.name,
                    device: format!("Ollama · {gpu}"),
                    idle: if m.expires_at.is_empty() { String::new() } else { "kept in memory for a while".into() },
                    memory: human_bytes(m.size),
                });
            }
        }
        Err(e) => ollama_error = Some(e.to_string()),
    }
    let best_vram = s.gpus.iter().filter(|g| !g.unified_memory).filter_map(|g| g.vram_total).max();
    let guide = size_guide(s.total_ram, s.available_ram, best_vram);
    let page = SystemPage {
        ctx: ctx.clone(),
        guide,
        has_gpu_memory: best_vram.is_some(),
        live,
        os: format!("{} {}", s.os_name, s.os_version).trim().to_string(),
        kernel: s.kernel.clone(),
        arch: s.arch.clone(),
        hostname: s.hostname.clone(),
        cpu: s.cpu_brand.clone(),
        cores: match s.physical_cores {
            Some(p) => format!("{p} cores · {} threads", s.logical_cores),
            None => format!("{} threads", s.logical_cores),
        },
        freq: if s.cpu_freq_mhz > 0 { format!("{:.2} GHz", s.cpu_freq_mhz as f64 / 1000.0) } else { String::new() },
        features: s.cpu_features.clone(),
        ram: meter("Memory (RAM)".into(), s.total_ram.saturating_sub(s.available_ram), s.total_ram),
        swap: human_bytes(s.total_swap),
        gpus: s
            .gpus
            .iter()
            .map(|g| GpuView {
                name: g.name.clone(),
                vendor: g.vendor.clone(),
                vram: g.vram_total.map(human_bytes).unwrap_or_else(|| {
                    if g.unified_memory { "Shared with system memory".into() } else { "Unknown".into() }
                }),
                free: g.vram_free.map(human_bytes).unwrap_or_default(),
                driver: g.driver.clone().unwrap_or_default(),
                unified: g.unified_memory,
                source: g.source.clone(),
            })
            .collect(),
        disks: s
            .disks
            .iter()
            .map(|d| {
                meter(format!("{} ({})", d.mount_point, d.file_system), d.total.saturating_sub(d.available), d.total)
            })
            .collect(),
        accel,
        loaded,
        ollama_error,
        server_url: state.server.url(),
        uptime: human_duration(state.server.started_at.elapsed().as_secs()),
        data_dir: state.paths.data_dir.to_string_lossy().to_string(),
        models_dir: settings.models_dir(&state.paths).to_string_lossy().to_string(),
        whisper_loaded: state.inference.whisper.loaded_id().unwrap_or_default(),
        image_loaded: state.inference.diffusion.loaded_id().unwrap_or_default(),
    };
    render(&ctx, &page)
}

#[derive(Deserialize, Default)]
pub struct UnloadForm {
    #[serde(default)]
    model_ref: String,
}

pub async fn unload(State(state): State<SharedState>, Form(f): Form<UnloadForm>) -> Response {
    match crate::inference::ModelRef::parse(&f.model_ref) {
        Some(crate::inference::ModelRef::Local(id)) => {
            state.inference.unload_local(&id);
            redirect_flash("/system", "ok", "Unloaded.")
        }
        Some(crate::inference::ModelRef::Ollama(name)) => match state.ollama.set_loaded(&name, false).await {
            Ok(()) => redirect_flash("/system", "ok", &format!("Ollama unloaded {name}.")),
            Err(e) => redirect_flash("/system", "error", &e.user_message()),
        },
        None if f.model_ref == "whisper" => {
            state.inference.unload_whisper();
            redirect_flash("/system", "ok", "Speech model unloaded.")
        }
        None if f.model_ref == "image" => {
            state.inference.unload_image_model();
            redirect_flash("/system", "ok", "Image model unloaded.")
        }
        None => redirect_flash("/system", "error", "Unknown model."),
    }
}
