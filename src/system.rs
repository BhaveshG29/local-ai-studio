//! Hardware and system information, plus memory-requirement estimates.
//!
//! GPU information is gathered from vendor tools when present (`nvidia-smi`, `system_profiler`
//! on macOS, PowerShell CIM on Windows) and from sysfs on Linux. Detection of a GPU does **not**
//! imply that an inference engine can use it: acceleration availability is reported separately
//! per engine in [`AccelInfo`].

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use sysinfo::{CpuRefreshKind, Disks, MemoryRefreshKind, RefreshKind, System};

#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub vram_total: Option<u64>,
    pub vram_free: Option<u64>,
    pub driver: Option<String>,
    /// Apple Silicon and most integrated GPUs share system memory.
    pub unified_memory: bool,
    /// Which probe found it ("nvidia-smi", "sysfs", ...).
    pub source: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: String,
    pub file_system: String,
    pub total: u64,
    pub available: u64,
    pub kind: String,
    pub removable: bool,
}

/// What the built-in engine can actually use on this machine with this build.
#[derive(Clone, Debug, Serialize)]
pub struct AccelInfo {
    pub cuda_compiled: bool,
    pub cuda_usable: bool,
    pub cuda_detail: String,
    pub metal_compiled: bool,
    pub metal_usable: bool,
    pub metal_detail: String,
    pub cpu_simd: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SystemInfo {
    pub os_name: String,
    pub os_version: String,
    pub kernel: String,
    pub arch: String,
    pub hostname: String,
    pub cpu_brand: String,
    pub cpu_vendor: String,
    pub physical_cores: Option<usize>,
    pub logical_cores: usize,
    pub cpu_freq_mhz: u64,
    pub cpu_features: Vec<String>,
    pub total_ram: u64,
    pub available_ram: u64,
    pub used_ram: u64,
    pub total_swap: u64,
    pub gpus: Vec<GpuInfo>,
    pub disks: Vec<DiskInfo>,
    pub accel: AccelInfo,
}

impl SystemInfo {
    /// Best-guess free GPU memory across detected discrete GPUs (largest single GPU).
    pub fn best_gpu_free(&self) -> Option<u64> {
        self.gpus.iter().filter(|g| !g.unified_memory).filter_map(|g| g.vram_free.or(g.vram_total)).max()
    }

    /// The disk that holds `path` (longest matching mount point).
    pub fn disk_for(&self, path: &Path) -> Option<&DiskInfo> {
        let p = canonical_or_parent(path);
        self.disks.iter().filter(|d| p.starts_with(&d.mount_point)).max_by_key(|d| d.mount_point.len())
    }
}

/// The canonical form of `path`, or of its nearest parent that exists, for matching against
/// the mount points that disks are listed under.
fn canonical_or_parent(path: &Path) -> PathBuf {
    let mut cur = path.to_path_buf();
    loop {
        if let Ok(c) = cur.canonicalize() {
            // On Windows `canonicalize` returns "verbatim" paths (`\\?\C:\…`), which never
            // match the plain mount points (`C:\`) disks are listed under.
            return if cfg!(windows) { without_verbatim_prefix(c) } else { c };
        }
        if !cur.pop() {
            return path.to_path_buf();
        }
    }
}

/// `\\?\C:\dir` → `C:\dir` and `\\?\UNC\server\share\dir` → `\\server\share\dir`; any
/// other path is returned unchanged.
fn without_verbatim_prefix(path: PathBuf) -> PathBuf {
    let Some(s) = path.to_str() else { return path };
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    match s.strip_prefix(r"\\?\") {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => PathBuf::from(rest),
        _ => path,
    }
}

/// Caches system snapshots so pages can be rendered cheaply.
pub struct SystemMonitor {
    snapshot: Mutex<Option<(Instant, Arc<SystemInfo>)>>,
    gpus: Mutex<Option<(Instant, Vec<GpuInfo>)>>,
    accel: AccelInfo,
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemMonitor {
    pub fn new() -> Self {
        Self { snapshot: Mutex::new(None), gpus: Mutex::new(None), accel: probe_acceleration() }
    }

    pub fn accel(&self) -> &AccelInfo {
        &self.accel
    }

    /// Returns a snapshot no older than ~3 seconds.
    pub async fn snapshot(&self) -> Arc<SystemInfo> {
        if let Some((at, snap)) = &*lock(&self.snapshot)
            && at.elapsed() < Duration::from_secs(3)
        {
            return snap.clone();
        }
        let gpus = self.gpus().await;
        let accel = self.accel.clone();
        let info = tokio::task::spawn_blocking(move || collect(gpus, accel))
            .await
            .unwrap_or_else(|_| collect(Vec::new(), probe_acceleration()));
        let info = Arc::new(info);
        *lock(&self.snapshot) = Some((Instant::now(), info.clone()));
        info
    }

    async fn gpus(&self) -> Vec<GpuInfo> {
        if let Some((at, g)) = &*lock(&self.gpus)
            && at.elapsed() < Duration::from_secs(30)
        {
            return g.clone();
        }
        let g = detect_gpus().await;
        *lock(&self.gpus) = Some((Instant::now(), g.clone()));
        g
    }

    /// Free space on the disk that contains `path`.
    pub async fn available_space(&self, path: &Path) -> Option<u64> {
        let p = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let disks = Disks::new_with_refreshed_list();
            let target = canonical_or_parent(&p);
            disks
                .list()
                .iter()
                .filter(|d| target.starts_with(d.mount_point()))
                .max_by_key(|d| d.mount_point().as_os_str().len())
                .map(|d| d.available_space())
        })
        .await
        .ok()
        .flatten()
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn collect(gpus: Vec<GpuInfo>, accel: AccelInfo) -> SystemInfo {
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()).with_cpu(CpuRefreshKind::everything()),
    );
    sys.refresh_memory();
    let cpus = sys.cpus();
    let (brand, vendor, freq) = cpus
        .first()
        .map(|c| (c.brand().trim().to_string(), c.vendor_id().to_string(), c.frequency()))
        .unwrap_or_else(|| ("Unknown CPU".into(), String::new(), 0));
    let disks = Disks::new_with_refreshed_list();
    let mut disk_list: Vec<DiskInfo> = disks
        .list()
        .iter()
        .filter(|d| d.total_space() > 0)
        .map(|d| DiskInfo {
            name: d.name().to_string_lossy().to_string(),
            mount_point: d.mount_point().to_string_lossy().to_string(),
            file_system: d.file_system().to_string_lossy().to_string(),
            total: d.total_space(),
            available: d.available_space(),
            kind: format!("{:?}", d.kind()),
            removable: d.is_removable(),
        })
        .filter(|d| !matches!(d.file_system.as_str(), "squashfs" | "overlay" | "tmpfs" | "devtmpfs"))
        .collect();
    disk_list.sort_by(|a, b| a.mount_point.cmp(&b.mount_point));
    disk_list.dedup_by(|a, b| a.mount_point == b.mount_point);

    SystemInfo {
        os_name: System::name().unwrap_or_else(|| std::env::consts::OS.to_string()),
        os_version: System::long_os_version().or_else(System::os_version).unwrap_or_default(),
        kernel: System::kernel_version().unwrap_or_default(),
        arch: System::cpu_arch(),
        hostname: System::host_name().unwrap_or_default(),
        cpu_brand: if brand.is_empty() { "Unknown CPU".into() } else { brand },
        cpu_vendor: vendor,
        physical_cores: System::physical_core_count(),
        logical_cores: cpus.len().max(1),
        cpu_freq_mhz: freq,
        cpu_features: cpu_features(),
        total_ram: sys.total_memory(),
        available_ram: sys.available_memory(),
        used_ram: sys.used_memory(),
        total_swap: sys.total_swap(),
        gpus,
        disks: disk_list,
        accel,
    }
}

/// SIMD features relevant to CPU inference speed.
/// How many threads the built-in engine uses when Settings → Inference leaves it empty.
///
/// Generating a token is a chain of small matrix products that every thread must finish
/// before the next one starts, so a few fast cores beat many slow ones. On processors with
/// performance and efficiency cores (Intel hybrid on Linux, Apple Silicon) only the
/// performance cores are used; elsewhere one thread per physical core.
pub fn default_engine_threads() -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if let Some(p) = performance_cores() {
        return p.clamp(1, logical);
    }
    System::physical_core_count().unwrap_or(logical).clamp(1, logical)
}

/// Physical performance cores on CPUs that also have efficiency cores; `None` otherwise.
fn performance_cores() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        // Hybrid Intel CPUs list their performance cores here (efficiency cores under cpu_atom).
        let list = std::fs::read_to_string("/sys/devices/cpu_core/cpus").ok()?;
        std::fs::metadata("/sys/devices/cpu_atom").ok()?;
        let mut cores = std::collections::HashSet::new();
        for cpu in parse_cpu_list(list.trim())? {
            let topo = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
            let core = std::fs::read_to_string(format!("{topo}/core_id")).ok()?;
            let pkg = std::fs::read_to_string(format!("{topo}/physical_package_id")).unwrap_or_default();
            cores.insert((pkg.trim().to_string(), core.trim().to_string()));
        }
        (!cores.is_empty()).then_some(cores.len())
    }
    #[cfg(target_os = "macos")]
    {
        // Apple Silicon: performance level 0 is the P-cluster. Only meaningful when there is
        // a second (efficiency) level.
        let read = |key: &str| -> Option<usize> {
            let out = std::process::Command::new("sysctl").args(["-n", key]).output().ok()?;
            String::from_utf8_lossy(&out.stdout).trim().parse().ok()
        };
        let levels = read("hw.nperflevels")?;
        if levels < 2 {
            return None;
        }
        read("hw.perflevel0.physicalcpu").filter(|n| *n > 0)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Parses a Linux CPU list such as `0-11,16,18-19`.
pub fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (a.trim().parse::<usize>().ok()?, b.trim().parse::<usize>().ok()?);
                if b < a || b - a > 65_536 {
                    return None;
                }
                out.extend(a..=b);
            }
            None => out.push(part.parse().ok()?),
        }
    }
    (!out.is_empty()).then_some(out)
}

pub fn cpu_features() -> Vec<String> {
    #[allow(unused_mut)]
    let mut f: Vec<String> = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.2") {
            f.push("SSE4.2".into());
        }
        if std::arch::is_x86_feature_detected!("avx") {
            f.push("AVX".into());
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            f.push("AVX2".into());
        }
        if std::arch::is_x86_feature_detected!("fma") {
            f.push("FMA".into());
        }
        if std::arch::is_x86_feature_detected!("f16c") {
            f.push("F16C".into());
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            f.push("AVX-512".into());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            f.push("NEON".into());
        }
        if std::arch::is_aarch64_feature_detected!("fp16") {
            f.push("FP16".into());
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            f.push("DOTPROD".into());
        }
    }
    f
}

/// Probes which accelerators the built-in Candle engine can use.
pub fn probe_acceleration() -> AccelInfo {
    let cuda_compiled = candle_core::utils::cuda_is_available();
    let metal_compiled = candle_core::utils::metal_is_available();
    let (cuda_usable, cuda_detail) = if cuda_compiled {
        match candle_core::Device::new_cuda(0) {
            Ok(_) => (true, "CUDA device 0 is available.".to_string()),
            Err(e) => (false, format!("Built with CUDA, but no usable CUDA device: {e}")),
        }
    } else {
        (
            false,
            "This build does not include CUDA. Rebuild with `--features cuda` (needs the CUDA toolkit) to use an NVIDIA GPU with the built-in engine.".to_string(),
        )
    };
    let (metal_usable, metal_detail) = if metal_compiled {
        match candle_core::Device::new_metal(0) {
            Ok(_) => (true, "Metal device 0 is available.".to_string()),
            Err(e) => (false, format!("Built with Metal, but no usable Metal device: {e}")),
        }
    } else if cfg!(target_os = "macos") {
        (
            false,
            "This build does not include Metal. Rebuild with `--features metal` to use the Apple GPU with the built-in engine.".to_string(),
        )
    } else {
        (false, "Metal is only available on macOS.".to_string())
    };
    let mut cpu_simd = Vec::new();
    if candle_core::utils::with_avx() {
        cpu_simd.push("AVX".to_string());
    }
    if candle_core::utils::with_neon() {
        cpu_simd.push("NEON".to_string());
    }
    if candle_core::utils::with_f16c() {
        cpu_simd.push("F16C".to_string());
    }
    AccelInfo { cuda_compiled, cuda_usable, cuda_detail, metal_compiled, metal_usable, metal_detail, cpu_simd }
}

async fn run_probe(program: &str, args: &[&str]) -> Option<String> {
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args).stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null()).kill_on_drop(true);
    #[cfg(windows)]
    {
        // CREATE_NO_WINDOW: do not flash a console window.
        cmd.creation_flags(0x0800_0000);
    }
    let out = tokio::time::timeout(Duration::from_secs(4), cmd.output()).await.ok()?.ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).to_string())
}

/// Detects GPUs using whatever the platform offers. Never fails; returns an empty list
/// when nothing can be determined.
pub async fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    if let Some(out) = run_probe(
        "nvidia-smi",
        &["--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
    )
    .await
    {
        gpus.extend(parse_nvidia_smi(&out));
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(out) = run_probe("system_profiler", &["SPDisplaysDataType", "-json"]).await {
            gpus.extend(parse_system_profiler(&out));
        }
    }
    #[cfg(target_os = "linux")]
    {
        let have_nvidia = !gpus.is_empty();
        gpus.extend(tokio::task::spawn_blocking(move || linux_sysfs_gpus(have_nvidia)).await.unwrap_or_default());
    }
    #[cfg(windows)]
    {
        if gpus.is_empty() {
            if let Some(out) = run_probe(
                "powershell",
                &[
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json",
                ],
            )
            .await
            {
                gpus.extend(parse_windows_cim(&out));
            }
        }
    }
    gpus
}

/// Parses `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits`.
pub fn parse_nvidia_smi(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            if parts.len() < 4 || parts[0].is_empty() {
                return None;
            }
            let mib = |s: &str| s.parse::<u64>().ok().map(|v| v * 1024 * 1024);
            Some(GpuInfo {
                name: parts[0].to_string(),
                vendor: "NVIDIA".into(),
                vram_total: mib(parts[1]),
                vram_free: mib(parts[2]),
                driver: Some(parts[3].to_string()).filter(|s| !s.is_empty()),
                unified_memory: false,
                source: "nvidia-smi".into(),
            })
        })
        .collect()
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_system_profiler(out: &str) -> Vec<GpuInfo> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(out) else {
        return Vec::new();
    };
    v.get("SPDisplaysDataType")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .map(|g| {
                    let name = g.get("sppci_model").and_then(|s| s.as_str()).unwrap_or("GPU").to_string();
                    let apple = name.starts_with("Apple");
                    GpuInfo {
                        vendor: if apple {
                            "Apple".into()
                        } else {
                            g.get("spdisplays_vendor")
                                .and_then(|s| s.as_str())
                                .unwrap_or("")
                                .trim_start_matches("sppci_vendor_")
                                .to_string()
                        },
                        name,
                        vram_total: None,
                        vram_free: None,
                        driver: None,
                        unified_memory: apple,
                        source: "system_profiler".into(),
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(target_os = "linux")]
fn linux_sysfs_gpus(skip_nvidia: bool) -> Vec<GpuInfo> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return out;
    };
    let mut seen = std::collections::HashSet::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = e.path().join("device");
        let Ok(real) = dev.canonicalize() else { continue };
        if !seen.insert(real.clone()) {
            continue;
        }
        let read = |f: &str| std::fs::read_to_string(dev.join(f)).ok().map(|s| s.trim().to_string());
        let vendor_id = read("vendor").unwrap_or_default();
        let (vendor, unified) = match vendor_id.as_str() {
            "0x10de" => ("NVIDIA", false),
            "0x1002" => ("AMD", false),
            "0x8086" => ("Intel", true),
            _ => continue,
        };
        if vendor == "NVIDIA" && skip_nvidia {
            continue;
        }
        let vram_total = read("mem_info_vram_total").and_then(|s| s.parse().ok());
        let vram_used: Option<u64> = read("mem_info_vram_used").and_then(|s| s.parse().ok());
        let label = read("product_name").filter(|s| !s.is_empty()).unwrap_or_else(|| match vendor {
            "Intel" => "Intel integrated graphics".to_string(),
            "AMD" => "AMD Radeon graphics".to_string(),
            _ => "NVIDIA GPU (install the NVIDIA driver for details)".to_string(),
        });
        out.push(GpuInfo {
            name: label,
            vendor: vendor.into(),
            vram_total,
            vram_free: match (vram_total, vram_used) {
                (Some(t), Some(u)) => Some(t.saturating_sub(u)),
                _ => None,
            },
            driver: std::fs::read_link(dev.join("driver"))
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string())),
            unified_memory: unified && vram_total.is_none(),
            source: "sysfs".into(),
        });
    }
    out
}

#[cfg_attr(not(windows), allow(dead_code))]
fn parse_windows_cim(out: &str) -> Vec<GpuInfo> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(out) else {
        return Vec::new();
    };
    let items = match v {
        serde_json::Value::Array(a) => a,
        other => vec![other],
    };
    items
        .iter()
        .filter_map(|g| {
            let name = g.get("Name")?.as_str()?.to_string();
            let vendor = if name.contains("NVIDIA") {
                "NVIDIA"
            } else if name.contains("AMD") || name.contains("Radeon") {
                "AMD"
            } else if name.contains("Intel") {
                "Intel"
            } else {
                ""
            };
            // AdapterRAM is a 32-bit field and saturates at 4 GB, so treat it as a lower bound.
            let ram = g.get("AdapterRAM").and_then(|r| r.as_u64()).filter(|r| *r > 0);
            Some(GpuInfo {
                name,
                vendor: vendor.into(),
                vram_total: ram,
                vram_free: None,
                driver: g.get("DriverVersion").and_then(|d| d.as_str()).map(String::from),
                unified_memory: vendor == "Intel",
                source: "windows-cim".into(),
            })
        })
        .collect()
}

/// Memory needed to run a model, split into parts.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct MemoryEstimate {
    pub weights: u64,
    pub kv_cache: u64,
    pub overhead: u64,
    pub total: u64,
}

/// Rough memory estimate for a transformer LLM.
/// `kv_bytes` is bytes per cached element (4 for f32 caches on CPU, 2 for f16).
pub fn estimate_llm_memory(
    weights_bytes: u64,
    layers: Option<u64>,
    kv_heads: Option<u64>,
    head_dim: Option<u64>,
    context: u64,
    kv_bytes: u64,
) -> MemoryEstimate {
    let kv_cache = match (layers, kv_heads, head_dim) {
        (Some(l), Some(h), Some(d)) => 2 * l * h * d * context * kv_bytes,
        // Without architecture details assume ~10% of the weights for a 4k context.
        _ => weights_bytes / 10 * (context.max(1) / 4096).max(1),
    };
    let overhead = 300 * 1024 * 1024 + weights_bytes / 20;
    MemoryEstimate { weights: weights_bytes, kv_cache, overhead, total: weights_bytes + kv_cache + overhead }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Fit {
    /// Fits with room to spare.
    Comfortable,
    /// Fits in total memory but not in what is free right now.
    Tight,
    /// Larger than the machine's total memory.
    TooLarge,
    Unknown,
}

/// Judges an estimate against system memory. Never a guarantee — the UI says so.
pub fn assess_fit(total_needed: u64, available: u64, total: u64) -> (Fit, String) {
    use crate::util::human_bytes;
    if total_needed == 0 || total == 0 {
        return (Fit::Unknown, "Not enough information to estimate memory use.".into());
    }
    if total_needed > total {
        (
            Fit::TooLarge,
            format!(
                "Needs about {} but this computer has {} of memory in total. It is very unlikely to run.",
                human_bytes(total_needed),
                human_bytes(total)
            ),
        )
    } else if total_needed > available {
        (
            Fit::Tight,
            format!(
                "Needs about {} and {} is free right now. Close other apps, or expect slowdowns.",
                human_bytes(total_needed),
                human_bytes(available)
            ),
        )
    } else {
        (Fit::Comfortable, format!("Needs about {} — {} is free.", human_bytes(total_needed), human_bytes(available)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbatim_windows_paths_are_simplified() {
        let s = |p: &str| without_verbatim_prefix(PathBuf::from(p)).to_string_lossy().into_owned();
        assert_eq!(s(r"\\?\C:\Users\me\models"), r"C:\Users\me\models");
        assert_eq!(s(r"\\?\D:\"), r"D:\");
        assert_eq!(s(r"\\?\UNC\nas\share\models"), r"\\nas\share\models");
        // Volume GUID paths have no drive letter to fall back to.
        assert_eq!(
            s(r"\\?\Volume{1b3b1146-4076-11e1-84aa-806e6f6e6963}\x"),
            r"\\?\Volume{1b3b1146-4076-11e1-84aa-806e6f6e6963}\x"
        );
        assert_eq!(s(r"C:\Users\me"), r"C:\Users\me");
        assert_eq!(s("/home/me/models"), "/home/me/models");
    }

    #[test]
    fn free_space_is_found_for_a_folder() {
        // The folder's disk must be found on every platform, or downloads skip the
        // free-space check (this happened on Windows).
        let dir = tempfile::tempdir().unwrap();
        let monitor = SystemMonitor::new();
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        assert!(
            rt.block_on(monitor.available_space(dir.path())).is_some(),
            "no disk found for {}",
            dir.path().display()
        );
        assert!(rt.block_on(monitor.available_space(&dir.path().join("not/yet/created"))).is_some());
    }

    #[test]
    fn cpu_lists_parse() {
        assert_eq!(parse_cpu_list("0-3,8,10-11"), Some(vec![0, 1, 2, 3, 8, 10, 11]));
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("a-b"), None);
    }

    #[test]
    fn nvidia_smi_output_is_parsed() {
        let out = "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7811, 560.35.03\n";
        let g = parse_nvidia_smi(out);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].name, "NVIDIA GeForce RTX 4070 Laptop GPU");
        assert_eq!(g[0].vram_total, Some(8188 * 1024 * 1024));
        assert_eq!(g[0].vram_free, Some(7811 * 1024 * 1024));
        assert_eq!(g[0].driver.as_deref(), Some("560.35.03"));
        assert!(parse_nvidia_smi("garbage").is_empty());
    }

    #[test]
    fn system_profiler_json_is_parsed() {
        let out = r#"{"SPDisplaysDataType":[{"sppci_model":"Apple M2 Pro","sppci_cores":"19"}]}"#;
        let g = parse_system_profiler(out);
        assert_eq!(g.len(), 1);
        assert!(g[0].unified_memory);
        assert_eq!(g[0].vendor, "Apple");
    }

    #[test]
    fn windows_cim_json_is_parsed() {
        let out = r#"{"Name":"NVIDIA GeForce RTX 3060","AdapterRAM":4293918720,"DriverVersion":"31.0.15.5222"}"#;
        let g = parse_windows_cim(out);
        assert_eq!(g[0].vendor, "NVIDIA");
        assert_eq!(g[0].vram_total, Some(4293918720));
    }

    #[test]
    fn memory_estimates_and_fit() {
        // Llama-3.2-1B-like: 16 layers, 8 kv heads, head dim 64, 4096 ctx, f32 cache.
        let e = estimate_llm_memory(800 << 20, Some(16), Some(8), Some(64), 4096, 4);
        assert_eq!(e.kv_cache, 2 * 16 * 8 * 64 * 4096 * 4);
        assert!(e.total > e.weights);
        assert_eq!(assess_fit(e.total, 64 << 30, 64 << 30).0, Fit::Comfortable);
        assert_eq!(assess_fit(e.total, 1 << 30, 64 << 30).0, Fit::Tight);
        assert_eq!(assess_fit(e.total, 1 << 30, 1 << 30).0, Fit::TooLarge);
        assert_eq!(assess_fit(0, 1, 1).0, Fit::Unknown);
    }

    #[tokio::test]
    async fn snapshot_has_basic_fields() {
        let m = SystemMonitor::new();
        let s = m.snapshot().await;
        assert!(s.total_ram > 0);
        assert!(s.logical_cores >= 1);
        let t = default_engine_threads();
        assert!(t >= 1 && t <= s.logical_cores, "{t}");
        assert!(!s.arch.is_empty());
        // Acceleration info is always populated with a human explanation.
        assert!(!s.accel.cuda_detail.is_empty());
    }
}
