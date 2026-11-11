//! Download manager: resumable, verified downloads from Hugging Face and pulls through Ollama.
//!
//! * Files are written to `<partial dir>/<job>/<path>.part` and renamed into the model folder
//!   only after their size and checksum (SHA-256 for LFS/Xet files, git blob SHA-1 for small
//!   files) have been verified.
//! * Pause keeps partial files; resume continues with an HTTP `Range` request. If a server
//!   ignores the range, the file restarts from zero (correctness over speed).
//! * Jobs are persisted; jobs interrupted by a shutdown come back as "paused".
//! * A semaphore limits concurrent jobs; each job's state transitions are serialised through
//!   its own task, so there are no races between pause/cancel and progress updates.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use sha2::Digest;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::config::{AppPaths, Settings};
use crate::db::downloads::{DownloadFile, DownloadKind, DownloadOptions, DownloadRow, DownloadState};
use crate::db::{self, Db};
use crate::error::{AppError, AppResult};
use crate::hub::HubClient;
use crate::ollama::OllamaClient;
use crate::system::SystemMonitor;
use crate::util::{human_bytes, now_rfc3339, safe_join, sanitize_component};

const INTENT_NONE: u8 = 0;
const INTENT_PAUSE: u8 = 1;
const INTENT_CANCEL: u8 = 2;

/// Live, in-memory progress of a running job.
struct Progress {
    done: AtomicU64,
    total: AtomicU64,
    current_file: Mutex<String>,
    samples: Mutex<VecDeque<(Instant, u64)>>,
}

impl Progress {
    fn new(done: u64, total: u64) -> Self {
        Self {
            done: AtomicU64::new(done),
            total: AtomicU64::new(total),
            current_file: Mutex::new(String::new()),
            samples: Mutex::new(VecDeque::new()),
        }
    }

    /// Bytes per second over the last ~5 seconds.
    fn speed(&self) -> f64 {
        let now = Instant::now();
        let done = self.done.load(Ordering::Relaxed);
        let mut s = self.samples.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if s.back().is_none_or(|(t, _)| now.duration_since(*t) >= Duration::from_millis(250)) {
            s.push_back((now, done));
        }
        while s.len() > 2 && now.duration_since(s[0].0) > Duration::from_secs(5) {
            s.pop_front();
        }
        match (s.front(), s.back()) {
            (Some((t0, b0)), Some((t1, b1))) if t1 > t0 => {
                (b1.saturating_sub(*b0)) as f64 / t1.duration_since(*t0).as_secs_f64()
            }
            _ => 0.0,
        }
    }
}

struct ActiveJob {
    cancel: CancellationToken,
    intent: Arc<AtomicU8>,
    progress: Arc<Progress>,
}

/// What the UI shows for one job.
#[derive(Clone, Debug, serde::Serialize)]
pub struct JobView {
    pub row: DownloadRow,
    pub done_bytes: u64,
    pub total_bytes: u64,
    pub percent: f64,
    pub speed_bps: f64,
    pub eta_secs: Option<u64>,
    pub current_file: String,
}

impl JobView {
    pub fn speed_label(&self) -> String {
        if self.speed_bps <= 0.0 { String::new() } else { format!("{}/s", human_bytes(self.speed_bps as u64)) }
    }
    pub fn eta_label(&self) -> String {
        self.eta_secs.map(|s| format!("{} left", crate::util::human_duration(s))).unwrap_or_default()
    }
}

struct Inner {
    db: Db,
    http: reqwest::Client,
    hub: HubClient,
    ollama: OllamaClient,
    paths: AppPaths,
    settings: Arc<RwLock<Settings>>,
    system: Arc<SystemMonitor>,
    active: Mutex<HashMap<String, ActiveJob>>,
    semaphore: Arc<Semaphore>,
}

#[derive(Clone)]
pub struct DownloadManager {
    inner: Arc<Inner>,
}

impl DownloadManager {
    pub fn new(
        db: Db,
        http: reqwest::Client,
        hub: HubClient,
        ollama: OllamaClient,
        paths: AppPaths,
        settings: Arc<RwLock<Settings>>,
        system: Arc<SystemMonitor>,
    ) -> Self {
        let permits =
            settings.read().unwrap_or_else(std::sync::PoisonError::into_inner).storage.max_concurrent_downloads.max(1)
                as usize;
        Self {
            inner: Arc::new(Inner {
                db,
                http,
                hub,
                ollama,
                paths,
                settings,
                system,
                active: Mutex::new(HashMap::new()),
                semaphore: Arc::new(Semaphore::new(permits)),
            }),
        }
    }

    fn settings(&self) -> Settings {
        self.inner.settings.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// Directory where a Hugging Face repository's files are stored.
    pub fn repo_dir(&self, repo_id: &str) -> PathBuf {
        let s = self.settings();
        let mut dir = s.models_dir(&self.inner.paths).join("huggingface");
        for part in repo_id.split('/') {
            dir.push(sanitize_component(part));
        }
        dir
    }

    fn partial_path(&self, job_id: &str, local_path: &str) -> Option<PathBuf> {
        let s = self.settings();
        let base = s.partial_dir(&self.inner.paths).join(sanitize_component(job_id));
        safe_join(&base, &format!("{local_path}.part"))
    }

    fn partial_job_dir(&self, job_id: &str) -> PathBuf {
        self.settings().partial_dir(&self.inner.paths).join(sanitize_component(job_id))
    }

    /// Queues a Hugging Face download. Rejects exact duplicates of finished or running jobs.
    pub async fn enqueue_hf(
        &self,
        repo_id: &str,
        commit: &str,
        files: Vec<DownloadFile>,
        options: DownloadOptions,
    ) -> AppResult<String> {
        crate::hub::validate_repo_id(repo_id)?;
        if files.is_empty() {
            return Err(AppError::bad_request("Choose at least one file to download."));
        }
        for f in &files {
            if safe_join(Path::new("/x"), &f.local_path).is_none() || safe_join(Path::new("/x"), &f.path).is_none() {
                return Err(AppError::bad_request(format!("Unsafe file path: {}", f.path)));
            }
        }
        // Duplicate detection: same repo + primary file already in the library.
        let repo = repo_id.to_string();
        let primary = options.primary_file.clone();
        let existing = self.inner.db.call(move |c| db::models::find_by_repo(c, &repo, primary.as_deref())).await?;
        if let Some(m) = existing
            && Path::new(&m.path).exists()
        {
            return Err(AppError::conflict(format!(
                "\"{}\" is already in your library. Open it from the Library page.",
                m.display_name
            )));
        }
        let jobs = self.inner.db.call(db::downloads::list).await?;
        if let Some(j) = jobs
            .iter()
            .find(|j| j.repo_id == repo_id && !j.state.is_finished() && j.options.primary_file == options.primary_file)
        {
            return Err(AppError::conflict(format!(
                "\"{}\" is already in the download list ({}).",
                j.title,
                j.state.label().to_lowercase()
            )));
        }
        let total: u64 = files.iter().map(|f| f.size).sum();
        let now = now_rfc3339();
        let title = options.display_name.clone().unwrap_or_else(|| repo_id.to_string());
        let mut options = options;
        options.commit = Some(commit.to_string());
        let row = DownloadRow {
            id: crate::util::new_id(),
            kind: DownloadKind::HuggingFace,
            title,
            repo_id: repo_id.to_string(),
            revision: Some(commit.to_string()),
            files,
            state: DownloadState::Queued,
            total_bytes: total as i64,
            done_bytes: 0,
            error: None,
            model_id: None,
            options,
            created_at: now.clone(),
            updated_at: now,
        };
        let r = row.clone();
        self.inner.db.call(move |c| db::downloads::insert(c, &r)).await?;
        let id = row.id.clone();
        self.spawn(row);
        Ok(id)
    }

    /// Queues an Ollama pull (`name` or `name:tag`).
    pub async fn enqueue_ollama(&self, name: &str) -> AppResult<String> {
        crate::ollama::validate_model_name(name)?;
        if !self.inner.ollama.enabled() {
            return Err(AppError::unavailable("Ollama integration is turned off in Settings."));
        }
        self.inner.ollama.version().await?;
        let jobs = self.inner.db.call(db::downloads::list).await?;
        if let Some(j) =
            jobs.iter().find(|j| j.kind == DownloadKind::Ollama && j.repo_id == name && !j.state.is_finished())
        {
            return Err(AppError::conflict(format!(
                "{} is already in the download list ({}).",
                j.title,
                j.state.label().to_lowercase()
            )));
        }
        let now = now_rfc3339();
        let row = DownloadRow {
            id: crate::util::new_id(),
            kind: DownloadKind::Ollama,
            title: name.to_string(),
            repo_id: name.to_string(),
            revision: None,
            files: Vec::new(),
            state: DownloadState::Queued,
            total_bytes: 0,
            done_bytes: 0,
            error: None,
            model_id: None,
            options: DownloadOptions::default(),
            created_at: now.clone(),
            updated_at: now,
        };
        let r = row.clone();
        self.inner.db.call(move |c| db::downloads::insert(c, &r)).await?;
        let id = row.id.clone();
        self.spawn(row);
        Ok(id)
    }

    fn spawn(&self, row: DownloadRow) {
        let progress = Arc::new(Progress::new(row.done_bytes.max(0) as u64, row.total_bytes.max(0) as u64));
        let job = ActiveJob {
            cancel: CancellationToken::new(),
            intent: Arc::new(AtomicU8::new(INTENT_NONE)),
            progress: progress.clone(),
        };
        let cancel = job.cancel.clone();
        let intent = job.intent.clone();
        {
            let mut active = self.inner.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if active.contains_key(&row.id) {
                return;
            }
            active.insert(row.id.clone(), job);
        }
        let me = self.clone();
        tokio::spawn(async move {
            let id = row.id.clone();
            me.run_job(row, cancel, intent, progress).await;
            me.inner.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&id);
        });
    }

    async fn save(&self, row: &DownloadRow) {
        let r = row.clone();
        if let Err(e) = self.inner.db.call(move |c| db::downloads::save(c, &r)).await {
            tracing::error!("could not save download state: {e:#}");
        }
    }

    async fn run_job(
        &self,
        mut row: DownloadRow,
        cancel: CancellationToken,
        intent: Arc<AtomicU8>,
        progress: Arc<Progress>,
    ) {
        row.state = DownloadState::Queued;
        row.error = None;
        row.updated_at = now_rfc3339();
        self.save(&row).await;
        let permit = tokio::select! {
            p = self.inner.semaphore.clone().acquire_owned() => p.ok(),
            _ = cancel.cancelled() => None,
        };
        let result = if permit.is_some() {
            row.state = DownloadState::Downloading;
            row.updated_at = now_rfc3339();
            self.save(&row).await;
            match row.kind {
                DownloadKind::HuggingFace => self.run_hf(&mut row, &cancel, &progress).await,
                DownloadKind::Ollama => self.run_ollama(&mut row, &cancel, &progress).await,
            }
        } else {
            Err(AppError::conflict("cancelled"))
        };
        drop(permit);
        row.done_bytes = progress.done.load(Ordering::Relaxed) as i64;
        row.total_bytes = row.total_bytes.max(progress.total.load(Ordering::Relaxed) as i64);
        match result {
            Ok(()) => {
                row.state = DownloadState::Completed;
                row.done_bytes = row.total_bytes;
                row.error = None;
            }
            Err(e) => match intent.load(Ordering::SeqCst) {
                INTENT_PAUSE => row.state = DownloadState::Paused,
                INTENT_CANCEL => {
                    row.state = DownloadState::Cancelled;
                    row.error = None;
                    self.remove_partials(&row).await;
                }
                _ => {
                    row.state = DownloadState::Failed;
                    row.error = Some(e.user_message());
                    tracing::warn!(job = %row.id, "download failed: {}", e.user_message());
                }
            },
        }
        row.updated_at = now_rfc3339();
        self.save(&row).await;
    }

    async fn remove_partials(&self, row: &DownloadRow) {
        if row.kind == DownloadKind::HuggingFace {
            let dir = self.partial_job_dir(&row.id);
            let _ = tokio::fs::remove_dir_all(dir).await;
        }
    }

    async fn run_hf(
        &self,
        row: &mut DownloadRow,
        cancel: &CancellationToken,
        progress: &Arc<Progress>,
    ) -> AppResult<()> {
        let model_dir = self.repo_dir(&row.repo_id);
        tokio::fs::create_dir_all(&model_dir).await?;
        // Progress already completed (files done in an earlier session).
        let mut done: u64 = 0;
        for f in &row.files {
            if f.done {
                done += f.size;
            } else if let Some(p) = self.partial_path(&row.id, &f.local_path) {
                done += tokio::fs::metadata(&p).await.map(|m| m.len()).unwrap_or(0).min(f.size);
            }
        }
        let total: u64 = row.files.iter().map(|f| f.size).sum();
        progress.done.store(done, Ordering::Relaxed);
        progress.total.store(total, Ordering::Relaxed);

        // Disk space check before starting.
        let remaining = total.saturating_sub(done);
        let partial_dir = self.settings().partial_dir(&self.inner.paths);
        tokio::fs::create_dir_all(&partial_dir).await?;
        if let Some(free) = self.inner.system.available_space(&partial_dir).await {
            let margin = 50 * 1024 * 1024;
            if free < remaining + margin {
                return Err(AppError::bad_request(format!(
                    "Not enough disk space: this download needs {} more, but only {} is free on that disk. Free some space or choose another models folder in Settings.",
                    human_bytes(remaining + margin),
                    human_bytes(free)
                )));
            }
        }

        for idx in 0..row.files.len() {
            if row.files[idx].done {
                continue;
            }
            if cancel.is_cancelled() {
                return Err(AppError::conflict("cancelled"));
            }
            let file = row.files[idx].clone();
            *progress.current_file.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = file.local_path.clone();
            let final_path =
                safe_join(&model_dir, &file.local_path).ok_or_else(|| AppError::bad_request("unsafe path"))?;
            // Already present from an earlier attempt or another copy?
            if let Ok(meta) = tokio::fs::metadata(&final_path).await
                && meta.len() == file.size
            {
                row.files[idx].done = true;
                self.save(row).await;
                continue;
            }
            if let Some(src) = self.find_duplicate(&file).await {
                if let Some(parent) = final_path.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                let before = tokio::fs::metadata(self.partial_path(&row.id, &file.local_path).unwrap_or_default())
                    .await
                    .map(|m| m.len())
                    .unwrap_or(0);
                if std::fs::hard_link(&src, &final_path).is_ok() || tokio::fs::copy(&src, &final_path).await.is_ok() {
                    tracing::info!("reused identical file {} for {}", src.display(), file.local_path);
                    progress.done.fetch_add(file.size.saturating_sub(before), Ordering::Relaxed);
                    row.files[idx].done = true;
                    self.save(row).await;
                    continue;
                }
            }
            let mut attempt = 0u32;
            loop {
                attempt += 1;
                match self.download_file(row, &file, &final_path, cancel, progress).await {
                    Ok(()) => break,
                    Err(_) if cancel.is_cancelled() => return Err(AppError::conflict("cancelled")),
                    Err(DlError::Fatal(e)) => return Err(e),
                    Err(DlError::Retryable(e)) if attempt < 4 => {
                        tracing::warn!("download attempt {attempt} for {} failed: {}", file.path, e.user_message());
                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))) => {},
                            _ = cancel.cancelled() => return Err(AppError::conflict("cancelled")),
                        }
                    }
                    Err(DlError::Retryable(e)) => return Err(e),
                    Err(DlError::Corrupt(_)) if attempt < 2 => {
                        tracing::warn!("{} failed verification; downloading it once more", file.path);
                    }
                    Err(DlError::Corrupt(e)) => return Err(e),
                }
            }
            row.files[idx].done = true;
            row.done_bytes = progress.done.load(Ordering::Relaxed) as i64;
            self.save(row).await;
        }
        let _ = tokio::fs::remove_dir_all(self.partial_job_dir(&row.id)).await;

        row.state = DownloadState::Verifying;
        self.save(row).await;
        let model_id =
            crate::library::register_hf_download(&self.inner.db, row, &model_dir).await.map_err(AppError::Internal)?;
        row.model_id = Some(model_id);
        Ok(())
    }

    /// A previously downloaded file with the same SHA-256 (from another repo or variant).
    async fn find_duplicate(&self, file: &DownloadFile) -> Option<PathBuf> {
        let sha = file.sha256.clone()?;
        let models = self.inner.db.call(db::models::list).await.ok()?;
        for m in models {
            for f in &m.files {
                if f.sha256.as_deref() == Some(sha.as_str()) && f.size == file.size {
                    let p = Path::new(&m.path).join(&f.path);
                    if tokio::fs::metadata(&p).await.map(|x| x.len() == file.size).unwrap_or(false) {
                        return Some(p);
                    }
                }
            }
        }
        None
    }

    async fn download_file(
        &self,
        row: &DownloadRow,
        file: &DownloadFile,
        final_path: &Path,
        cancel: &CancellationToken,
        progress: &Arc<Progress>,
    ) -> Result<(), DlError> {
        let part = self
            .partial_path(&row.id, &file.local_path)
            .ok_or_else(|| DlError::Fatal(AppError::bad_request("unsafe path")))?;
        if let Some(parent) = part.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(fatal_io)?;
        }
        let mut start = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);
        if start > file.size {
            tokio::fs::remove_file(&part).await.ok();
            progress.done.fetch_sub(start.min(progress.done.load(Ordering::Relaxed)), Ordering::Relaxed);
            start = 0;
        }
        let url = self.inner.hub.file_url(&file.repo_id, &file.revision, &file.path);
        let mut req = self.inner.http.get(&url);
        if start > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={start}-"));
        }
        if let Some(t) = self.inner.hub.token().await {
            req = req.bearer_auth(t);
        }
        let resp = tokio::select! {
            r = req.send() => r.map_err(|e| {
                if e.is_connect() || e.is_timeout() {
                    DlError::Retryable(AppError::unavailable("Lost connection to Hugging Face. Check your internet connection; the download will resume where it stopped."))
                } else {
                    DlError::Retryable(AppError::upstream(format!("Network error: {e}")))
                }
            })?,
            _ = cancel.cancelled() => return Err(DlError::Fatal(AppError::conflict("cancelled"))),
        };
        let status = resp.status();
        let mut append = false;
        match status.as_u16() {
            206 => append = true,
            200 => {
                if start > 0 {
                    // Server ignored the Range header: restart this file from zero.
                    progress.done.fetch_sub(start.min(progress.done.load(Ordering::Relaxed)), Ordering::Relaxed);
                    start = 0;
                }
            }
            416 if start == file.size => {
                return self.finalize_file(file, &part, final_path).await;
            }
            416 => {
                tokio::fs::remove_file(&part).await.ok();
                progress.done.fetch_sub(start.min(progress.done.load(Ordering::Relaxed)), Ordering::Relaxed);
                return Err(DlError::Retryable(AppError::upstream(
                    "The server rejected the resume position; restarting this file.",
                )));
            }
            401 | 403 => {
                return Err(DlError::Fatal(AppError::Forbidden(format!(
                    "\"{}\" requires permission. Open https://huggingface.co/{} in your browser, accept the model's terms, then add a Hugging Face token in Settings and press Retry.",
                    file.repo_id, file.repo_id
                ))));
            }
            404 => {
                return Err(DlError::Fatal(AppError::not_found(format!(
                    "{} no longer exists in {}. Open the model page again to see its current files.",
                    file.path, file.repo_id
                ))));
            }
            s if s >= 500 || s == 429 => {
                return Err(DlError::Retryable(AppError::upstream(format!(
                    "Hugging Face is having trouble ({status}). Retrying…"
                ))));
            }
            _ => {
                return Err(DlError::Fatal(AppError::upstream(format!(
                    "Unexpected response {status} while downloading {}",
                    file.path
                ))));
            }
        }
        let mut out = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)
            .await
            .map_err(fatal_io)?;
        let mut written = start;
        let mut stream = resp.bytes_stream();
        let mut last_persist = Instant::now();
        loop {
            let next = tokio::select! {
                n = tokio::time::timeout(Duration::from_secs(60), stream.next()) => n,
                _ = cancel.cancelled() => {
                    out.flush().await.ok();
                    return Err(DlError::Fatal(AppError::conflict("cancelled")));
                }
            };
            let chunk = match next {
                Err(_) => {
                    out.flush().await.ok();
                    return Err(DlError::Retryable(AppError::upstream(
                        "The download stalled for 60 seconds. Retrying…",
                    )));
                }
                Ok(None) => break,
                Ok(Some(Err(e))) => {
                    out.flush().await.ok();
                    return Err(DlError::Retryable(AppError::upstream(format!("Connection interrupted: {e}"))));
                }
                Ok(Some(Ok(b))) => b,
            };
            out.write_all(&chunk).await.map_err(fatal_io)?;
            written += chunk.len() as u64;
            progress.done.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            if written > file.size {
                drop(out);
                tokio::fs::remove_file(&part).await.ok();
                return Err(DlError::Fatal(AppError::upstream(format!(
                    "{} is larger than Hugging Face reported; the download was stopped. Refresh the model page and try again.",
                    file.path
                ))));
            }
            if last_persist.elapsed() > Duration::from_secs(3) {
                last_persist = Instant::now();
                let mut snapshot = row.clone();
                snapshot.done_bytes = progress.done.load(Ordering::Relaxed) as i64;
                snapshot.state = DownloadState::Downloading;
                snapshot.updated_at = now_rfc3339();
                self.save(&snapshot).await;
            }
        }
        out.flush().await.map_err(fatal_io)?;
        out.sync_all().await.map_err(fatal_io)?;
        drop(out);
        if written != file.size {
            return Err(DlError::Retryable(AppError::upstream(format!(
                "{} ended early ({} of {}). Resuming…",
                file.path,
                human_bytes(written),
                human_bytes(file.size)
            ))));
        }
        self.finalize_file(file, &part, final_path).await
    }

    /// Verifies a completed partial file and moves it into place.
    async fn finalize_file(&self, file: &DownloadFile, part: &Path, final_path: &Path) -> Result<(), DlError> {
        let ok = verify_file(part, file).await.map_err(fatal_io)?;
        if !ok {
            tokio::fs::remove_file(part).await.ok();
            return Err(DlError::Corrupt(AppError::upstream(format!(
                "{} failed its integrity check twice: the downloaded data doesn't match the checksum Hugging Face publishes, so it was deleted. The file may be corrupted on the server; try again later or pick another file.",
                file.path
            ))));
        }
        if let Some(parent) = final_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(fatal_io)?;
        }
        if tokio::fs::rename(part, final_path).await.is_err() {
            // Different disks: copy then remove.
            tokio::fs::copy(part, final_path).await.map_err(fatal_io)?;
            tokio::fs::remove_file(part).await.ok();
        }
        Ok(())
    }

    async fn run_ollama(
        &self,
        row: &mut DownloadRow,
        cancel: &CancellationToken,
        progress: &Arc<Progress>,
    ) -> AppResult<()> {
        let layers: Mutex<HashMap<String, (u64, u64)>> = Mutex::new(HashMap::new());
        let p = progress.clone();
        self.inner
            .ollama
            .pull(&row.repo_id, cancel.clone(), |ev| {
                if let (Some(d), Some(t)) = (&ev.digest, ev.total) {
                    let mut l = layers.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    l.insert(d.clone(), (t, ev.completed.unwrap_or(0)));
                    let total: u64 = l.values().map(|v| v.0).sum();
                    let done: u64 = l.values().map(|v| v.1.min(v.0)).sum();
                    p.total.store(total, Ordering::Relaxed);
                    p.done.store(done, Ordering::Relaxed);
                }
                *p.current_file.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = ev.status.clone();
            })
            .await?;
        row.total_bytes = progress.total.load(Ordering::Relaxed) as i64;
        let name = row.repo_id.clone();
        let now = now_rfc3339();
        let _ = self.inner.db.call(move |c| db::models::touch_meta(c, &format!("ollama:{name}"), &now)).await;
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // Controls
    // ---------------------------------------------------------------------------------------

    fn signal(&self, id: &str, intent: u8) -> bool {
        let active = self.inner.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(job) = active.get(id) {
            job.intent.store(intent, Ordering::SeqCst);
            job.cancel.cancel();
            true
        } else {
            false
        }
    }

    async fn row(&self, id: &str) -> AppResult<DownloadRow> {
        let i = id.to_string();
        self.inner
            .db
            .call(move |c| db::downloads::get(c, &i))
            .await?
            .ok_or_else(|| AppError::not_found("That download no longer exists."))
    }

    pub async fn pause(&self, id: &str) -> AppResult<()> {
        if self.signal(id, INTENT_PAUSE) {
            return Ok(());
        }
        let row = self.row(id).await?;
        if row.state.is_finished() {
            return Err(AppError::conflict("This download has already finished."));
        }
        Ok(())
    }

    pub async fn resume(&self, id: &str) -> AppResult<()> {
        if self.inner.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner).contains_key(id) {
            return Ok(());
        }
        let row = self.row(id).await?;
        match row.state {
            DownloadState::Paused | DownloadState::Failed | DownloadState::Queued => {
                if row.kind == DownloadKind::Ollama {
                    self.inner.ollama.version().await?;
                }
                self.spawn(row);
                Ok(())
            }
            DownloadState::Completed => Err(AppError::conflict("This download is already complete.")),
            DownloadState::Cancelled => {
                Err(AppError::conflict("This download was cancelled. Start it again from the model page."))
            }
            _ => Ok(()),
        }
    }

    pub async fn cancel(&self, id: &str) -> AppResult<()> {
        if self.signal(id, INTENT_CANCEL) {
            return Ok(());
        }
        let mut row = self.row(id).await?;
        if row.state.is_finished() {
            return Ok(());
        }
        self.remove_partials(&row).await;
        row.state = DownloadState::Cancelled;
        row.updated_at = now_rfc3339();
        self.save(&row).await;
        Ok(())
    }

    /// Removes a finished/paused job from the list (partial files are deleted).
    pub async fn remove(&self, id: &str) -> AppResult<()> {
        if self.inner.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner).contains_key(id) {
            return Err(AppError::conflict("Pause or cancel the download first."));
        }
        let row = self.row(id).await?;
        self.remove_partials(&row).await;
        let i = id.to_string();
        self.inner.db.call(move |c| db::downloads::delete(c, &i)).await?;
        Ok(())
    }

    pub async fn clear_finished(&self) -> AppResult<usize> {
        Ok(self.inner.db.call(db::downloads::delete_finished).await?)
    }

    /// Pauses every running job (used at shutdown so they resume next time).
    pub fn pause_all(&self) {
        let active = self.inner.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for job in active.values() {
            job.intent.store(INTENT_PAUSE, Ordering::SeqCst);
            job.cancel.cancel();
        }
    }

    pub fn active_count(&self) -> usize {
        self.inner.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner).len()
    }

    /// Waits (bounded) for running jobs to record their final state.
    pub async fn wait_idle(&self, max: Duration) {
        let start = Instant::now();
        while self.active_count() > 0 && start.elapsed() < max {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// All jobs with live progress merged in.
    pub async fn list(&self) -> AppResult<Vec<JobView>> {
        let rows = self.inner.db.call(db::downloads::list).await?;
        let active = self.inner.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(rows
            .into_iter()
            .map(|row| {
                let (done, total, speed, current) = match active.get(&row.id) {
                    Some(j) => (
                        j.progress.done.load(Ordering::Relaxed),
                        j.progress.total.load(Ordering::Relaxed),
                        j.progress.speed(),
                        j.progress.current_file.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone(),
                    ),
                    None => (row.done_bytes.max(0) as u64, row.total_bytes.max(0) as u64, 0.0, String::new()),
                };
                let total = total.max(done);
                let percent = if total > 0 {
                    (done as f64 / total as f64 * 100.0).min(100.0)
                } else if row.state == DownloadState::Completed {
                    100.0
                } else {
                    0.0
                };
                let eta = (speed > 1.0 && total > done).then(|| ((total - done) as f64 / speed) as u64);
                JobView {
                    row,
                    done_bytes: done,
                    total_bytes: total,
                    percent,
                    speed_bps: speed,
                    eta_secs: eta,
                    current_file: current,
                }
            })
            .collect())
    }
}

enum DlError {
    /// Worth retrying (network hiccup).
    Retryable(AppError),
    /// The finished file failed its checksum: worth one more try (a corrupted transfer), but
    /// not more (the published file itself may not match its checksum).
    Corrupt(AppError),
    /// Won't get better by retrying (auth, missing file, disk error, cancel).
    Fatal(AppError),
}

fn fatal_io(e: std::io::Error) -> DlError {
    let msg = if e.raw_os_error() == Some(28) || e.to_string().contains("No space left") {
        "Your disk is full. Free some space, then press Retry — the download will resume.".to_string()
    } else {
        format!("Could not write the file: {e}")
    };
    DlError::Fatal(AppError::Internal(anyhow::anyhow!(msg)))
}

/// Checks a file's size and checksum. LFS files: SHA-256; small files: git blob SHA-1.
pub async fn verify_file(path: &Path, file: &DownloadFile) -> std::io::Result<bool> {
    let meta = tokio::fs::metadata(path).await?;
    if meta.len() != file.size {
        return Ok(false);
    }
    if file.sha256.is_none() && file.git_oid.is_none() {
        return Ok(true);
    }
    let mut f = tokio::fs::File::open(path).await?;
    let mut sha256 = sha2::Sha256::new();
    let mut sha1 = sha1::Sha1::new();
    if file.git_oid.is_some() {
        sha1.update(format!("blob {}\0", file.size).as_bytes());
    }
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        if file.sha256.is_some() {
            sha256.update(&buf[..n]);
        }
        if file.git_oid.is_some() {
            sha1.update(&buf[..n]);
        }
    }
    if let Some(expected) = &file.sha256 {
        return Ok(hex::encode(sha256.finalize()).eq_ignore_ascii_case(expected));
    }
    if let Some(expected) = &file.git_oid {
        return Ok(hex::encode(sha1.finalize()).eq_ignore_ascii_case(expected));
    }
    Ok(true)
}

/// Builds the job's file list from repository metadata and the chosen file paths.
pub fn files_from_siblings(
    info: &crate::hub::HubModelInfo,
    commit: &str,
    paths: &[String],
) -> AppResult<Vec<DownloadFile>> {
    let mut out = Vec::new();
    for p in paths {
        let s =
            info.sibling(p).ok_or_else(|| AppError::bad_request(format!("{p} is not part of {}", info.summary.id)))?;
        let sha = s.sha256();
        out.push(DownloadFile {
            repo_id: info.summary.id.clone(),
            revision: commit.to_string(),
            path: p.clone(),
            local_path: p.clone(),
            size: s.size(),
            git_oid: if sha.is_none() { s.blob_id.clone() } else { None },
            sha256: sha,
            done: false,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn verification_uses_sha256_and_git_oid() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        tokio::fs::write(&p, b"hello world").await.unwrap();
        let mut f = DownloadFile {
            size: 11,
            sha256: Some("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9".into()),
            ..Default::default()
        };
        assert!(verify_file(&p, &f).await.unwrap());
        f.sha256 = Some("00".repeat(32));
        assert!(!verify_file(&p, &f).await.unwrap());
        // git hash-object of "hello world" (no newline)
        f.sha256 = None;
        f.git_oid = Some("95d09f2b10159347eece71399a7e2e907ea3df4f".into());
        assert!(verify_file(&p, &f).await.unwrap());
        f.size = 12;
        assert!(!verify_file(&p, &f).await.unwrap());
    }
}
