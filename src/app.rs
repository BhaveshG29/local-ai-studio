//! Application state shared by all request handlers and background tasks.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;

use crate::catalog::IndexStatus;
use crate::config::{AppPaths, Settings};
use crate::db::{self, Db};
use crate::downloads::DownloadManager;
use crate::hub::HubClient;
use crate::inference::chat::ChatService;
use crate::inference::manager::InferenceManager;
use crate::ollama::OllamaClient;
use crate::secrets::Secrets;
use crate::system::SystemMonitor;

/// Progress of a background job (image generation, transcription) shown by auto-refreshing
/// pages.
#[derive(Clone, Debug, Default)]
pub struct JobProgress {
    pub label: String,
    pub step: usize,
    pub total: usize,
    pub done: bool,
    pub error: Option<String>,
}

impl JobProgress {
    /// Completion in percent (0 while the total is still unknown).
    pub fn percent(&self) -> u32 {
        (self.step * 100).checked_div(self.total).map(|p| p.min(100) as u32).unwrap_or(0)
    }
}

#[derive(Default)]
pub struct JobRegistry {
    jobs: Mutex<HashMap<String, (JobProgress, CancellationToken)>>,
}

impl JobRegistry {
    fn map(&self) -> std::sync::MutexGuard<'_, HashMap<String, (JobProgress, CancellationToken)>> {
        self.jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn start(&self, id: &str, label: &str) -> CancellationToken {
        let token = CancellationToken::new();
        self.map()
            .insert(id.to_string(), (JobProgress { label: label.to_string(), ..Default::default() }, token.clone()));
        token
    }

    pub fn update(&self, id: &str, f: impl FnOnce(&mut JobProgress)) {
        if let Some((p, _)) = self.map().get_mut(id) {
            f(p);
        }
    }

    pub fn get(&self, id: &str) -> Option<JobProgress> {
        self.map().get(id).map(|(p, _)| p.clone())
    }

    pub fn cancel(&self, id: &str) -> bool {
        match self.map().get(id) {
            Some((p, t)) if !p.done => {
                t.cancel();
                true
            }
            _ => false,
        }
    }

    pub fn finish(&self, id: &str, error: Option<String>) {
        self.update(id, |p| {
            p.done = true;
            p.error = error;
        });
    }

    pub fn remove(&self, id: &str) {
        self.map().remove(id);
    }

    pub fn active_count(&self) -> usize {
        self.map().values().filter(|(p, _)| !p.done).count()
    }

    pub fn cancel_all(&self) {
        for (_, t) in self.map().values() {
            t.cancel();
        }
    }
}

/// Information about the running server.
pub struct ServerInfo {
    pub addr: RwLock<Option<SocketAddr>>,
    /// Required from non-loopback clients when network access is enabled.
    pub access_token: String,
    pub started_at: Instant,
}

impl ServerInfo {
    pub fn addr(&self) -> Option<SocketAddr> {
        *self.addr.read().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    pub fn url(&self) -> String {
        match self.addr() {
            Some(a) if a.ip().is_unspecified() => format!("http://127.0.0.1:{}", a.port()),
            Some(a) => format!("http://{a}"),
            None => String::new(),
        }
    }
}

/// Everything the app needs, created once at startup.
pub struct AppState {
    pub paths: AppPaths,
    pub settings: Arc<RwLock<Settings>>,
    pub db: Db,
    pub http: reqwest::Client,
    pub secrets: Arc<Secrets>,
    pub hub: HubClient,
    pub ollama: OllamaClient,
    pub downloads: DownloadManager,
    pub inference: InferenceManager,
    pub chat: ChatService,
    pub system: Arc<SystemMonitor>,
    pub jobs: JobRegistry,
    pub index: IndexStatus,
    pub shutdown: CancellationToken,
    pub server: ServerInfo,
    /// Whether settings changes are written to `config.toml`.
    persist_settings: bool,
}

pub type SharedState = Arc<AppState>;

/// Options used to construct the state (CLI flags, tests).
#[derive(Clone, Debug, Default)]
pub struct AppOptions {
    pub data_dir: Option<PathBuf>,
    /// Store secrets in the OS keychain (disabled in tests).
    pub use_keychain: bool,
    /// Override the Hugging Face endpoint (tests point this at a local fake server).
    pub hf_endpoint: Option<String>,
    /// Override the Ollama server URL.
    pub ollama_url: Option<String>,
    /// Override ollama.com / registry hosts (tests).
    pub ollama_library: Option<(String, String)>,
    /// Write config changes to disk (disabled for throwaway test states).
    pub persist_settings: bool,
    /// Ignore `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` (tests).
    pub ignore_env_token: bool,
}

impl AppState {
    /// Builds the state: directories, settings, database, clients and managers.
    pub fn build(opts: AppOptions) -> Result<SharedState> {
        let paths = AppPaths::resolve(opts.data_dir.clone())?;
        paths.ensure()?;
        let mut settings = Settings::load(&paths.config_file)?;
        if !paths.config_file.exists() && opts.persist_settings {
            settings.save(&paths.config_file).context("writing the default config.toml")?;
        }
        if let Some(e) = &opts.hf_endpoint {
            settings.huggingface.endpoint = e.clone();
        }
        if let Some(u) = &opts.ollama_url {
            settings.ollama.base_url = u.clone();
        }
        settings.normalize();
        std::fs::create_dir_all(settings.models_dir(&paths))
            .with_context(|| format!("creating models folder {}", settings.models_dir(&paths).display()))?;

        let db = Db::open(&paths.db_file)?;
        // Recover state left by a previous run that ended abruptly.
        db.call_blocking(|c| {
            let now = crate::util::now_rfc3339();
            db::conversations::recover_interrupted(c)?;
            db::downloads::pause_interrupted(c, &now)?;
            db::jobs::fail_interrupted(c)?;
            Ok(())
        })?;

        let http = reqwest::Client::builder()
            .user_agent(concat!("LocalAIStudio/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(15))
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .context("creating the HTTP client")?;
        let settings = Arc::new(RwLock::new(settings));
        let secrets = Arc::new(Secrets::with_env(opts.use_keychain, !opts.ignore_env_token));
        let system = Arc::new(SystemMonitor::new());
        let hub = HubClient::new(http.clone(), settings.clone(), secrets.clone(), db.clone());
        let mut ollama = OllamaClient::new(http.clone(), settings.clone(), db.clone());
        if let Some((lib, reg)) = &opts.ollama_library {
            ollama = ollama.with_library_base(lib, reg);
        }
        let downloads = DownloadManager::new(
            db.clone(),
            http.clone(),
            hub.clone(),
            ollama.clone(),
            paths.clone(),
            settings.clone(),
            system.clone(),
        );
        let inference = InferenceManager::new(db.clone(), settings.clone(), system.clone(), ollama.clone());
        let state = Arc::new(AppState {
            paths,
            settings,
            db,
            http,
            secrets,
            hub,
            ollama,
            downloads,
            inference,
            chat: ChatService::default(),
            system,
            jobs: JobRegistry::default(),
            index: IndexStatus::default(),
            shutdown: CancellationToken::new(),
            server: ServerInfo {
                addr: RwLock::new(None),
                access_token: crate::util::random_token(),
                started_at: Instant::now(),
            },
            persist_settings: opts.persist_settings,
        });
        Ok(state)
    }

    /// A copy of the current settings.
    pub fn settings(&self) -> Settings {
        self.settings.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// Applies a change, normalises it and saves `config.toml`.
    pub fn update_settings(&self, f: impl FnOnce(&mut Settings)) -> Result<Settings> {
        let mut guard = self.settings.write().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut next = guard.clone();
        f(&mut next);
        next.normalize();
        if self.persist_settings {
            next.save(&self.paths.config_file)?;
        }
        *guard = next.clone();
        Ok(next)
    }

    /// Starts periodic housekeeping (idle model unloading).
    pub fn spawn_background_tasks(self: &Arc<Self>) {
        let st = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = tick.tick() => st.inference.reap_idle(),
                    _ = st.shutdown.cancelled() => break,
                }
            }
        });
    }

    /// Stops everything gracefully: generations save partial answers, downloads pause.
    pub async fn shutdown_gracefully(&self) {
        self.shutdown.cancel();
        self.chat.stop_all();
        self.jobs.cancel_all();
        self.downloads.pause_all();
        self.chat.wait_idle(Duration::from_secs(5)).await;
        self.downloads.wait_idle(Duration::from_secs(5)).await;
        self.inference.unload_all();
    }
}
