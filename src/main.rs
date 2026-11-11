//! Local AI Studio — command-line entry point. Starts the local server and opens the browser.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use local_ai_studio::app::{AppOptions, AppState};
use local_ai_studio::config::{AppPaths, Settings};
use local_ai_studio::web::{self, BindOutcome};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser, Debug)]
#[command(name = "local-ai-studio", version, about = "Discover, download and run AI models on your own computer.")]
struct Cli {
    /// Folder for settings, database, models and logs (default: your platform's app data folder).
    #[arg(long, env = "LOCAL_AI_STUDIO_HOME")]
    data_dir: Option<PathBuf>,
    /// Address to listen on (default from settings: 127.0.0.1).
    #[arg(long)]
    host: Option<String>,
    /// Port to listen on (default from settings: 8765).
    #[arg(long)]
    port: Option<u16>,
    /// Don't open the browser on start.
    #[arg(long)]
    no_browser: bool,
    /// Allow listening on a non-loopback address for this run (LAN access; requires the
    /// access token shown in Settings).
    #[arg(long)]
    allow_network: bool,
    /// Keep the Hugging Face token in memory only (don't use the OS keychain).
    #[arg(long)]
    no_keychain: bool,
    /// Log level (error, warn, info, debug, trace). Overrides the setting.
    #[arg(long)]
    log_level: Option<String>,
    /// Print system information as JSON and exit.
    #[arg(long)]
    system_info: bool,
}

fn init_logging(
    settings: &Settings,
    paths: &AppPaths,
    cli_level: Option<&str>,
) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let level = cli_level.unwrap_or(&settings.logging.level).to_string();
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "{level},hyper=warn,hyper_util=warn,reqwest=warn,h2=warn,rustls=warn,tower_http=warn,zbus=warn"
        ))
    });
    let console = tracing_subscriber::fmt::layer().with_target(false);
    if settings.logging.to_file {
        let appender = tracing_appender::rolling::daily(&paths.logs_dir, "local-ai-studio.log");
        let (writer, guard) = tracing_appender::non_blocking(appender);
        let file = tracing_subscriber::fmt::layer().with_ansi(false).with_writer(writer);
        tracing_subscriber::registry().with(filter).with(console).with(file).init();
        Some(guard)
    } else {
        tracing_subscriber::registry().with(filter).with(console).init();
        None
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let paths = AppPaths::resolve(cli.data_dir.clone())?;
    paths.ensure()?;
    let settings = Settings::load(&paths.config_file).context("loading settings")?;

    // Prompts are read on all cores and tokens generated on the performance cores, unless
    // Settings (or RAYON_NUM_THREADS) fixes one number for both.
    let plan = local_ai_studio::inference::threads::configure(settings.inference.threads);
    if std::env::var_os("RAYON_NUM_THREADS").is_none() && settings.inference.threads.is_some() {
        // Rayon's global pool (prompt reading) reads this once, at first use.
        // SAFETY: called at the very start of `main`, before any other thread exists,
        // so no other thread can be reading the environment concurrently.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", plan.prompt.to_string()) };
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("las-worker")
        .build()
        .context("starting the async runtime")?;
    runtime.block_on(run(cli, paths, settings))
}

async fn run(cli: Cli, paths: AppPaths, settings: Settings) -> anyhow::Result<()> {
    if cli.system_info {
        let info = local_ai_studio::system::SystemMonitor::new().snapshot().await;
        println!("{}", serde_json::to_string_pretty(&*info)?);
        return Ok(());
    }
    let _log_guard = init_logging(&settings, &paths, cli.log_level.as_deref());
    tracing::info!("{} {} starting", local_ai_studio::APP_NAME, local_ai_studio::APP_VERSION);
    tracing::info!("data folder: {}", paths.data_dir.display());
    let plan = local_ai_studio::inference::threads::plan();
    tracing::info!("built-in engine threads: {} for reading prompts, {} for generating", plan.prompt, plan.generate);

    let state = AppState::build(AppOptions {
        data_dir: Some(paths.data_dir.clone()),
        use_keychain: !cli.no_keychain,
        persist_settings: true,
        ..Default::default()
    })?;
    let s = state.settings();
    let host = cli.host.clone().unwrap_or_else(|| s.server.host.clone());
    if !local_ai_studio::config::is_loopback_host(&host) && !(s.server.allow_network || cli.allow_network) {
        anyhow::bail!(
            "Refusing to listen on {host}: that would expose Local AI Studio to your network. Pass --allow-network (or enable it in Settings → Server) if you really want this."
        );
    }
    let port = cli.port.unwrap_or(s.server.port);
    let listener = match web::bind(&host, port, cli.port.is_some()).await? {
        BindOutcome::Bound(l) => l,
        BindOutcome::AlreadyRunning(url) => {
            println!("Local AI Studio is already running at {url}");
            if s.server.open_browser && !cli.no_browser {
                let _ = webbrowser::open(&url);
            }
            return Ok(());
        }
    };
    let addr = listener.local_addr()?;
    let url =
        if addr.ip().is_unspecified() { format!("http://127.0.0.1:{}", addr.port()) } else { format!("http://{addr}") };
    if !addr.ip().is_loopback() {
        tracing::warn!(
            "Local AI Studio is reachable from your network at {addr}. Other devices need the access token from Settings → Server."
        );
        println!("\n  ⚠  Network access is ON. Access token for other devices: {}\n", state.server.access_token);
    }
    println!("\n  Local AI Studio is running at {url}\n  Press Ctrl+C to stop.\n");
    state.spawn_background_tasks();
    if s.server.open_browser && !cli.no_browser {
        let u = url.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            if let Err(e) = webbrowser::open(&u) {
                tracing::warn!("couldn't open the browser ({e}); open {u} yourself");
            }
        });
    }
    web::serve(state.clone(), listener).await?;
    tracing::info!("stopped cleanly");
    Ok(())
}
