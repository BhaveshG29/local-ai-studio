//! Application directories and user settings.
//!
//! Settings live in a human-editable `config.toml` inside the data directory (see
//! `config/local-ai-studio.example.toml` in the repository). Secrets are never stored here:
//! the Hugging Face token lives in the operating-system keychain (see [`crate::secrets`]).

use std::fmt;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// All directories the application uses. Created on startup.
#[derive(Clone, Debug)]
pub struct AppPaths {
    pub data_dir: PathBuf,
    pub config_file: PathBuf,
    pub db_file: PathBuf,
    pub default_models_dir: PathBuf,
    pub uploads_dir: PathBuf,
    pub outputs_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl AppPaths {
    /// Resolves the data directory: explicit override, then `LOCAL_AI_STUDIO_HOME`,
    /// then the platform default (e.g. `~/.local/share/local-ai-studio` on Linux,
    /// `~/Library/Application Support/dev.LocalAIStudio.local-ai-studio` on macOS,
    /// `%APPDATA%\LocalAIStudio\local-ai-studio\data` on Windows).
    pub fn resolve(override_dir: Option<PathBuf>) -> Result<Self> {
        let data_dir = if let Some(dir) = override_dir {
            dir
        } else if let Some(env) = std::env::var_os("LOCAL_AI_STUDIO_HOME") {
            PathBuf::from(env)
        } else {
            directories::ProjectDirs::from("dev", "LocalAIStudio", "local-ai-studio")
                .map(|d| d.data_dir().to_path_buf())
                .context("could not determine a data directory for this platform; pass --data-dir")?
        };
        // Paths are shown to people and handed to other programs (Ollama), so a relative
        // `--data-dir data` must not depend on where those programs run from.
        let data_dir = std::path::absolute(&data_dir)
            .with_context(|| format!("could not resolve the data directory {}", data_dir.display()))?;
        Ok(Self::from_data_dir(data_dir))
    }

    pub fn from_data_dir(data_dir: PathBuf) -> Self {
        Self {
            config_file: data_dir.join("config.toml"),
            db_file: data_dir.join("studio.db"),
            default_models_dir: data_dir.join("models"),
            uploads_dir: data_dir.join("uploads"),
            outputs_dir: data_dir.join("outputs"),
            logs_dir: data_dir.join("logs"),
            cache_dir: data_dir.join("cache"),
            data_dir,
        }
    }

    pub fn ensure(&self) -> Result<()> {
        for dir in [
            &self.data_dir,
            &self.default_models_dir,
            &self.uploads_dir,
            &self.outputs_dir,
            &self.logs_dir,
            &self.cache_dir,
        ] {
            std::fs::create_dir_all(dir).with_context(|| format!("could not create directory {}", dir.display()))?;
        }
        Ok(())
    }
}

/// The six UI themes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Theme {
    #[default]
    Midnight,
    Graphite,
    Neon,
    Minimal,
    Warm,
    Glass,
}

impl Theme {
    pub const ALL: [Theme; 6] =
        [Theme::Midnight, Theme::Graphite, Theme::Neon, Theme::Minimal, Theme::Warm, Theme::Glass];

    pub fn slug(self) -> &'static str {
        match self {
            Theme::Midnight => "midnight",
            Theme::Graphite => "graphite",
            Theme::Neon => "neon",
            Theme::Minimal => "minimal",
            Theme::Warm => "warm",
            Theme::Glass => "glass",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Theme::Midnight => "Midnight",
            Theme::Graphite => "Graphite",
            Theme::Neon => "Neon Cyberpunk",
            Theme::Minimal => "Pure Minimal",
            Theme::Warm => "Warm Studio",
            Theme::Glass => "Modern Glass",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Theme::Midnight => "Night-sky indigo with a faint star field, moonlight blue and gold markers.",
            Theme::Graphite => "Drafting-table charcoal on a faint grid. Square edges, chalk-white buttons.",
            Theme::Neon => "Magenta and cyan neon on violet-black, with HUD corners and cut-edge buttons.",
            Theme::Minimal => "White paper and black ink. Hairline borders and plenty of space.",
            Theme::Warm => "Linen workspace, walnut sidebar, amber key-cap buttons and serif headings.",
            Theme::Glass => "Frosted glass panels floating over a soft pastel aurora.",
        }
    }

    pub fn is_dark(self) -> bool {
        matches!(self, Theme::Midnight | Theme::Graphite | Theme::Neon)
    }
}

impl FromStr for Theme {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Theme::ALL.into_iter().find(|t| t.slug() == s).ok_or_else(|| format!("unknown theme `{s}`"))
    }
}

impl fmt::Display for Theme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// Which engine to prefer when a model could run on more than one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum BackendPreference {
    #[default]
    Auto,
    Builtin,
    Ollama,
}

/// Hardware acceleration preference for the built-in (Candle) engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Acceleration {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
}

impl Acceleration {
    pub fn label(self) -> &'static str {
        match self {
            Acceleration::Auto => "Automatic",
            Acceleration::Cpu => "CPU only",
            Acceleration::Cuda => "NVIDIA CUDA",
            Acceleration::Metal => "Apple Metal",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerSettings {
    /// Address to bind. Loopback by default; anything else requires `allow_network`.
    pub host: String,
    pub port: u16,
    /// Open the default browser when the app starts.
    pub open_browser: bool,
    /// Explicit opt-in to bind a non-loopback address (LAN access). Remote clients must
    /// then unlock the app with the access token shown in Settings.
    pub allow_network: bool,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self { host: "127.0.0.1".into(), port: 8765, open_browser: true, allow_network: false }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiSettings {
    pub theme: Theme,
    /// Open the most recent conversation when visiting the chat page.
    pub restore_last_chat: bool,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self { theme: Theme::Midnight, restore_last_chat: true }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageSettings {
    /// Where finished model files are stored. Defaults to `<data dir>/models`.
    pub models_dir: Option<PathBuf>,
    /// Where in-progress downloads are written. Defaults to a `.partial` folder inside
    /// the models directory (same disk, so completed files can be moved atomically).
    pub downloads_dir: Option<PathBuf>,
    /// Folder generated images are copied to with "Save to folder". Defaults to
    /// `<data dir>/outputs`.
    pub outputs_dir: Option<PathBuf>,
    pub max_concurrent_downloads: u8,
}

impl Default for StorageSettings {
    fn default() -> Self {
        Self { models_dir: None, downloads_dir: None, outputs_dir: None, max_concurrent_downloads: 2 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OllamaSettings {
    pub enabled: bool,
    /// Base URL of the Ollama server.
    pub base_url: String,
    /// How long Ollama keeps a model in memory after a request (Ollama syntax, e.g. "10m").
    pub keep_alive: String,
}

impl Default for OllamaSettings {
    fn default() -> Self {
        Self { enabled: true, base_url: "http://127.0.0.1:11434".into(), keep_alive: "10m".into() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HuggingFaceSettings {
    pub endpoint: String,
    /// Hide repositories that need a token unless one is configured.
    pub show_gated: bool,
}

impl Default for HuggingFaceSettings {
    fn default() -> Self {
        Self { endpoint: "https://huggingface.co".into(), show_gated: true }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceSettings {
    pub preferred_backend: BackendPreference,
    pub acceleration: Acceleration,
    /// CPU threads for the built-in engine (None = automatic, see
    /// [`crate::system::default_engine_threads`]). Applied at start-up.
    pub threads: Option<usize>,
    /// Unload built-in models after this many idle minutes (0 = never).
    pub unload_after_minutes: u32,
    /// Maximum number of built-in models kept in memory at once.
    pub max_loaded_models: u8,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            preferred_backend: BackendPreference::Auto,
            acceleration: Acceleration::Auto,
            threads: None,
            unload_after_minutes: 15,
            max_loaded_models: 1,
        }
    }
}

/// Writes an `f32` setting as the short decimal a person typed (0.3, not 0.30000001…).
fn tidy_f32<S: serde::Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(((*v as f64) * 10_000.0).round() / 10_000.0)
}

/// Default generation parameters for new conversations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationDefaults {
    #[serde(serialize_with = "tidy_f32")]
    pub temperature: f32,
    #[serde(serialize_with = "tidy_f32")]
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    #[serde(serialize_with = "tidy_f32")]
    pub repeat_penalty: f32,
    pub context_length: u32,
    pub seed: Option<u64>,
    pub system_prompt: String,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 1024,
            repeat_penalty: 1.1,
            context_length: 4096,
            seed: None,
            system_prompt: "You are a helpful assistant.".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingSettings {
    /// `error`, `warn`, `info`, `debug` or `trace`.
    pub level: String,
    /// Also write daily-rotated log files to `<data dir>/logs`.
    pub to_file: bool,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self { level: "info".into(), to_file: true }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MediaSettings {
    /// Path to `ffmpeg` (used for video frame extraction). None = search PATH.
    pub ffmpeg_path: Option<PathBuf>,
    /// How many frames to sample from a video for vision models.
    pub video_frames: u8,
}

impl Default for MediaSettings {
    fn default() -> Self {
        Self { ffmpeg_path: None, video_frames: 6 }
    }
}

/// Everything in `config.toml`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub server: ServerSettings,
    pub ui: UiSettings,
    pub storage: StorageSettings,
    pub ollama: OllamaSettings,
    pub huggingface: HuggingFaceSettings,
    pub inference: InferenceSettings,
    pub generation: GenerationDefaults,
    pub logging: LoggingSettings,
    pub media: MediaSettings,
}

pub const LOG_LEVELS: [&str; 5] = ["error", "warn", "info", "debug", "trace"];

impl Settings {
    /// Loads settings, falling back to defaults when the file does not exist yet.
    /// A malformed file is an error (we never silently overwrite a user's config).
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let mut s: Settings = toml::from_str(&text)
                    .with_context(|| format!("{} is not valid; fix or delete it", path.display()))?;
                s.normalize();
                Ok(s)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Settings::default()),
            Err(e) => Err(e).with_context(|| format!("could not read {}", path.display())),
        }
    }

    /// Writes settings atomically (temp file + rename).
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = toml::to_string_pretty(self).context("serialising settings")?;
        let header = "# Local AI Studio configuration. Edit while the app is stopped, or use the Settings page.\n# Secrets (Hugging Face token) are stored in your operating system's keychain, never here.\n\n";
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, format!("{header}{text}")).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
        Ok(())
    }

    /// Clamps values into supported ranges so a hand-edited file cannot break the app.
    pub fn normalize(&mut self) {
        let g = &mut self.generation;
        g.temperature = crate::util::clamp_f32(g.temperature, 0.0, 2.0);
        g.top_p = crate::util::clamp_f32(g.top_p, 0.0, 1.0);
        g.top_k = g.top_k.min(1000);
        g.max_tokens = g.max_tokens.clamp(1, 32_768);
        g.repeat_penalty = crate::util::clamp_f32(g.repeat_penalty, 0.5, 2.0);
        g.context_length = g.context_length.clamp(256, 262_144);
        self.storage.max_concurrent_downloads = self.storage.max_concurrent_downloads.clamp(1, 6);
        self.inference.max_loaded_models = self.inference.max_loaded_models.clamp(1, 8);
        self.media.video_frames = self.media.video_frames.clamp(1, 16);
        if !LOG_LEVELS.contains(&self.logging.level.as_str()) {
            self.logging.level = "info".into();
        }
        if self.server.port == 0 {
            self.server.port = ServerSettings::default().port;
        }
        let trimmed = self.ollama.base_url.trim().trim_end_matches('/').to_string();
        self.ollama.base_url = if trimmed.is_empty() { OllamaSettings::default().base_url } else { trimmed };
        let hf = self.huggingface.endpoint.trim().trim_end_matches('/').to_string();
        self.huggingface.endpoint = if hf.is_empty() { HuggingFaceSettings::default().endpoint } else { hf };
    }

    pub fn models_dir(&self, paths: &AppPaths) -> PathBuf {
        self.storage.models_dir.clone().unwrap_or_else(|| paths.default_models_dir.clone())
    }

    pub fn partial_dir(&self, paths: &AppPaths) -> PathBuf {
        self.storage.downloads_dir.clone().unwrap_or_else(|| self.models_dir(paths).join(".partial"))
    }

    pub fn outputs_dir(&self, paths: &AppPaths) -> PathBuf {
        self.storage.outputs_dir.clone().unwrap_or_else(|| paths.outputs_dir.clone())
    }

    /// True when the configured host is a loopback address (or `localhost`).
    pub fn binds_loopback_only(&self) -> bool {
        is_loopback_host(&self.server.host)
    }
}

/// Whether a host string refers to the local machine only.
pub fn is_loopback_host(host: &str) -> bool {
    let h = host.trim().trim_start_matches('[').trim_end_matches(']');
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    h.parse::<IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(false)
}

/// Validates a user-entered directory: must be absolute; created if missing; must be writable.
pub fn validate_writable_dir(input: &str) -> Result<PathBuf, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Please enter a folder path.".into());
    }
    let expanded = if let Some(rest) = trimmed.strip_prefix("~/") {
        match directories::BaseDirs::new() {
            Some(b) => b.home_dir().join(rest),
            None => PathBuf::from(trimmed),
        }
    } else {
        PathBuf::from(trimmed)
    };
    if !expanded.is_absolute() {
        return Err("Please enter a full (absolute) folder path.".into());
    }
    std::fs::create_dir_all(&expanded).map_err(|e| format!("Could not create {}: {e}", expanded.display()))?;
    let probe = expanded.join(".las-write-test");
    std::fs::write(&probe, b"ok").map_err(|e| format!("{} is not writable: {e}", expanded.display()))?;
    let _ = std::fs::remove_file(probe);
    Ok(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relative_data_dir_becomes_absolute() {
        let p = AppPaths::resolve(Some(PathBuf::from("some/data"))).unwrap();
        assert!(p.data_dir.is_absolute(), "{}", p.data_dir.display());
        assert!(p.data_dir.ends_with("some/data"));
        assert!(p.db_file.is_absolute());
    }

    #[test]
    fn defaults_round_trip_through_toml() {
        let s = Settings::default();
        let text = toml::to_string_pretty(&s).unwrap();
        let back: Settings = toml::from_str(&text).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn floats_are_written_tidily() {
        let mut s = Settings::default();
        s.generation.temperature = 0.3;
        let text = toml::to_string_pretty(&s).unwrap();
        assert!(text.contains("temperature = 0.3\n"), "{text}");
    }

    #[test]
    fn partial_files_are_accepted() {
        let s: Settings = toml::from_str("[ui]\ntheme = \"neon\"\n").unwrap();
        assert_eq!(s.ui.theme, Theme::Neon);
        assert_eq!(s.server.port, 8765);
    }

    #[test]
    fn values_are_clamped() {
        let mut s: Settings = toml::from_str(
            "[generation]\ntemperature = 9.0\ntop_p = -1\nmax_tokens = 0\n[logging]\nlevel = \"loud\"\n",
        )
        .unwrap();
        s.normalize();
        assert_eq!(s.generation.temperature, 2.0);
        assert_eq!(s.generation.top_p, 0.0);
        assert_eq!(s.generation.max_tokens, 1);
        assert_eq!(s.logging.level, "info");
    }

    #[test]
    fn save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut s = Settings::default();
        s.ui.theme = Theme::Glass;
        s.save(&path).unwrap();
        let loaded = Settings::load(&path).unwrap();
        assert_eq!(loaded.ui.theme, Theme::Glass);
    }

    #[test]
    fn malformed_config_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "this is = = not toml").unwrap();
        assert!(Settings::load(&path).is_err());
    }

    #[test]
    fn loopback_detection() {
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("::1"));
        assert!(is_loopback_host("[::1]"));
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.1.10"));
    }

    /// The documented example must use real keys and show the real defaults.
    #[test]
    fn example_config_matches_the_code() {
        let text = include_str!("../config/local-ai-studio.example.toml");
        let example: Settings = toml::from_str(text).expect("example config parses");
        assert_eq!(example, Settings::default());
        let defaults = toml::Value::try_from(Settings::default()).unwrap();
        let written: toml::Value = toml::from_str(text).unwrap();
        for (section, table) in written.as_table().unwrap() {
            let known = defaults
                .get(section)
                .and_then(|v| v.as_table())
                .unwrap_or_else(|| panic!("unknown section [{section}] in the example config"));
            for key in table.as_table().unwrap().keys() {
                assert!(known.contains_key(key), "unknown key {section}.{key} in the example config");
            }
        }
    }

    #[test]
    fn theme_slugs_round_trip() {
        for t in Theme::ALL {
            assert_eq!(t.slug().parse::<Theme>().unwrap(), t);
        }
        assert!("pink".parse::<Theme>().is_err());
    }
}
