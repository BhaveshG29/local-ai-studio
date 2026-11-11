//! Secure storage of the Hugging Face access token.
//!
//! Order of precedence when reading: `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` environment
//! variables, then a token stored in the operating-system keychain (macOS Keychain,
//! Windows Credential Manager, Secret Service on Linux). If no keychain is available
//! (e.g. a headless Linux box without a Secret Service), a token entered in the UI is kept
//! **in memory only** for the current session and the UI says so. Tokens are never written
//! to `config.toml`, the database or logs.

use std::path::PathBuf;
use std::sync::Mutex;

const SERVICE: &str = "local-ai-studio";
const ACCOUNT: &str = "huggingface-token";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenSource {
    Environment,
    Keychain,
    SessionOnly,
    None,
}

impl TokenSource {
    pub fn label(self) -> &'static str {
        match self {
            TokenSource::Environment => "from the HF_TOKEN environment variable",
            TokenSource::Keychain => "stored in your system keychain",
            TokenSource::SessionOnly => "kept in memory until the app closes (no keychain available)",
            TokenSource::None => "not set",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TokenStatus {
    pub source: TokenSource,
    /// e.g. `hf_…a1B2`; never the full token.
    pub masked: Option<String>,
    pub keychain_available: bool,
}

pub struct Secrets {
    use_keychain: bool,
    /// Read `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` (off in tests, so a developer's own token
    /// never leaks into them).
    use_env: bool,
    /// Token entered this session when the keychain is unavailable.
    session: Mutex<Option<String>>,
    /// Cache of the keychain value: None = not loaded yet.
    cached: Mutex<Option<Option<String>>>,
    keychain_ok: Mutex<Option<bool>>,
}

impl Secrets {
    /// `use_keychain = false` keeps everything in memory (tests, `--no-keychain`).
    pub fn new(use_keychain: bool) -> Self {
        Self::with_env(use_keychain, true)
    }

    /// Like [`Secrets::new`], optionally ignoring tokens from environment variables.
    pub fn with_env(use_keychain: bool, use_env: bool) -> Self {
        let disabled_by_env = std::env::var_os("LOCAL_AI_STUDIO_NO_KEYCHAIN").is_some();
        Self {
            use_keychain: use_keychain && !disabled_by_env,
            use_env,
            session: Mutex::new(None),
            cached: Mutex::new(None),
            keychain_ok: Mutex::new(None),
        }
    }

    fn env_token(&self) -> Option<String> {
        if !self.use_env {
            return None;
        }
        ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
            .iter()
            .filter_map(|k| std::env::var(k).ok())
            .map(|v| v.trim().to_string())
            .find(|v| !v.is_empty())
    }

    /// The token to send to Hugging Face, if any.
    pub async fn hf_token(&self) -> Option<String> {
        if let Some(t) = self.env_token() {
            return Some(t);
        }
        if let Some(t) = lock(&self.session).clone() {
            return Some(t);
        }
        self.keychain_value().await
    }

    async fn keychain_value(&self) -> Option<String> {
        if !self.use_keychain {
            return None;
        }
        if let Some(v) = lock(&self.cached).clone() {
            return v;
        }
        let result = tokio::task::spawn_blocking(|| {
            let entry = keyring::Entry::new(SERVICE, ACCOUNT)?;
            entry.get_password()
        })
        .await;
        let value = match result {
            Ok(Ok(v)) => {
                *lock(&self.keychain_ok) = Some(true);
                Some(v)
            }
            Ok(Err(keyring::Error::NoEntry)) => {
                *lock(&self.keychain_ok) = Some(true);
                None
            }
            Ok(Err(e)) => {
                tracing::warn!("system keychain unavailable: {e}");
                *lock(&self.keychain_ok) = Some(false);
                None
            }
            Err(e) => {
                tracing::warn!("keychain task failed: {e}");
                None
            }
        };
        *lock(&self.cached) = Some(value.clone());
        value
    }

    pub async fn status(&self) -> TokenStatus {
        let keychain_available = self.keychain_available().await;
        if let Some(t) = self.env_token() {
            return TokenStatus { source: TokenSource::Environment, masked: Some(mask(&t)), keychain_available };
        }
        if let Some(t) = lock(&self.session).clone() {
            return TokenStatus { source: TokenSource::SessionOnly, masked: Some(mask(&t)), keychain_available };
        }
        match self.keychain_value().await {
            Some(t) => TokenStatus { source: TokenSource::Keychain, masked: Some(mask(&t)), keychain_available },
            None => TokenStatus { source: TokenSource::None, masked: None, keychain_available },
        }
    }

    /// Whether the OS keychain works on this machine (probed once).
    pub async fn keychain_available(&self) -> bool {
        if !self.use_keychain {
            return false;
        }
        if let Some(ok) = *lock(&self.keychain_ok) {
            return ok;
        }
        let _ = self.keychain_value().await;
        lock(&self.keychain_ok).unwrap_or(false)
    }

    /// Stores a token. Returns where it was stored.
    pub async fn set_hf_token(&self, token: &str) -> Result<TokenSource, String> {
        let token = token.trim().to_string();
        validate_token_format(&token)?;
        if self.use_keychain {
            let t = token.clone();
            let res = tokio::task::spawn_blocking(move || {
                let entry = keyring::Entry::new(SERVICE, ACCOUNT)?;
                entry.set_password(&t)
            })
            .await
            .map_err(|e| format!("keychain task failed: {e}"))?;
            match res {
                Ok(()) => {
                    *lock(&self.cached) = Some(Some(token));
                    *lock(&self.keychain_ok) = Some(true);
                    *lock(&self.session) = None;
                    return Ok(TokenSource::Keychain);
                }
                Err(e) => {
                    tracing::warn!("could not store token in keychain, keeping it for this session only: {e}");
                    *lock(&self.keychain_ok) = Some(false);
                }
            }
        }
        *lock(&self.session) = Some(token);
        Ok(TokenSource::SessionOnly)
    }

    pub async fn clear_hf_token(&self) -> Result<(), String> {
        *lock(&self.session) = None;
        if self.use_keychain {
            let res = tokio::task::spawn_blocking(|| {
                let entry = keyring::Entry::new(SERVICE, ACCOUNT)?;
                entry.delete_credential()
            })
            .await
            .map_err(|e| format!("keychain task failed: {e}"))?;
            match res {
                Ok(()) | Err(keyring::Error::NoEntry) => {}
                Err(e) => return Err(format!("Could not remove the token from your keychain: {e}")),
            }
        }
        *lock(&self.cached) = Some(None);
        Ok(())
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Shows only a prefix and the last four characters.
pub fn mask(token: &str) -> String {
    let tail: String = token.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    if token.starts_with("hf_") { format!("hf_…{tail}") } else { format!("…{tail}") }
}

/// Basic sanity checks so typos are caught before a network round-trip.
pub fn validate_token_format(token: &str) -> Result<(), String> {
    if token.is_empty() {
        return Err("Please paste a token.".into());
    }
    if token.len() < 8 || token.len() > 512 {
        return Err("That doesn't look like a Hugging Face token (they start with \"hf_\").".into());
    }
    if token.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("The token contains spaces or invisible characters; copy it again.".into());
    }
    Ok(())
}

/// Location where `huggingface-cli login` / `hf auth login` stores its token.
pub fn hf_cli_token_path() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(home).join("token"));
    }
    if let Some(cache) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(cache).join("huggingface").join("token"));
    }
    directories::BaseDirs::new().map(|b| b.home_dir().join(".cache").join("huggingface").join("token"))
}

/// Reads the token saved by the Hugging Face CLI, if present.
pub fn read_hf_cli_token() -> Option<String> {
    let path = hf_cli_token_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    let t = text.trim().to_string();
    (!t.is_empty()).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masking_hides_the_token() {
        assert_eq!(mask("hf_abcdefghijklmnop"), "hf_…mnop");
        assert_eq!(mask("abcdefgh"), "…efgh");
    }

    #[test]
    fn token_format_validation() {
        assert!(validate_token_format("").is_err());
        assert!(validate_token_format("hf_ab cd").is_err());
        assert!(validate_token_format("short").is_err());
        assert!(validate_token_format("hf_0123456789abcdefghij").is_ok());
    }

    #[tokio::test]
    async fn session_fallback_without_keychain() {
        let s = Secrets::with_env(false, false);
        assert!(s.hf_token().await.is_none());
        let src = s.set_hf_token("hf_0123456789abcdef").await.unwrap();
        assert_eq!(src, TokenSource::SessionOnly);
        assert_eq!(s.hf_token().await.as_deref(), Some("hf_0123456789abcdef"));
        let st = s.status().await;
        assert_eq!(st.source, TokenSource::SessionOnly);
        assert_eq!(st.masked.as_deref(), Some("hf_…cdef"));
        s.clear_hf_token().await.unwrap();
        assert!(s.hf_token().await.is_none());
    }
}
