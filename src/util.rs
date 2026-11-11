//! Small helpers shared across the crate: ids, formatting, escaping and path safety.

use std::path::{Component, Path, PathBuf};

use chrono::{DateTime, Utc};

/// Current time as an RFC 3339 string (UTC, second precision).
pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// A new random identifier (32 lowercase hex characters).
pub fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// A random secret suitable for access tokens (64 hex characters, 244 bits of entropy).
pub fn random_token() -> String {
    format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple())
}

/// Constant-time string comparison for secrets.
pub fn secure_eq(a: &str, b: &str) -> bool {
    use subtle::ConstantTimeEq;
    a.len() == b.len() && bool::from(a.as_bytes().ct_eq(b.as_bytes()))
}

/// Formats a byte count using binary units ("1.4 GB").
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if value >= 100.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else if value >= 10.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Formats large counts compactly ("5.6M", "12.3K").
pub fn human_count(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => trim_decimal(n as f64 / 1_000.0, "K"),
        1_000_000..=999_999_999 => trim_decimal(n as f64 / 1_000_000.0, "M"),
        _ => trim_decimal(n as f64 / 1_000_000_000.0, "B"),
    }
}

fn trim_decimal(v: f64, suffix: &str) -> String {
    if v >= 100.0 {
        format!("{v:.0}{suffix}")
    } else {
        let s = format!("{v:.1}");
        let s = s.strip_suffix(".0").unwrap_or(&s).to_string();
        format!("{s}{suffix}")
    }
}

/// Formats a parameter count ("494M", "7.6B").
pub fn human_params(n: u64) -> String {
    if n >= 1_000_000_000 {
        trim_decimal(n as f64 / 1e9, "B")
    } else if n >= 1_000_000 {
        trim_decimal(n as f64 / 1e6, "M")
    } else if n >= 1_000 {
        trim_decimal(n as f64 / 1e3, "K")
    } else {
        n.to_string()
    }
}

/// Formats a duration in seconds as "1h 02m", "3m 05s" or "12s".
pub fn human_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

/// "3 days ago" style relative time for an RFC 3339 timestamp; falls back to the raw text.
pub fn relative_time(ts: &str) -> String {
    let Ok(t) = DateTime::parse_from_rfc3339(ts) else {
        return ts.to_string();
    };
    relative_time_from(t.with_timezone(&Utc), Utc::now())
}

/// Relative time between `t` and `now` (exposed for tests).
pub fn relative_time_from(t: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let secs = (now - t).num_seconds();
    if secs < 0 {
        return "just now".into();
    }
    let secs = secs as u64;
    let plural = |n: u64, unit: &str| {
        if n == 1 { format!("1 {unit} ago") } else { format!("{n} {unit}s ago") }
    };
    match secs {
        0..=44 => "just now".into(),
        45..=3_599 => plural((secs / 60).max(1), "minute"),
        3_600..=86_399 => plural(secs / 3_600, "hour"),
        86_400..=2_591_999 => plural(secs / 86_400, "day"),
        2_592_000..=31_535_999 => plural(secs / 2_592_000, "month"),
        _ => plural(secs / 31_536_000, "year"),
    }
}

/// Short local date-time for message timestamps ("2026-11-10 21:36").
pub fn short_datetime(ts: &str) -> String {
    match DateTime::parse_from_rfc3339(ts) {
        Ok(t) => t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string(),
        Err(_) => ts.to_string(),
    }
}

/// Escapes text for safe inclusion in HTML element content and attribute values.
pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + s.len() / 8);
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Percent-encodes a string for use in a URL query value or path segment.
pub fn url_encode(s: &str) -> String {
    percent_encoding::utf8_percent_encode(s, percent_encoding::NON_ALPHANUMERIC).to_string()
}

/// Turns an arbitrary name into a safe single path component (no separators, no dots-only names).
pub fn sanitize_component(name: &str) -> String {
    let mut out: String =
        name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '-' }).collect();
    while out.contains("..") {
        out = out.replace("..", ".");
    }
    let out = out.trim_matches(|c| c == '.' || c == '-').to_string();
    if out.is_empty() { "unnamed".to_string() } else { out.chars().take(120).collect() }
}

/// Joins a *relative* path (e.g. a file path inside a model repository) onto `base`,
/// rejecting absolute paths, `..` components and other escapes. Returns `None` if unsafe.
pub fn safe_join(base: &Path, rel: &str) -> Option<PathBuf> {
    if rel.is_empty() || rel.contains('\0') || rel.contains('\\') {
        return None;
    }
    let rel_path = Path::new(rel);
    let mut out = base.to_path_buf();
    for comp in rel_path.components() {
        match comp {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => return None,
        }
    }
    if out == base {
        return None;
    }
    Some(out)
}

/// Extracts a llama.cpp style quantisation label from a file name, e.g. `Q4_K_M`, `IQ3_XS`, `F16`.
pub fn parse_quantization(file_name: &str) -> Option<String> {
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r"(?i)(?:^|[-_.])((?:I?Q[1-8](?:_[0-9])?(?:_K)?(?:_(?:XXS|XS|S|M|L|XL|NL))?)|TQ[12]_0|MXFP4|BF16|F16|F32|FP16|FP8)(?:[-_.]|$)",
        )
        .expect("valid regex")
    });
    let stem = file_name.rsplit('/').next().unwrap_or(file_name).trim_end_matches(".gguf");
    RE.captures(stem).and_then(|c| c.get(1)).map(|m| m.as_str().to_uppercase().replace("FP16", "F16"))
}

/// Returns true when a GGUF file name looks like one shard of a split model
/// (`model-00001-of-00003.gguf`).
pub fn is_split_gguf(file_name: &str) -> bool {
    static RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)-\d{5}-of-\d{5}\.gguf$").expect("valid regex"));
    RE.is_match(file_name)
}

/// The common base name of a split GGUF shard (`m-Q4_K_M-00001-of-00003.gguf` → `m-Q4_K_M`).
pub fn split_gguf_base(file_name: &str) -> String {
    static RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)-\d{5}-of-\d{5}\.gguf$").expect("valid regex"));
    RE.replace(file_name, "").to_string()
}

/// Clamp helper for user supplied numbers.
pub fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() { lo } else { v.clamp(lo, hi) }
}

/// Truncates a string to at most `max` characters, appending an ellipsis when cut.
pub fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

/// "1 file", "3 files".
pub fn count_noun(n: impl Into<u64>, one: &str, many: &str) -> String {
    let n = n.into();
    format!("{n} {}", if n == 1 { one } else { many })
}

/// A short conversation title from the first message: its first line without Markdown
/// markers, cut at a word boundary. `None` when nothing readable is left.
pub fn chat_title_from(text: &str) -> Option<String> {
    const MAX: usize = 48;
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    let line = line.trim_start_matches(['#', '>', '-', '*', ' ']).trim();
    let words: Vec<&str> =
        line.split_whitespace().map(|w| w.trim_matches(['*', '_', '`'])).filter(|w| !w.is_empty()).collect();
    if words.is_empty() {
        return None;
    }
    let mut title = String::new();
    for w in &words {
        let extra = if title.is_empty() { 0 } else { 1 };
        if title.chars().count() + extra + w.chars().count() > MAX {
            if title.is_empty() {
                // One very long word: cut it.
                return Some(truncate_chars(w, MAX));
            }
            title = title.trim_end_matches([',', ';', ':', '.', '-']).to_string();
            title.push('…');
            return Some(title);
        }
        if extra == 1 {
            title.push(' ');
        }
        title.push_str(w);
    }
    Some(title)
}

/// Opens a directory in the operating system's file manager.
pub fn open_in_file_manager(path: &Path) -> std::io::Result<()> {
    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = std::process::Command::new("explorer");
        c.arg(path);
        c
    } else if cfg!(target_os = "macos") {
        let mut c = std::process::Command::new("open");
        c.arg(path);
        c
    } else {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(path);
        c
    };
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    cmd.spawn().map(|_| ())
}

/// Searches `PATH` for an executable.
pub fn find_executable(name: &str) -> Option<PathBuf> {
    let exe_names: Vec<String> = if cfg!(windows) {
        vec![format!("{name}.exe"), format!("{name}.cmd"), name.to_string()]
    } else {
        vec![name.to_string()]
    };
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for exe in &exe_names {
            let candidate = dir.join(exe);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counted_nouns() {
        assert_eq!(count_noun(1u32, "file", "files"), "1 file");
        assert_eq!(count_noun(0u32, "file", "files"), "0 files");
        assert_eq!(count_noun(3u64, "image", "images"), "3 images");
    }

    #[test]
    fn chat_titles_come_from_the_first_line() {
        assert_eq!(chat_title_from("  \n## What is a VAE?\nMore text").as_deref(), Some("What is a VAE?"));
        assert_eq!(
            chat_title_from("In three sentences, explain the reparameterization trick used to train a VAE.").as_deref(),
            Some("In three sentences, explain the…")
        );
        assert_eq!(chat_title_from("**Bold** `code` question").as_deref(), Some("Bold code question"));
        assert_eq!(chat_title_from(&"x".repeat(80)).unwrap().chars().count(), 48);
        assert_eq!(chat_title_from("   \n  "), None);
        assert_eq!(chat_title_from("***"), None);
    }

    #[test]
    fn bytes_are_human_readable() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.50 KB");
        assert_eq!(human_bytes(415_182_688), "396 MB");
        assert_eq!(human_bytes(7_800_000_000), "7.26 GB");
    }

    #[test]
    fn counts_are_compact() {
        assert_eq!(human_count(999), "999");
        assert_eq!(human_count(5_616_244), "5.6M");
        assert_eq!(human_count(1_000), "1K");
        assert_eq!(human_count(124_280), "124K");
        assert_eq!(human_params(494_032_768), "494M");
        assert_eq!(human_params(7_615_616_512), "7.6B");
    }

    #[test]
    fn html_is_escaped() {
        assert_eq!(
            html_escape("<script>alert('x')</script> & \"q\""),
            "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt; &amp; &quot;q&quot;"
        );
    }

    #[test]
    fn safe_join_rejects_traversal() {
        let base = Path::new("/models/x");
        assert!(safe_join(base, "../etc/passwd").is_none());
        assert!(safe_join(base, "/etc/passwd").is_none());
        assert!(safe_join(base, "a/../../b").is_none());
        assert!(safe_join(base, "").is_none());
        assert!(safe_join(base, "a\\..\\b").is_none());
        assert_eq!(safe_join(base, "unet/config.json").unwrap(), PathBuf::from("/models/x/unet/config.json"));
    }

    #[test]
    fn components_are_sanitized() {
        assert_eq!(sanitize_component("Qwen/Qwen2.5"), "Qwen-Qwen2.5");
        assert_eq!(sanitize_component("../.."), "unnamed");
        assert_eq!(sanitize_component("a b:c"), "a-b-c");
    }

    #[test]
    fn quantization_is_parsed_from_file_names() {
        assert_eq!(parse_quantization("qwen2.5-0.5b-instruct-q4_k_m.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(parse_quantization("Llama-3.2-1B-Instruct-IQ3_XS.gguf").as_deref(), Some("IQ3_XS"));
        assert_eq!(parse_quantization("model-q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(parse_quantization("qwen2.5-0.5b-instruct-fp16.gguf").as_deref(), Some("F16"));
        assert_eq!(parse_quantization("model.gguf"), None);
    }

    #[test]
    fn split_ggufs_are_detected() {
        assert!(is_split_gguf("big-model-Q4_K_M-00001-of-00003.gguf"));
        assert!(!is_split_gguf("small-Q4_K_M.gguf"));
    }

    #[test]
    fn relative_times() {
        let now = Utc::now();
        assert_eq!(relative_time_from(now, now), "just now");
        assert_eq!(relative_time_from(now - chrono::Duration::minutes(5), now), "5 minutes ago");
        assert_eq!(relative_time_from(now - chrono::Duration::days(1), now), "1 day ago");
    }

    #[test]
    fn secrets_compare() {
        assert!(secure_eq("abc", "abc"));
        assert!(!secure_eq("abc", "abd"));
        assert!(!secure_eq("abc", "abcd"));
    }
}
