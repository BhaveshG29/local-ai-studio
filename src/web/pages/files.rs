//! Serves user uploads (chat attachments) and generated images, with strict path checks.

use std::io::SeekFrom;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::SharedState;

#[derive(Deserialize, Default)]
pub struct FileQuery {
    #[serde(default)]
    download: Option<String>,
}

const ALLOWED: &[(&str, &str)] = &[
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
    ("wav", "audio/wav"),
    ("mp3", "audio/mpeg"),
    ("flac", "audio/flac"),
    ("ogg", "audio/ogg"),
    ("m4a", "audio/mp4"),
    ("mp4", "video/mp4"),
    ("mov", "video/quicktime"),
    ("webm", "video/webm"),
    ("mkv", "video/x-matroska"),
    ("avi", "video/x-msvideo"),
];

/// A single `Range: bytes=…` request, which media players send to seek; Safari won't play
/// audio or video at all without range support. Returns the inclusive byte range, `Err` when
/// it lies outside the file (416), and `None` for no or unusable ranges, which get the whole
/// file (several ranges in one request are allowed to be answered that way).
fn parse_range(value: &str, len: u64) -> Option<Result<(u64, u64), ()>> {
    let spec = value.trim().strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    let (a, b) = (a.trim(), b.trim());
    if len == 0 {
        return Some(Err(()));
    }
    let last = len - 1;
    let (start, end) = if a.is_empty() {
        // `bytes=-500`: the last 500 bytes.
        let n: u64 = b.parse().ok()?;
        if n == 0 {
            return Some(Err(()));
        }
        (len.saturating_sub(n), last)
    } else {
        let start: u64 = a.parse().ok()?;
        let end = match b {
            "" => last,
            b => {
                let end: u64 = b.parse().ok()?;
                // Ends before it starts: invalid, so the header is ignored (RFC 9110).
                if end < start {
                    return None;
                }
                end.min(last)
            }
        };
        (start, end)
    };
    if start > end {
        return Some(Err(()));
    }
    Some(Ok((start, end)))
}

pub async fn serve(
    State(state): State<SharedState>,
    Path((kind, path)): Path<(String, String)>,
    Query(q): Query<FileQuery>,
    headers: HeaderMap,
) -> Response {
    let base = match kind.as_str() {
        "uploads" => state.paths.uploads_dir.clone(),
        "outputs" => state.settings().outputs_dir(&state.paths),
        _ => return (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    let Some(full) = crate::util::safe_join(&base, &path) else {
        return (StatusCode::BAD_REQUEST, "invalid path").into_response();
    };
    let ext = full.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let Some((_, mime)) = ALLOWED.iter().find(|(e, _)| *e == ext) else {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    };
    let mut file = match tokio::fs::File::open(&full).await {
        Ok(f) => f,
        Err(_) => return (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    let len = file.metadata().await.map(|m| m.len()).unwrap_or(0);
    let range = headers.get(header::RANGE).and_then(|v| v.to_str().ok()).and_then(|v| parse_range(v, len));
    let (status, body, body_len, content_range) = match range {
        Some(Ok((start, end))) => {
            if file.seek(SeekFrom::Start(start)).await.is_err() {
                return (StatusCode::INTERNAL_SERVER_ERROR, "could not read the file").into_response();
            }
            let n = end - start + 1;
            let body = Body::from_stream(tokio_util::io::ReaderStream::new(file.take(n)));
            (StatusCode::PARTIAL_CONTENT, body, n, Some(format!("bytes {start}-{end}/{len}")))
        }
        Some(Err(())) => {
            let mut r = StatusCode::RANGE_NOT_SATISFIABLE.into_response();
            if let Ok(v) = HeaderValue::from_str(&format!("bytes */{len}")) {
                r.headers_mut().insert(header::CONTENT_RANGE, v);
            }
            return r;
        }
        None => (StatusCode::OK, Body::from_stream(tokio_util::io::ReaderStream::new(file)), len, None),
    };
    let mut r = body.into_response();
    *r.status_mut() = status;
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(mime));
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    if let Ok(v) = HeaderValue::from_str(&body_len.to_string()) {
        h.insert(header::CONTENT_LENGTH, v);
    }
    if let Some(v) = content_range.and_then(|c| HeaderValue::from_str(&c).ok()) {
        h.insert(header::CONTENT_RANGE, v);
    }
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("private, max-age=86400"));
    // User files must never be able to run code in the app's origin.
    h.insert(
        "content-security-policy",
        HeaderValue::from_static("default-src 'none'; img-src 'self'; media-src 'self'; sandbox"),
    );
    let name = full.file_name().map(|n| n.to_string_lossy().replace('"', "")).unwrap_or_else(|| "file".into());
    let disp = if q.download.is_some() {
        format!("attachment; filename=\"{name}\"")
    } else {
        format!("inline; filename=\"{name}\"")
    };
    if let Ok(v) = HeaderValue::from_str(&disp) {
        h.insert(header::CONTENT_DISPOSITION, v);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::parse_range;

    #[test]
    fn byte_ranges() {
        assert_eq!(parse_range("bytes=0-99", 1000), Some(Ok((0, 99))));
        assert_eq!(parse_range("bytes=500-", 1000), Some(Ok((500, 999))));
        assert_eq!(parse_range("bytes=-100", 1000), Some(Ok((900, 999))));
        // Ends past the file are cut to its end; a suffix longer than the file is all of it.
        assert_eq!(parse_range("bytes=900-5000", 1000), Some(Ok((900, 999))));
        assert_eq!(parse_range("bytes=-5000", 1000), Some(Ok((0, 999))));
        // Outside the file: 416.
        assert_eq!(parse_range("bytes=1000-", 1000), Some(Err(())));
        assert_eq!(parse_range("bytes=0-", 0), Some(Err(())));
        // Unusable or several ranges: the whole file.
        assert_eq!(parse_range("bytes=0-1,5-6", 1000), None);
        assert_eq!(parse_range("items=0-1", 1000), None);
        assert_eq!(parse_range("bytes=a-b", 1000), None);
        assert_eq!(parse_range("bytes=5-1", 1000), None);
    }
}
