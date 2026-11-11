//! Serves user uploads (chat attachments) and generated images, with strict path checks.

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

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

pub async fn serve(
    State(state): State<SharedState>,
    Path((kind, path)): Path<(String, String)>,
    Query(q): Query<FileQuery>,
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
    let file = match tokio::fs::File::open(&full).await {
        Ok(f) => f,
        Err(_) => return (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    let len = file.metadata().await.map(|m| m.len()).unwrap_or(0);
    let stream = tokio_util::io::ReaderStream::new(file);
    let mut r = Body::from_stream(stream).into_response();
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(mime));
    if let Ok(v) = HeaderValue::from_str(&len.to_string()) {
        h.insert(header::CONTENT_LENGTH, v);
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
