//! Static assets compiled into the binary (CSS, icons, fonts), served with ETags.

use std::sync::LazyLock;

use axum::extract::Path;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

struct Asset {
    path: &'static str,
    mime: &'static str,
    bytes: &'static [u8],
}

macro_rules! asset {
    ($path:literal, $mime:literal) => {
        Asset { path: $path, mime: $mime, bytes: include_bytes!(concat!("../../assets/", $path)) }
    };
}

static ASSETS: &[Asset] = &[
    asset!("app.css", "text/css; charset=utf-8"),
    asset!("themes.css", "text/css; charset=utf-8"),
    asset!("icons.svg", "image/svg+xml"),
    asset!("logo.svg", "image/svg+xml"),
    asset!("fonts/Inter.woff", "font/woff"),
    asset!("fonts/Manrope.woff", "font/woff"),
    asset!("fonts/IBMPlexSans.woff", "font/woff"),
    asset!("fonts/SpaceGrotesk.woff", "font/woff"),
    asset!("fonts/Orbitron.woff", "font/woff"),
    asset!("fonts/Fraunces.woff", "font/woff"),
    asset!("fonts/NunitoSans.woff", "font/woff"),
    asset!("fonts/PlusJakartaSans.woff", "font/woff"),
    asset!("fonts/JetBrainsMono.woff", "font/woff"),
];

static ETAGS: LazyLock<Vec<String>> = LazyLock::new(|| {
    use sha2::Digest;
    ASSETS.iter().map(|a| format!("\"{}\"", &hex::encode(sha2::Sha256::digest(a.bytes))[..16])).collect()
});

/// Changes whenever a stylesheet changes, so pages can link `app.css?v=…` and browsers
/// never keep an outdated copy after an update.
pub fn stylesheet_rev() -> &'static str {
    static REV: LazyLock<String> = LazyLock::new(|| {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        for a in ASSETS.iter().filter(|a| a.path.ends_with(".css")) {
            h.update(a.bytes);
        }
        hex::encode(h.finalize())[..12].to_string()
    });
    &REV
}

fn find(path: &str) -> Option<(usize, &'static Asset)> {
    ASSETS.iter().enumerate().find(|(_, a)| a.path == path)
}

pub async fn serve(Path(path): Path<String>, headers: HeaderMap) -> Response {
    let Some((idx, asset)) = find(&path) else {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    };
    let etag = &ETAGS[idx];
    if headers.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()).map(|v| v == etag).unwrap_or(false) {
        return StatusCode::NOT_MODIFIED.into_response();
    }
    let mut r = asset.bytes.into_response();
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(asset.mime));
    // Always revalidate (a cheap 304 from localhost), so an updated app never shows old
    // icons, fonts or styles.
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    if let Ok(v) = HeaderValue::from_str(etag) {
        h.insert(header::ETAG, v);
    }
    r
}

pub async fn favicon() -> Response {
    serve(Path("logo.svg".to_string()), HeaderMap::new()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_assets_are_present_and_nonempty() {
        for a in ASSETS {
            assert!(!a.bytes.is_empty(), "{} is empty", a.path);
        }
        assert_eq!(ETAGS.len(), ASSETS.len());
    }
}
