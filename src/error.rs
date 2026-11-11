//! Application error type. Every handler returns [`AppResult`]; errors are rendered as a
//! themed HTML page for browser routes and as JSON for `/api` routes (see `web::errors`).

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Errors surfaced to users. Messages are written for people, not developers.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    Forbidden(String),
    #[error("{0}")]
    Conflict(String),
    /// A remote service (Hugging Face, ollama.com, a local Ollama server) failed or refused.
    #[error("{0}")]
    Upstream(String),
    /// A required local component is not available (e.g. Ollama not running, ffmpeg missing).
    #[error("{0}")]
    Unavailable(String),
    #[error("{0}")]
    TooLarge(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

pub type AppResult<T> = Result<T, AppError>;

impl AppError {
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest(msg.into())
    }
    pub fn upstream(msg: impl Into<String>) -> Self {
        Self::Upstream(msg.into())
    }
    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self::Unavailable(msg.into())
    }
    pub fn conflict(msg: impl Into<String>) -> Self {
        Self::Conflict(msg.into())
    }

    pub fn status(&self) -> StatusCode {
        match self {
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::Forbidden(_) => StatusCode::FORBIDDEN,
            AppError::Conflict(_) => StatusCode::CONFLICT,
            AppError::Upstream(_) => StatusCode::BAD_GATEWAY,
            AppError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Short, friendly heading for the error page.
    pub fn title(&self) -> &'static str {
        match self {
            AppError::NotFound(_) => "Not found",
            AppError::BadRequest(_) => "That didn't work",
            AppError::Forbidden(_) => "Not allowed",
            AppError::Conflict(_) => "Already in progress",
            AppError::Upstream(_) => "A service didn't respond as expected",
            AppError::Unavailable(_) => "Something needed isn't available",
            AppError::TooLarge(_) => "Too large",
            AppError::Internal(_) => "Something went wrong",
        }
    }

    /// Message safe to show to users (internal errors keep their chain, which never
    /// contains secrets because tokens are never formatted into errors).
    pub fn user_message(&self) -> String {
        match self {
            AppError::Internal(e) => format!("{e:#}"),
            other => other.to_string(),
        }
    }
}

/// Marker extension attached to error responses so the error-page middleware can render
/// them with the user's theme (HTML) or as JSON (API routes).
#[derive(Clone, Debug)]
pub struct ErrorInfo {
    pub status: StatusCode,
    pub title: String,
    pub message: String,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();
        if status.is_server_error() {
            tracing::error!(error = %self.user_message(), "request failed");
        } else {
            tracing::debug!(error = %self.user_message(), "request rejected");
        }
        let info = ErrorInfo { status, title: self.title().to_string(), message: self.user_message() };
        let mut resp = (status, info.message.clone()).into_response();
        resp.extensions_mut().insert(info);
        resp
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(e: rusqlite::Error) -> Self {
        AppError::Internal(anyhow::Error::new(e).context("database error"))
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Internal(anyhow::Error::new(e))
    }
}
