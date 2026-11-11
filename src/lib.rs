//! Local AI Studio — discover, download, manage and run AI models locally.
//!
//! The crate is organised as a library (used by the binary in `main.rs` and by the
//! integration tests in `tests/`) with one module per subsystem:
//!
//! * [`config`]      – data directories and the user-editable `config.toml`.
//! * [`db`]          – SQLite persistence (conversations, models, downloads, catalog index…).
//! * [`secrets`]     – Hugging Face token storage in the operating-system keychain.
//! * [`system`]      – hardware detection and memory estimates.
//! * [`hub`]         – Hugging Face Hub API client and compatibility analysis.
//! * [`ollama`]      – Ollama local API client and ollama.com library catalog.
//! * [`catalog`]     – local, searchable metadata index of both catalogs.
//! * [`downloads`]   – resumable, verified download manager.
//! * [`library`]     – the local model library (analysis of downloaded files).
//! * [`inference`]   – inference backends (Ollama, Candle) and generation orchestration.
//! * [`media`]       – audio decoding/resampling, video frame extraction, image handling.
//! * [`render`]      – Markdown rendering with syntax highlighting.
//! * [`web`]         – the Axum server, HTML pages and JSON API.

pub mod app;
pub mod catalog;
pub mod config;
pub mod db;
pub mod downloads;
pub mod error;
pub mod hub;
pub mod inference;
pub mod library;
pub mod math;
pub mod media;
pub mod ollama;
pub mod render;
pub mod secrets;
pub mod system;
pub mod util;
pub mod web;

pub use app::{AppState, SharedState};

/// Human readable application name.
pub const APP_NAME: &str = "Local AI Studio";
/// Crate version, shown in the UI and returned by the health endpoint.
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
