//! SQLite persistence.
//!
//! A single connection (WAL mode) guarded by a mutex; every query runs on Tokio's blocking
//! thread pool via [`Db::call`] so async tasks never block on disk I/O. Schema changes are
//! applied with versioned migrations tracked in `PRAGMA user_version`.

pub mod catalog;
pub mod conversations;
pub mod downloads;
pub mod jobs;
pub mod kv;
pub mod models;

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context, Result};
use rusqlite::Connection;

#[derive(Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
}

impl Db {
    /// Opens (or creates) the database file and applies pending migrations.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("opening database {}", path.display()))?;
        Self::init(conn)
    }

    /// In-memory database (tests).
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        migrate(&conn)?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }

    /// Runs `f` with exclusive access to the connection on the blocking thread pool.
    pub async fn call<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        self.call_mut(move |c| f(c)).await
    }

    /// Like [`Db::call`], for work that needs a mutable connection (transactions).
    pub async fn call_mut<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = conn.lock().unwrap_or_else(PoisonError::into_inner);
            f(&mut guard)
        })
        .await
        .context("database task panicked")?
        .context("database query failed")
    }

    /// Synchronous access, for startup code and tests that are not inside a runtime.
    pub fn call_blocking<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> rusqlite::Result<T>,
    {
        let mut guard = self.conn.lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut guard).context("database query failed")
    }
}

const MIGRATIONS: &[&str] = &[
    // v1: initial schema
    r#"
    CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        model_ref TEXT,
        system_prompt TEXT NOT NULL DEFAULT '',
        params_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        attachments_json TEXT NOT NULL DEFAULT '[]',
        model_ref TEXT,
        status TEXT NOT NULL DEFAULT 'done',
        error TEXT,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        duration_ms INTEGER,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_messages_conversation ON messages(conversation_id, seq);

    CREATE TABLE models (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        repo_id TEXT,
        revision TEXT,
        display_name TEXT NOT NULL,
        path TEXT NOT NULL,
        primary_file TEXT,
        format TEXT NOT NULL,
        architecture TEXT,
        kind TEXT NOT NULL,
        size_bytes INTEGER NOT NULL DEFAULT 0,
        quantization TEXT,
        parameters INTEGER,
        context_length INTEGER,
        analysis_json TEXT NOT NULL DEFAULT '{}',
        files_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        last_used_at TEXT
    );
    CREATE INDEX idx_models_repo ON models(repo_id);

    CREATE TABLE model_meta (
        model_ref TEXT PRIMARY KEY,
        display_name TEXT,
        last_used_at TEXT
    );

    CREATE TABLE downloads (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        repo_id TEXT NOT NULL,
        revision TEXT,
        files_json TEXT NOT NULL DEFAULT '[]',
        state TEXT NOT NULL,
        total_bytes INTEGER NOT NULL DEFAULT 0,
        done_bytes INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        model_id TEXT,
        options_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE catalog (
        rowid INTEGER PRIMARY KEY,
        source TEXT NOT NULL,
        model_id TEXT NOT NULL,
        name TEXT NOT NULL,
        author TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',
        task TEXT NOT NULL DEFAULT '',
        library TEXT NOT NULL DEFAULT '',
        tags TEXT NOT NULL DEFAULT '',
        downloads INTEGER NOT NULL DEFAULT 0,
        likes INTEGER NOT NULL DEFAULT 0,
        updated TEXT NOT NULL DEFAULT '',
        params TEXT NOT NULL DEFAULT '',
        formats TEXT NOT NULL DEFAULT '',
        json TEXT NOT NULL DEFAULT '{}',
        indexed_at TEXT NOT NULL,
        UNIQUE(source, model_id)
    );
    CREATE VIRTUAL TABLE catalog_fts USING fts5(
        name, author, description, tags,
        content='catalog', content_rowid='rowid',
        tokenize='unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER catalog_ai AFTER INSERT ON catalog BEGIN
        INSERT INTO catalog_fts(rowid, name, author, description, tags)
        VALUES (new.rowid, new.name, new.author, new.description, new.tags);
    END;
    CREATE TRIGGER catalog_ad AFTER DELETE ON catalog BEGIN
        INSERT INTO catalog_fts(catalog_fts, rowid, name, author, description, tags)
        VALUES ('delete', old.rowid, old.name, old.author, old.description, old.tags);
    END;
    CREATE TRIGGER catalog_au AFTER UPDATE ON catalog BEGIN
        INSERT INTO catalog_fts(catalog_fts, rowid, name, author, description, tags)
        VALUES ('delete', old.rowid, old.name, old.author, old.description, old.tags);
        INSERT INTO catalog_fts(rowid, name, author, description, tags)
        VALUES (new.rowid, new.name, new.author, new.description, new.tags);
    END;

    CREATE TABLE http_cache (
        key TEXT PRIMARY KEY,
        body TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    );

    CREATE TABLE generations (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        model_id TEXT NOT NULL,
        model_name TEXT NOT NULL DEFAULT '',
        prompt TEXT NOT NULL,
        negative_prompt TEXT NOT NULL DEFAULT '',
        params_json TEXT NOT NULL DEFAULT '{}',
        outputs_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL,
        error TEXT,
        duration_ms INTEGER,
        created_at TEXT NOT NULL
    );

    CREATE TABLE transcriptions (
        id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        model_name TEXT NOT NULL DEFAULT '',
        file_name TEXT NOT NULL,
        stored_file TEXT NOT NULL DEFAULT '',
        language TEXT,
        task TEXT NOT NULL DEFAULT 'transcribe',
        text TEXT NOT NULL DEFAULT '',
        segments_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL,
        error TEXT,
        duration_ms INTEGER,
        created_at TEXT NOT NULL
    );

    CREATE TABLE kv (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    "#,
    // v2: generation speed as measured by the engine (excludes model loading)
    r#"
    ALTER TABLE messages ADD COLUMN tokens_per_sec REAL;
    "#,
];

fn migrate(conn: &Connection) -> Result<()> {
    let current: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    let target = MIGRATIONS.len() as i64;
    if current > target {
        anyhow::bail!(
            "the database was created by a newer version of Local AI Studio (schema v{current}, this build supports v{target}); please update the app"
        );
    }
    for (i, sql) in MIGRATIONS.iter().enumerate().skip(current as usize) {
        let version = i as i64 + 1;
        conn.execute_batch(&format!("BEGIN;\n{sql}\nPRAGMA user_version = {version};\nCOMMIT;"))
            .with_context(|| format!("applying database migration v{version}"))?;
        tracing::info!(version, "applied database migration");
    }
    Ok(())
}

/// Serialises a value to JSON for storage (infallible for our plain data types).
pub(crate) fn to_json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "null".into())
}

/// Parses a stored JSON column, falling back to the type's default on corruption.
pub(crate) fn from_json<T: serde::de::DeserializeOwned + Default>(s: &str) -> T {
    serde_json::from_str(s).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_apply_and_are_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        {
            let db = Db::open(&path).unwrap();
            let v: i64 = db.call_blocking(|c| c.query_row("PRAGMA user_version", [], |r| r.get(0))).unwrap();
            assert_eq!(v, MIGRATIONS.len() as i64);
        }
        // Re-opening must not re-run migrations.
        let db = Db::open(&path).unwrap();
        let n: i64 = db
            .call_blocking(|c| {
                c.query_row("SELECT count(*) FROM sqlite_master WHERE name = 'catalog_fts'", [], |r| r.get(0))
            })
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn newer_schema_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        {
            let c = Connection::open(&path).unwrap();
            c.pragma_update(None, "user_version", 999).unwrap();
        }
        let err = Db::open(&path).err().expect("must fail");
        assert!(format!("{err:#}").contains("newer version"));
    }
}
