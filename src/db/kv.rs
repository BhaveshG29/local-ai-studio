//! Tiny key/value store for flags and bookkeeping (e.g. when the index was last refreshed).

use rusqlite::{Connection, OptionalExtension, params};

pub fn get(c: &Connection, key: &str) -> rusqlite::Result<Option<String>> {
    c.query_row("SELECT value FROM kv WHERE key = ?1", params![key], |r| r.get(0)).optional()
}

pub fn set(c: &Connection, key: &str, value: &str) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO kv (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

pub fn delete(c: &Connection, key: &str) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM kv WHERE key = ?1", params![key])
}
