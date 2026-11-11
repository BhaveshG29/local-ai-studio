//! Local metadata index (full-text searchable) and the HTTP response cache.

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct CatalogEntry {
    /// `huggingface` or `ollama`.
    pub source: String,
    /// Repo id (HF) or library name (Ollama).
    pub model_id: String,
    pub name: String,
    pub author: String,
    pub description: String,
    pub task: String,
    pub library: String,
    /// Space-separated tags / capabilities.
    pub tags: String,
    pub downloads: i64,
    pub likes: i64,
    pub updated: String,
    /// Parameter sizes ("7b 13b") or a parameter count.
    pub params: String,
    /// "gguf safetensors" etc.
    pub formats: String,
    pub indexed_at: String,
}

pub fn upsert_many(c: &mut Connection, entries: &[CatalogEntry]) -> rusqlite::Result<usize> {
    let tx = c.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO catalog (source, model_id, name, author, description, task, library, tags, downloads, likes,
                                  updated, params, formats, json, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, '{}', ?14)
             ON CONFLICT(source, model_id) DO UPDATE SET
                name = excluded.name, author = excluded.author, description = excluded.description,
                task = excluded.task, library = excluded.library, tags = excluded.tags,
                downloads = excluded.downloads, likes = excluded.likes, updated = excluded.updated,
                params = excluded.params, formats = excluded.formats, indexed_at = excluded.indexed_at",
        )?;
        for e in entries {
            stmt.execute(params![
                e.source,
                e.model_id,
                e.name,
                e.author,
                e.description,
                e.task,
                e.library,
                e.tags,
                e.downloads,
                e.likes,
                e.updated,
                e.params,
                e.formats,
                e.indexed_at
            ])?;
        }
    }
    tx.commit()?;
    Ok(entries.len())
}

/// Turns free text into a safe FTS5 query: every word becomes a quoted prefix term.
pub fn fts_query(text: &str) -> Option<String> {
    let terms: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric() && c != '.' && c != '-' && c != '_')
        .filter(|t| !t.is_empty())
        .take(8)
        .map(|t| format!("\"{}\"*", t.replace('"', "")))
        .collect();
    if terms.is_empty() { None } else { Some(terms.join(" ")) }
}

#[derive(Clone, Debug, Default)]
pub struct CatalogQuery {
    pub text: String,
    pub source: Option<String>,
    pub task: Option<String>,
    pub format: Option<String>,
    pub sort: CatalogSort,
    pub limit: usize,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CatalogSort {
    #[default]
    Relevance,
    Downloads,
    Likes,
    Updated,
}

pub fn search(c: &Connection, q: &CatalogQuery) -> rusqlite::Result<(Vec<CatalogEntry>, i64)> {
    let mut where_sql = vec!["1 = 1".to_string()];
    let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    let fts = fts_query(&q.text);
    let from = if let Some(f) = &fts {
        args.push(Box::new(f.clone()));
        where_sql.push(format!("catalog_fts MATCH ?{}", args.len()));
        "catalog c JOIN catalog_fts ON catalog_fts.rowid = c.rowid"
    } else {
        "catalog c"
    };
    if let Some(s) = &q.source {
        args.push(Box::new(s.clone()));
        where_sql.push(format!("c.source = ?{}", args.len()));
    }
    if let Some(t) = &q.task {
        args.push(Box::new(t.clone()));
        where_sql.push(format!("(c.task = ?{n} OR (' ' || c.tags || ' ') LIKE '% ' || ?{n} || ' %')", n = args.len()));
    }
    if let Some(f) = &q.format {
        args.push(Box::new(format!("%{f}%")));
        where_sql.push(format!("c.formats LIKE ?{}", args.len()));
    }
    let order = match (q.sort, fts.is_some()) {
        (CatalogSort::Relevance, true) => "bm25(catalog_fts) ASC, c.downloads DESC",
        (CatalogSort::Likes, _) => "c.likes DESC",
        (CatalogSort::Updated, _) => "c.updated DESC",
        _ => "c.downloads DESC",
    };
    let where_clause = where_sql.join(" AND ");
    let count_sql = format!("SELECT count(*) FROM {from} WHERE {where_clause}");
    let total: i64 =
        c.query_row(&count_sql, rusqlite::params_from_iter(args.iter().map(|a| a.as_ref())), |r| r.get(0))?;
    let sql = format!(
        "SELECT c.source, c.model_id, c.name, c.author, c.description, c.task, c.library, c.tags, c.downloads,
                c.likes, c.updated, c.params, c.formats, c.indexed_at
         FROM {from} WHERE {where_clause} ORDER BY {order} LIMIT {} OFFSET {}",
        q.limit.clamp(1, 200),
        q.offset
    );
    let mut stmt = c.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(args.iter().map(|a| a.as_ref())), |r| {
        Ok(CatalogEntry {
            source: r.get(0)?,
            model_id: r.get(1)?,
            name: r.get(2)?,
            author: r.get(3)?,
            description: r.get(4)?,
            task: r.get(5)?,
            library: r.get(6)?,
            tags: r.get(7)?,
            downloads: r.get(8)?,
            likes: r.get(9)?,
            updated: r.get(10)?,
            params: r.get(11)?,
            formats: r.get(12)?,
            indexed_at: r.get(13)?,
        })
    })?;
    Ok((rows.collect::<rusqlite::Result<Vec<_>>>()?, total))
}

pub fn count_by_source(c: &Connection) -> rusqlite::Result<Vec<(String, i64)>> {
    let mut stmt = c.prepare("SELECT source, count(*) FROM catalog GROUP BY source ORDER BY source")?;
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
    rows.collect()
}

pub fn clear(c: &Connection) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM catalog", [])
}

/// Cached body for a key if it is newer than `max_age_secs` (or any age when None).
pub fn cache_get(c: &Connection, key: &str, max_age_secs: Option<i64>) -> rusqlite::Result<Option<(String, String)>> {
    let row: Option<(String, String)> = c
        .query_row("SELECT body, fetched_at FROM http_cache WHERE key = ?1", params![key], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?;
    Ok(row.filter(|(_, fetched)| match max_age_secs {
        None => true,
        Some(max) => chrono::DateTime::parse_from_rfc3339(fetched)
            .map(|t| (chrono::Utc::now() - t.with_timezone(&chrono::Utc)).num_seconds() <= max)
            .unwrap_or(false),
    }))
}

pub fn cache_put(c: &Connection, key: &str, body: &str, now: &str) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO http_cache (key, body, fetched_at) VALUES (?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET body = excluded.body, fetched_at = excluded.fetched_at",
        params![key, body, now],
    )?;
    Ok(())
}

pub fn cache_clear(c: &Connection) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM http_cache", [])
}

pub fn cache_stats(c: &Connection) -> rusqlite::Result<(i64, i64)> {
    c.query_row("SELECT count(*), COALESCE(sum(length(body)), 0) FROM http_cache", [], |r| Ok((r.get(0)?, r.get(1)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;

    fn entry(source: &str, id: &str, desc: &str, downloads: i64) -> CatalogEntry {
        CatalogEntry {
            source: source.into(),
            model_id: id.into(),
            name: id.rsplit('/').next().unwrap().into(),
            author: id.split('/').next().unwrap().into(),
            description: desc.into(),
            task: "text-generation".into(),
            tags: "gguf chat".into(),
            downloads,
            formats: "gguf".into(),
            indexed_at: crate::util::now_rfc3339(),
            ..Default::default()
        }
    }

    #[test]
    fn fts_search_finds_and_ranks() {
        let db = Db::open_in_memory().unwrap();
        db.call_blocking(|c| {
            upsert_many(
                c,
                &[
                    entry("huggingface", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "small chat model", 10),
                    entry("huggingface", "meta-llama/Llama-3.2-1B", "llama model", 500),
                    entry("ollama", "llama3.2", "Meta's Llama 3.2 goes small", 1000),
                ],
            )?;
            // update path keeps FTS in sync
            upsert_many(c, &[entry("ollama", "llama3.2", "Meta Llama 3.2 small models", 2000)])?;
            let (rows, total) = search(c, &CatalogQuery { text: "llama".into(), limit: 10, ..Default::default() })?;
            assert_eq!(total, 2);
            assert!(rows.iter().all(|r| r.model_id.to_lowercase().contains("llama")));
            let (rows, _) = search(
                c,
                &CatalogQuery {
                    text: "qwen2.5".into(),
                    source: Some("huggingface".into()),
                    limit: 10,
                    ..Default::default()
                },
            )?;
            assert_eq!(rows.len(), 1);
            let (rows, total) = search(
                c,
                &CatalogQuery { text: "".into(), sort: CatalogSort::Downloads, limit: 10, ..Default::default() },
            )?;
            assert_eq!(total, 3);
            assert_eq!(rows[0].model_id, "llama3.2");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn fts_query_is_sanitized() {
        assert_eq!(fts_query("  "), None);
        assert_eq!(fts_query("llama \"3\" OR x").unwrap(), "\"llama\"* \"3\"* \"OR\"* \"x\"*");
    }

    #[test]
    fn http_cache_respects_age() {
        let db = Db::open_in_memory().unwrap();
        db.call_blocking(|c| {
            cache_put(c, "k", "body", "2000-01-01T00:00:00Z")?;
            assert!(cache_get(c, "k", Some(60))?.is_none());
            assert_eq!(cache_get(c, "k", None)?.unwrap().0, "body");
            cache_put(c, "k", "new", &crate::util::now_rfc3339())?;
            assert_eq!(cache_get(c, "k", Some(60))?.unwrap().0, "new");
            Ok(())
        })
        .unwrap();
    }
}
