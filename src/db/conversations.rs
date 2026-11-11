//! Conversations and messages.

use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};

use super::{from_json, to_json};
use crate::inference::params::GenParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    /// `ollama:<name>` or `local:<model id>`; None until a model is picked.
    pub model_ref: Option<String>,
    pub system_prompt: String,
    pub params: GenParams,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub model_ref: Option<String>,
    pub updated_at: String,
    pub message_count: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
    pub fn parse(s: &str) -> Role {
        match s {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            _ => Role::User,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageStatus {
    Done,
    Generating,
    Stopped,
    Error,
}

impl MessageStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            MessageStatus::Done => "done",
            MessageStatus::Generating => "generating",
            MessageStatus::Stopped => "stopped",
            MessageStatus::Error => "error",
        }
    }
    pub fn parse(s: &str) -> MessageStatus {
        match s {
            "generating" => MessageStatus::Generating,
            "stopped" => MessageStatus::Stopped,
            "error" => MessageStatus::Error,
            _ => MessageStatus::Done,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AttachmentKind {
    #[default]
    Image,
    Audio,
    Video,
}

/// A file attached to a user message. `file` is relative to the uploads directory.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Attachment {
    pub kind: AttachmentKind,
    pub file: String,
    pub name: String,
    pub mime: String,
    pub size: u64,
    /// For videos: extracted frame image files (relative to the uploads directory).
    #[serde(default)]
    pub frames: Vec<String>,
    /// For audio: the transcript produced by the selected speech-to-text model.
    #[serde(default)]
    pub transcript: Option<String>,
    /// Video duration in seconds, when known.
    #[serde(default)]
    pub duration_secs: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub seq: i64,
    pub role: Role,
    pub content: String,
    pub attachments: Vec<Attachment>,
    pub model_ref: Option<String>,
    pub status: MessageStatus,
    pub error: Option<String>,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub duration_ms: Option<i64>,
    /// Generation speed reported by the engine (tokens per second, without loading time).
    #[serde(default)]
    pub tokens_per_sec: Option<f64>,
    pub created_at: String,
}

fn conversation_from_row(r: &Row<'_>) -> rusqlite::Result<Conversation> {
    let params_json: String = r.get("params_json")?;
    Ok(Conversation {
        id: r.get("id")?,
        title: r.get("title")?,
        model_ref: r.get("model_ref")?,
        system_prompt: r.get("system_prompt")?,
        params: from_json(&params_json),
        created_at: r.get("created_at")?,
        updated_at: r.get("updated_at")?,
    })
}

fn message_from_row(r: &Row<'_>) -> rusqlite::Result<Message> {
    let role: String = r.get("role")?;
    let status: String = r.get("status")?;
    let attachments: String = r.get("attachments_json")?;
    Ok(Message {
        id: r.get("id")?,
        conversation_id: r.get("conversation_id")?,
        seq: r.get("seq")?,
        role: Role::parse(&role),
        content: r.get("content")?,
        attachments: from_json(&attachments),
        model_ref: r.get("model_ref")?,
        status: MessageStatus::parse(&status),
        error: r.get("error")?,
        prompt_tokens: r.get("prompt_tokens")?,
        completion_tokens: r.get("completion_tokens")?,
        duration_ms: r.get("duration_ms")?,
        tokens_per_sec: r.get("tokens_per_sec")?,
        created_at: r.get("created_at")?,
    })
}

pub fn insert_conversation(c: &Connection, conv: &Conversation) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO conversations (id, title, model_ref, system_prompt, params_json, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            conv.id,
            conv.title,
            conv.model_ref,
            conv.system_prompt,
            to_json(&conv.params),
            conv.created_at,
            conv.updated_at
        ],
    )?;
    Ok(())
}

pub fn get_conversation(c: &Connection, id: &str) -> rusqlite::Result<Option<Conversation>> {
    c.query_row("SELECT * FROM conversations WHERE id = ?1", params![id], conversation_from_row).optional()
}

pub fn list_conversations(c: &Connection, limit: usize) -> rusqlite::Result<Vec<ConversationSummary>> {
    let mut stmt = c.prepare(
        "SELECT c.id, c.title, c.model_ref, c.updated_at,
                (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS n
         FROM conversations c ORDER BY c.updated_at DESC, c.created_at DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |r| {
        Ok(ConversationSummary {
            id: r.get(0)?,
            title: r.get(1)?,
            model_ref: r.get(2)?,
            updated_at: r.get(3)?,
            message_count: r.get(4)?,
        })
    })?;
    rows.collect()
}

pub fn latest_conversation_id(c: &Connection) -> rusqlite::Result<Option<String>> {
    c.query_row("SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1", [], |r| r.get(0)).optional()
}

/// Names a conversation after its first message, unless someone already renamed it.
pub fn title_if_untitled(c: &Connection, id: &str, title: &str) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE conversations SET title = ?2 WHERE id = ?1 AND title = 'New chat'
           AND (SELECT COUNT(*) FROM messages WHERE conversation_id = ?1 AND role = 'user') <= 1",
        params![id, title],
    )
}

pub fn rename_conversation(c: &Connection, id: &str, title: &str, now: &str) -> rusqlite::Result<usize> {
    c.execute("UPDATE conversations SET title = ?2, updated_at = ?3 WHERE id = ?1", params![id, title, now])
}

pub fn update_conversation_settings(
    c: &Connection,
    id: &str,
    model_ref: Option<&str>,
    system_prompt: &str,
    params: &GenParams,
    now: &str,
) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE conversations SET model_ref = ?2, system_prompt = ?3, params_json = ?4, updated_at = ?5 WHERE id = ?1",
        params![id, model_ref, system_prompt, to_json(params), now],
    )
}

pub fn touch_conversation(c: &Connection, id: &str, now: &str) -> rusqlite::Result<usize> {
    c.execute("UPDATE conversations SET updated_at = ?2 WHERE id = ?1", params![id, now])
}

pub fn delete_conversation(c: &Connection, id: &str) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM conversations WHERE id = ?1", params![id])
}

pub fn delete_all_conversations(c: &Connection) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM conversations", [])
}

pub fn messages(c: &Connection, conversation_id: &str) -> rusqlite::Result<Vec<Message>> {
    let mut stmt = c.prepare("SELECT * FROM messages WHERE conversation_id = ?1 ORDER BY seq ASC")?;
    let rows = stmt.query_map(params![conversation_id], message_from_row)?;
    rows.collect()
}

pub fn get_message(c: &Connection, id: &str) -> rusqlite::Result<Option<Message>> {
    c.query_row("SELECT * FROM messages WHERE id = ?1", params![id], message_from_row).optional()
}

/// Inserts a message at the end of the conversation (assigns `seq`) and returns it.
pub fn append_message(c: &Connection, mut msg: Message) -> rusqlite::Result<Message> {
    let next: i64 = c.query_row(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?1",
        params![msg.conversation_id],
        |r| r.get(0),
    )?;
    msg.seq = next;
    c.execute(
        "INSERT INTO messages (id, conversation_id, seq, role, content, attachments_json, model_ref, status, error,
                               prompt_tokens, completion_tokens, duration_ms, tokens_per_sec, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            msg.id,
            msg.conversation_id,
            msg.seq,
            msg.role.as_str(),
            msg.content,
            to_json(&msg.attachments),
            msg.model_ref,
            msg.status.as_str(),
            msg.error,
            msg.prompt_tokens,
            msg.completion_tokens,
            msg.duration_ms,
            msg.tokens_per_sec,
            msg.created_at
        ],
    )?;
    Ok(msg)
}

/// Final state of an assistant message after generation.
#[derive(Clone, Debug, Default)]
pub struct MessageCompletion {
    pub content: String,
    pub status: Option<MessageStatus>,
    pub error: Option<String>,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub duration_ms: Option<i64>,
    pub tokens_per_sec: Option<f64>,
}

pub fn complete_message(c: &Connection, id: &str, done: &MessageCompletion) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE messages SET content = ?2, status = ?3, error = ?4, prompt_tokens = ?5,
                completion_tokens = ?6, duration_ms = ?7, tokens_per_sec = ?8 WHERE id = ?1",
        params![
            id,
            done.content,
            done.status.unwrap_or(MessageStatus::Done).as_str(),
            done.error,
            done.prompt_tokens,
            done.completion_tokens,
            done.duration_ms,
            done.tokens_per_sec.filter(|v| v.is_finite())
        ],
    )
}

pub fn update_message_content(c: &Connection, id: &str, content: &str) -> rusqlite::Result<usize> {
    c.execute("UPDATE messages SET content = ?2 WHERE id = ?1", params![id, content])
}

/// Deletes all messages of a conversation with `seq >= from_seq` (used by edit/regenerate).
pub fn delete_messages_from(c: &Connection, conversation_id: &str, from_seq: i64) -> rusqlite::Result<usize> {
    c.execute("DELETE FROM messages WHERE conversation_id = ?1 AND seq >= ?2", params![conversation_id, from_seq])
}

/// Marks messages left in `generating` state (e.g. after a crash) as stopped.
pub fn recover_interrupted(c: &Connection) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE messages SET status = 'stopped', error = COALESCE(error, 'Interrupted: the app was closed while this response was being written.') WHERE status = 'generating'",
        [],
    )
}

/// All upload file names referenced by any message (for cache clean-up safety).
pub fn referenced_uploads(c: &Connection) -> rusqlite::Result<Vec<String>> {
    let mut stmt = c.prepare("SELECT attachments_json FROM messages WHERE attachments_json != '[]'")?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
    let mut out = Vec::new();
    for json in rows {
        let atts: Vec<Attachment> = from_json(&json?);
        for a in atts {
            out.push(a.file);
            out.extend(a.frames);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;

    fn conv(id: &str) -> Conversation {
        Conversation {
            id: id.into(),
            title: "Test".into(),
            model_ref: Some("ollama:llama3.2".into()),
            system_prompt: "Be brief.".into(),
            params: GenParams::default(),
            created_at: crate::util::now_rfc3339(),
            updated_at: crate::util::now_rfc3339(),
        }
    }

    fn msg(conv: &str, role: Role, content: &str) -> Message {
        Message {
            id: crate::util::new_id(),
            conversation_id: conv.into(),
            seq: 0,
            role,
            content: content.into(),
            attachments: vec![],
            model_ref: None,
            status: MessageStatus::Done,
            error: None,
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: None,
            tokens_per_sec: None,
            created_at: crate::util::now_rfc3339(),
        }
    }

    #[test]
    fn conversation_lifecycle() {
        let db = Db::open_in_memory().unwrap();
        db.call_blocking(|c| {
            insert_conversation(c, &conv("c1"))?;
            let a = append_message(c, msg("c1", Role::User, "hi"))?;
            let b = append_message(c, msg("c1", Role::Assistant, "hello"))?;
            assert_eq!(a.seq, 1);
            assert_eq!(b.seq, 2);
            let list = list_conversations(c, 10)?;
            assert_eq!(list.len(), 1);
            assert_eq!(list[0].message_count, 2);
            delete_messages_from(c, "c1", 2)?;
            assert_eq!(messages(c, "c1")?.len(), 1);
            rename_conversation(c, "c1", "Renamed", "2026-01-01T00:00:00Z")?;
            assert_eq!(get_conversation(c, "c1")?.unwrap().title, "Renamed");
            delete_conversation(c, "c1")?;
            // cascade removed the messages too
            assert!(messages(c, "c1")?.is_empty());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn interrupted_generations_are_recovered() {
        let db = Db::open_in_memory().unwrap();
        db.call_blocking(|c| {
            insert_conversation(c, &conv("c1"))?;
            let mut m = msg("c1", Role::Assistant, "partial");
            m.status = MessageStatus::Generating;
            let m = append_message(c, m)?;
            assert_eq!(recover_interrupted(c)?, 1);
            let m = get_message(c, &m.id)?.unwrap();
            assert_eq!(m.status, MessageStatus::Stopped);
            assert!(m.error.is_some());
            Ok(())
        })
        .unwrap();
    }
}
