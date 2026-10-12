//! Persistent, categorized conversation memory.
//!
//! Memory is a small JSON store kept per *profile* (a named persona). Before a
//! session it is rendered into the assistant's system prompt so the model
//! recalls earlier conversations; at the end of a session a fast, stateless
//! model rewrites it from the new transcript.
//!
//! Three categories are kept, each with its own expiry:
//!
//! - `recent` — a one-line summary per past conversation, trimmed to the newest
//!   [`MemoryLimits::recent_max`] (count-based expiry).
//! - `fundamental` — durable facts, opinions, and preferences, capped by total
//!   length ([`MemoryLimits::fundamental_max_chars`]) and otherwise permanent.
//! - `todo` — concrete follow-ups ("ask how the interview went") that expire
//!   after [`MemoryLimits::todo_ttl_days`] days or once resolved.

use std::error::Error;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::traits::{Llm, LlmResponse};

/// Instructions for the summarizer model that rewrites the memory store.
pub const MEMORY_SYSTEM_PROMPT: &str = "\
You maintain the compact long-term memory of a spoken voice assistant. \
You are given the current memory as JSON and a transcript of the latest \
conversation. Return the UPDATED memory as a single JSON object and nothing else.

Schema:
{
  \"recent\": [ { \"at\": <unix seconds>, \"summary\": \"<1-2 sentences>\" } ],
  \"fundamental\": [ { \"text\": \"<durable fact>\", \"updated_at\": <unix seconds> } ],
  \"todo\": [ { \"text\": \"<follow-up>\", \"added_at\": <unix seconds>, \"done\": false } ]
}

Rules:
- recent: add ONE short summary of the new conversation, keeping only the most \
recent entries up to the stated limit.
- fundamental: stable, durable facts only — the user's personality, opinions, \
preferences, ongoing projects, and important people. Do not record small talk \
or one-off questions. Keep the combined text under the stated character limit, \
merging or dropping entries to fit. Change these only when clearly superseded.
- todo: concrete things to follow up on or ask about next time. Drop items that \
were resolved in this conversation or are older than the stated TTL. Keep at \
most the stated number of items.
- Every entry must carry its timestamp field, set to CURRENT_TIME_UNIX.
- Output JSON only: no markdown fences, no commentary.";

/// Default number of past conversation summaries retained.
const DEFAULT_RECENT_MAX: usize = 8;
/// Default cap (characters) on the combined durable-facts text.
const DEFAULT_FUNDAMENTAL_MAX_CHARS: usize = 1200;
/// Default age (days) after which an unresolved follow-up is dropped.
const DEFAULT_TODO_TTL_DAYS: u64 = 14;
/// Default cap on the number of open follow-ups.
const DEFAULT_TODO_MAX: usize = 10;
/// Upper bound on transcript characters handed to the summarizer.
const MAX_TRANSCRIPT_CHARS: usize = 12_000;

fn default_version() -> u32 {
    1
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// The persisted memory store for one profile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Memory {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent: Vec<RecentChat>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fundamental: Vec<Fundamental>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub todo: Vec<Todo>,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            version: default_version(),
            recent: Vec::new(),
            fundamental: Vec::new(),
            todo: Vec::new(),
        }
    }
}

/// A one-line summary of a past conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecentChat {
    /// Unix seconds when the conversation finished.
    #[serde(default)]
    pub at: u64,
    #[serde(default)]
    pub summary: String,
}

/// A durable fact, opinion, or preference about the user.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fundamental {
    #[serde(default)]
    pub text: String,
    /// Unix seconds when the fact was last confirmed or changed.
    #[serde(default)]
    pub updated_at: u64,
}

/// A concrete follow-up to raise in a future conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Todo {
    #[serde(default)]
    pub text: String,
    /// Unix seconds when the follow-up was recorded.
    #[serde(default)]
    pub added_at: u64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub done: bool,
}

/// Per-category caps and expiry thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLimits {
    /// Maximum number of recent-conversation summaries retained.
    pub recent_max: usize,
    /// Maximum combined characters of durable facts.
    pub fundamental_max_chars: usize,
    /// Days after which an unresolved follow-up is dropped.
    pub todo_ttl_days: u64,
    /// Maximum number of open follow-ups retained.
    pub todo_max: usize,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            recent_max: DEFAULT_RECENT_MAX,
            fundamental_max_chars: DEFAULT_FUNDAMENTAL_MAX_CHARS,
            todo_ttl_days: DEFAULT_TODO_TTL_DAYS,
            todo_max: DEFAULT_TODO_MAX,
        }
    }
}

/// How the memory store is rendered for injection into the system prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryFormat {
    /// A compact, human-readable Markdown block (default).
    Markdown,
    /// The raw store as pretty-printed JSON.
    Json,
}

impl MemoryFormat {
    /// Parse a format name (`md`/`markdown` or `json`).
    pub fn parse(value: &str) -> Option<MemoryFormat> {
        match value.trim().to_ascii_lowercase().as_str() {
            "md" | "markdown" => Some(MemoryFormat::Markdown),
            "json" => Some(MemoryFormat::Json),
            _ => None,
        }
    }
}

/// Seconds since the Unix epoch, or 0 if the clock is unavailable.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Memory {
    /// True when every category is empty.
    pub fn is_empty(&self) -> bool {
        self.recent.is_empty() && self.fundamental.is_empty() && self.todo.is_empty()
    }

    /// Load a profile's store, returning an empty store when the file is absent.
    pub fn load(path: &Path) -> Result<Memory, String> {
        if !path.exists() {
            return Ok(Memory::default());
        }
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read memory '{}': {e}", path.display()))?;
        serde_json::from_str(&data)
            .map_err(|e| format!("failed to parse memory '{}': {e}", path.display()))
    }

    /// Write the store to `path`, creating the parent directory if needed.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize memory: {e}"))?;
        std::fs::write(path, format!("{json}\n"))
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    /// Drop blank entries and enforce every per-category limit and expiry.
    pub fn sanitize(&mut self, limits: &MemoryLimits, now: u64) {
        self.recent.retain(|r| !r.summary.trim().is_empty());
        self.fundamental.retain(|f| !f.text.trim().is_empty());
        self.todo.retain(|t| !t.text.trim().is_empty());

        if self.recent.len() > limits.recent_max {
            let excess = self.recent.len() - limits.recent_max;
            self.recent.drain(0..excess);
        }

        let ttl_secs = limits.todo_ttl_days.saturating_mul(86_400);
        self.todo
            .retain(|t| !t.done && (t.added_at == 0 || now.saturating_sub(t.added_at) <= ttl_secs));
        if self.todo.len() > limits.todo_max {
            let excess = self.todo.len() - limits.todo_max;
            self.todo.drain(0..excess);
        }

        // Enforce the durable-facts cap by dropping the oldest-updated entries,
        // then truncating if a single entry still exceeds the cap.
        let char_count = |f: &Fundamental| f.text.chars().count();
        let mut total: usize = self.fundamental.iter().map(char_count).sum();
        while total > limits.fundamental_max_chars && !self.fundamental.is_empty() {
            let oldest = self
                .fundamental
                .iter()
                .enumerate()
                .min_by_key(|(_, f)| f.updated_at)
                .map(|(i, _)| i)
                .unwrap_or(0);
            total -= char_count(&self.fundamental[oldest]);
            self.fundamental.remove(oldest);
        }
        if total > limits.fundamental_max_chars {
            if let Some(f) = self.fundamental.first_mut() {
                f.text = f.text.chars().take(limits.fundamental_max_chars).collect();
            }
        }
    }

    /// Render the store for injection into the system prompt. Returns an empty
    /// string when there is nothing to inject.
    pub fn render(&self, format: MemoryFormat, now: u64) -> String {
        match format {
            MemoryFormat::Json => {
                if self.is_empty() {
                    String::new()
                } else {
                    serde_json::to_string_pretty(self).unwrap_or_default()
                }
            }
            MemoryFormat::Markdown => self.render_markdown(now),
        }
    }

    fn render_markdown(&self, now: u64) -> String {
        if self.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "## Persistent memory (from earlier conversations)\n\
             Use this to stay consistent with the user; do not read it aloud or \
             mention it unless it is directly relevant.\n",
        );
        if !self.fundamental.is_empty() {
            out.push_str("\n### Stable facts & preferences\n");
            for f in &self.fundamental {
                out.push_str(&format!("- {}\n", f.text.trim()));
            }
        }
        if !self.todo.is_empty() {
            out.push_str("\n### Follow-ups (may be stale)\n");
            for t in &self.todo {
                out.push_str(&format!(
                    "- {} ({})\n",
                    t.text.trim(),
                    age_phrase(t.added_at, now)
                ));
            }
        }
        if !self.recent.is_empty() {
            out.push_str("\n### Recent conversations\n");
            for r in &self.recent {
                out.push_str(&format!(
                    "- {}: {}\n",
                    age_phrase(r.at, now),
                    r.summary.trim()
                ));
            }
        }
        out
    }

    /// Rewrite the store from `transcript` using `llm`. On any failure the
    /// caller should keep the existing store; this returns the new one.
    pub fn update(
        &self,
        llm: &dyn Llm,
        transcript: &str,
        limits: &MemoryLimits,
        now: u64,
    ) -> Result<Memory, Box<dyn Error + Send + Sync>> {
        let existing = serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string());
        let user = format!(
            "CURRENT_TIME_UNIX: {now}\n\
             Limits: recent_max={}, fundamental_max_chars={}, todo_ttl_days={}, todo_max={}.\n\n\
             CURRENT_MEMORY_JSON:\n{existing}\n\n\
             NEW_CONVERSATION_TRANSCRIPT:\n{transcript}\n\n\
             Return only the updated memory JSON.",
            limits.recent_max, limits.fundamental_max_chars, limits.todo_ttl_days, limits.todo_max,
        );
        let resp = llm.generate(&user)?;
        let text = collect_response(resp)?;
        let mut updated =
            parse_memory(&text).ok_or("summarizer did not return valid memory JSON")?;
        updated.sanitize(limits, now);
        Ok(updated)
    }
}

/// Read a full (possibly streamed) LLM response into a single string.
fn collect_response(mut resp: LlmResponse) -> Result<String, Box<dyn Error + Send + Sync>> {
    if let Some(stream) = resp.stream.take() {
        let mut out = String::new();
        for item in stream {
            out.push_str(&item?);
        }
        if !out.trim().is_empty() {
            return Ok(out);
        }
    }
    Ok(resp.full)
}

/// Extract the outermost JSON object from `text` and parse it as a [`Memory`].
/// Tolerates surrounding prose and Markdown code fences.
fn parse_memory(text: &str) -> Option<Memory> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    serde_json::from_str(&text[start..=end]).ok()
}

/// Truncate a transcript to the summarizer's input budget, keeping the tail
/// (the most recent turns matter most).
pub fn trim_transcript(transcript: &str) -> String {
    if transcript.chars().count() <= MAX_TRANSCRIPT_CHARS {
        return transcript.to_string();
    }
    let skip = transcript.chars().count() - MAX_TRANSCRIPT_CHARS;
    transcript.chars().skip(skip).collect()
}

/// A human phrase for how long ago `then` was, relative to `now`.
fn age_phrase(then: u64, now: u64) -> String {
    if then == 0 || now <= then {
        return "recently".to_string();
    }
    let days = now.saturating_sub(then) / 86_400;
    match days {
        0 => "today".to_string(),
        1 => "yesterday".to_string(),
        n => format!("{n} days ago"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(recent: &[&str], facts: &[&str], todos: &[&str]) -> Memory {
        Memory {
            version: 1,
            recent: recent
                .iter()
                .enumerate()
                .map(|(i, s)| RecentChat {
                    at: 1_000 + i as u64,
                    summary: (*s).to_string(),
                })
                .collect(),
            fundamental: facts
                .iter()
                .map(|s| Fundamental {
                    text: (*s).to_string(),
                    updated_at: 1_000,
                })
                .collect(),
            todo: todos
                .iter()
                .map(|s| Todo {
                    text: (*s).to_string(),
                    added_at: 1_000,
                    done: false,
                })
                .collect(),
        }
    }

    #[test]
    fn trims_recent_to_newest() {
        let mut m = mem(&["a", "b", "c", "d"], &[], &[]);
        m.sanitize(
            &MemoryLimits {
                recent_max: 2,
                ..Default::default()
            },
            2_000,
        );
        assert_eq!(m.recent.len(), 2);
        assert_eq!(m.recent[0].summary, "c");
        assert_eq!(m.recent[1].summary, "d");
    }

    #[test]
    fn drops_expired_and_done_todos() {
        let now = 1_000 + 100 * 86_400;
        let mut m = mem(&[], &[], &["open", "old", "finished"]);
        m.todo[0].added_at = now; // fresh
        m.todo[1].added_at = 1_000; // 100 days before `now`
        m.todo[2].done = true;
        m.sanitize(
            &MemoryLimits {
                todo_ttl_days: 14,
                ..Default::default()
            },
            now,
        );
        assert_eq!(m.todo.len(), 1);
        assert_eq!(m.todo[0].text, "open");
    }

    #[test]
    fn caps_fundamental_text_length() {
        let mut m = mem(&[], &["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"], &[]);
        m.sanitize(
            &MemoryLimits {
                fundamental_max_chars: 15,
                ..Default::default()
            },
            2_000,
        );
        let total: usize = m.fundamental.iter().map(|f| f.text.chars().count()).sum();
        assert!(total <= 15);
        assert!(!m.fundamental.is_empty());
    }

    #[test]
    fn renders_markdown_sections() {
        let m = mem(
            &["talked about Rust"],
            &["prefers metric"],
            &["ask about the trip"],
        );
        let md = m.render(MemoryFormat::Markdown, 2_000);
        assert!(md.contains("Stable facts & preferences"));
        assert!(md.contains("- prefers metric"));
        assert!(md.contains("Follow-ups"));
        assert!(md.contains("Recent conversations"));
        assert!(md.contains("talked about Rust"));
    }

    #[test]
    fn empty_render_is_blank() {
        let m = Memory::default();
        assert!(m.render(MemoryFormat::Markdown, 0).is_empty());
        assert!(m.render(MemoryFormat::Json, 0).is_empty());
    }

    #[test]
    fn round_trips_through_disk() {
        let path = std::env::temp_dir().join(format!(
            "audiochat-mem-{}-{}.json",
            std::process::id(),
            now_unix()
        ));
        let m = mem(&["s"], &["f"], &["t"]);
        m.save(&path).unwrap();
        let back = Memory::load(&path).unwrap();
        assert_eq!(m, back);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_yields_empty_store() {
        let path = std::env::temp_dir().join("audiochat-mem-does-not-exist.json");
        let _ = std::fs::remove_file(&path);
        assert!(Memory::load(&path).unwrap().is_empty());
    }

    #[test]
    fn parses_json_embedded_in_prose_and_fences() {
        let text = "Sure!\n```json\n{\"recent\":[{\"at\":1,\"summary\":\"hi\"}]}\n```";
        let m = parse_memory(text).unwrap();
        assert_eq!(m.recent.len(), 1);
        assert_eq!(m.recent[0].summary, "hi");
    }

    #[test]
    fn rejects_non_json() {
        assert!(parse_memory("no object here").is_none());
    }

    #[test]
    fn trims_long_transcripts_keeping_the_tail() {
        let long = "x".repeat(MAX_TRANSCRIPT_CHARS + 50);
        let trimmed = trim_transcript(&long);
        assert_eq!(trimmed.chars().count(), MAX_TRANSCRIPT_CHARS);

        let short = "hello";
        assert_eq!(trim_transcript(short), "hello");
    }

    #[test]
    fn update_uses_summarizer_output_and_sanitizes() {
        struct FakeLlm;
        impl Llm for FakeLlm {
            fn generate(&self, _prompt: &str) -> Result<LlmResponse, Box<dyn Error + Send + Sync>> {
                let json =
                    r#"{"recent":[{"at":2000,"summary":"new chat"}],"fundamental":[],"todo":[]}"#;
                Ok(LlmResponse {
                    stream: None,
                    full: format!("```json\n{json}\n```"),
                })
            }
            fn model(&self) -> &str {
                "fake"
            }
        }

        let m = Memory::default();
        let updated = m
            .update(&FakeLlm, "user: hi", &MemoryLimits::default(), 2_000)
            .unwrap();
        assert_eq!(updated.recent.len(), 1);
        assert_eq!(updated.recent[0].summary, "new chat");
    }
}
