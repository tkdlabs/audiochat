//! Structured conversation logging.
//!
//! Each completed turn is appended as a single-line JSON object (JSONL) to a
//! log file, so transcripts can be replayed, searched, or reviewed after a
//! session. Turns are only logged once the reply has been received.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// A single completed conversation turn, serialized to JSON.
#[derive(Serialize)]
struct Turn<'a> {
    /// Unix timestamp (seconds) when the turn completed.
    timestamp: u64,
    /// Monotonic turn number within this session.
    turn: u64,
    /// The LLM model that produced the reply.
    model: &'a str,
    /// The user's transcribed question.
    user: &'a str,
    /// The assistant's full reply.
    assistant: &'a str,
    /// Timing breakdown in milliseconds (0 when unavailable).
    timing: Timing,
}

/// Per-turn latency, in milliseconds, included in each JSON record.
#[derive(Serialize)]
pub struct Timing {
    pub stt_ms: u64,
    pub first_token_ms: u64,
    pub first_audio_ms: u64,
    pub stream_ms: u64,
    pub total_ms: u64,
}

/// Appends completed turns to a JSONL file. Cheap to create and discard; each
/// turn is written and flushed on the calling (main) thread.
pub struct ConversationLogger {
    writer: BufWriter<File>,
    turn: u64,
}

impl ConversationLogger {
    /// Open a logger appending to `path` (parent directories must exist).
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::options().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            turn: 0,
        })
    }

    /// Append a completed turn to the log.
    pub fn log(&mut self, model: &str, user: &str, assistant: &str, timing: Timing) {
        let turn = Turn {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            turn: self.turn,
            model,
            user,
            assistant,
            timing,
        };
        match serde_json::to_string(&turn) {
            Ok(line) => {
                let _ = writeln!(self.writer, "{line}");
                let _ = self.writer.flush();
            }
            Err(e) => eprintln!("audiochat: failed to serialize conversation log: {e}"),
        }
        self.turn += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_jsonl_records() {
        let path =
            std::env::temp_dir().join(format!("audiochat-test-{}.jsonl", std::process::id()));
        let mut logger = ConversationLogger::open(&path).unwrap();
        let timing = Timing {
            stt_ms: 90,
            first_token_ms: 120,
            first_audio_ms: 250,
            stream_ms: 400,
            total_ms: 900,
        };
        logger.log("gemma4:e2b", "Hello", "Hi there!", timing);
        drop(logger);

        let lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(lines.len(), 1);
        let v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(v["model"], "gemma4:e2b");
        assert_eq!(v["user"], "Hello");
        assert_eq!(v["assistant"], "Hi there!");
        assert_eq!(v["timing"]["stt_ms"], 90);
        assert_eq!(v["turn"], 0);
        let _ = std::fs::remove_file(&path);
    }
}
