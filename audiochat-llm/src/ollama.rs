//! Ollama LLM backend using `/api/chat` with multi-turn conversation history.

use std::error::Error;
use std::fmt;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use audiochat_core::{Llm, LlmResponse, LlmStreamItem};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE: &str = "http://localhost:11434";
const DEFAULT_MAX_TURNS: usize = 10;

/// A message in the conversation history (role + content).
#[derive(Serialize, Clone)]
pub struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// An LLM client for a local [Ollama](https://ollama.com) server.
pub struct Ollama {
    base_url: String,
    model: String,
    client: ureq::Agent,
    history: Arc<Mutex<Vec<ChatMessage>>>,
    max_turns: usize,
    system: Option<String>,
}

/// Default system prompt: encourage short, conversational replies rather than
/// lecture-style answers, since they are spoken aloud.
const DEFAULT_SYSTEM_PROMPT: &str = "\
You are an assistant having a natural spoken conversation. \
Keep replies short, conversational, and to the point. \
Speak like a dialog partner, not a lecture. \
Avoid headings, lists, and long-winded explanations. \
Use one or two sentences unless the user explicitly asks for more detail.";

/// Request body for `/api/chat`.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
}

/// Per-chunk response line from the streaming `/api/chat` endpoint.
#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<ChatMessageOut>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChatMessageOut {
    #[serde(default)]
    content: String,
}

#[derive(Debug)]
pub struct OllamaError(String);

impl fmt::Display for OllamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for OllamaError {}

impl Ollama {
    /// Connect to Ollama on localhost with the given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_base(DEFAULT_BASE, model)
    }

    /// Connect to a custom Ollama base URL.
    pub fn with_base(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let client = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(300)))
            .timeout_recv_body(Some(Duration::from_secs(300)))
            .build()
            .new_agent();
        Self {
            base_url: base_url.into(),
            model: model.into(),
            client,
            history: Arc::new(Mutex::new(Vec::new())),
            max_turns: DEFAULT_MAX_TURNS,
            system: Some(DEFAULT_SYSTEM_PROMPT.to_string()),
        }
    }

    /// The model this client is configured for.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Override the system prompt. Pass `None` to run with no system prompt.
    pub fn with_system_prompt(mut self, prompt: Option<impl Into<String>>) -> Self {
        self.system = prompt.map(|p| p.into());
        self
    }

    /// Number of conversation turns (user+assistant pairs) retained.
    pub fn max_turns(&self) -> usize {
        self.max_turns
    }

    /// Set the maximum number of turns kept in context (oldest dropped).
    pub fn set_max_turns(&mut self, turns: usize) {
        self.max_turns = turns.max(1);
    }

    /// Clear the conversation history.
    pub fn reset_conversation(&self) {
        if let Ok(mut h) = self.history.lock() {
            h.clear();
        }
    }

    /// Number of messages currently held in context.
    pub fn history_len(&self) -> usize {
        self.history.lock().map(|h| h.len()).unwrap_or(0)
    }

    fn trim_history(&self) {
        if let Ok(mut h) = self.history.lock() {
            let max_msgs = self.max_turns * 2;
            if h.len() > max_msgs {
                let excess = h.len() - max_msgs;
                h.drain(0..excess);
            }
        }
    }
}

impl Llm for Ollama {
    fn generate(&self, prompt: &str) -> Result<LlmResponse, Box<dyn Error + Send + Sync>> {
        // Record the user turn, then hand the full history to the model.
        if let Ok(mut h) = self.history.lock() {
            h.push(ChatMessage::user(prompt));
        }
        self.trim_history();

        let url = format!("{}/api/chat", self.base_url);
        let snapshot = self.history.lock().map(|h| h.clone()).unwrap_or_default();
        let body = ChatRequest {
            model: &self.model,
            messages: &snapshot,
            stream: true,
            system: self.system.as_deref(),
        };

        let resp = self
            .client
            .post(&url)
            .send_json(body)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(OllamaError(format!("{e}"))))?;

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(Box::<dyn Error + Send + Sync>::from(OllamaError(format!(
                "ollama returned status {status}"
            ))));
        }

        let reader = BufReader::new(resp.into_body().into_reader());
        let stream = OllamaStream {
            reader,
            accum: String::new(),
            history: Some(self.history.clone()),
        };
        Ok(LlmResponse {
            stream: Some(Box::new(stream)),
            full: String::new(),
        })
    }
}

/// Lazily parses the newline-delimited Ollama `/api/chat` stream into text
/// chunks, accumulating the full assistant reply into the conversation history.
struct OllamaStream {
    reader: BufReader<ureq::BodyReader<'static>>,
    accum: String,
    history: Option<Arc<Mutex<Vec<ChatMessage>>>>,
}

impl Iterator for OllamaStream {
    type Item = LlmStreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => {
                // Malformed stream ended without a done flag; save what we have.
                self.save_assistant();
                None
            }
            Ok(_) => {
                let line = line.trim();
                if line.is_empty() {
                    return Some(Ok(String::new()));
                }
                match serde_json::from_str::<ChatChunk>(line) {
                    Ok(chunk) => {
                        if let Some(err) = chunk.error {
                            Some(Err(Box::<dyn Error + Send + Sync>::from(OllamaError(err))))
                        } else if chunk.done {
                            self.save_assistant();
                            None
                        } else {
                            let text = chunk.message.map(|m| m.content).unwrap_or_default();
                            self.accum.push_str(&text);
                            Some(Ok(text))
                        }
                    }
                    Err(e) => Some(Err(Box::<dyn Error + Send + Sync>::from(OllamaError(
                        format!("bad stream chunk: {e}"),
                    )))),
                }
            }
            Err(e) => Some(Err(Box::<dyn Error + Send + Sync>::from(OllamaError(
                format!("read stream: {e}"),
            )))),
        }
    }
}

impl OllamaStream {
    /// Append the accumulated reply as an assistant turn, then clear it.
    fn save_assistant(&mut self) {
        let content = std::mem::take(&mut self.accum);
        if content.is_empty() {
            return;
        }
        if let Some(history) = &self.history {
            if let Ok(mut h) = history.lock() {
                h.push(ChatMessage::assistant(content));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_trim_drops_oldest_turns() {
        let mut client = Ollama::with_base("http://localhost:11434", "test-model");
        client.set_max_turns(2);
        // Simulate the message growth a multi-turn conversation would produce.
        for i in 0..6 {
            {
                let mut h = client.history.lock().unwrap();
                h.push(ChatMessage::user(format!("user {i}")));
                h.push(ChatMessage::assistant(format!("assistant {i}")));
            }
            client.trim_history();
        }
        // max_turns=2 -> 4 messages retained (2 most recent pairs).
        assert_eq!(client.history_len(), 4);
        let h = client.history.lock().unwrap();
        assert_eq!(h[0].content, "user 4");
        assert_eq!(h[1].content, "assistant 4");
        assert_eq!(h[3].content, "assistant 5");
    }

    #[test]
    fn reset_conversation_clears_history() {
        let client = Ollama::with_base("http://localhost:11434", "test-model");
        {
            let mut h = client.history.lock().unwrap();
            h.push(ChatMessage::user("hi"));
        }
        assert_eq!(client.history_len(), 1);
        client.reset_conversation();
        assert_eq!(client.history_len(), 0);
    }

    #[test]
    fn set_max_turns_is_at_least_one() {
        let mut client = Ollama::with_base("http://localhost:11434", "test-model");
        client.set_max_turns(0);
        assert_eq!(client.max_turns(), 1);
    }
}
