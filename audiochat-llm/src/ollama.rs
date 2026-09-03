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

/// Trim the oldest messages so at most `max_turns` user+assistant pairs remain.
fn trim_messages(messages: &mut Vec<ChatMessage>, max_turns: usize) {
    let max_msgs = max_turns * 2;
    if messages.len() > max_msgs {
        let excess = messages.len() - max_msgs;
        messages.drain(0..excess);
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
    /// Sampling temperature (None = model default).
    temperature: Option<f32>,
    /// Context window size in tokens (None = model default).
    num_ctx: Option<u32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatOptions>,
}

/// Sampling options forwarded to `/api/chat`. Absent fields keep Ollama's defaults.
#[derive(Serialize, Default)]
struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
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
            temperature: None,
            num_ctx: None,
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

    /// Set the sampling temperature (0..2). `None` uses the model default.
    pub fn with_temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature.map(|t| t.clamp(0.0, 2.0));
        self
    }

    /// Set the context window size in tokens. `None` uses the model default.
    pub fn with_num_ctx(mut self, num_ctx: Option<u32>) -> Self {
        self.num_ctx = num_ctx;
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
            trim_messages(&mut h, self.max_turns);
        }
    }
}

impl Llm for Ollama {
    fn generate(&self, prompt: &str) -> Result<LlmResponse, Box<dyn Error + Send + Sync>> {
        // Build the request messages as a snapshot: existing history plus the
        // new user turn. We do NOT mutate the persistent history yet, so a
        // failed attempt (or a retry) never appends a duplicate user message.
        let mut messages = self.history.lock().map(|h| h.clone()).unwrap_or_default();
        messages.push(ChatMessage::user(prompt));
        trim_messages(&mut messages, self.max_turns);

        let url = format!("{}/api/chat", self.base_url);
        let body = ChatRequest {
            model: &self.model,
            messages: &messages,
            stream: true,
            system: self.system.as_deref(),
            options: if self.temperature.is_none() && self.num_ctx.is_none() {
                None
            } else {
                Some(ChatOptions {
                    temperature: self.temperature,
                    num_ctx: self.num_ctx,
                })
            },
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

        // The request was accepted: commit the user turn. The assistant turn is
        // appended lazily as the stream is read.
        if let Ok(mut h) = self.history.lock() {
            h.push(ChatMessage::user(prompt));
        }
        self.trim_history();

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
        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    // Malformed stream ended without a done flag; save what we have.
                    self.save_assistant();
                    return None;
                }
                Ok(_) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<ChatChunk>(line) {
                        Ok(chunk) => {
                            if let Some(err) = chunk.error {
                                return Some(Err(Box::<dyn Error + Send + Sync>::from(
                                    OllamaError(err),
                                )));
                            }
                            if chunk.done {
                                // Some backends emit the final content on the
                                // done chunk; flush it before saving.
                                if let Some(msg) = chunk.message {
                                    if !msg.content.trim().is_empty() {
                                        self.accum.push_str(&msg.content);
                                    }
                                }
                                self.save_assistant();
                                return None;
                            }
                            let text = chunk.message.map(|m| m.content).unwrap_or_default();
                            if text.is_empty() {
                                continue;
                            }
                            self.accum.push_str(&text);
                            return Some(Ok(text));
                        }
                        Err(e) => {
                            return Some(Err(Box::<dyn Error + Send + Sync>::from(OllamaError(
                                format!("bad stream chunk: {e}"),
                            ))));
                        }
                    }
                }
                Err(e) => {
                    return Some(Err(Box::<dyn Error + Send + Sync>::from(OllamaError(
                        format!("read stream: {e}"),
                    ))));
                }
            }
        }
    }
}

impl OllamaStream {
    /// Append the accumulated reply as an assistant turn, then clear it.
    fn save_assistant(&mut self) {
        let content = std::mem::take(&mut self.accum);
        if content.trim().is_empty() {
            return;
        }
        if let Some(history) = &self.history {
            if let Ok(mut h) = history.lock() {
                h.push(ChatMessage::assistant(content));
            }
        }
    }
}

impl Drop for OllamaStream {
    /// If the stream is dropped early (e.g. the user barged in mid-reply), still
    /// commit whatever was streamed so the conversation history stays coherent.
    fn drop(&mut self) {
        self.save_assistant();
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

    #[test]
    fn request_serializes_options_and_omits_defaults() {
        let client = Ollama::with_base("http://localhost:11434", "m")
            .with_temperature(Some(0.7))
            .with_num_ctx(Some(4096));
        let body = ChatRequest {
            model: "m",
            messages: &[],
            stream: true,
            system: None,
            options: Some(ChatOptions {
                temperature: client.temperature,
                num_ctx: client.num_ctx,
            }),
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"num_ctx\":4096"));
        assert!(!json.contains("system"));

        // No options configured -> the field is omitted entirely (mirrors the
        // real `generate`, which passes `options: None` when unset).
        let body = ChatRequest {
            model: "m",
            messages: &[],
            stream: true,
            system: None,
            options: None,
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(!json.contains("options"));
    }
}
