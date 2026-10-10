//! OpenAI-compatible LLM backend using the streaming `/chat/completions` endpoint.
//!
//! Works with any OpenAI-compatible server (OpenCode Go, Ollama's `/v1`, vLLM,
//! etc.). Replies stream token-by-token and are kept in a bounded multi-turn
//! conversation history, mirroring the [`crate::Ollama`] backend.
//!
//! Reasoning models (e.g. DeepSeek V4.1 Flash) think by default; thinking can be
//! disabled per request with `{"thinking": {"type": "disabled"}}`.

use std::error::Error;
use std::fmt;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use audiochat_core::{Llm, LlmResponse, LlmStreamItem};
use serde::{Deserialize, Serialize};

use crate::chat::{build_request_messages, trim_messages, ChatMessage, DEFAULT_SYSTEM_PROMPT};

/// Default base URL: OpenCode Go's OpenAI-compatible API.
const DEFAULT_BASE: &str = "https://opencode.ai/zen/go/v1";
const DEFAULT_MAX_TURNS: usize = 10;
/// Identify ourselves as a coding-agent client rather than a generic HTTP library,
/// as OpenCode Go asks of its clients.
const DEFAULT_USER_AGENT: &str = "audiochat/0.1";

/// An LLM client for an OpenAI-compatible `/chat/completions` server.
pub struct OpenAiCompatible {
    base_url: String,
    model: String,
    client: ureq::Agent,
    api_key: Option<String>,
    user_agent: String,
    /// Stable per-conversation id sent as `x-opencode-session`.
    session_id: String,
    history: Arc<Mutex<Vec<ChatMessage>>>,
    max_turns: usize,
    system: Option<String>,
    /// Sampling temperature (None = model default).
    temperature: Option<f32>,
    /// Thinking mode: `Some(false)` disables, `Some(true)` enables, `None` omits
    /// the field and leaves the server/model default.
    thinking: Option<bool>,
}

/// Request body for `/chat/completions`.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Thinking>,
}

/// DeepSeek-style thinking toggle, sent as `{"thinking": {"type": "..."}}`.
#[derive(Serialize)]
struct Thinking {
    #[serde(rename = "type")]
    kind: &'static str,
}

impl Thinking {
    fn disabled() -> Self {
        Self { kind: "disabled" }
    }
    fn enabled() -> Self {
        Self { kind: "enabled" }
    }
}

/// A streamed SSE chunk from `/chat/completions`.
#[derive(Deserialize, Default)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    error: Option<ApiError>,
}

#[derive(Deserialize, Default)]
struct Choice {
    #[serde(default)]
    delta: Option<Delta>,
    #[serde(default)]
    message: Option<Delta>,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize, Default)]
struct ApiError {
    #[serde(default)]
    message: String,
}

#[derive(Debug)]
pub struct OpenAiError(String);

impl fmt::Display for OpenAiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for OpenAiError {}

impl OpenAiCompatible {
    /// Connect to OpenCode Go with the given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_base(DEFAULT_BASE, model)
    }

    /// Connect to a custom OpenAI-compatible base URL (e.g. `.../v1`).
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
            api_key: None,
            user_agent: DEFAULT_USER_AGENT.to_string(),
            session_id: default_session_id(),
            history: Arc::new(Mutex::new(Vec::new())),
            max_turns: DEFAULT_MAX_TURNS,
            system: Some(DEFAULT_SYSTEM_PROMPT.to_string()),
            temperature: None,
            thinking: None,
        }
    }

    /// The model this client is configured for.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Set the bearer token sent in the `Authorization` header.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        let key = api_key.into();
        if !key.trim().is_empty() {
            self.api_key = Some(key);
        }
        self
    }

    /// Override the `User-Agent` header.
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Set the stable session id sent as `x-opencode-session`.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
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

    /// Configure thinking mode: `Some(false)` disables it, `Some(true)` enables
    /// it, `None` omits the field (server/model default).
    pub fn with_thinking(mut self, thinking: Option<bool>) -> Self {
        self.thinking = thinking;
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

impl Llm for OpenAiCompatible {
    fn generate(&self, prompt: &str) -> Result<LlmResponse, Box<dyn Error + Send + Sync>> {
        // Snapshot the history plus the new user turn; the persistent history is
        // only committed once the request is accepted, so retries don't duplicate.
        let messages = {
            let history = self.history.lock().map(|h| h.clone()).unwrap_or_default();
            build_request_messages(&history, prompt, self.system.as_deref(), self.max_turns)
        };

        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatRequest {
            model: &self.model,
            messages: &messages,
            stream: true,
            temperature: self.temperature,
            thinking: self.thinking.map(|enabled| {
                if enabled {
                    Thinking::enabled()
                } else {
                    Thinking::disabled()
                }
            }),
        };

        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("User-Agent", &self.user_agent)
            .header("x-opencode-session", &self.session_id);
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", &format!("Bearer {key}"));
        }

        let resp = req
            .send_json(body)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(OpenAiError(format!("{e}"))))?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp.into_body().read_to_string().unwrap_or_default();
            let detail = detail.trim();
            let detail = if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            };
            return Err(Box::<dyn Error + Send + Sync>::from(OpenAiError(format!(
                "openai-compatible server returned status {status}{detail}"
            ))));
        }

        // Accepted: commit the user turn. The assistant turn is appended as the
        // stream is read (or on drop if interrupted).
        if let Ok(mut h) = self.history.lock() {
            h.push(ChatMessage::user(prompt));
        }
        self.trim_history();

        let reader = BufReader::new(resp.into_body().into_reader());
        let stream = OpenAiStream {
            reader,
            accum: String::new(),
            history: Some(self.history.clone()),
        };
        Ok(LlmResponse {
            stream: Some(Box::new(stream)),
            full: String::new(),
        })
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Parse the SSE `/chat/completions` stream, accumulating the reply into history.
struct OpenAiStream {
    reader: BufReader<ureq::BodyReader<'static>>,
    accum: String,
    history: Option<Arc<Mutex<Vec<ChatMessage>>>>,
}

impl Iterator for OpenAiStream {
    type Item = LlmStreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    self.save_assistant();
                    return None;
                }
                Ok(_) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let payload = match line.strip_prefix("data:") {
                        Some(rest) => rest.trim(),
                        // Some servers return a plain JSON object (no SSE framing).
                        None if line.starts_with('{') => line,
                        None => continue,
                    };
                    if payload == "[DONE]" {
                        self.save_assistant();
                        return None;
                    }
                    match serde_json::from_str::<StreamChunk>(payload) {
                        Ok(chunk) => {
                            if let Some(err) = chunk.error {
                                return Some(Err(Box::<dyn Error + Send + Sync>::from(
                                    OpenAiError(format!("stream error: {}", err.message)),
                                )));
                            }
                            let text = chunk
                                .choices
                                .into_iter()
                                .next()
                                .and_then(|c| c.delta.or(c.message))
                                .and_then(|d| d.content)
                                .unwrap_or_default();
                            if text.is_empty() {
                                continue;
                            }
                            self.accum.push_str(&text);
                            return Some(Ok(text));
                        }
                        Err(e) => {
                            return Some(Err(Box::<dyn Error + Send + Sync>::from(OpenAiError(
                                format!("bad stream chunk: {e}"),
                            ))));
                        }
                    }
                }
                Err(e) => {
                    return Some(Err(Box::<dyn Error + Send + Sync>::from(OpenAiError(
                        format!("read stream: {e}"),
                    ))));
                }
            }
        }
    }
}

impl OpenAiStream {
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

impl Drop for OpenAiStream {
    /// If the stream is dropped early (e.g. barge-in), still commit what streamed
    /// so the conversation history stays coherent.
    fn drop(&mut self) {
        self.save_assistant();
    }
}

/// A stable-ish per-process session id for the `x-opencode-session` header.
fn default_session_id() -> String {
    format!("audiochat-{}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_disables_thinking_by_default_when_configured() {
        let body = ChatRequest {
            model: "deepseek-v4.1-flash",
            messages: &[],
            stream: true,
            temperature: None,
            thinking: Some(Thinking::disabled()),
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"thinking\":{\"type\":\"disabled\"}"));
        assert!(!json.contains("temperature"));
    }

    #[test]
    fn request_omits_thinking_when_unset() {
        let body = ChatRequest {
            model: "m",
            messages: &[],
            stream: true,
            temperature: Some(0.5),
            thinking: None,
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(!json.contains("thinking"));
        assert!(json.contains("\"temperature\":0.5"));
    }

    #[test]
    fn parses_sse_content_and_ignores_reasoning() {
        let chunk = r#"{"choices":[{"delta":{"reasoning_content":"hmm","content":"Hi"}}]}"#;
        let parsed: StreamChunk = serde_json::from_str(chunk).unwrap();
        let text = parsed
            .choices
            .into_iter()
            .next()
            .unwrap()
            .delta
            .unwrap()
            .content;
        assert_eq!(text.as_deref(), Some("Hi"));

        // A reasoning-only chunk yields no content.
        let chunk = r#"{"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#;
        let parsed: StreamChunk = serde_json::from_str(chunk).unwrap();
        let text = parsed
            .choices
            .into_iter()
            .next()
            .unwrap()
            .delta
            .unwrap()
            .content;
        assert_eq!(text, None);
    }

    #[test]
    fn history_trims_oldest_turns() {
        let mut client = OpenAiCompatible::with_base("http://localhost/v1", "m");
        for _ in 0..3 {
            if let Ok(mut h) = client.history.lock() {
                h.push(ChatMessage::user("u"));
                h.push(ChatMessage::assistant("a"));
            }
        }
        client.set_max_turns(1);
        client.trim_history();
        assert_eq!(client.history_len(), 2);
    }
}
