//! Ollama LLM backend using the `/api/generate` streaming endpoint.

use std::error::Error;
use std::fmt;
use std::io::{BufRead, BufReader};
use std::time::Duration;

use audiochat_core::{Llm, LlmResponse, LlmStreamItem};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE: &str = "http://localhost:11434";

/// An LLM client for a local [Ollama](https://ollama.com) server.
pub struct Ollama {
    base_url: String,
    model: String,
    client: ureq::Agent,
}

/// Request body for `/api/generate`.
#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
}

/// Per-chunk response line from the streaming `/api/generate` endpoint.
#[derive(Deserialize)]
struct GenerateChunk {
    #[serde(default)]
    response: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
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
        }
    }

    /// The model this client is configured for.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl Llm for Ollama {
    fn generate(&self, prompt: &str) -> Result<LlmResponse, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/api/generate", self.base_url);
        let body = GenerateRequest {
            model: &self.model,
            prompt,
            stream: true,
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
        let stream = OllamaStream { reader };
        Ok(LlmResponse {
            stream: Some(Box::new(stream)),
            full: String::new(),
        })
    }
}

/// Lazily parses the newline-delimited Ollama stream into text chunks.
struct OllamaStream {
    reader: BufReader<ureq::BodyReader<'static>>,
}

impl Iterator for OllamaStream {
    type Item = LlmStreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => {
                let line = line.trim();
                if line.is_empty() {
                    return Some(Ok(String::new()));
                }
                match serde_json::from_str::<GenerateChunk>(line) {
                    Ok(chunk) => {
                        if let Some(err) = chunk.error {
                            Some(Err(Box::<dyn Error + Send + Sync>::from(OllamaError(err))))
                        } else if chunk.done {
                            None
                        } else {
                            Some(Ok(chunk.response))
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
