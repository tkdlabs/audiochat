//! Pluggable LLM clients (Ollama, OpenAI-compatible, ...).

mod ollama;

pub use ollama::Ollama;
