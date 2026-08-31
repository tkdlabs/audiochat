//! Pluggable pipeline traits.

use std::error::Error;

/// A chunk of a streamed LLM reply.
pub type LlmStreamItem = Result<String, Box<dyn Error + Send + Sync>>;

/// An iterator of streamed reply chunks.
pub type LlmStream = Box<dyn Iterator<Item = LlmStreamItem> + Send>;

/// A speech-to-text engine. Implementations are pluggable (e.g. whisper.cpp).
pub trait SpeechRecognizer {
    /// Transcribe a complete audio utterance (PCM 16 kHz mono) to text.
    fn transcribe(&mut self, pcm: &[i16]) -> Result<String, Box<dyn Error + Send + Sync>>;
}

/// A text-to-speech engine. Implementations are pluggable (e.g. Piper).
pub trait TextToSpeech {
    /// Synthesize a text chunk to PCM audio (16 kHz mono i16).
    /// Returns `None` when there is no more audio for this chunk.
    fn synthesize(&mut self, text: &str) -> Result<Vec<i16>, Box<dyn Error + Send + Sync>>;
}

/// A prompt response stream from an LLM backend.
pub struct LlmResponse {
    /// Streaming text of the model's reply.
    pub stream: Option<LlmStream>,
    /// Full accumulated text (useful when streaming is unavailable).
    pub full: String,
}

/// An LLM client. Implementations are pluggable (Ollama, OpenAI-compatible).
pub trait Llm {
    /// Send a prompt and return a (streaming) response.
    fn generate(&self, prompt: &str) -> Result<LlmResponse, Box<dyn Error + Send + Sync>>;
}
