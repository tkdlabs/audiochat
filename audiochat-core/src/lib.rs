//! Core types and pluggable pipeline traits shared across audiochat crates.

pub mod audio_sink;
pub mod capture;
pub mod config;
pub mod markdown;
pub mod pipeline;
pub mod resample;
pub mod session;
pub mod traits;
pub mod vad;

pub use traits::{Llm, LlmResponse, LlmStream, LlmStreamItem, SpeechRecognizer, TextToSpeech};

pub use audio_sink::AudioSink;
pub use capture::MicCapture;
pub use config::AudioConfig;
pub use markdown::strip_markdown;
pub use pipeline::{Pipeline, TurnTiming};
pub use resample::LinearResampler;
pub use session::Session;
pub use vad::EnergyVad;

/// Default capture sample rate in Hz.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;
/// Default number of audio channels.
pub const CHANNELS: u16 = 1;
