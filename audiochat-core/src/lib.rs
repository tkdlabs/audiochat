//! Core types and pluggable pipeline traits shared across audiochat crates.

pub mod capture;
pub mod config;
pub mod playback;
pub mod resample;
pub mod traits;
pub mod vad;

pub use traits::{Llm, SpeechRecognizer, TextToSpeech};

pub use capture::MicCapture;
pub use config::AudioConfig;
pub use playback::play_pcm;
pub use resample::LinearResampler;
pub use vad::EnergyVad;

/// Default capture sample rate in Hz.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;
/// Default number of audio channels.
pub const CHANNELS: u16 = 1;
