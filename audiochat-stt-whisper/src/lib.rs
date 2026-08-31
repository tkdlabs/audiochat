//! whisper.cpp-backed speech-to-text implementation.

use std::error::Error;
use std::fmt;
use std::path::Path;

use audiochat_core::SpeechRecognizer;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// A speech recognizer backed by whisper.cpp (via whisper-rs).
pub struct WhisperRecognizer {
    ctx: WhisperContext,
    language: Option<String>,
}

#[derive(Debug)]
pub struct WhisperError(String);

impl fmt::Display for WhisperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for WhisperError {}

impl WhisperRecognizer {
    /// Create a recognizer from a GGML model file.
    ///
    /// `model_path` should point to an `ggml-*.bin` (e.g. `ggml-base.en.bin`).
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Self::with_config(model_path, None)
    }

    /// Create a recognizer with an optional language hint.
    pub fn with_config(
        model_path: impl AsRef<Path>,
        language: Option<String>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(model_path.as_ref(), params)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(WhisperError(format!("{e}"))))?;
        Ok(Self { ctx, language })
    }
}

impl SpeechRecognizer for WhisperRecognizer {
    fn transcribe(&mut self, pcm: &[i16]) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(WhisperError(format!("{e}"))))?;

        // Convert i16 PCM to f32 in [-1.0, 1.0], as expected by whisper.
        let samples: Vec<f32> = pcm.iter().map(|&s| s as f32 / i16::MAX as f32).collect();

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(self.language.as_deref());
        params.set_n_threads(4);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(false);

        state
            .full(params, &samples)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(WhisperError(format!("{e}"))))?;

        let mut out = String::new();
        let num_segments = state.full_n_segments();
        for i in 0..num_segments {
            if let Some(seg) = state.get_segment(i) {
                out.push_str(&seg.to_string());
            }
        }

        Ok(out.trim().to_string())
    }
}
