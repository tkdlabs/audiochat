//! Piper-backed text-to-speech, running Piper as a subprocess.
//!
//! Piper (pip install piper-tts) synthesizes text read on stdin and writes raw
//! 22.05 kHz mono 16-bit PCM to stdout. We feed it text and resample the PCM to
//! the pipeline's 16 kHz mono format.

use std::error::Error;
use std::fmt;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Mutex;

use audiochat_core::{LinearResampler, TextToSpeech};

const PIPER_OUTPUT_RATE: u32 = 22_050;

/// A text-to-speech engine backed by the `piper` CLI subprocess.
pub struct Piper {
    /// Absolute/relative path to the piper executable.
    bin: String,
    /// Path to the ONNX voice model.
    model: String,
    /// Serializes synthesis calls (piper subprocess isn't thread-safe in use).
    lock: Mutex<()>,
}

#[derive(Debug)]
pub struct PiperError(String);

impl fmt::Display for PiperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for PiperError {}

impl Piper {
    /// Create a Piper engine for `model` using the `piper` executable.
    pub fn new(model: impl AsRef<Path>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Self::with_bin("piper", model)
    }

    /// Create a Piper engine with a custom executable path.
    pub fn with_bin(
        bin: impl Into<String>,
        model: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let bin = bin.into();
        let model = model.as_ref().display().to_string();
        // Validate the executable exists early for a friendlier error.
        if Command::new(&bin)
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return Err(Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                "piper executable '{bin}' not found or not runnable. Is piper-tts installed?"
            ))));
        }
        Ok(Self {
            bin,
            model,
            lock: Mutex::new(()),
        })
    }
}

impl TextToSpeech for Piper {
    /// Synthesize `text` to 16 kHz mono i16 PCM by running piper and resampling.
    fn synthesize(&mut self, text: &str) -> Result<Vec<i16>, Box<dyn Error + Send + Sync>> {
        let _guard = self.lock.lock().unwrap();

        let mut child = Command::new(&self.bin)
            .arg("-m")
            .arg(&self.model)
            .arg("--output-raw")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                    "failed to spawn piper: {e}"
                )))
            })?;

        {
            let mut stdin = child.stdin.take().ok_or_else(|| {
                Box::<dyn Error + Send + Sync>::from(PiperError("no piper stdin".into()))
            })?;
            stdin.write_all(text.as_bytes()).map_err(|e| {
                Box::<dyn Error + Send + Sync>::from(PiperError(format!("write stdin: {e}")))
            })?;
            // Closing stdin signals end of input to piper.
        } // stdin dropped -> closed

        let output = child.wait_with_output().map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(PiperError(format!("piper wait: {e}")))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                "piper failed: {stderr}"
            ))));
        }

        // Piper outputs raw 16-bit PCM at 22.05 kHz. `as_chunks` drops any
        // trailing odd byte.
        let bytes = &output.stdout;
        let pcm22050: Vec<i16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b: &[u8; 2]| i16::from_le_bytes([b[0], b[1]]))
            .collect();

        // Convert to 16 kHz mono i16 via linear resampling.
        let mono_f32: Vec<f32> = pcm22050
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();
        let mut resampler = LinearResampler::new(PIPER_OUTPUT_RATE, 16_000);
        let out_f32 = resampler.process(&mono_f32);
        Ok(out_f32
            .into_iter()
            .map(|v| (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect())
    }
}
