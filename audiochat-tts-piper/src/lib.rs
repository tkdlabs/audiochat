//! Piper-backed text-to-speech, running Piper as a persistent subprocess.
//!
//! Piper (pip install piper-tts) normally synthesizes a whole text and exits.
//! Here we run a small Python helper (see `piper_stream.py`) that keeps a
//! single voice model loaded and serves length-prefixed synthesis requests over
//! stdin/stdout, so we avoid re-loading the ONNX model on every chunk. The
//! helper streams 16-bit mono PCM at the voice's native rate; we resample it to
//! the pipeline's 16 kHz mono format.

use std::error::Error;
use std::fmt;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

use audiochat_core::{LinearResampler, TextToSpeech};

/// Embedded helper that keeps a single Piper voice model loaded and serves
/// length-prefixed synthesis requests over stdin/stdout.
const STREAM_SCRIPT: &str = include_str!("piper_stream.py");

/// Default Python interpreter used to run the helper.
const DEFAULT_PYTHON: &str = "python3";

/// Safety cap on the size of a single synthesis reply, to avoid a huge
/// allocation from a corrupt length prefix.
const MAX_PCM_BYTES: usize = 256 * 1024 * 1024;

/// A text-to-speech engine backed by a persistent Piper subprocess.
pub struct Piper {
    /// Python interpreter used to run the streaming helper.
    python: String,
    /// Path to the ONNX voice model.
    model: String,
    /// Lazily-spawned helper process, kept alive across synthesis calls.
    process: Option<PiperProcess>,
}

/// A running Piper helper process with an established framing protocol.
struct PiperProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    /// The voice model's native sample rate (from the startup header).
    sample_rate: u32,
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
    /// Create a Piper engine for `model` using the system `python3`.
    pub fn new(model: impl AsRef<Path>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Self::with_python(DEFAULT_PYTHON, model)
    }

    /// Create a Piper engine using a specific Python interpreter.
    pub fn with_python(
        python: impl Into<String>,
        model: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let python = python.into();
        let model = model.as_ref().display().to_string();
        // Validate the interpreter exists early for a friendlier error.
        if Command::new(&python)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return Err(Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                "python interpreter '{python}' not found or not runnable"
            ))));
        }
        Ok(Self {
            python,
            model,
            process: None,
        })
    }

    /// Lazily spawn the persistent helper and read its startup header.
    fn ensure_process(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if self.process.is_some() {
            return Ok(());
        }

        let mut child = Command::new(&self.python)
            .arg("-c")
            .arg(STREAM_SCRIPT)
            .arg(&self.model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                    "failed to spawn piper helper: {e}"
                )))
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            Box::<dyn Error + Send + Sync>::from(PiperError("no piper helper stdin".into()))
        })?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or_else(|| {
            Box::<dyn Error + Send + Sync>::from(PiperError("no piper helper stdout".into()))
        })?);

        let sample_rate = read_u32(&mut stdout).map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                "piper helper failed to start (is the voice model valid?): {e}"
            )))
        })?;

        self.process = Some(PiperProcess {
            child,
            stdin,
            stdout,
            sample_rate,
        });
        Ok(())
    }

    /// Kill and drop the helper process, so a subsequent call respawns it.
    fn kill_process(&mut self) {
        if let Some(mut proc) = self.process.take() {
            drop(proc.stdin);
            let _ = proc.child.kill();
            let _ = proc.child.wait();
        }
    }
}

impl TextToSpeech for Piper {
    /// Synthesize `text` to 16 kHz mono i16 PCM via the persistent helper.
    fn synthesize(&mut self, text: &str) -> Result<Vec<i16>, Box<dyn Error + Send + Sync>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let result = self.synthesize_inner(text);
        if result.is_err() {
            // The helper may have died mid-stream; drop it so a retry respawns.
            self.kill_process();
        }
        result
    }
}

impl Piper {
    fn synthesize_inner(&mut self, text: &str) -> Result<Vec<i16>, Box<dyn Error + Send + Sync>> {
        self.ensure_process()?;
        let proc = self.process.as_mut().unwrap();

        let text_bytes = text.as_bytes();
        proc.stdin
            .write_all(&(text_bytes.len() as u32).to_le_bytes())
            .map_err(|e| {
                Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                    "write text length to piper helper: {e}"
                )))
            })?;
        proc.stdin.write_all(text_bytes).map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                "write text to piper helper: {e}"
            )))
        })?;
        proc.stdin.flush().map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(PiperError(format!("flush piper helper: {e}")))
        })?;

        let pcm_len = read_u32(&mut proc.stdout)? as usize;
        if pcm_len > MAX_PCM_BYTES {
            return Err(Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                "piper helper returned implausibly large audio ({pcm_len} bytes)"
            ))));
        }
        let mut pcm_bytes = vec![0u8; pcm_len];
        proc.stdout.read_exact(&mut pcm_bytes).map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(PiperError(format!(
                "read audio from piper helper: {e}"
            )))
        })?;

        let sample_rate = proc.sample_rate;
        let pcm: Vec<i16> = pcm_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b: &[u8; 2]| i16::from_le_bytes([b[0], b[1]]))
            .collect();

        // Convert to 16 kHz mono i16 via linear resampling.
        let mono_f32: Vec<f32> = pcm.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
        let mut resampler = LinearResampler::new(sample_rate, 16_000);
        let out_f32 = resampler.process(&mono_f32);
        Ok(out_f32
            .into_iter()
            .map(|v| (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect())
    }
}

impl Drop for Piper {
    fn drop(&mut self) {
        self.kill_process();
    }
}

/// Read a little-endian `u32` from a reader.
fn read_u32(r: &mut impl Read) -> Result<u32, Box<dyn Error + Send + Sync>> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| Box::<dyn Error + Send + Sync>::from(PiperError(format!("read: {e}"))))?;
    Ok(u32::from_le_bytes(buf))
}
