//! audio.cpp-backed speech-to-text over its OpenAI-compatible HTTP API.
//!
//! [audio.cpp](https://github.com/0xShug0/audio.cpp)'s `audiocpp_server` exposes
//! `POST /v1/audio/transcriptions` (OpenAI Whisper-compatible). This backend
//! uploads a completed utterance as a WAV and returns the transcript, so STT can
//! run on a fast GPU host instead of locally. The server also offers a streaming
//! `/v1/audio/transcriptions/live` endpoint, which a future streaming recognizer
//! can use.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use audiochat_core::SpeechRecognizer;
use serde::Deserialize;

/// Default audio.cpp server base URL (override with `with_base`).
const DEFAULT_BASE: &str = "http://localhost:8080";
/// Default server-side ASR model id (fast English streaming model).
const DEFAULT_MODEL: &str = "moonshine-asr";
/// Multipart boundary; fixed is fine for our single-shot uploads.
const BOUNDARY: &str = "----audiochat-boundary-7f3a9c";
/// Sample rate the pipeline produces and the server expects.
const SAMPLE_RATE: u32 = 16_000;

/// A speech recognizer backed by an audio.cpp server.
pub struct AudioCppStt {
    base_url: String,
    model: String,
    language: Option<String>,
    client: ureq::Agent,
}

/// Successful transcription response (extra fields are ignored).
#[derive(Deserialize)]
struct Transcription {
    #[serde(default)]
    text: String,
}

#[derive(Debug)]
pub struct AudioCppSttError(String);

impl fmt::Display for AudioCppSttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for AudioCppSttError {}

impl AudioCppStt {
    /// Create a recognizer against the default audio.cpp host and model.
    pub fn new() -> Self {
        Self::with_base(DEFAULT_BASE, DEFAULT_MODEL)
    }

    /// Create a recognizer for a specific base URL and server-side model id.
    pub fn with_base(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let client = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(120)))
            .timeout_recv_body(Some(Duration::from_secs(120)))
            .build()
            .new_agent();
        Self {
            base_url: base_url.into(),
            model: model.into(),
            language: None,
            client,
        }
    }

    /// Set the recognition language (e.g. `en-US`). `None` lets the model
    /// auto-detect.
    pub fn with_language(mut self, language: Option<String>) -> Self {
        self.language = language.filter(|l| !l.trim().is_empty());
        self
    }

    /// The server-side model id this recognizer is configured for.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl Default for AudioCppStt {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechRecognizer for AudioCppStt {
    fn transcribe(&mut self, pcm: &[i16]) -> Result<String, Box<dyn Error + Send + Sync>> {
        if pcm.is_empty() {
            return Ok(String::new());
        }

        let wav = encode_wav(pcm)?;
        let body = build_multipart(&self.model, self.language.as_deref(), &wav);

        let url = format!("{}/v1/audio/transcriptions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header(
                "Content-Type",
                &format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .send(body)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(AudioCppSttError(format!("{e}"))))?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp.into_body().read_to_string().unwrap_or_default();
            let detail = detail.trim();
            let detail = if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            };
            return Err(Box::<dyn Error + Send + Sync>::from(AudioCppSttError(
                format!("audio.cpp returned status {status}{detail}"),
            )));
        }

        let parsed: Transcription = resp.into_body().read_json().map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(AudioCppSttError(format!("parse response: {e}")))
        })?;
        Ok(parsed.text.trim().to_string())
    }
}

/// Encode 16 kHz mono i16 PCM as a WAV byte buffer.
fn encode_wav(pcm: &[i16]) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(AudioCppSttError(format!("encode wav: {e}")))
        })?;
        for &sample in pcm {
            writer.write_sample(sample).map_err(|e| {
                Box::<dyn Error + Send + Sync>::from(AudioCppSttError(format!("encode wav: {e}")))
            })?;
        }
        writer.finalize().map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(AudioCppSttError(format!("encode wav: {e}")))
        })?;
    }
    Ok(buf.into_inner())
}

/// Build a `multipart/form-data` body with `model`, optional `language`, and the
/// WAV `file` field.
fn build_multipart(model: &str, language: Option<&str>, wav: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(wav.len() + 256);
    push_field(&mut body, "model", model);
    if let Some(language) = language {
        push_field(&mut body, "language", language);
    }
    push_file(&mut body, "file", "audio.wav", "audio/wav", wav);
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    body
}

fn push_field(body: &mut Vec<u8>, name: &str, value: &str) {
    body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
    );
    body.extend_from_slice(value.as_bytes());
    body.extend_from_slice(b"\r\n");
}

fn push_file(body: &mut Vec<u8>, name: &str, filename: &str, content_type: &str, data: &[u8]) {
    body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(data);
    body.extend_from_slice(b"\r\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_contains_fields_and_file() {
        let wav = b"RIFFfake";
        let body = build_multipart("nemotron-asr", Some("en-US"), wav);
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"model\""));
        assert!(text.contains("nemotron-asr"));
        assert!(text.contains("name=\"language\""));
        assert!(text.contains("en-US"));
        assert!(text.contains("name=\"file\"; filename=\"audio.wav\""));
        assert!(text.contains("Content-Type: audio/wav"));
        assert!(text.contains("RIFFfake"));
        assert!(text.ends_with(&format!("--{BOUNDARY}--\r\n")));
    }

    #[test]
    fn multipart_omits_language_when_none() {
        let body = build_multipart("m", None, b"x");
        let text = String::from_utf8_lossy(&body);
        assert!(!text.contains("name=\"language\""));
    }

    #[test]
    fn encodes_a_valid_wav_header() {
        let wav = encode_wav(&[0, 1000, -1000]).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        // 44-byte canonical header + 3 samples * 2 bytes.
        assert_eq!(wav.len(), 44 + 6);
    }
}
