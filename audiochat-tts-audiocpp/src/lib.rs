//! audio.cpp-backed text-to-speech over its OpenAI-compatible HTTP API.
//!
//! [audio.cpp](https://github.com/0xShug0/audio.cpp) exposes an
//! `audiocpp_server` with `POST /v1/audio/speech` returning a WAV. This backend
//! sends text there, decodes the WAV, downmixes to mono, and resamples to the
//! pipeline's 16 kHz mono i16 format. Running the model on a GPU host keeps TTS
//! very fast and off the local box; pass its URL to `with_base`.

use std::error::Error;
use std::fmt;
use std::io::Cursor;
use std::time::Duration;

use audiochat_core::{LinearResampler, TextToSpeech};
use serde::Serialize;

/// Default audio.cpp server base URL (override with `with_base`).
const DEFAULT_BASE: &str = "http://localhost:8080";
/// Default server-side model id (a Qwen3-TTS CustomVoice model).
const DEFAULT_MODEL: &str = "qwen3-tts";
/// Default built-in speaker for the CustomVoice model.
const DEFAULT_VOICE: &str = "Serena";
/// Output rate expected by the rest of the pipeline.
const TARGET_RATE: u32 = 16_000;

/// A text-to-speech engine backed by an audio.cpp server.
pub struct AudioCppTts {
    base_url: String,
    model: String,
    voice: String,
    speed: Option<f32>,
    instruct: Option<String>,
    client: ureq::Agent,
}

/// Request body for `POST /v1/audio/speech`.
#[derive(Serialize)]
struct SpeechRequest<'a> {
    model: &'a str,
    input: &'a str,
    voice: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    /// Style/emotion/pacing instruction. audio.cpp maps the top-level
    /// `instructions` field onto the model's `instruction` request option
    /// (Qwen3-TTS CustomVoice and VoiceDesign consume it).
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
}

#[derive(Debug)]
pub struct AudioCppError(String);

impl fmt::Display for AudioCppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for AudioCppError {}

impl AudioCppTts {
    /// Create an engine against the default audio.cpp host and voice.
    pub fn new() -> Self {
        Self::with_base(DEFAULT_BASE, DEFAULT_MODEL)
    }

    /// Create an engine for a specific base URL and server-side model id.
    pub fn with_base(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let client = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(120)))
            .timeout_recv_body(Some(Duration::from_secs(120)))
            .http_status_as_error(false)
            .build()
            .new_agent();
        Self {
            base_url: base_url.into(),
            model: model.into(),
            voice: DEFAULT_VOICE.to_string(),
            speed: None,
            instruct: None,
            client,
        }
    }

    /// Set the server-side voice/speaker id (e.g. `Serena`, `Dylan`).
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    /// Set a speech-rate multiplier. `None` uses the model default.
    pub fn with_speed(mut self, speed: Option<f32>) -> Self {
        self.speed = speed.filter(|s| *s > 0.0);
        self
    }

    /// Set a natural-language style/emotion/pacing instruction (e.g. "Speak
    /// softly and slowly"). Blank values are ignored. Only models that accept
    /// an instruction (Qwen3-TTS CustomVoice/VoiceDesign) act on it; audio.cpp
    /// ignores it for other families.
    pub fn with_instruct(mut self, instruct: Option<String>) -> Self {
        self.instruct = instruct.filter(|s| !s.trim().is_empty());
        self
    }

    /// The server-side model id this engine is configured for.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl Default for AudioCppTts {
    fn default() -> Self {
        Self::new()
    }
}

impl TextToSpeech for AudioCppTts {
    /// Synthesize `text` to 16 kHz mono i16 PCM via the audio.cpp server.
    fn synthesize(&mut self, text: &str) -> Result<Vec<i16>, Box<dyn Error + Send + Sync>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/v1/audio/speech", self.base_url);
        let body = SpeechRequest {
            model: &self.model,
            input: text,
            voice: &self.voice,
            speed: self.speed,
            instructions: self.instruct.as_deref(),
        };

        let resp = self
            .client
            .post(&url)
            .send_json(body)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(AudioCppError(format!("{e}"))))?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp.into_body().read_to_string().unwrap_or_default();
            let detail = detail.trim();
            let detail = if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            };
            return Err(Box::<dyn Error + Send + Sync>::from(AudioCppError(
                format!("audio.cpp returned status {status}{detail}"),
            )));
        }

        let bytes = resp.into_body().read_to_vec().map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(AudioCppError(format!("read wav: {e}")))
        })?;

        decode_wav_to_16k_mono(&bytes)
    }
}

/// Decode a WAV byte buffer to 16 kHz mono i16 PCM.
fn decode_wav_to_16k_mono(bytes: &[u8]) -> Result<Vec<i16>, Box<dyn Error + Send + Sync>> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|e| {
        Box::<dyn Error + Send + Sync>::from(AudioCppError(format!("decode wav: {e}")))
    })?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    // Read samples as f32 in -1..1 regardless of the file's bit depth/format.
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
            Box::<dyn Error + Send + Sync>::from(AudioCppError(format!("wav samples: {e}")))
        })?,
        (hound::SampleFormat::Int, bits) if bits <= 16 => {
            let scale = (1i64 << (bits.max(1) - 1)) as f32;
            reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    Box::<dyn Error + Send + Sync>::from(AudioCppError(format!("wav samples: {e}")))
                })?
        }
        (hound::SampleFormat::Int, bits) => {
            let scale = (1i64 << (bits.min(32) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    Box::<dyn Error + Send + Sync>::from(AudioCppError(format!("wav samples: {e}")))
                })?
        }
    };

    // Downmix to mono by averaging channels.
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    };

    // Resample to the pipeline's 16 kHz if needed, then quantize to i16.
    let mono = if spec.sample_rate == TARGET_RATE {
        mono
    } else {
        let mut resampler = LinearResampler::new(spec.sample_rate, TARGET_RATE);
        resampler.process(&mono)
    };

    Ok(mono
        .into_iter()
        .map(|v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode `samples` as a mono 16-bit PCM WAV.
    fn wav_bytes(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
            for &s in samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn decodes_16k_mono_passthrough() {
        let input: Vec<i16> = vec![0, 1000, -1000, i16::MAX, i16::MIN];
        let out = decode_wav_to_16k_mono(&wav_bytes(16_000, 1, &input)).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn resamples_24k_to_16k() {
        // 24 kHz -> 16 kHz: length shrinks by 2/3.
        let input = vec![1000i16; 24_000];
        let out = decode_wav_to_16k_mono(&wav_bytes(24_000, 1, &input)).unwrap();
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn downmixes_stereo_to_mono() {
        // Two interleaved channels -> half as many mono samples.
        let input = vec![1000i16, 3000, 1000, 3000];
        let out = decode_wav_to_16k_mono(&wav_bytes(16_000, 2, &input)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 2000);
        assert_eq!(out[1], 2000);
    }

    #[test]
    fn request_sends_instruction_as_instructions() {
        let body = SpeechRequest {
            model: "qwen3-tts",
            input: "hi",
            voice: "serena",
            speed: None,
            instructions: Some("Speak slowly and warmly."),
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"instructions\":\"Speak slowly and warmly.\""));
        // Absent optional fields are omitted entirely.
        assert!(!json.contains("speed"));
    }

    #[test]
    fn request_omits_instructions_when_unset() {
        let body = SpeechRequest {
            model: "qwen3-tts",
            input: "hi",
            voice: "serena",
            speed: Some(1.0),
            instructions: None,
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(!json.contains("instructions"));
    }

    #[test]
    fn with_instruct_ignores_blank() {
        let engine = AudioCppTts::with_base("http://h", "qwen3-tts")
            .with_instruct(Some("   ".into()));
        assert!(engine.instruct.is_none());
    }
}
