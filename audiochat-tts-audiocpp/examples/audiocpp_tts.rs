//! Offline TTS test: synthesize text via an audio.cpp server to a 16 kHz mono WAV.
//!
//! Usage:
//!   cargo run -p audiochat-tts-audiocpp --example audiocpp_tts -- "some text"
//! Environment:
//!   AUDIOCHAT_TTS_URL    audio.cpp base URL (default: http://localhost:8080)
//!   AUDIOCHAT_TTS_MODEL  server-side model id (default: qwen3-tts)
//!   AUDIOCHAT_TTS_VOICE  server-side voice id (default: Serena)

use std::error::Error;

use audiochat_core::TextToSpeech;
use audiochat_tts_audiocpp::AudioCppTts;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: tts \"text to speak\"");
        std::process::exit(2);
    }
    let text = args[1..].join(" ");

    let base = std::env::var("AUDIOCHAT_TTS_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());
    let model = std::env::var("AUDIOCHAT_TTS_MODEL").unwrap_or_else(|_| "qwen3-tts".to_string());
    let voice = std::env::var("AUDIOCHAT_TTS_VOICE").unwrap_or_else(|_| "Serena".to_string());

    let mut tts = AudioCppTts::with_base(base, model).with_voice(voice);
    let pcm = tts.synthesize(&text)?;
    eprintln!(
        "synthesized {} samples ({} ms) at 16 kHz",
        pcm.len(),
        pcm.len() * 1000 / 16_000
    );

    let out_path = "tts_out.wav";
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out_path, spec)
        .map_err(|e| Box::<dyn Error + Send + Sync>::from(e.to_string()))?;
    for &s in &pcm {
        writer
            .write_sample(s)
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(e.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|e| Box::<dyn Error + Send + Sync>::from(e.to_string()))?;
    eprintln!("wrote {out_path}");
    Ok(())
}
