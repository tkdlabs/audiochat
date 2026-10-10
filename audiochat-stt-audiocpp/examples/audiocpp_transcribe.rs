//! Offline STT test: transcribe a 16 kHz mono WAV via an audio.cpp server.
//!
//! Usage:
//!   cargo run -p audiochat-stt-audiocpp --example audiocpp_transcribe -- <input.wav>
//! Environment:
//!   AUDIOCHAT_STT_URL     audio.cpp base URL (default: http://localhost:8080)
//!   AUDIOCHAT_STT_MODEL   server-side model id (default: moonshine-asr)
//!   AUDIOCHAT_STT_LANGUAGE recognition language, e.g. en-US (default: auto)

use std::error::Error;
use std::path::PathBuf;

use audiochat_core::SpeechRecognizer;
use audiochat_stt_audiocpp::AudioCppStt;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: audiocpp_transcribe <input.wav>");
        std::process::exit(2);
    }
    let wav = PathBuf::from(&args[1]);

    let base = std::env::var("AUDIOCHAT_STT_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());
    let model =
        std::env::var("AUDIOCHAT_STT_MODEL").unwrap_or_else(|_| "moonshine-asr".to_string());
    let language = std::env::var("AUDIOCHAT_STT_LANGUAGE").ok();

    let mut reader = hound::WavReader::open(&wav)?;
    let spec = reader.spec();
    eprintln!(
        "input wav: rate={} channels={} bits={}",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
    let mono: Vec<i16> = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(spec.channels as usize)
            .map(|ch| (ch.iter().map(|&s| s as i32).sum::<i32>() / spec.channels as i32) as i16)
            .collect()
    };

    let mut stt = AudioCppStt::with_base(base, model).with_language(language);
    let text = stt.transcribe(&mono)?;
    println!("TRANSCRIPT: {text}");
    Ok(())
}
