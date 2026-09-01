//! Offline STT test: transcribe a 16 kHz mono 16-bit PCM WAV file.
//!
//! Usage: cargo run -p audiochat-stt-whisper --example transcribe -- <model.bin> <input.wav>

use std::error::Error;
use std::path::PathBuf;

use audiochat_core::SpeechRecognizer;
use audiochat_stt_whisper::WhisperRecognizer;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: transcribe <whisper-model.bin> <input.wav>");
        std::process::exit(2);
    }
    let model = PathBuf::from(&args[1]);
    let wav = PathBuf::from(&args[2]);

    let mut reader = hound::WavReader::open(&wav)?;
    let spec = reader.spec();
    eprintln!(
        "input wav: rate={} channels={} bits={}",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;

    // Downmix to mono if needed by averaging channels.
    let mono: Vec<i16> = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(spec.channels as usize)
            .map(|ch| (ch.iter().map(|&s| s as i32).sum::<i32>() / spec.channels as i32) as i16)
            .collect()
    };

    eprintln!(
        "transcribing {} samples ({} s)...",
        mono.len(),
        mono.len() / spec.sample_rate as usize
    );
    let mut recognizer = WhisperRecognizer::new(model)?;
    let text = recognizer.transcribe(&mono)?;
    println!("TRANSCRIPT: {text}");
    Ok(())
}
