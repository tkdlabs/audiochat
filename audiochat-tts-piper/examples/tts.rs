//! Offline TTS test: synthesize text to a 16 kHz mono 16-bit PCM WAV.
//!
//! Usage:
//!   cargo run -p audiochat-tts-piper --example tts -- <piper-model.onnx> "some text"
//! Environment:
//!   AUDIOCHAT_PYTHON  python interpreter with piper-tts (default: "python3")

use std::error::Error;

use audiochat_core::TextToSpeech;
use audiochat_tts_piper::Piper;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: tts <piper-model.onnx> \"text to speak\"");
        std::process::exit(2);
    }
    let model = &args[1];
    let text = args[2..].join(" ");

    let python = std::env::var("AUDIOCHAT_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut piper = Piper::with_python(python, model)?;

    let pcm = piper.synthesize(&text)?;
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
