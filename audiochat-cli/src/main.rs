//! M1: mic -> VAD -> speech-to-text.

use std::path::PathBuf;

use audiochat_core::{AudioConfig, EnergyVad, MicCapture, SpeechRecognizer};
use audiochat_stt_whisper::WhisperRecognizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(PathBuf::from)
        .ok_or("usage: audiochat <whisper-model.bin>")?;

    let mut recognizer = WhisperRecognizer::new(model_path)?;
    let mut vad = EnergyVad::new(16_000);

    println!("audiochat: listening... (Ctrl-C to stop)");
    let mic = MicCapture::start(AudioConfig::default())?;

    while let Ok(pcm) = mic.rx.recv() {
        for utterance in vad.feed(&pcm) {
            let text = recognizer.transcribe(&utterance)?;
            println!("> {text}");
        }
    }

    if let Some(utterance) = vad.flush() {
        let text = recognizer.transcribe(&utterance)?;
        println!("> {text}");
    }

    Ok(())
}
