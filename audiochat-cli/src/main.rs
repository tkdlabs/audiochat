//! M1: mic -> VAD -> speech-to-text.

use std::path::PathBuf;

use audiochat_core::{AudioConfig, EnergyVad, MicCapture, SpeechRecognizer};
use audiochat_stt_whisper::WhisperRecognizer;

const USAGE: &str = "usage: audiochat [--device <name>] <whisper-model.bin>";

struct Opts {
    model: PathBuf,
    device: Option<String>,
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut model: Option<PathBuf> = None;
    let mut device: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--device" | "-d" => {
                i += 1;
                let name = args.get(i).ok_or("--device requires a value")?;
                device = Some(name.clone());
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option: {flag}\n{USAGE}"))
            }
            pos if model.is_none() => model = Some(PathBuf::from(pos)),
            _ => return Err(format!("unexpected argument: {}\n{USAGE}", args[i])),
        }
        i += 1;
    }
    let model = model.ok_or_else(|| USAGE.to_string())?;
    Ok(Opts { model, device })
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts =
        parse_args(&args)        .map_err(Box::<dyn std::error::Error + Send + Sync>::from)?;

    let mut recognizer = WhisperRecognizer::new(opts.model)?;
    let mut vad = EnergyVad::new(16_000);

    let mic = MicCapture::start_with_device(AudioConfig::default(), opts.device.as_deref())?;

    println!("audiochat: listening... (Ctrl-C to stop)");
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
