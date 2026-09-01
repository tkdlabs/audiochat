//! audiochat CLI: speech-to-text (default) or text-to-speech test (`--speak`).

use std::path::PathBuf;

use audiochat_core::{
    play_pcm, AudioConfig, EnergyVad, MicCapture, SpeechRecognizer, TextToSpeech,
};
use audiochat_stt_whisper::WhisperRecognizer;
use audiochat_tts_piper::Piper;

const USAGE: &str = "\
usage: audiochat [OPTIONS] <whisper-model.bin>

Modes:
  (default)   Live mic -> text using the whisper model.
  --speak T   Synthesize T with Piper and play it (requires --tts-model).

Options:
  -d, --device NAME   Match an input device by name (case-insensitive substring).
  --tts-model PATH    Piper ONNX voice model for --speak mode.
  --tts-bin PATH      Piper executable (default: $PIPER_BIN or \"piper\").
  -h, --help          Show this help.";

struct Opts {
    model: Option<PathBuf>,
    device: Option<String>,
    speak: Option<String>,
    tts_model: Option<PathBuf>,
    tts_bin: Option<String>,
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut model: Option<PathBuf> = None;
    let mut device: Option<String> = None;
    let mut speak: Option<String> = None;
    let mut tts_model: Option<PathBuf> = None;
    let mut tts_bin: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--device" | "-d" => {
                i += 1;
                let name = args.get(i).ok_or("--device requires a value")?;
                device = Some(name.clone());
            }
            "--speak" => {
                i += 1;
                let text = args.get(i).ok_or("--speak requires text")?;
                speak = Some(text.clone());
            }
            "--tts-model" => {
                i += 1;
                let p = args.get(i).ok_or("--tts-model requires a path")?;
                tts_model = Some(PathBuf::from(p));
            }
            "--tts-bin" => {
                i += 1;
                let p = args.get(i).ok_or("--tts-bin requires a path")?;
                tts_bin = Some(p.clone());
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
    let model = model;
    Ok(Opts {
        model,
        device,
        speak,
        tts_model,
        tts_bin,
    })
}

fn run_tts(opts: &Opts) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tts_model = opts
        .tts_model
        .as_ref()
        .ok_or("--speak requires --tts-model <piper.onnx>")?;
    let bin = opts
        .tts_bin
        .clone()
        .or_else(|| std::env::var("PIPER_BIN").ok())
        .unwrap_or_else(|| "piper".to_string());
    let mut piper = Piper::with_bin(bin, tts_model)?;
    let text = opts.speak.as_deref().unwrap_or_default();
    let pcm = piper.synthesize(text)?;
    eprintln!(
        "audiochat: synthesized {} samples ({} ms)",
        pcm.len(),
        pcm.len() * 1000 / 16_000
    );
    play_pcm(&pcm, AudioConfig::default())?;
    Ok(())
}

fn run_stt(opts: &Opts) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = opts
        .model
        .as_ref()
        .ok_or_else(|| Box::<dyn std::error::Error + Send + Sync>::from(USAGE.to_string()))?;
    let mut recognizer = WhisperRecognizer::new(model)?;
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

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = parse_args(&args).map_err(Box::<dyn std::error::Error + Send + Sync>::from)?;

    if opts.speak.is_some() {
        run_tts(&opts)
    } else {
        run_stt(&opts)
    }
}
