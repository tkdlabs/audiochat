//! audiochat CLI: speech-to-text (default), TTS test (`--speak`), LLM test
//! (`--prompt`), or full speech-to-speech (`--s2s`).

use std::path::PathBuf;

use audiochat_core::{
    play_pcm, AudioConfig, EnergyVad, Llm, MicCapture, Pipeline, SpeechRecognizer, TextToSpeech,
};
use audiochat_llm::Ollama;
use audiochat_stt_whisper::WhisperRecognizer;
use audiochat_tts_piper::Piper;

const USAGE: &str = "\
usage: audiochat [OPTIONS] <whisper-model.bin>

Modes:
  (default)   Live mic -> text using the whisper model.
  --speak T   Synthesize T with Piper and play it (requires --tts-model).
  --prompt T  Send T to an LLM (Ollama) and print the streamed reply.
  --s2s       Full speech-to-speech loop: mic -> STT -> LLM -> Piper.

Options:
  -d, --device NAME   Match an input device by name (case-insensitive substring).
  --tts-model PATH    Piper ONNX voice model for --speak/--s2s.
  --tts-bin PATH      Piper executable (default: $PIPER_BIN or \"piper\").
  --llm-model NAME    Ollama model name for --prompt/--s2s.
  --llm-url URL       Ollama base URL (default: http://localhost:11434).
  --silent            In --s2s, print replies but do not speak them.
  -h, --help          Show this help.";

struct Opts {
    model: Option<PathBuf>,
    device: Option<String>,
    speak: Option<String>,
    tts_model: Option<PathBuf>,
    tts_bin: Option<String>,
    prompt: Option<String>,
    llm_model: Option<String>,
    llm_url: Option<String>,
    s2s: bool,
    silent: bool,
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut model: Option<PathBuf> = None;
    let mut device: Option<String> = None;
    let mut speak: Option<String> = None;
    let mut tts_model: Option<PathBuf> = None;
    let mut tts_bin: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut llm_model: Option<String> = None;
    let mut llm_url: Option<String> = None;
    let mut s2s = false;
    let mut silent = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--device" | "-d" => {
                i += 1;
                let name = args.get(i).ok_or("--device requires a value")?;
                device = Some(name.clone());
            }
            "--s2s" => s2s = true,
            "--silent" => silent = true,
            "--speak" => {
                i += 1;
                let text = args.get(i).ok_or("--speak requires text")?;
                speak = Some(text.clone());
            }
            "--prompt" => {
                i += 1;
                let text = args.get(i).ok_or("--prompt requires text")?;
                prompt = Some(text.clone());
            }
            "--llm-model" => {
                i += 1;
                let name = args.get(i).ok_or("--llm-model requires a name")?;
                llm_model = Some(name.clone());
            }
            "--llm-url" => {
                i += 1;
                let m = args.get(i).ok_or("--llm-url requires a URL")?;
                llm_url = Some(m.clone());
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
    Ok(Opts {
        model,
        device,
        speak,
        tts_model,
        tts_bin,
        prompt,
        llm_model,
        llm_url,
        s2s,
        silent,
    })
}

fn run_llm(opts: &Opts) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = opts
        .llm_model
        .clone()
        .ok_or("--prompt requires --llm-model <ollama-model>")?;
    let base = opts
        .llm_url
        .clone()
        .unwrap_or_else(|| "http://localhost:11434".to_string());
    let client = Ollama::with_base(base, model);
    let prompt = opts.prompt.as_deref().unwrap_or_default();

    println!("audiochat: asking ollama ({})...", client.model());
    let mut resp = client.generate(prompt)?;
    if let Some(stream) = resp.stream.take() {
        for item in stream {
            let chunk = item?;
            print!("{chunk}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
    if !resp.full.is_empty() {
        print!("{}", resp.full);
    }
    println!();
    Ok(())
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

fn run_s2s(opts: &Opts) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = opts
        .model
        .as_ref()
        .ok_or_else(|| Box::<dyn std::error::Error + Send + Sync>::from(USAGE.to_string()))?;
    let tts_model = opts
        .tts_model
        .as_ref()
        .ok_or("--s2s requires --tts-model <piper.onnx>")?;
    let llm_model = opts
        .llm_model
        .clone()
        .ok_or("--s2s requires --llm-model <ollama-model>")?;
    let llm_url = opts
        .llm_url
        .clone()
        .unwrap_or_else(|| "http://localhost:11434".to_string());
    let tts_bin = opts
        .tts_bin
        .clone()
        .or_else(|| std::env::var("PIPER_BIN").ok())
        .unwrap_or_else(|| "piper".to_string());

    let stt = Box::new(WhisperRecognizer::new(model)?);
    let llm = Box::new(Ollama::with_base(llm_url, llm_model));
    let tts = Box::new(Piper::with_bin(tts_bin, tts_model)?);

    let mut pipeline = Pipeline::new(stt, llm, tts);
    pipeline.speak_replies = !opts.silent;
    let mic = MicCapture::start_with_device(AudioConfig::default(), opts.device.as_deref())?;

    println!("audiochat: speech-to-speech mode. Speak to ask; Ctrl-C to stop.");
    while let Ok(pcm) = mic.rx.recv() {
        pipeline.feed(&pcm)?;
    }
    pipeline.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let opts = parse_args(&args).map_err(Box::<dyn std::error::Error + Send + Sync>::from)?;

    if opts.s2s {
        return run_s2s(&opts);
    }
    if opts.prompt.is_some() {
        return run_llm(&opts);
    }
    if opts.speak.is_some() {
        run_tts(&opts)
    } else {
        run_stt(&opts)
    }
}
