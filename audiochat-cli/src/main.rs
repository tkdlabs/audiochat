//! audiochat CLI: speech-to-text (default), TTS test (`--speak`), LLM test
//! (`--prompt`), or full speech-to-speech (`--s2s`).

mod config;
mod discover;
mod template;
mod wizard;

use std::path::PathBuf;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use audiochat_core::{
    AudioConfig, AudioSink, ConversationLogger, EnergyVad, Llm, MicCapture, Pipeline, Session,
    SpeechRecognizer, TextToSpeech,
};
use audiochat_llm::{opencode_go_api_key, Ollama, OpenAiCompatible};
use audiochat_stt_audiocpp::AudioCppStt;
use audiochat_stt_whisper::WhisperRecognizer;
use audiochat_tts_audiocpp::AudioCppTts;
use audiochat_tts_piper::Piper;

const USAGE: &str = "\
usage: audiochat [OPTIONS] <whisper-model.bin>

Modes:
  (default)   Live mic -> text using the STT backend.
  --speak T   Synthesize T with TTS and play it (requires --tts-model for Piper).
  --prompt T  Send T to an LLM and print the streamed reply.
  --s2s       Full speech-to-speech loop: mic -> STT -> LLM -> TTS.

Options:
  -d, --device NAME   Match an input device by name (case-insensitive substring).
  --stt-backend NAME  STT backend: whisper (default) or audiocpp.
  --stt-url URL       audio.cpp base URL (default: http://localhost:8080).
  --stt-model NAME    audio.cpp server-side ASR model id (default: moonshine-asr).
  --stt-language LANG audio.cpp recognition language (e.g. en-US; default: auto).
  --tts-backend NAME  TTS backend: piper (default) or audiocpp.
  --tts-model PATH    Piper ONNX voice model (piper), or server-side model id
                      (audiocpp, default: qwen3-tts).
  --tts-python PATH   Python interpreter with piper-tts installed (default: python3).
  --tts-url URL       audio.cpp base URL (default: http://localhost:8080).
  --tts-voice NAME    audio.cpp voice/speaker id (default: Serena).
  --llm-provider NAME LLM backend: ollama (default), openai, or opencode-go.
  --llm-model NAME    Model name (Ollama model, or OpenAI-compatible model id).
  --llm-url URL       Base URL (Ollama: http://localhost:11434; opencode-go:
                      https://opencode.ai/zen/go/v1).
  --llm-api-key KEY   Bearer token for OpenAI-compatible backends (or env).
  --llm-thinking      Enable reasoning/thinking mode (default: off).
  --llm-temperature F Sampling temperature (0..2); default is the model's.
  --llm-num-ctx N     Context window size in tokens; default is the model's.
  --system-prompt P   Override the default assistant system prompt (spoken style).
  --vad-threshold F   Minimum VAD RMS threshold (0..1). With adaptive noise
                      (default) the threshold auto-rises above this as the room
                      gets noisier; with --no-adaptive it's a fixed threshold.
  --noise-ratio F     Adaptive noise multiplier (default 2.0): effective threshold
                      = max(vad-threshold, measured_noise * F).
  --no-adaptive       Disable adaptive noise tracking; use --vad-threshold as a
                      fixed absolute threshold.
  --vad-silence MS    Trailing silence (ms) that ends a candidate segment (default 600).
                      Mid-sentence segments are held up to one more window by the
                      STT endpointer, so brief pauses don't split a turn.
  --pause MS          How long (ms) to hold a mid-sentence transcript waiting for
                      you to resume (default: same as --vad-silence). Raise this
                      to tolerate long thinking pauses; tune live with '+'/'-'.
  --stop-word WORD    Spoken code word that finalizes the current turn immediately
                      instead of waiting out the pause (default: stop).
  --no-stop-word      Disable the spoken stop word.
  --barge-in          Let the user interrupt a reply by speaking over it. Use
                      headphones so the mic doesn't hear the assistant's own voice.
  --barge-threshold F RMS threshold (0..1) that triggers barge-in (default 0.06).
                      Raise if the assistant interrupts itself; lower if it misses
                      a deliberate stop.
  --conversation-log F Write each turn as a JSON line to F (JSONL).
                      AUDIOCHAT_CONVERSATION_LOG is the env fallback.
  -v, --verbose       Print per-turn latency metrics in --s2s.
  --silent            In --s2s, print replies but do not speak them.

Templates:
  --wizard            Interactive setup for STT/TTS/LLM/system prompt; saves a
                      template and/or launches speech-to-speech.
  --template NAME     Load a saved template (explicit flags still win; a full
                      STT+TTS+LLM template enables --s2s automatically).
  --save-template N   Save the current options as template N and continue.
  --edit-template N   Open template N in the wizard and save changes back.
  --list-templates    List saved templates.
  --delete-template N Delete saved template N.
  Templates live in ~/.config/audiochat/templates/ (override with
  AUDIOCHAT_CONFIG_DIR or XDG_CONFIG_HOME).

Defaults:
  --save-config       Save the current options as global defaults (hosts, model
                      ids, voice) to ~/.config/audiochat/config.json. These are
                      loaded automatically and are the lowest-priority layer,
                      below flags, environment variables, and templates.

  -h, --help          Show this help.

Environment (flag takes precedence):
  AUDIOCHAT_DEVICE        input device substring
  AUDIOCHAT_STT_BACKEND   stt backend: whisper or audiocpp
  AUDIOCHAT_STT_URL       audio.cpp base URL
  AUDIOCHAT_STT_MODEL     audio.cpp server-side ASR model id
  AUDIOCHAT_STT_LANGUAGE  audio.cpp recognition language (e.g. en-US)
  AUDIOCHAT_TTS_BACKEND   tts backend: piper or audiocpp
  AUDIOCHAT_TTS_MODEL     piper voice path, or audiocpp server model id
  AUDIOCHAT_TTS_URL       audio.cpp base URL
  AUDIOCHAT_TTS_VOICE     audio.cpp voice id
  AUDIOCHAT_PYTHON        python interpreter for piper (default python3)
  AUDIOCHAT_LLM_PROVIDER  llm backend: ollama, openai, or opencode-go
  AUDIOCHAT_LLM_MODEL     model name
  AUDIOCHAT_LLM_URL       base URL
  AUDIOCHAT_LLM_API_KEY   bearer token (falls back to OPENCODE_API_KEY, then
                          opencode's auth.json for the opencode-go provider)
  AUDIOCHAT_LLM_THINKING  set to 1/true to enable thinking mode
  AUDIOCHAT_PAUSE         endpointer pause window (ms)
  AUDIOCHAT_STOP_WORD     spoken stop word (default: stop)";

struct Opts {
    model: Option<PathBuf>,
    device: Option<String>,
    stt_backend: Option<String>,
    stt_url: Option<String>,
    stt_model: Option<String>,
    stt_language: Option<String>,
    speak: Option<String>,
    tts_backend: Option<String>,
    tts_model: Option<PathBuf>,
    tts_url: Option<String>,
    tts_voice: Option<String>,
    tts_python: Option<String>,
    prompt: Option<String>,
    llm_provider: Option<String>,
    llm_model: Option<String>,
    llm_url: Option<String>,
    llm_api_key: Option<String>,
    llm_thinking: bool,
    llm_temperature: Option<f32>,
    llm_num_ctx: Option<u32>,
    s2s: bool,
    silent: bool,
    verbose: bool,
    vad_threshold: Option<f32>,
    noise_ratio: Option<f32>,
    adaptive_noise: bool,
    vad_silence_ms: Option<u64>,
    pause_ms: Option<u64>,
    stop_word: Option<String>,
    barge_in: bool,
    barge_threshold: Option<f32>,
    conversation_log: Option<String>,
    system_prompt: Option<String>,
    wizard: bool,
    template: Option<String>,
    save_template: Option<String>,
    edit_template: Option<String>,
    list_templates: bool,
    delete_template: Option<String>,
    save_config: bool,
}

/// Resolve a CLI flag value, falling back to an environment variable.
fn opt_or_env(cli: Option<String>, env: &str) -> Option<String> {
    cli.or_else(|| std::env::var(env).ok().filter(|s| !s.is_empty()))
}

/// Resolve a CLI float flag, falling back to a parsed environment variable.
fn opt_float_or_env(cli: Option<f32>, env: &str) -> Option<f32> {
    cli.or_else(|| std::env::var(env).ok().and_then(|s| s.parse().ok()))
}

/// Resolve a CLI integer flag, falling back to a parsed environment variable.
fn opt_u32_or_env(cli: Option<u32>, env: &str) -> Option<u32> {
    cli.or_else(|| std::env::var(env).ok().and_then(|s| s.parse().ok()))
}

/// Resolve a CLI u64 flag, falling back to a parsed environment variable.
fn opt_u64_or_env(cli: Option<u64>, env: &str) -> Option<u64> {
    cli.or_else(|| std::env::var(env).ok().and_then(|s| s.parse().ok()))
}

/// Whether a truthy boolean env var is set ("1", "true", "yes", "on").
fn env_flag(env: &str) -> bool {
    matches!(
        std::env::var(env).ok().map(|s| s.trim().to_ascii_lowercase()),
        Some(v) if matches!(v.as_str(), "1" | "true" | "yes" | "on")
    )
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut model: Option<PathBuf> = None;
    let mut device: Option<String> = None;
    let mut stt_backend: Option<String> = None;
    let mut stt_url: Option<String> = None;
    let mut stt_model: Option<String> = None;
    let mut stt_language: Option<String> = None;
    let mut speak: Option<String> = None;
    let mut tts_backend: Option<String> = None;
    let mut tts_model: Option<PathBuf> = None;
    let mut tts_url: Option<String> = None;
    let mut tts_voice: Option<String> = None;
    let mut tts_python: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut llm_provider: Option<String> = None;
    let mut llm_model: Option<String> = None;
    let mut llm_url: Option<String> = None;
    let mut llm_api_key: Option<String> = None;
    let mut llm_thinking = false;
    let mut llm_temperature: Option<f32> = None;
    let mut llm_num_ctx: Option<u32> = None;
    let mut s2s = false;
    let mut silent = false;
    let mut verbose = false;
    let mut vad_threshold: Option<f32> = None;
    let mut noise_ratio: Option<f32> = None;
    let mut adaptive_noise = true;
    let mut vad_silence_ms: Option<u64> = None;
    let mut pause_ms: Option<u64> = None;
    let mut stop_word: Option<String> = None;
    let mut barge_in = false;
    let mut barge_threshold: Option<f32> = None;
    let mut conversation_log: Option<String> = None;
    let mut system_prompt: Option<String> = None;
    let mut wizard = false;
    let mut template: Option<String> = None;
    let mut save_template: Option<String> = None;
    let mut edit_template: Option<String> = None;
    let mut list_templates = false;
    let mut delete_template: Option<String> = None;
    let mut save_config = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--device" | "-d" => {
                i += 1;
                let name = args.get(i).ok_or("--device requires a value")?;
                device = Some(name.clone());
            }
            "--stt-backend" => {
                i += 1;
                let name = args.get(i).ok_or("--stt-backend requires a value")?;
                stt_backend = Some(name.clone());
            }
            "--stt-url" => {
                i += 1;
                let u = args.get(i).ok_or("--stt-url requires a URL")?;
                stt_url = Some(u.clone());
            }
            "--stt-model" => {
                i += 1;
                let m = args.get(i).ok_or("--stt-model requires a value")?;
                stt_model = Some(m.clone());
            }
            "--stt-language" => {
                i += 1;
                let l = args.get(i).ok_or("--stt-language requires a value")?;
                stt_language = Some(l.clone());
            }
            "--s2s" => s2s = true,
            "--silent" => silent = true,
            "-v" | "--verbose" => verbose = true,
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
            "--llm-provider" => {
                i += 1;
                let name = args.get(i).ok_or("--llm-provider requires a value")?;
                llm_provider = Some(name.clone());
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
            "--llm-api-key" => {
                i += 1;
                let k = args.get(i).ok_or("--llm-api-key requires a value")?;
                llm_api_key = Some(k.clone());
            }
            "--llm-thinking" => llm_thinking = true,
            "--llm-temperature" => {
                i += 1;
                let s = args.get(i).ok_or("--llm-temperature requires a value")?;
                let v: f32 = s
                    .parse()
                    .map_err(|_| "--llm-temperature must be a number")?;
                if !(0.0..=2.0).contains(&v) {
                    return Err("--llm-temperature must be in 0..2".into());
                }
                llm_temperature = Some(v);
            }
            "--llm-num-ctx" => {
                i += 1;
                let s = args.get(i).ok_or("--llm-num-ctx requires a value")?;
                let v: u32 = s.parse().map_err(|_| "--llm-num-ctx must be a number")?;
                llm_num_ctx = Some(v);
            }
            "--tts-backend" => {
                i += 1;
                let name = args.get(i).ok_or("--tts-backend requires a value")?;
                tts_backend = Some(name.clone());
            }
            "--tts-model" => {
                i += 1;
                let p = args.get(i).ok_or("--tts-model requires a path")?;
                tts_model = Some(PathBuf::from(p));
            }
            "--tts-url" => {
                i += 1;
                let u = args.get(i).ok_or("--tts-url requires a URL")?;
                tts_url = Some(u.clone());
            }
            "--tts-voice" => {
                i += 1;
                let v = args.get(i).ok_or("--tts-voice requires a value")?;
                tts_voice = Some(v.clone());
            }
            "--tts-python" => {
                i += 1;
                let p = args.get(i).ok_or("--tts-python requires a path")?;
                tts_python = Some(p.clone());
            }
            "--vad-threshold" => {
                i += 1;
                let s = args.get(i).ok_or("--vad-threshold requires a value")?;
                let v: f32 = s.parse().map_err(|_| "--vad-threshold must be a number")?;
                if !(0.0..=1.0).contains(&v) {
                    return Err("--vad-threshold must be in 0..1".into());
                }
                vad_threshold = Some(v);
            }
            "--noise-ratio" => {
                i += 1;
                let s = args.get(i).ok_or("--noise-ratio requires a value")?;
                let v: f32 = s.parse().map_err(|_| "--noise-ratio must be a number")?;
                if v < 1.0 {
                    return Err("--noise-ratio must be >= 1.0".into());
                }
                noise_ratio = Some(v);
            }
            "--no-adaptive" => adaptive_noise = false,
            "--vad-silence" => {
                i += 1;
                let s = args.get(i).ok_or("--vad-silence requires a value")?;
                let v: u64 = s.parse().map_err(|_| "--vad-silence must be a number")?;
                vad_silence_ms = Some(v);
            }
            "--pause" => {
                i += 1;
                let s = args.get(i).ok_or("--pause requires a value")?;
                let v: u64 = s.parse().map_err(|_| "--pause must be a number")?;
                pause_ms = Some(v);
            }
            "--stop-word" => {
                i += 1;
                let w = args.get(i).ok_or("--stop-word requires a value")?;
                stop_word = Some(w.clone());
            }
            "--no-stop-word" => stop_word = Some(String::new()),
            "--barge-in" => barge_in = true,
            "--barge-threshold" => {
                i += 1;
                let s = args.get(i).ok_or("--barge-threshold requires a value")?;
                let v: f32 = s
                    .parse()
                    .map_err(|_| "--barge-threshold must be a number")?;
                if !(0.0..=1.0).contains(&v) {
                    return Err("--barge-threshold must be in 0..1".into());
                }
                barge_threshold = Some(v);
            }
            "--conversation-log" => {
                i += 1;
                let p = args.get(i).ok_or("--conversation-log requires a path")?;
                conversation_log = Some(p.clone());
            }
            "--system-prompt" => {
                i += 1;
                let p = args.get(i).ok_or("--system-prompt requires a value")?;
                system_prompt = Some(p.clone());
            }
            "--wizard" => wizard = true,
            "--template" => {
                i += 1;
                let n = args.get(i).ok_or("--template requires a name")?;
                template = Some(n.clone());
            }
            "--save-template" => {
                i += 1;
                let n = args.get(i).ok_or("--save-template requires a name")?;
                save_template = Some(n.clone());
            }
            "--edit-template" => {
                i += 1;
                let n = args.get(i).ok_or("--edit-template requires a name")?;
                edit_template = Some(n.clone());
            }
            "--list-templates" => list_templates = true,
            "--delete-template" => {
                i += 1;
                let n = args.get(i).ok_or("--delete-template requires a name")?;
                delete_template = Some(n.clone());
            }
            "--save-config" => save_config = true,
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
        device: opt_or_env(device, "AUDIOCHAT_DEVICE"),
        stt_backend: opt_or_env(stt_backend, "AUDIOCHAT_STT_BACKEND"),
        stt_url: opt_or_env(stt_url, "AUDIOCHAT_STT_URL"),
        stt_model: opt_or_env(stt_model, "AUDIOCHAT_STT_MODEL"),
        stt_language: opt_or_env(stt_language, "AUDIOCHAT_STT_LANGUAGE"),
        speak,
        tts_backend: opt_or_env(tts_backend, "AUDIOCHAT_TTS_BACKEND"),
        tts_model: tts_model
            .or_else(|| std::env::var("AUDIOCHAT_TTS_MODEL").ok().map(PathBuf::from)),
        tts_url: opt_or_env(tts_url, "AUDIOCHAT_TTS_URL"),
        tts_voice: opt_or_env(tts_voice, "AUDIOCHAT_TTS_VOICE"),
        tts_python: opt_or_env(tts_python, "AUDIOCHAT_PYTHON"),
        prompt,
        llm_provider: opt_or_env(llm_provider, "AUDIOCHAT_LLM_PROVIDER"),
        llm_model: opt_or_env(llm_model, "AUDIOCHAT_LLM_MODEL"),
        llm_url: opt_or_env(llm_url, "AUDIOCHAT_LLM_URL"),
        llm_api_key: opt_or_env(llm_api_key, "AUDIOCHAT_LLM_API_KEY"),
        llm_thinking: llm_thinking || env_flag("AUDIOCHAT_LLM_THINKING"),
        llm_temperature: opt_float_or_env(llm_temperature, "AUDIOCHAT_LLM_TEMPERATURE"),
        llm_num_ctx: opt_u32_or_env(llm_num_ctx, "AUDIOCHAT_LLM_NUM_CTX"),
        s2s,
        silent,
        verbose,
        vad_threshold,
        noise_ratio,
        adaptive_noise,
        vad_silence_ms,
        pause_ms: opt_u64_or_env(pause_ms, "AUDIOCHAT_PAUSE"),
        stop_word: opt_or_env(stop_word, "AUDIOCHAT_STOP_WORD"),
        barge_in,
        barge_threshold,
        conversation_log: opt_or_env(conversation_log, "AUDIOCHAT_CONVERSATION_LOG"),
        system_prompt: opt_or_env(system_prompt, "AUDIOCHAT_SYSTEM_PROMPT"),
        wizard,
        template,
        save_template,
        edit_template,
        list_templates,
        delete_template,
        save_config,
    })
}

pub(crate) const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
pub(crate) const OPENCODE_GO_URL: &str = "https://opencode.ai/zen/go/v1";
pub(crate) const OPENCODE_GO_DEFAULT_MODEL: &str = "deepseek-v4.1-flash";
pub(crate) const DEFAULT_AUDIOCPP_URL: &str = "http://localhost:8080";
pub(crate) const DEFAULT_AUDIOCPP_STT_MODEL: &str = "moonshine-asr";
pub(crate) const DEFAULT_AUDIOCPP_MODEL: &str = "qwen3-tts";
pub(crate) const DEFAULT_AUDIOCPP_VOICE: &str = "Serena";
/// The configured STT backend name (default: whisper).
fn stt_backend(opts: &Opts) -> &str {
    opts.stt_backend.as_deref().unwrap_or("whisper")
}

/// Build the speech recognizer selected by `--stt-backend`.
fn build_stt(
    opts: &Opts,
) -> Result<Box<dyn SpeechRecognizer>, Box<dyn std::error::Error + Send + Sync>> {
    match stt_backend(opts) {
        "whisper" => {
            let model = opts.model.as_ref().ok_or(
                "the whisper STT backend requires a whisper model path (positional argument)",
            )?;
            Ok(Box::new(WhisperRecognizer::new(model)?))
        }
        "audiocpp" => {
            let base = opts
                .stt_url
                .clone()
                .unwrap_or_else(|| DEFAULT_AUDIOCPP_URL.to_string());
            let model = opts
                .stt_model
                .clone()
                .unwrap_or_else(|| DEFAULT_AUDIOCPP_STT_MODEL.to_string());
            let stt = AudioCppStt::with_base(base, model).with_language(opts.stt_language.clone());
            Ok(Box::new(stt))
        }
        other => {
            Err(format!("unknown --stt-backend '{other}' (expected whisper or audiocpp)").into())
        }
    }
}

/// The configured LLM provider name (default: ollama).
fn llm_provider(opts: &Opts) -> &str {
    opts.llm_provider.as_deref().unwrap_or("ollama")
}

/// The configured TTS backend name (default: piper).
fn tts_backend(opts: &Opts) -> &str {
    opts.tts_backend.as_deref().unwrap_or("piper")
}

/// Build the LLM client selected by `--llm-provider`.
fn build_llm(opts: &Opts) -> Result<Box<dyn Llm>, Box<dyn std::error::Error + Send + Sync>> {
    match llm_provider(opts) {
        "ollama" => {
            let base = opts
                .llm_url
                .clone()
                .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
            let model = opts.llm_model.clone().unwrap_or_default();
            let mut client = Ollama::with_base(base, model);
            if let Some(p) = &opts.system_prompt {
                client = client.with_system_prompt(Some(p.clone()));
            }
            if let Some(t) = opts.llm_temperature {
                client = client.with_temperature(Some(t));
            }
            if let Some(n) = opts.llm_num_ctx {
                client = client.with_num_ctx(Some(n));
            }
            Ok(Box::new(client))
        }
        provider @ ("openai" | "opencode-go") => {
            let base = match opts.llm_url.clone() {
                Some(url) => url,
                None if provider == "opencode-go" => OPENCODE_GO_URL.to_string(),
                None => return Err("--llm-provider openai requires --llm-url".into()),
            };
            let model = opts.llm_model.clone().unwrap_or_else(|| {
                if provider == "opencode-go" {
                    OPENCODE_GO_DEFAULT_MODEL.to_string()
                } else {
                    String::new()
                }
            });
            let mut client = OpenAiCompatible::with_base(base, model);
            let key = opts.llm_api_key.clone().or_else(|| {
                if provider == "opencode-go" {
                    opencode_go_api_key()
                } else {
                    None
                }
            });
            if let Some(key) = key {
                client = client.with_api_key(key);
            }
            if let Some(p) = &opts.system_prompt {
                client = client.with_system_prompt(Some(p.clone()));
            }
            if let Some(t) = opts.llm_temperature {
                client = client.with_temperature(Some(t));
            }
            // Thinking is off for opencode-go unless explicitly requested; other
            // OpenAI-compatible servers keep their own default.
            let thinking = if opts.llm_thinking {
                Some(true)
            } else if provider == "opencode-go" {
                Some(false)
            } else {
                None
            };
            client = client.with_thinking(thinking);
            Ok(Box::new(client))
        }
        other => Err(format!(
            "unknown --llm-provider '{other}' (expected ollama, openai, or opencode-go)"
        )
        .into()),
    }
}

/// Build the TTS engine selected by `--tts-backend`.
fn build_tts(
    opts: &Opts,
) -> Result<Box<dyn TextToSpeech>, Box<dyn std::error::Error + Send + Sync>> {
    match tts_backend(opts) {
        "piper" => {
            let model = opts
                .tts_model
                .as_ref()
                .ok_or("--speak/--s2s requires --tts-model <piper.onnx> for the piper backend")?;
            let python = opts
                .tts_python
                .clone()
                .unwrap_or_else(|| "python3".to_string());
            Ok(Box::new(Piper::with_python(python, model)?))
        }
        "audiocpp" => {
            let base = opts
                .tts_url
                .clone()
                .unwrap_or_else(|| DEFAULT_AUDIOCPP_URL.to_string());
            let model = opts
                .tts_model
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| DEFAULT_AUDIOCPP_MODEL.to_string());
            let voice = opts
                .tts_voice
                .clone()
                .unwrap_or_else(|| DEFAULT_AUDIOCPP_VOICE.to_string());
            Ok(Box::new(
                AudioCppTts::with_base(base, model).with_voice(voice),
            ))
        }
        other => {
            Err(format!("unknown --tts-backend '{other}' (expected piper or audiocpp)").into())
        }
    }
}

fn run_llm(opts: &Opts) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = build_llm(opts)?;
    if client.model().is_empty() {
        return Err("--prompt requires --llm-model <name>".into());
    }
    let prompt = opts.prompt.as_deref().unwrap_or_default();

    println!(
        "audiochat: asking {} ({})...",
        llm_provider(opts),
        client.model()
    );
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
    let mut tts = build_tts(opts)?;
    let text = opts.speak.as_deref().unwrap_or_default();
    let pcm = tts.synthesize(text)?;
    eprintln!(
        "audiochat: synthesized {} samples ({} ms)",
        pcm.len(),
        pcm.len() * 1000 / 16_000
    );
    let sink = AudioSink::new(AudioConfig::default())?;
    sink.push(pcm);
    sink.drain();
    Ok(())
}

fn run_stt(opts: &Opts) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut recognizer = build_stt(opts)?;
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
    let llm = build_llm(opts)?;
    if llm.model().is_empty() {
        return Err("--s2s requires --llm-model <name>".into());
    }
    let tts = build_tts(opts)?;
    let stt = build_stt(opts)?;

    let mut pipeline = Pipeline::new(llm, tts);
    pipeline.speak_replies = !opts.silent;
    pipeline.verbose = opts.verbose;
    if let Some(path) = &opts.conversation_log {
        let logger = ConversationLogger::open(std::path::Path::new(path))
            .map_err(|e| format!("failed to open conversation log {path}: {e}"))?;
        pipeline = pipeline.with_conversation_log(Some(logger));
    }
    let mic = MicCapture::start_with_device(AudioConfig::default(), opts.device.as_deref())?;

    let mut session = Session::new();
    if let Some(t) = opts.vad_threshold {
        session = session.with_vad_threshold(t);
    }
    if let Some(r) = opts.noise_ratio {
        session = session.with_noise_ratio(r);
    }
    if !opts.adaptive_noise {
        session = session.with_adaptive_noise(false);
    }
    if let Some(ms) = opts.vad_silence_ms {
        session = session.with_vad_max_silence(ms);
    }
    if let Some(ms) = opts.pause_ms {
        session = session.with_pause(ms);
    }
    if let Some(word) = &opts.stop_word {
        if word.is_empty() {
            session = session.with_stop_word(None);
        } else {
            session = session.with_stop_word(Some(word.clone()));
        }
    }
    session = session.with_barge_in(opts.barge_in);
    if let Some(t) = opts.barge_threshold {
        session = session.with_barge_in_threshold(t);
    }
    session = session.with_verbose(opts.verbose);

    let stop = Arc::new(AtomicBool::new(false));
    let stop_h = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        stop_h.store(true, Ordering::Relaxed);
    })
    .map_err(|e| format!("failed to install signal handler: {e}"))?;

    println!("audiochat: speech-to-speech mode. Speak to ask; Ctrl-C to stop.");
    session.run(mic, stt, pipeline, Some(stop))?;
    println!("\naudiochat: stopped.");
    Ok(())
}

/// Whether a run mode was requested explicitly (so a template shouldn't decide).
fn has_explicit_mode(opts: &Opts) -> bool {
    opts.s2s || opts.prompt.is_some() || opts.speak.is_some()
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let mut opts = parse_args(&args).map_err(Box::<dyn std::error::Error + Send + Sync>::from)?;

    // Template management needs no backend, so handle it before anything runs.
    if opts.list_templates {
        let names = template::list_templates();
        if names.is_empty() {
            println!(
                "no templates yet (dir: {})",
                template::templates_dir().display()
            );
        } else {
            for name in names {
                println!("{name}");
            }
        }
        return Ok(());
    }
    if let Some(name) = opts.delete_template.take() {
        template::delete_template(&name)?;
        println!("deleted template '{name}'");
        return Ok(());
    }

    // The wizard produces a template which may then be saved and/or applied.
    if opts.wizard || opts.edit_template.is_some() {
        let baseline = match &opts.edit_template {
            Some(name) => template::Template::load(name)?,
            None => template::Template::default(),
        };
        match wizard::run_wizard(&baseline)? {
            wizard::WizardOutcome::Cancelled => {
                println!("cancelled; nothing changed.");
                return Ok(());
            }
            wizard::WizardOutcome::Configured { template, run_now } => {
                template.apply_to_opts(&mut opts);
                if run_now {
                    opts.s2s = true;
                    opts.prompt = None;
                    opts.speak = None;
                } else {
                    return Ok(());
                }
            }
        }
    } else if let Some(name) = opts.template.take() {
        let loaded = template::Template::load(&name)?;
        loaded.apply_to_opts(&mut opts);
        // A template that configures a full stack implies speech-to-speech when
        // the user didn't pick a mode explicitly.
        if !has_explicit_mode(&opts) && loaded.effective_mode() == Some("s2s") {
            opts.s2s = true;
            eprintln!("audiochat: template '{name}' -> speech-to-speech mode (--s2s)");
        }
    }

    // Global defaults (hosts, model ids, voice) from the config file are the
    // lowest-priority layer: flags, env, and templates already filled their
    // fields, so this only supplies what is still unset.
    if let Some(defaults) = config::load_defaults()? {
        defaults.apply_to_opts(&mut opts);
    }

    if opts.save_config {
        let path = config::save_defaults(&template::Template::from_opts(&opts))?;
        println!("saved defaults to {}", path.display());
        // Persisting defaults shouldn't start capturing from the mic.
        if !opts.s2s && opts.prompt.is_none() && opts.speak.is_none() {
            return Ok(());
        }
    }

    if let Some(name) = opts.save_template.take() {
        let path = template::Template::from_opts(&opts).save(&name)?;
        println!("saved template '{name}' to {}", path.display());
        // Saving settings on their own shouldn't start capturing from the mic.
        if !opts.s2s && opts.prompt.is_none() && opts.speak.is_none() {
            return Ok(());
        }
    }

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
