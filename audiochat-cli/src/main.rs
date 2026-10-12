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
    AudioConfig, AudioSink, ConversationLogger, EnergyVad, Llm, Memory, MemoryFormat, MemoryLimits,
    MicCapture, Pipeline, Session, SpeechRecognizer, TextToSpeech,
};
use audiochat_llm::{
    opencode_go_api_key, Ollama, OpenAiCompatible, DEFAULT_SYSTEM_PROMPT, FILLER_SYSTEM_PROMPT,
};
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
  --tts-instruct TEXT audiocpp style/emotion/pacing instruction, e.g. speak
                      softly and slowly (Qwen3-TTS CustomVoice/VoiceDesign).
  --llm-provider NAME LLM backend: ollama (default), openai, or opencode-go.
  --llm-model NAME    Model name (Ollama model, or OpenAI-compatible model id).
  --llm-url URL       Base URL (Ollama: http://localhost:11434; opencode-go:
                      https://opencode.ai/zen/go/v1).
  --llm-api-key KEY   Bearer token for OpenAI-compatible backends (or env).
  --llm-thinking      Enable reasoning/thinking mode (default: off).
  --thinking-filler   With --llm-thinking, speak a short speculative holding
                      line while the main model thinks, pre-empted by the real
                      reply. Needs an OpenAI-compatible provider.
  --filler-model NAME Model for the holding line (openai-compatible backends);
                      opencode-go defaults to deepseek-v4.1-flash.
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
  --memory            Load persistent memory into the prompt and rewrite it from
                      the session at exit (default: off).
  --no-memory         Disable persistent memory.
  --memory-profile N  Memory profile to read/write (default: template name or
                      'default'). Stores live in <config>/memory/<profile>.json.
  --memory-format F   Inject memory as 'md' (default) or 'json'.
  --memory-model N    Summarizer model (default: holding-line model, then the
                      provider default).
  --memory-recent N   Recent-chat summaries retained (default: 8).
  --memory-fundamental-chars N  Durable-facts character cap (default: 1200).
  --memory-todo-ttl-days N      Follow-up expiry in days (default: 14).
  --memory-todo-max N Follow-ups retained (default: 10).
  --show-memory       Print the profile's memory (Markdown) and exit.
  --forget-memory     Delete the profile's memory store and exit.
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
  AUDIOCHAT_TTS_INSTRUCT  audiocpp style/emotion instruction
  AUDIOCHAT_PYTHON        python interpreter for piper (default python3)
  AUDIOCHAT_LLM_PROVIDER  llm backend: ollama, openai, or opencode-go
  AUDIOCHAT_LLM_MODEL     model name
  AUDIOCHAT_LLM_URL       base URL
  AUDIOCHAT_LLM_API_KEY   bearer token (falls back to OPENCODE_API_KEY, then
                          opencode's auth.json for the opencode-go provider)
  AUDIOCHAT_LLM_THINKING  set to 1/true to enable thinking mode
  AUDIOCHAT_THINKING_FILLER set to 1/true to enable the speculative holding line
  AUDIOCHAT_FILLER_MODEL  model id for the holding line
  AUDIOCHAT_PAUSE         endpointer pause window (ms)
  AUDIOCHAT_STOP_WORD     spoken stop word (default: stop)
  AUDIOCHAT_MEMORY        set to 1/true to enable persistent memory
  AUDIOCHAT_MEMORY_PROFILE   memory profile name
  AUDIOCHAT_MEMORY_FORMAT    memory injection format: md or json
  AUDIOCHAT_MEMORY_MODEL     summarizer model id
  AUDIOCHAT_MEMORY_RECENT    recent-chat summaries retained
  AUDIOCHAT_MEMORY_FUNDAMENTAL_CHARS  durable-facts character cap
  AUDIOCHAT_MEMORY_TODO_TTL_DAYS      follow-up expiry in days
  AUDIOCHAT_MEMORY_TODO_MAX  follow-ups retained";

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
    tts_instruct: Option<String>,
    tts_python: Option<String>,
    prompt: Option<String>,
    llm_provider: Option<String>,
    llm_model: Option<String>,
    llm_url: Option<String>,
    llm_api_key: Option<String>,
    llm_thinking: bool,
    thinking_filler: bool,
    filler_model: Option<String>,
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
    memory_enabled: Option<bool>,
    memory_profile: Option<String>,
    memory_format: Option<String>,
    memory_model: Option<String>,
    memory_recent: Option<usize>,
    memory_fundamental_chars: Option<usize>,
    memory_todo_ttl_days: Option<u64>,
    memory_todo_max: Option<usize>,
    show_memory: bool,
    forget_memory: bool,
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

/// Resolve a CLI usize flag, falling back to a parsed environment variable.
fn opt_usize_or_env(cli: Option<usize>, env: &str) -> Option<usize> {
    cli.or_else(|| std::env::var(env).ok().and_then(|s| s.parse().ok()))
}

/// Whether a truthy boolean env var is set ("1", "true", "yes", "on").
fn env_flag(env: &str) -> bool {
    matches!(
        std::env::var(env).ok().map(|s| s.trim().to_ascii_lowercase()),
        Some(v) if matches!(v.as_str(), "1" | "true" | "yes" | "on")
    )
}

/// Parse a boolean env var: truthy -> `Some(true)`, falsy -> `Some(false)`.
fn env_bool(env: &str) -> Option<bool> {
    match std::env::var(env)
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
    {
        Some(v) if matches!(v.as_str(), "1" | "true" | "yes" | "on") => Some(true),
        Some(v) if matches!(v.as_str(), "0" | "false" | "no" | "off") => Some(false),
        _ => None,
    }
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
    let mut tts_instruct: Option<String> = None;
    let mut tts_python: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut llm_provider: Option<String> = None;
    let mut llm_model: Option<String> = None;
    let mut llm_url: Option<String> = None;
    let mut llm_api_key: Option<String> = None;
    let mut llm_thinking = false;
    let mut thinking_filler = false;
    let mut filler_model: Option<String> = None;
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
    let mut memory_enabled: Option<bool> = None;
    let mut memory_profile: Option<String> = None;
    let mut memory_format: Option<String> = None;
    let mut memory_model: Option<String> = None;
    let mut memory_recent: Option<usize> = None;
    let mut memory_fundamental_chars: Option<usize> = None;
    let mut memory_todo_ttl_days: Option<u64> = None;
    let mut memory_todo_max: Option<usize> = None;
    let mut show_memory = false;
    let mut forget_memory = false;
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
            "--thinking-filler" => thinking_filler = true,
            "--filler-model" => {
                i += 1;
                let name = args.get(i).ok_or("--filler-model requires a name")?;
                filler_model = Some(name.clone());
            }
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
            "--tts-instruct" => {
                i += 1;
                let t = args.get(i).ok_or("--tts-instruct requires text")?;
                tts_instruct = Some(t.clone());
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
            "--memory" => memory_enabled = Some(true),
            "--no-memory" => memory_enabled = Some(false),
            "--memory-profile" => {
                i += 1;
                let n = args.get(i).ok_or("--memory-profile requires a name")?;
                memory_profile = Some(n.clone());
            }
            "--memory-format" => {
                i += 1;
                let f = args.get(i).ok_or("--memory-format requires a value")?;
                if audiochat_core::MemoryFormat::parse(f).is_none() {
                    return Err("--memory-format must be 'md' or 'json'".into());
                }
                memory_format = Some(f.clone());
            }
            "--memory-model" => {
                i += 1;
                let n = args.get(i).ok_or("--memory-model requires a name")?;
                memory_model = Some(n.clone());
            }
            "--memory-recent" => {
                i += 1;
                let s = args.get(i).ok_or("--memory-recent requires a value")?;
                memory_recent = Some(s.parse().map_err(|_| "--memory-recent must be a number")?);
            }
            "--memory-fundamental-chars" => {
                i += 1;
                let s = args
                    .get(i)
                    .ok_or("--memory-fundamental-chars requires a value")?;
                memory_fundamental_chars = Some(
                    s.parse()
                        .map_err(|_| "--memory-fundamental-chars must be a number")?,
                );
            }
            "--memory-todo-ttl-days" => {
                i += 1;
                let s = args
                    .get(i)
                    .ok_or("--memory-todo-ttl-days requires a value")?;
                memory_todo_ttl_days = Some(
                    s.parse()
                        .map_err(|_| "--memory-todo-ttl-days must be a number")?,
                );
            }
            "--memory-todo-max" => {
                i += 1;
                let s = args.get(i).ok_or("--memory-todo-max requires a value")?;
                memory_todo_max = Some(
                    s.parse()
                        .map_err(|_| "--memory-todo-max must be a number")?,
                );
            }
            "--show-memory" => show_memory = true,
            "--forget-memory" => forget_memory = true,
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
        tts_instruct: opt_or_env(tts_instruct, "AUDIOCHAT_TTS_INSTRUCT"),
        tts_python: opt_or_env(tts_python, "AUDIOCHAT_PYTHON"),
        prompt,
        llm_provider: opt_or_env(llm_provider, "AUDIOCHAT_LLM_PROVIDER"),
        llm_model: opt_or_env(llm_model, "AUDIOCHAT_LLM_MODEL"),
        llm_url: opt_or_env(llm_url, "AUDIOCHAT_LLM_URL"),
        llm_api_key: opt_or_env(llm_api_key, "AUDIOCHAT_LLM_API_KEY"),
        llm_thinking: llm_thinking || env_flag("AUDIOCHAT_LLM_THINKING"),
        thinking_filler: thinking_filler || env_flag("AUDIOCHAT_THINKING_FILLER"),
        filler_model: opt_or_env(filler_model, "AUDIOCHAT_FILLER_MODEL"),
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
        memory_enabled: memory_enabled.or_else(|| env_bool("AUDIOCHAT_MEMORY")),
        memory_profile: opt_or_env(memory_profile, "AUDIOCHAT_MEMORY_PROFILE"),
        memory_format: opt_or_env(memory_format, "AUDIOCHAT_MEMORY_FORMAT"),
        memory_model: opt_or_env(memory_model, "AUDIOCHAT_MEMORY_MODEL"),
        memory_recent: opt_usize_or_env(memory_recent, "AUDIOCHAT_MEMORY_RECENT"),
        memory_fundamental_chars: opt_usize_or_env(
            memory_fundamental_chars,
            "AUDIOCHAT_MEMORY_FUNDAMENTAL_CHARS",
        ),
        memory_todo_ttl_days: opt_u64_or_env(
            memory_todo_ttl_days,
            "AUDIOCHAT_MEMORY_TODO_TTL_DAYS",
        ),
        memory_todo_max: opt_usize_or_env(memory_todo_max, "AUDIOCHAT_MEMORY_TODO_MAX"),
        show_memory,
        forget_memory,
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

/// Build the speculative "holding line" client, if enabled. It is a stateless,
/// non-thinking OpenAI-compatible client that fills the silence while the main
/// thinking model works; the first real reply token pre-empts it.
fn build_filler(opts: &Opts) -> Option<Box<dyn Llm + Send>> {
    if !opts.thinking_filler {
        return None;
    }
    if !opts.llm_thinking {
        eprintln!("audiochat: --thinking-filler needs --llm-thinking; ignoring");
        return None;
    }
    let provider = llm_provider(opts);
    if !matches!(provider, "openai" | "opencode-go") {
        eprintln!("audiochat: --thinking-filler needs an OpenAI-compatible provider; ignoring");
        return None;
    }
    let base = match opts.llm_url.clone() {
        Some(url) => url,
        None if provider == "opencode-go" => OPENCODE_GO_URL.to_string(),
        None => {
            eprintln!("audiochat: --thinking-filler needs --llm-url for the openai provider");
            return None;
        }
    };
    let model = opts
        .filler_model
        .clone()
        .or_else(|| (provider == "opencode-go").then(|| OPENCODE_GO_DEFAULT_MODEL.to_string()));
    let Some(model) = model.filter(|m| !m.trim().is_empty()) else {
        eprintln!("audiochat: --thinking-filler needs --filler-model for the openai provider");
        return None;
    };

    let mut client = OpenAiCompatible::with_base(base, model)
        .with_stateless(true)
        .with_thinking(Some(false))
        .with_system_prompt(Some(FILLER_SYSTEM_PROMPT))
        .with_temperature(Some(0.6));
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
    Some(Box::new(client))
}

/// Build the summarizer client used to rewrite memory at the end of a session.
/// It is stateless (OpenAI-compatible) or a fresh client (Ollama), so it never
/// pollutes the conversation history. Returns `None` when no model is available.
fn build_memory_llm(opts: &Opts) -> Option<Box<dyn Llm>> {
    let provider = llm_provider(opts);
    let system = audiochat_core::memory::MEMORY_SYSTEM_PROMPT.to_string();
    match provider {
        "ollama" => {
            let base = opts
                .llm_url
                .clone()
                .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
            let model = opts
                .memory_model
                .clone()
                .or_else(|| opts.llm_model.clone())?;
            if model.trim().is_empty() {
                return None;
            }
            Some(Box::new(
                Ollama::with_base(base, model)
                    .with_system_prompt(Some(system))
                    .with_temperature(Some(0.2)),
            ))
        }
        provider @ ("openai" | "opencode-go") => {
            let base = match opts.llm_url.clone() {
                Some(url) => url,
                None if provider == "opencode-go" => OPENCODE_GO_URL.to_string(),
                None => return None,
            };
            let model = opts
                .memory_model
                .clone()
                .or_else(|| opts.filler_model.clone())
                .or_else(|| {
                    (provider == "opencode-go").then(|| OPENCODE_GO_DEFAULT_MODEL.to_string())
                })?;
            if model.trim().is_empty() {
                return None;
            }
            let mut client = OpenAiCompatible::with_base(base, model)
                .with_stateless(true)
                .with_thinking(Some(false))
                .with_system_prompt(Some(system))
                .with_temperature(Some(0.2));
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
            Some(Box::new(client))
        }
        _ => None,
    }
}

/// Loaded memory state for the active run: the store injected into the prompt,
/// where to write the update, the enforced limits, and the summarizer.
struct ActiveMemory {
    profile: String,
    path: PathBuf,
    store: Memory,
    limits: MemoryLimits,
    llm: Option<Box<dyn Llm>>,
}

/// Resolve the memory profile: explicit name, else the template name, else
/// `default`.
fn resolve_memory_profile(opts: &Opts, template_name: Option<&str>) -> String {
    opts.memory_profile
        .clone()
        .filter(|p| !p.trim().is_empty())
        .or_else(|| template_name.map(str::to_string))
        .unwrap_or_else(|| "default".to_string())
}

/// Resolve the effective per-category limits from the flags/env.
fn memory_limits(opts: &Opts) -> MemoryLimits {
    let d = MemoryLimits::default();
    MemoryLimits {
        recent_max: opts.memory_recent.unwrap_or(d.recent_max),
        fundamental_max_chars: opts
            .memory_fundamental_chars
            .unwrap_or(d.fundamental_max_chars),
        todo_ttl_days: opts.memory_todo_ttl_days.unwrap_or(d.todo_ttl_days),
        todo_max: opts.memory_todo_max.unwrap_or(d.todo_max),
    }
}

/// Load the profile's memory and fold it into `opts.system_prompt` so it reaches
/// the next request. Runs after any `--save-*` so the augmented prompt is never
/// persisted back into a template or the defaults file.
fn setup_memory(opts: &mut Opts, template_name: Option<&str>) -> Result<ActiveMemory, String> {
    let profile = resolve_memory_profile(opts, template_name);
    let path = template::memory_path(&profile)?;
    let store = Memory::load(&path)?;
    let format = opts
        .memory_format
        .as_deref()
        .and_then(MemoryFormat::parse)
        .unwrap_or(MemoryFormat::Markdown);
    let block = store.render(format, audiochat_core::memory::now_unix());
    if !block.is_empty() {
        let base = opts
            .system_prompt
            .clone()
            .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
        opts.system_prompt = Some(format!("{base}\n\n{block}"));
    }
    let llm = build_memory_llm(opts);
    if llm.is_none() {
        eprintln!(
            "audiochat: memory is on but no summarizer model is available; \
             the existing memory will be injected but not updated"
        );
    }
    Ok(ActiveMemory {
        profile,
        path,
        store,
        limits: memory_limits(opts),
        llm,
    })
}

/// Rewrite and persist memory from this session's turns (best effort).
fn update_memory(mem: &ActiveMemory, turns: &[(String, String)]) {
    if turns.is_empty() {
        return;
    }
    let Some(llm) = mem.llm.as_ref() else {
        return;
    };
    let mut transcript = String::new();
    for (user, assistant) in turns {
        transcript.push_str(&format!("User: {user}\nAssistant: {assistant}\n"));
    }
    let transcript = audiochat_core::memory::trim_transcript(&transcript);
    let now = audiochat_core::memory::now_unix();
    match mem
        .store
        .update(llm.as_ref(), &transcript, &mem.limits, now)
    {
        Ok(updated) => match updated.save(&mem.path) {
            Ok(()) => println!("audiochat: memory '{}' updated.", mem.profile),
            Err(e) => eprintln!("audiochat: failed to save memory: {e}"),
        },
        Err(e) => eprintln!("audiochat: memory update failed: {e}"),
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
                AudioCppTts::with_base(base, model)
                    .with_voice(voice)
                    .with_instruct(opts.tts_instruct.clone()),
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

fn run_s2s(
    opts: &Opts,
    memory: Option<&ActiveMemory>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let llm = build_llm(opts)?;
    if llm.model().is_empty() {
        return Err("--s2s requires --llm-model <name>".into());
    }
    let tts = build_tts(opts)?;
    let stt = build_stt(opts)?;

    let mut pipeline = Pipeline::new(llm, tts).with_filler(build_filler(opts));
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
    let turns = session.run(mic, stt, pipeline, Some(stop))?;
    println!("\naudiochat: stopped.");
    if let Some(mem) = memory {
        update_memory(mem, &turns);
    }
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

    // Global defaults (hosts, model ids, voice) from ~/.config/audiochat/config.json.
    // They seed the wizard and act as the lowest-priority layer for a normal run.
    let global_defaults = config::load_defaults()?;

    // The active template's name, used as the default memory profile.
    let mut active_template_name: Option<String> = None;

    // The wizard produces a template which may then be saved and/or applied.
    if opts.wizard || opts.edit_template.is_some() {
        let mut baseline = match &opts.edit_template {
            Some(name) => template::Template::load(name)?,
            None => template::Template::default(),
        };
        // Seed the wizard from the global defaults (a template being edited wins).
        if let Some(defaults) = &global_defaults {
            baseline.fill_from(defaults);
        }
        match wizard::run_wizard(&baseline)? {
            wizard::WizardOutcome::Cancelled => {
                println!("cancelled; nothing changed.");
                return Ok(());
            }
            wizard::WizardOutcome::Configured { template, run_now } => {
                template.apply_to_opts(&mut opts);
                active_template_name = template.name.clone();
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
        active_template_name = Some(name.clone());
        // A template that configures a full stack implies speech-to-speech when
        // the user didn't pick a mode explicitly.
        if !has_explicit_mode(&opts) && loaded.effective_mode() == Some("s2s") {
            opts.s2s = true;
            eprintln!("audiochat: template '{name}' -> speech-to-speech mode (--s2s)");
        }
    }

    // Global defaults are the lowest-priority layer: flags, env, and templates
    // already filled their fields, so this only supplies what is still unset.
    if let Some(defaults) = &global_defaults {
        defaults.apply_to_opts(&mut opts);
    }

    // Memory inspection/reset is terminal, like template management.
    if opts.show_memory || opts.forget_memory {
        let profile = resolve_memory_profile(&opts, active_template_name.as_deref());
        let path = template::memory_path(&profile)?;
        if opts.forget_memory {
            match std::fs::remove_file(&path) {
                Ok(()) => println!("deleted memory '{profile}'"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    println!("no memory for profile '{profile}'");
                }
                Err(e) => {
                    return Err(format!("failed to delete {}: {e}", path.display()).into());
                }
            }
        } else {
            let store = Memory::load(&path)?;
            if store.is_empty() {
                println!("memory '{profile}' is empty ({})", path.display());
            } else {
                println!(
                    "{}",
                    store.render(MemoryFormat::Markdown, audiochat_core::memory::now_unix())
                );
            }
        }
        return Ok(());
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

    // Load persistent memory into the prompt. Done after any --save-* so the
    // augmented system prompt is never written back into a template.
    let memory = if opts.memory_enabled == Some(true) {
        Some(setup_memory(&mut opts, active_template_name.as_deref())?)
    } else {
        None
    };

    if opts.s2s {
        return run_s2s(&opts, memory.as_ref());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(args: &[&str]) -> Opts {
        let args: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
        parse_args(&args).unwrap()
    }

    #[test]
    fn parses_thinking_filler_flags() {
        let o = opts(&["--thinking-filler", "--filler-model", "flash"]);
        assert!(o.thinking_filler);
        assert_eq!(o.filler_model.as_deref(), Some("flash"));
    }

    #[test]
    fn filler_needs_thinking_enabled() {
        let o = opts(&["--thinking-filler", "--llm-provider", "opencode-go"]);
        assert!(build_filler(&o).is_none());
    }

    #[test]
    fn filler_needs_openai_compatible_provider() {
        let o = opts(&[
            "--thinking-filler",
            "--llm-thinking",
            "--llm-provider",
            "ollama",
        ]);
        assert!(build_filler(&o).is_none());
    }

    #[test]
    fn filler_defaults_to_flash_for_opencode_go() {
        let o = opts(&[
            "--thinking-filler",
            "--llm-thinking",
            "--llm-provider",
            "opencode-go",
        ]);
        assert!(build_filler(&o).is_some());
    }

    #[test]
    fn openai_filler_needs_a_model() {
        let o = opts(&[
            "--thinking-filler",
            "--llm-thinking",
            "--llm-provider",
            "openai",
            "--llm-url",
            "http://localhost:8000/v1",
        ]);
        assert!(build_filler(&o).is_none());
    }
}
