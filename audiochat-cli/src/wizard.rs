//! Interactive setup wizard: pick STT, TTS, LLM, and a system prompt, then save
//! the result as a named template and/or launch a speech-to-speech session.

use inquire::{Confirm, Editor, InquireError, Select, Text};

use crate::discover::{self, ModelOption};
use crate::template::{AudioTemplate, LlmTemplate, SttTemplate, Template, TtsTemplate};
use crate::{
    DEFAULT_AUDIOCPP_MODEL, DEFAULT_AUDIOCPP_STT_MODEL, DEFAULT_AUDIOCPP_URL,
    DEFAULT_AUDIOCPP_VOICE, DEFAULT_OLLAMA_URL, OPENCODE_GO_DEFAULT_MODEL, OPENCODE_GO_URL,
};

const DEFAULT_WHISPER_MODEL: &str = "models/ggml-tiny.en.bin";
const DEFAULT_PIPER_MODEL: &str = "models/voices/en_US-lessac-medium.onnx";
const DEFAULT_OLLAMA_MODEL: &str = "gemma4:e2b";
const DEFAULT_PYTHON: &str = "python3";

/// Menu entry that drops into free-text entry.
const MANUAL: &str = "Type manually…";

/// The outcome of a wizard run.
pub enum WizardOutcome {
    /// The user backed out (Ctrl-C / Esc) without changing anything.
    Cancelled,
    /// A template was configured. `run_now` launches a speech-to-speech session.
    Configured {
        template: Box<Template>,
        run_now: bool,
    },
}

/// Run the wizard, using `baseline` as the initial values (e.g. a template
/// being edited, or `Default::default()` for a fresh setup).
pub fn run_wizard(baseline: &Template) -> Result<WizardOutcome, String> {
    match wizard_inner(baseline) {
        Ok(outcome) => Ok(outcome),
        Err(InquireError::OperationCanceled) | Err(InquireError::OperationInterrupted) => {
            Ok(WizardOutcome::Cancelled)
        }
        Err(e) => Err(format!("wizard failed: {e}")),
    }
}

fn wizard_inner(baseline: &Template) -> Result<WizardOutcome, InquireError> {
    println!("audiochat setup wizard — Ctrl-C to cancel at any point.\n");

    let stt = ask_stt(&baseline.stt)?;
    let tts = ask_tts(&baseline.tts)?;
    let llm = ask_llm(&baseline.llm)?;
    let system_prompt = ask_system_prompt(baseline.system_prompt.as_deref())?;
    let device = text_optional("Input device (blank = default)", baseline.device.as_deref())?;
    let audio = ask_advanced(&baseline.audio)?;

    let template = Template {
        name: baseline.name.clone(),
        description: baseline.description.clone(),
        mode: Some("s2s".into()),
        stt,
        tts,
        llm,
        system_prompt,
        device,
        audio,
    };

    println!("\n{}", describe(&template));

    let actions = [
        "Save as template and run (speech-to-speech)",
        "Save as template only",
        "Run once without saving",
        "Cancel",
    ];
    let action = select_index("What next?", &actions, 0)?;
    let save = matches!(action, 0 | 1);
    let run_now = matches!(action, 0 | 2);

    let mut template = template;
    if save {
        let suggested = baseline
            .name
            .clone()
            .unwrap_or_else(|| "my-setup".to_string());
        let name = Text::new("Template name:")
            .with_default(&suggested)
            .with_help_message("letters, digits, '-' and '_'")
            .prompt()?;
        match template.save(name.trim()) {
            Ok(path) => println!("saved: {}", path.display()),
            Err(e) => eprintln!("could not save template: {e}"),
        }
        template.name = Some(name.trim().to_string());
    }

    Ok(WizardOutcome::Configured {
        template: Box::new(template),
        run_now,
    })
}

fn ask_stt(baseline: &SttTemplate) -> Result<SttTemplate, InquireError> {
    let backends = ["whisper (local)", "audiocpp (remote)"];
    let default = if baseline.backend.as_deref() == Some("audiocpp") {
        1
    } else {
        0
    };
    let idx = select_index("Speech-to-text backend:", &backends, default)?;

    if idx == 0 {
        let model = choose_model(
            "Whisper model:",
            discover::local_whisper_models(),
            baseline.model.as_deref(),
            Some(DEFAULT_WHISPER_MODEL),
            "found under models/ (*.bin)",
        )?;
        Ok(SttTemplate {
            backend: Some("whisper".into()),
            model: Some(model),
            url: None,
            language: None,
        })
    } else {
        let url = text(
            "audio.cpp base URL:",
            baseline.url.as_deref().or(Some(DEFAULT_AUDIOCPP_URL)),
        )?;
        eprintln!("querying {url}/v1/models for ASR models…");
        let model = choose_model(
            "ASR model:",
            discover::audiocpp_models(&url, "asr"),
            baseline.model.as_deref(),
            Some(DEFAULT_AUDIOCPP_STT_MODEL),
            "reported by the audio.cpp server",
        )?;
        let language = choose_language(baseline.language.as_deref())?;
        Ok(SttTemplate {
            backend: Some("audiocpp".into()),
            model: Some(model),
            url: Some(url),
            language,
        })
    }
}

fn ask_tts(baseline: &TtsTemplate) -> Result<TtsTemplate, InquireError> {
    let backends = ["piper (local)", "audiocpp (remote)"];
    let default = if baseline.backend.as_deref() == Some("audiocpp") {
        1
    } else {
        0
    };
    let idx = select_index("Text-to-speech backend:", &backends, default)?;

    if idx == 0 {
        let model = choose_model(
            "Piper voice:",
            discover::local_piper_voices(),
            baseline.model.as_deref(),
            Some(DEFAULT_PIPER_MODEL),
            "found under models/ (*.onnx)",
        )?;
        let python = text(
            "Python interpreter with piper-tts:",
            baseline.python.as_deref().or(Some(DEFAULT_PYTHON)),
        )?;
        Ok(TtsTemplate {
            backend: Some("piper".into()),
            model: Some(model),
            voice: None,
            instruct: None,
            url: None,
            python: Some(python),
        })
    } else {
        let url = text(
            "audio.cpp base URL:",
            baseline.url.as_deref().or(Some(DEFAULT_AUDIOCPP_URL)),
        )?;
        eprintln!("querying {url}/v1/models for TTS models…");
        let model = choose_model(
            "TTS model:",
            discover::audiocpp_models(&url, "tts"),
            baseline.model.as_deref(),
            Some(DEFAULT_AUDIOCPP_MODEL),
            "reported by the audio.cpp server",
        )?;
        eprintln!("querying {url}/v1/audio/voices for voices…");
        let voice = choose_model(
            "Voice/speaker:",
            discover::audiocpp_voices(&url, &model),
            baseline.voice.as_deref(),
            Some(DEFAULT_AUDIOCPP_VOICE),
            "built-in voices for the selected model",
        )?;
        let instruct = text_optional(
            "Style/emotion instruction (blank = none):",
            baseline.instruct.as_deref(),
        )?;
        Ok(TtsTemplate {
            backend: Some("audiocpp".into()),
            model: Some(model),
            voice: Some(voice),
            instruct,
            url: Some(url),
            python: None,
        })
    }
}

fn ask_llm(baseline: &LlmTemplate) -> Result<LlmTemplate, InquireError> {
    let providers = ["ollama", "openai (compatible)", "opencode-go"];
    let default = match baseline.provider.as_deref() {
        Some("openai") => 1,
        Some("opencode-go") => 2,
        _ => 0,
    };
    let idx = select_index("LLM provider:", &providers, default)?;

    let mut llm = LlmTemplate {
        provider: Some(["ollama", "openai", "opencode-go"][idx].to_string()),
        ..baseline.clone()
    };

    match idx {
        0 => {
            llm.url = Some(text(
                "Ollama base URL:",
                baseline.url.as_deref().or(Some(DEFAULT_OLLAMA_URL)),
            )?);
            let url = llm.url.clone().unwrap_or_default();
            eprintln!("querying {url}/api/tags for installed models…");
            llm.model = Some(choose_model(
                "Ollama model:",
                discover::ollama_models(&url),
                baseline.model.as_deref(),
                Some(DEFAULT_OLLAMA_MODEL),
                "installed on the Ollama server",
            )?);
            llm.api_key = None;
        }
        1 => {
            llm.url = Some(text(
                "OpenAI-compatible base URL (e.g. https://host/v1):",
                baseline.url.as_deref(),
            )?);
            llm.api_key = text_optional(
                "API key (blank to use AUDIOCHAT_LLM_API_KEY env):",
                baseline.api_key.as_deref(),
            )?;
            let url = llm.url.clone().unwrap_or_default();
            eprintln!("querying {url}/models…");
            llm.model = Some(choose_model(
                "Model id:",
                discover::openai_models(&url, llm.api_key.as_deref()),
                baseline.model.as_deref(),
                None,
                "reported by the server",
            )?);
            llm.thinking = Some(ask_thinking(baseline.thinking.unwrap_or(false))?);
        }
        _ => {
            llm.url = Some(text(
                "opencode-go base URL:",
                baseline.url.as_deref().or(Some(OPENCODE_GO_URL)),
            )?);
            let key = baseline
                .api_key
                .clone()
                .or_else(audiochat_llm::opencode_go_api_key);
            let url = llm.url.clone().unwrap_or_default();
            eprintln!("querying {url}/models…");
            llm.model = Some(choose_model(
                "Model id:",
                discover::openai_models(&url, key.as_deref()),
                baseline.model.as_deref(),
                Some(OPENCODE_GO_DEFAULT_MODEL),
                "available on your subscription",
            )?);
            llm.api_key = None;
            llm.thinking = Some(ask_thinking(baseline.thinking.unwrap_or(false))?);
        }
    }

    llm.temperature = optional_number::<f32>(
        "Sampling temperature 0..2 (blank = model default):",
        baseline.temperature,
        "lower is more deterministic",
    )?;
    llm.num_ctx = optional_number::<u32>(
        "Context window in tokens (blank = model default):",
        baseline.num_ctx,
        "e.g. 4096",
    )?;
    Ok(llm)
}

fn ask_thinking(default: bool) -> Result<bool, InquireError> {
    Confirm::new("Enable model thinking/reasoning mode?")
        .with_default(default)
        .with_help_message("off is faster and better for speech")
        .prompt()
}

fn ask_system_prompt(baseline: Option<&str>) -> Result<Option<String>, InquireError> {
    let current = baseline.unwrap_or(audiochat_llm::DEFAULT_SYSTEM_PROMPT);
    let choices = [
        "Keep the default assistant prompt",
        "Edit the prompt in $EDITOR",
        "Type a single-line prompt",
        "No system prompt",
    ];
    let idx = select_index("System prompt:", &choices, 0)?;
    match idx {
        0 => Ok(Some(current.to_string())),
        1 => match Editor::new("Editing system prompt")
            .with_predefined_text(current)
            .prompt()
        {
            Ok(text) => {
                let text = text.trim().to_string();
                Ok(if text.is_empty() { None } else { Some(text) })
            }
            // No $EDITOR available: fall back to a single-line prompt.
            Err(InquireError::OperationCanceled) | Err(InquireError::OperationInterrupted) => {
                Err(InquireError::OperationCanceled)
            }
            Err(_) => {
                let text = Text::new("System prompt:").with_default(current).prompt()?;
                let text = text.trim().to_string();
                Ok(if text.is_empty() { None } else { Some(text) })
            }
        },
        2 => {
            let text = Text::new("System prompt:").with_default(current).prompt()?;
            let text = text.trim().to_string();
            Ok(if text.is_empty() { None } else { Some(text) })
        }
        _ => Ok(None),
    }
}

fn ask_advanced(baseline: &AudioTemplate) -> Result<AudioTemplate, InquireError> {
    let customize = Confirm::new("Configure advanced audio / turn-taking options?")
        .with_default(false)
        .prompt()?;
    if !customize {
        return Ok(baseline.clone());
    }

    let mut audio = baseline.clone();
    audio.vad_threshold = optional_number::<f32>(
        "VAD threshold 0..1 (blank = 0.02):",
        baseline.vad_threshold,
        "minimum RMS to count as speech",
    )?;
    audio.noise_ratio = optional_number::<f32>(
        "Adaptive noise multiplier (blank = 2.0):",
        baseline.noise_ratio,
        ">= 1.0",
    )?;
    audio.adaptive_noise = Some(
        Confirm::new("Track background noise adaptively?")
            .with_default(baseline.adaptive_noise.unwrap_or(true))
            .prompt()?,
    );
    audio.vad_silence_ms = optional_number::<u64>(
        "VAD trailing silence in ms (blank = 600):",
        baseline.vad_silence_ms,
        "how long a pause ends a segment",
    )?;
    audio.pause_ms = optional_number::<u64>(
        "Endpointer pause in ms (blank = same as silence):",
        baseline.pause_ms,
        "how long to hold a mid-sentence transcript",
    )?;
    audio.stop_word = text_optional(
        "Spoken stop word (blank = 'stop'):",
        baseline.stop_word.as_deref(),
    )?;
    audio.barge_in = Some(
        Confirm::new("Allow barge-in (interrupt the reply by speaking)?")
            .with_default(baseline.barge_in.unwrap_or(false))
            .with_help_message("headphones recommended")
            .prompt()?,
    );
    if audio.barge_in == Some(true) {
        audio.barge_threshold = optional_number::<f32>(
            "Barge-in RMS threshold 0..1 (blank = 0.06):",
            baseline.barge_threshold,
            "raise if the assistant interrupts itself",
        )?;
    }
    Ok(audio)
}

/// A short, key=value description of a configured template.
fn describe(t: &Template) -> String {
    let stt = t.stt.backend.as_deref().unwrap_or("whisper");
    let tts = t.tts.backend.as_deref().unwrap_or("piper");
    let llm = t.llm.provider.as_deref().unwrap_or("ollama");
    let prompt = match &t.system_prompt {
        Some(p) => {
            let one_line: String = p.split_whitespace().collect::<Vec<_>>().join(" ");
            if one_line.chars().count() > 60 {
                format!("{}…", one_line.chars().take(60).collect::<String>())
            } else {
                one_line
            }
        }
        None => "(none)".to_string(),
    };
    let api_key = if t.llm.api_key.is_some() {
        " (key saved in template)"
    } else {
        ""
    };
    format!(
        "Summary\n  STT: {stt}{}\n  TTS: {tts}{}\n  LLM: {llm}{api_key}{}\n  Prompt: {prompt}{}",
        model_suffix(stt, &t.stt.model, &t.stt.url),
        model_suffix(tts, &t.tts.model, &t.tts.url),
        model_suffix(llm, &t.llm.model, &t.llm.url),
        t.device
            .as_deref()
            .map(|d| format!("\n  Device: {d}"))
            .unwrap_or_default(),
    )
}

fn model_suffix(backend: &str, model: &Option<String>, url: &Option<String>) -> String {
    let mut parts = Vec::new();
    if let Some(m) = model {
        if !m.is_empty() {
            parts.push(m.clone());
        }
    }
    if backend == "audiocpp" {
        if let Some(u) = url {
            if !u.is_empty() {
                parts.push(u.clone());
            }
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" ({})", parts.join(", "))
    }
}

fn select_index(msg: &str, options: &[&str], default: usize) -> Result<usize, InquireError> {
    let choices: Vec<String> = options.iter().map(|s| (*s).to_string()).collect();
    let cursor = default.min(choices.len().saturating_sub(1));
    Select::new(msg, choices)
        .with_starting_cursor(cursor)
        .prompt()
        .map(|picked| options.iter().position(|o| *o == picked).unwrap_or(cursor))
}

/// Choose a model id from a discovered list, with a "Type manually…" escape
/// hatch. Falls back to free text when nothing was discovered.
fn choose_model(
    msg: &str,
    discovered: Vec<ModelOption>,
    current: Option<&str>,
    fallback: Option<&str>,
    help: &str,
) -> Result<String, InquireError> {
    if discovered.is_empty() {
        return text(msg, current.or(fallback));
    }
    let mut choices: Vec<String> = discovered.iter().map(ModelOption::label).collect();
    choices.push(MANUAL.to_string());
    let cursor = current
        .and_then(|c| discovered.iter().position(|m| m.id == c))
        .unwrap_or(0);
    let picked = Select::new(msg, choices)
        .with_starting_cursor(cursor)
        .with_page_size(14)
        .with_help_message(help)
        .prompt()?;
    if picked == MANUAL {
        return text(&format!("{msg} (id)"), current.or(fallback));
    }
    Ok(discovered
        .into_iter()
        .find(|m| m.label() == picked)
        .map(|m| m.id)
        .unwrap_or(picked))
}

/// Pick from a fixed list of strings, with a "Type manually…" escape hatch.
fn select_choice(
    msg: &str,
    options: &[&str],
    current: Option<&str>,
    help: Option<&str>,
) -> Result<String, InquireError> {
    let mut choices: Vec<String> = options.iter().map(|s| (*s).to_string()).collect();
    choices.push(MANUAL.to_string());
    let cursor = current
        .and_then(|c| options.iter().position(|o| *o == c))
        .unwrap_or(0);
    let mut prompt = Select::new(msg, choices)
        .with_starting_cursor(cursor)
        .with_page_size(14);
    if let Some(h) = help {
        prompt = prompt.with_help_message(h);
    }
    prompt.prompt()
}

/// Choose an audio.cpp recognition language; `auto-detect` maps to `None`.
fn choose_language(current: Option<&str>) -> Result<Option<String>, InquireError> {
    const AUTO: &str = "auto-detect";
    let options = [
        AUTO, "en-US", "en-GB", "de-DE", "fr-FR", "es-ES", "it-IT", "pl-PL", "ja-JP", "zh-CN",
    ];
    let picked = select_choice(
        "Recognition language:",
        &options,
        current,
        Some("omitted = server auto-detect"),
    )?;
    if picked == AUTO {
        Ok(None)
    } else if picked == MANUAL {
        text_optional("Language code (e.g. en-US):", current)
    } else {
        Ok(Some(picked))
    }
}
/// Prompt for a required string, pre-filled with `default` when present.
fn text(msg: &str, default: Option<&str>) -> Result<String, InquireError> {
    let fallback = default.map(str::to_string);
    let mut prompt = Text::new(msg);
    if let Some(d) = &fallback {
        if !d.is_empty() {
            prompt = prompt.with_default(d.as_str());
        }
    }
    prompt.prompt()
}

/// Prompt for an optional string; blank yields `None`.
fn text_optional(msg: &str, default: Option<&str>) -> Result<Option<String>, InquireError> {
    let value = text(msg, default)?.trim().to_string();
    Ok(if value.is_empty() { None } else { Some(value) })
}

/// Prompt for an optional number; blank yields `None`, invalid input re-prompts.
fn optional_number<T>(msg: &str, default: Option<T>, help: &str) -> Result<Option<T>, InquireError>
where
    T: std::str::FromStr + std::fmt::Display + Copy,
{
    let fallback = default.map(|d| d.to_string());
    loop {
        let mut prompt = Text::new(msg).with_help_message(help);
        if let Some(d) = &fallback {
            prompt = prompt.with_default(d.as_str());
        }
        let input = prompt.prompt()?;
        let input = input.trim();
        if input.is_empty() {
            return Ok(None);
        }
        if let Ok(value) = input.parse::<T>() {
            return Ok(Some(value));
        }
        eprintln!("'{input}' is not a valid number; try again.");
    }
}
