//! Named configuration templates for the audiochat CLI.
//!
//! A template captures the STT/TTS/LLM backends, the system prompt, and a few
//! audio/turn-taking knobs so a setup can be saved, listed, edited, and reloaded.
//! Templates are stored as JSON under `~/.config/audiochat/templates/<name>.json`.
//! The root can be overridden with `AUDIOCHAT_CONFIG_DIR` or `XDG_CONFIG_HOME`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::Opts;

/// A saved setup: which backends to use, how they're reached, and the prompt.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct Template {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Intended run mode: `s2s`, `llm`, `tts`, or `stt`. When absent it is
    /// inferred from the configured backends (see [`Template::effective_mode`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default)]
    pub stt: SttTemplate,
    #[serde(default)]
    pub tts: TtsTemplate,
    #[serde(default)]
    pub llm: LlmTemplate,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default)]
    pub audio: AudioTemplate,
    #[serde(default)]
    pub memory: MemoryTemplate,
}

/// Speech-to-text selection. `model` is the whisper `.bin` path for the whisper
/// backend, or the server-side model id for the audiocpp backend.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct SttTemplate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// Text-to-speech selection. `model` is the Piper `.onnx` path for the piper
/// backend, or the server-side model id for the audiocpp backend.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct TtsTemplate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// audiocpp style/emotion/pacing instruction (Qwen3-TTS CustomVoice/VoiceDesign).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruct: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python: Option<String>,
}

/// LLM selection. `api_key` is optional and, when present, is stored in
/// plaintext; prefer the environment or opencode's `auth.json`.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmTemplate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    /// Speak a short speculative holding line while the (thinking) model works.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_filler: Option<bool>,
    /// Model id for the holding line; `None` uses the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filler_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
}

/// Capture/VAD/turn-taking knobs (all optional; omitted values keep defaults).
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioTemplate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vad_threshold: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_ratio: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive_noise: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vad_silence_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_word: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub barge_in: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub barge_threshold: Option<f32>,
}

/// Persistent-memory settings. The mutable store lives outside the template
/// (under `<config_dir>/memory/<profile>.json`); this only carries the knobs.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryTemplate {
    /// Whether to load and update persistent memory for this profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Memory profile name. Defaults to the template name, then `default`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// How memory is injected into the system prompt: `md` (default) or `json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Model used to summarize at session end; defaults to the holding-line
    /// model, then the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Maximum number of recent-conversation summaries retained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recent_max: Option<usize>,
    /// Maximum combined characters of durable facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fundamental_max_chars: Option<usize>,
    /// Days after which an unresolved follow-up is dropped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub todo_ttl_days: Option<u64>,
    /// Maximum number of open follow-ups retained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub todo_max: Option<usize>,
}

/// Root config directory, honouring `AUDIOCHAT_CONFIG_DIR`, then
/// `XDG_CONFIG_HOME`, then `~/.config`.
pub fn config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("AUDIOCHAT_CONFIG_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("audiochat");
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join(".config").join("audiochat");
    }
    PathBuf::from(".audiochat")
}

/// Directory holding the `<name>.json` template files.
pub fn templates_dir() -> PathBuf {
    config_dir().join("templates")
}

/// Validate a template or memory-profile name (letters, digits, `-`, `_`).
fn validate_name(name: &str) -> Result<&str, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("name must not be empty".into());
    }
    if name == "."
        || name == ".."
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        return Err(format!(
            "invalid name '{name}' (use letters, digits, '-', '_')"
        ));
    }
    Ok(name)
}

/// Validate a template name and resolve its file path.
fn template_path(name: &str) -> Result<PathBuf, String> {
    Ok(templates_dir().join(format!("{}.json", validate_name(name)?)))
}

/// Directory holding per-profile memory stores.
pub fn memory_dir() -> PathBuf {
    config_dir().join("memory")
}

/// Resolve a memory profile's store path, validating the name.
pub fn memory_path(profile: &str) -> Result<PathBuf, String> {
    Ok(memory_dir().join(format!("{}.json", validate_name(profile)?)))
}

/// Fill an `Option` slot only when it is currently empty (flags/env win).
fn fill<T: Clone>(slot: &mut Option<T>, value: Option<&T>) {
    if slot.is_none() {
        if let Some(v) = value {
            *slot = Some(v.clone());
        }
    }
}

impl SttTemplate {
    /// Copy fields that are still unset from `other`.
    fn fill_from(&mut self, other: &SttTemplate) {
        fill(&mut self.backend, other.backend.as_ref());
        fill(&mut self.model, other.model.as_ref());
        fill(&mut self.url, other.url.as_ref());
        fill(&mut self.language, other.language.as_ref());
    }
}

impl TtsTemplate {
    /// Copy fields that are still unset from `other`.
    fn fill_from(&mut self, other: &TtsTemplate) {
        fill(&mut self.backend, other.backend.as_ref());
        fill(&mut self.model, other.model.as_ref());
        fill(&mut self.voice, other.voice.as_ref());
        fill(&mut self.instruct, other.instruct.as_ref());
        fill(&mut self.url, other.url.as_ref());
        fill(&mut self.python, other.python.as_ref());
    }
}

impl LlmTemplate {
    /// Copy fields that are still unset from `other`.
    fn fill_from(&mut self, other: &LlmTemplate) {
        fill(&mut self.provider, other.provider.as_ref());
        fill(&mut self.model, other.model.as_ref());
        fill(&mut self.url, other.url.as_ref());
        fill(&mut self.api_key, other.api_key.as_ref());
        fill(&mut self.thinking, other.thinking.as_ref());
        fill(&mut self.thinking_filler, other.thinking_filler.as_ref());
        fill(&mut self.filler_model, other.filler_model.as_ref());
        fill(&mut self.temperature, other.temperature.as_ref());
        fill(&mut self.num_ctx, other.num_ctx.as_ref());
    }
}

impl AudioTemplate {
    /// Copy fields that are still unset from `other`.
    fn fill_from(&mut self, other: &AudioTemplate) {
        fill(&mut self.vad_threshold, other.vad_threshold.as_ref());
        fill(&mut self.noise_ratio, other.noise_ratio.as_ref());
        fill(&mut self.adaptive_noise, other.adaptive_noise.as_ref());
        fill(&mut self.vad_silence_ms, other.vad_silence_ms.as_ref());
        fill(&mut self.pause_ms, other.pause_ms.as_ref());
        fill(&mut self.stop_word, other.stop_word.as_ref());
        fill(&mut self.barge_in, other.barge_in.as_ref());
        fill(&mut self.barge_threshold, other.barge_threshold.as_ref());
    }
}

impl MemoryTemplate {
    /// Copy fields that are still unset from `other`.
    fn fill_from(&mut self, other: &MemoryTemplate) {
        fill(&mut self.enabled, other.enabled.as_ref());
        fill(&mut self.profile, other.profile.as_ref());
        fill(&mut self.format, other.format.as_ref());
        fill(&mut self.model, other.model.as_ref());
        fill(&mut self.recent_max, other.recent_max.as_ref());
        fill(
            &mut self.fundamental_max_chars,
            other.fundamental_max_chars.as_ref(),
        );
        fill(&mut self.todo_ttl_days, other.todo_ttl_days.as_ref());
        fill(&mut self.todo_max, other.todo_max.as_ref());
    }
}

impl Template {
    /// Fill any unset field from `other`, leaving set fields untouched. Used to
    /// seed the wizard baseline with the global defaults so a template being
    /// edited (and any explicit choice) still wins.
    pub fn fill_from(&mut self, other: &Template) {
        fill(&mut self.name, other.name.as_ref());
        fill(&mut self.description, other.description.as_ref());
        fill(&mut self.mode, other.mode.as_ref());
        fill(&mut self.system_prompt, other.system_prompt.as_ref());
        fill(&mut self.device, other.device.as_ref());
        self.stt.fill_from(&other.stt);
        self.tts.fill_from(&other.tts);
        self.llm.fill_from(&other.llm);
        self.audio.fill_from(&other.audio);
        self.memory.fill_from(&other.memory);
    }

    /// The mode this template should run in. Uses the explicit `mode` when set,
    /// otherwise infers `s2s` when both an LLM model and a TTS backend/model are
    /// configured (e.g. templates written before `mode` existed).
    pub fn effective_mode(&self) -> Option<&str> {
        if let Some(mode) = self.mode.as_deref() {
            if !mode.is_empty() {
                return Some(mode);
            }
        }
        let has_llm = self
            .llm
            .model
            .as_deref()
            .map(|m| !m.trim().is_empty())
            .unwrap_or(false);
        let has_tts = self
            .tts
            .model
            .as_deref()
            .map(|m| !m.trim().is_empty())
            .unwrap_or(false)
            || self.tts.backend.as_deref() == Some("audiocpp");
        if has_llm && has_tts {
            Some("s2s")
        } else {
            None
        }
    }

    /// Load a named template from disk.
    pub fn load(name: &str) -> Result<Template, String> {
        let path = template_path(name)?;
        let data = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read template '{}': {e}", path.display()))?;
        let mut template: Template = serde_json::from_str(&data)
            .map_err(|e| format!("failed to parse template '{}': {e}", path.display()))?;
        if template.name.is_none() {
            template.name = Some(name.trim().to_string());
        }
        Ok(template)
    }

    /// Write this template to `<name>.json`, creating the directory if needed.
    pub fn save(&self, name: &str) -> Result<PathBuf, String> {
        let mut copy = self.clone();
        copy.name = Some(name.trim().to_string());
        let path = template_path(name)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
        }
        let json = serde_json::to_string_pretty(&copy)
            .map_err(|e| format!("failed to serialize template: {e}"))?;
        std::fs::write(&path, format!("{json}\n"))
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
        Ok(path)
    }

    /// Copy template values into `opts`, filling only unset (flag/env-free)
    /// fields so explicit CLI flags always take precedence.
    pub fn apply_to_opts(&self, opts: &mut Opts) {
        fill(&mut opts.stt_backend, self.stt.backend.as_ref());
        fill(&mut opts.stt_url, self.stt.url.as_ref());
        fill(&mut opts.stt_language, self.stt.language.as_ref());
        if let Some(model) = &self.stt.model {
            let backend = opts
                .stt_backend
                .as_deref()
                .or(self.stt.backend.as_deref())
                .unwrap_or("whisper");
            if backend == "whisper" {
                if opts.model.is_none() {
                    opts.model = Some(PathBuf::from(model));
                }
            } else {
                fill(&mut opts.stt_model, Some(model));
            }
        }

        fill(&mut opts.tts_backend, self.tts.backend.as_ref());
        fill(&mut opts.tts_url, self.tts.url.as_ref());
        fill(&mut opts.tts_voice, self.tts.voice.as_ref());
        fill(&mut opts.tts_instruct, self.tts.instruct.as_ref());
        fill(&mut opts.tts_python, self.tts.python.as_ref());
        if let Some(model) = &self.tts.model {
            if opts.tts_model.is_none() {
                opts.tts_model = Some(PathBuf::from(model));
            }
        }

        fill(&mut opts.llm_provider, self.llm.provider.as_ref());
        fill(&mut opts.llm_model, self.llm.model.as_ref());
        fill(&mut opts.llm_url, self.llm.url.as_ref());
        fill(&mut opts.llm_api_key, self.llm.api_key.as_ref());
        if let Some(v) = self.llm.thinking {
            if !opts.llm_thinking {
                opts.llm_thinking = v;
            }
        }
        if let Some(v) = self.llm.thinking_filler {
            if v {
                opts.thinking_filler = true;
            }
        }
        fill(&mut opts.filler_model, self.llm.filler_model.as_ref());
        if let Some(v) = self.llm.temperature {
            if opts.llm_temperature.is_none() {
                opts.llm_temperature = Some(v);
            }
        }
        if let Some(v) = self.llm.num_ctx {
            if opts.llm_num_ctx.is_none() {
                opts.llm_num_ctx = Some(v);
            }
        }

        fill(&mut opts.system_prompt, self.system_prompt.as_ref());
        fill(&mut opts.device, self.device.as_ref());

        if let Some(v) = self.audio.vad_threshold {
            if opts.vad_threshold.is_none() {
                opts.vad_threshold = Some(v);
            }
        }
        if let Some(v) = self.audio.noise_ratio {
            if opts.noise_ratio.is_none() {
                opts.noise_ratio = Some(v);
            }
        }
        if let Some(v) = self.audio.adaptive_noise {
            if !v {
                opts.adaptive_noise = false;
            }
        }
        if let Some(v) = self.audio.vad_silence_ms {
            if opts.vad_silence_ms.is_none() {
                opts.vad_silence_ms = Some(v);
            }
        }
        if let Some(v) = self.audio.pause_ms {
            if opts.pause_ms.is_none() {
                opts.pause_ms = Some(v);
            }
        }
        fill(&mut opts.stop_word, self.audio.stop_word.as_ref());
        if let Some(v) = self.audio.barge_in {
            if v {
                opts.barge_in = true;
            }
        }
        if let Some(v) = self.audio.barge_threshold {
            if opts.barge_threshold.is_none() {
                opts.barge_threshold = Some(v);
            }
        }

        fill(&mut opts.memory_enabled, self.memory.enabled.as_ref());
        fill(&mut opts.memory_profile, self.memory.profile.as_ref());
        fill(&mut opts.memory_format, self.memory.format.as_ref());
        fill(&mut opts.memory_model, self.memory.model.as_ref());
        fill(&mut opts.memory_recent, self.memory.recent_max.as_ref());
        fill(
            &mut opts.memory_fundamental_chars,
            self.memory.fundamental_max_chars.as_ref(),
        );
        fill(
            &mut opts.memory_todo_ttl_days,
            self.memory.todo_ttl_days.as_ref(),
        );
        fill(&mut opts.memory_todo_max, self.memory.todo_max.as_ref());
    }

    /// Snapshot the resolved effective options as a template. The API key is
    /// deliberately not captured here (use the env var or `auth.json`).
    pub fn from_opts(opts: &Opts) -> Template {
        let stt_backend = opts
            .stt_backend
            .clone()
            .unwrap_or_else(|| "whisper".to_string());
        let stt = SttTemplate {
            model: if stt_backend == "whisper" {
                opts.model.as_ref().map(|p| p.display().to_string())
            } else {
                opts.stt_model.clone()
            },
            backend: Some(stt_backend),
            url: opts.stt_url.clone(),
            language: opts.stt_language.clone(),
        };
        let tts = TtsTemplate {
            backend: Some(
                opts.tts_backend
                    .clone()
                    .unwrap_or_else(|| "piper".to_string()),
            ),
            model: opts.tts_model.as_ref().map(|p| p.display().to_string()),
            voice: opts.tts_voice.clone(),
            instruct: opts.tts_instruct.clone(),
            url: opts.tts_url.clone(),
            python: opts.tts_python.clone(),
        };
        let llm = LlmTemplate {
            provider: Some(
                opts.llm_provider
                    .clone()
                    .unwrap_or_else(|| "ollama".to_string()),
            ),
            model: opts.llm_model.clone(),
            url: opts.llm_url.clone(),
            api_key: None,
            thinking: Some(opts.llm_thinking),
            thinking_filler: Some(opts.thinking_filler),
            filler_model: opts.filler_model.clone(),
            temperature: opts.llm_temperature,
            num_ctx: opts.llm_num_ctx,
        };
        let audio = AudioTemplate {
            vad_threshold: opts.vad_threshold,
            noise_ratio: opts.noise_ratio,
            adaptive_noise: Some(opts.adaptive_noise),
            vad_silence_ms: opts.vad_silence_ms,
            pause_ms: opts.pause_ms,
            stop_word: opts.stop_word.clone(),
            barge_in: Some(opts.barge_in),
            barge_threshold: opts.barge_threshold,
        };
        let memory = MemoryTemplate {
            enabled: opts.memory_enabled,
            profile: opts.memory_profile.clone(),
            format: opts.memory_format.clone(),
            model: opts.memory_model.clone(),
            recent_max: opts.memory_recent,
            fundamental_max_chars: opts.memory_fundamental_chars,
            todo_ttl_days: opts.memory_todo_ttl_days,
            todo_max: opts.memory_todo_max,
        };
        Template {
            name: None,
            description: None,
            mode: if opts.s2s {
                Some("s2s".to_string())
            } else if opts.prompt.is_some() {
                Some("llm".to_string())
            } else if opts.speak.is_some() {
                Some("tts".to_string())
            } else {
                None
            },
            stt,
            tts,
            llm,
            system_prompt: opts.system_prompt.clone(),
            device: opts.device.clone(),
            audio,
            memory,
        }
    }
}

/// Names of all saved templates, sorted.
pub fn list_templates() -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(templates_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
    }
    names.sort();
    names
}

/// Delete a saved template.
pub fn delete_template(name: &str) -> Result<(), String> {
    let path = template_path(name)?;
    std::fs::remove_file(&path)
        .map_err(|e| format!("failed to delete template '{}': {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let template = Template {
            name: Some("fast".into()),
            description: Some("remote GPU stack".into()),
            mode: Some("s2s".into()),
            stt: SttTemplate {
                backend: Some("audiocpp".into()),
                model: Some("moonshine-asr".into()),
                url: Some("http://host:8080".into()),
                language: Some("en-US".into()),
            },
            tts: TtsTemplate {
                backend: Some("audiocpp".into()),
                voice: Some("Serena".into()),
                ..Default::default()
            },
            llm: LlmTemplate {
                provider: Some("opencode-go".into()),
                model: Some("deepseek-v4.1-flash".into()),
                thinking: Some(false),
                temperature: Some(0.4),
                ..Default::default()
            },
            system_prompt: Some("Be brief.".into()),
            device: Some("C920".into()),
            audio: AudioTemplate {
                pause_ms: Some(2000),
                ..Default::default()
            },
            memory: MemoryTemplate {
                enabled: Some(true),
                profile: Some("work".into()),
                format: Some("json".into()),
                recent_max: Some(6),
                fundamental_max_chars: Some(800),
                todo_ttl_days: Some(7),
                todo_max: Some(5),
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&template).unwrap();
        let back: Template = serde_json::from_str(&json).unwrap();
        assert_eq!(template, back);
    }

    #[test]
    fn memory_round_trips_via_opts() {
        let opts = crate::parse_args(&[
            "--memory".into(),
            "--memory-profile".into(),
            "work".into(),
            "--memory-format".into(),
            "json".into(),
            "--memory-recent".into(),
            "6".into(),
        ])
        .unwrap();
        let template = Template::from_opts(&opts);
        assert_eq!(template.memory.enabled, Some(true));
        assert_eq!(template.memory.profile.as_deref(), Some("work"));
        assert_eq!(template.memory.format.as_deref(), Some("json"));
        assert_eq!(template.memory.recent_max, Some(6));

        let mut restored = crate::parse_args(&[]).unwrap();
        template.apply_to_opts(&mut restored);
        assert_eq!(restored.memory_enabled, Some(true));
        assert_eq!(restored.memory_profile.as_deref(), Some("work"));
        assert_eq!(restored.memory_format.as_deref(), Some("json"));
        assert_eq!(restored.memory_recent, Some(6));
    }

    #[test]
    fn partial_json_uses_defaults() {
        let template: Template = serde_json::from_str(r#"{"llm":{"provider":"ollama"}}"#).unwrap();
        assert_eq!(template.llm.provider.as_deref(), Some("ollama"));
        assert_eq!(template.stt.backend, None);
        assert_eq!(template.audio.pause_ms, None);
    }

    #[test]
    fn rejects_unsafe_names() {
        assert!(template_path("").is_err());
        assert!(template_path("..").is_err());
        assert!(template_path("a/b").is_err());
        assert!(template_path("ok-name_1").is_ok());
    }

    #[test]
    fn fill_does_not_override_existing() {
        let mut slot = Some("flag".to_string());
        fill(&mut slot, Some(&"template".to_string()));
        assert_eq!(slot.as_deref(), Some("flag"));
    }

    #[test]
    fn apply_keeps_flags_but_fills_unset_fields() {
        let mut opts = crate::parse_args(&["--llm-model".into(), "flag-model".into()]).unwrap();
        let template: Template = serde_json::from_str(
            r#"{"llm":{"provider":"opencode-go","model":"tpl-model","temperature":0.5}}"#,
        )
        .unwrap();
        template.apply_to_opts(&mut opts);
        assert_eq!(opts.llm_model.as_deref(), Some("flag-model"));
        assert_eq!(opts.llm_provider.as_deref(), Some("opencode-go"));
        assert_eq!(opts.llm_temperature, Some(0.5));
    }

    #[test]
    fn whisper_template_model_routes_to_positional_arg() {
        let mut opts = crate::parse_args(&[]).unwrap();
        let template: Template =
            serde_json::from_str(r#"{"stt":{"backend":"whisper","model":"models/x.bin"}}"#)
                .unwrap();
        template.apply_to_opts(&mut opts);
        assert_eq!(
            opts.model.as_deref(),
            Some(std::path::Path::new("models/x.bin"))
        );
    }

    #[test]
    fn audiocpp_template_model_routes_to_stt_model() {
        let mut opts = crate::parse_args(&[]).unwrap();
        let template: Template = serde_json::from_str(
            r#"{"stt":{"backend":"audiocpp","model":"nemotron-asr","url":"http://h:8080"}}"#,
        )
        .unwrap();
        template.apply_to_opts(&mut opts);
        assert_eq!(opts.stt_model.as_deref(), Some("nemotron-asr"));
        assert!(opts.model.is_none());
    }

    #[test]
    fn mode_inferred_as_s2s_for_full_stack() {
        let template: Template = serde_json::from_str(
            r#"{"stt":{"backend":"audiocpp"},"tts":{"backend":"piper","model":"v.onnx"},"llm":{"provider":"ollama","model":"m"}}"#,
        )
        .unwrap();
        assert_eq!(template.effective_mode(), Some("s2s"));
    }

    #[test]
    fn mode_none_for_stt_only() {
        let template: Template =
            serde_json::from_str(r#"{"stt":{"backend":"whisper","model":"m.bin"}}"#).unwrap();
        assert_eq!(template.effective_mode(), None);
    }

    #[test]
    fn explicit_mode_wins_over_inference() {
        let template = Template {
            mode: Some("stt".into()),
            llm: LlmTemplate {
                model: Some("m".into()),
                ..Default::default()
            },
            tts: TtsTemplate {
                model: Some("v.onnx".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(template.effective_mode(), Some("stt"));
    }

    #[test]
    fn from_opts_records_s2s_mode() {
        let opts = crate::parse_args(&["--s2s".into(), "--llm-model".into(), "m".into()]).unwrap();
        assert_eq!(Template::from_opts(&opts).mode.as_deref(), Some("s2s"));
    }

    #[test]
    fn fill_from_supplies_only_unset_fields() {
        let mut template: Template =
            serde_json::from_str(r#"{"stt":{"backend":"whisper"},"tts":{"voice":"Kept"}}"#)
                .unwrap();
        let defaults: Template = serde_json::from_str(
            r#"{"stt":{"backend":"audiocpp","url":"http://gpu:8080"},"tts":{"voice":"Dylan","model":"qwen3-tts"}}"#,
        )
        .unwrap();
        template.fill_from(&defaults);
        assert_eq!(template.stt.backend.as_deref(), Some("whisper"));
        assert_eq!(template.stt.url.as_deref(), Some("http://gpu:8080"));
        assert_eq!(template.tts.voice.as_deref(), Some("Kept"));
        assert_eq!(template.tts.model.as_deref(), Some("qwen3-tts"));
    }

    #[test]
    fn thinking_filler_round_trips_via_template() {
        let opts = crate::parse_args(&[
            "--thinking-filler".into(),
            "--llm-thinking".into(),
            "--filler-model".into(),
            "flash".into(),
        ])
        .unwrap();
        let template = Template::from_opts(&opts);
        assert_eq!(template.llm.thinking_filler, Some(true));
        assert_eq!(template.llm.filler_model.as_deref(), Some("flash"));

        let mut restored = crate::parse_args(&[]).unwrap();
        template.apply_to_opts(&mut restored);
        assert!(restored.thinking_filler);
        assert_eq!(restored.filler_model.as_deref(), Some("flash"));
    }

    #[test]
    fn tts_instruct_round_trips_via_template() {
        let opts = crate::parse_args(&["--tts-instruct".into(), "Speak slowly".into()]).unwrap();
        let template = Template::from_opts(&opts);
        assert_eq!(template.tts.instruct.as_deref(), Some("Speak slowly"));
        let mut restored = crate::parse_args(&[]).unwrap();
        template.apply_to_opts(&mut restored);
        assert_eq!(restored.tts_instruct.as_deref(), Some("Speak slowly"));
    }
}
