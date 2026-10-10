//! Best-effort discovery of the backends' available models.
//!
//! Backends report their catalogs over HTTP: audio.cpp's OpenAI-compatible
//! `GET /v1/models` (each entry carries a `task`: `asr`, `tts`, ...), Ollama's
//! `GET /api/tags`, and generic OpenAI-compatible `GET /v1/models`. Local
//! whisper/Piper files are found by scanning `models/`. Everything here is
//! best-effort: on any failure an empty list is returned and the wizard falls
//! back to free-text entry.

use std::path::{Path, PathBuf};

use serde_json::Value;

const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// A discovered choice: a stable `id` plus an optional detail for display.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOption {
    pub id: String,
    pub detail: Option<String>,
}

impl ModelOption {
    fn new(id: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            id: id.into(),
            detail: detail.filter(|d| !d.is_empty()),
        }
    }

    /// The string shown in a selection menu.
    pub fn label(&self) -> String {
        match &self.detail {
            Some(d) => format!("{}  ({})", self.id, d),
            None => self.id.clone(),
        }
    }
}

/// Fetch and parse a JSON document from `url`, optionally authenticated with a
/// bearer token. Returns `None` on any network/parse failure.
fn get_json(url: &str, api_key: Option<&str>) -> Option<Value> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(TIMEOUT))
        .timeout_recv_body(Some(TIMEOUT))
        .build()
        .new_agent();
    let mut req = agent.get(url);
    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        req = req.header("Authorization", &format!("Bearer {key}"));
    }
    let body = req.call().ok()?.into_body().read_to_string().ok()?;
    serde_json::from_str(&body).ok()
}

fn join(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), path)
}

/// Installed Ollama models via `GET /api/tags`.
pub fn ollama_models(base: &str) -> Vec<ModelOption> {
    let Some(json) = get_json(&join(base, "/api/tags"), None) else {
        return Vec::new();
    };
    json.get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    let id = m.get("name").and_then(Value::as_str)?;
                    let detail = m
                        .get("details")
                        .and_then(|d| d.get("parameter_size"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    Some(ModelOption::new(id, detail))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Models from an OpenAI-compatible `GET /v1/models` (base already includes
/// `/v1`). Used for the `openai`/`opencode-go` LLM providers.
pub fn openai_models(base: &str, api_key: Option<&str>) -> Vec<ModelOption> {
    let Some(json) = get_json(&join(base, "/models"), api_key) else {
        return Vec::new();
    };
    json.get("data")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    let id = m.get("id").and_then(Value::as_str)?;
                    Some(ModelOption::new(id, None))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// audio.cpp models whose `task` matches (e.g. `asr`, `tts`).
pub fn audiocpp_models(base: &str, task: &str) -> Vec<ModelOption> {
    let Some(json) = get_json(&join(base, "/v1/models"), None) else {
        return Vec::new();
    };
    json.get("data")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter(|m| m.get("task").and_then(Value::as_str) == Some(task))
                .filter_map(|m| {
                    let id = m.get("id").and_then(Value::as_str)?;
                    let detail = m
                        .get("mode")
                        .and_then(Value::as_str)
                        .map(|mode| format!("{task}, {mode}"));
                    Some(ModelOption::new(id, detail))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Built-in voices an audio.cpp TTS model exposes (`GET /v1/audio/voices`).
/// Falls back to the known Qwen3-TTS speakers when the server reports none.
pub fn audiocpp_voices(base: &str, model: &str) -> Vec<ModelOption> {
    let url = if model.is_empty() {
        join(base, "/v1/audio/voices")
    } else {
        format!("{}?model={model}", join(base, "/v1/audio/voices"))
    };
    if let Some(json) = get_json(&url, None) {
        if let Some(voices) = json.get("voices").and_then(Value::as_array) {
            if !voices.is_empty() {
                return voices
                    .iter()
                    .filter_map(|v| {
                        let id = v
                            .as_str()
                            .map(str::to_string)
                            .or_else(|| v.get("id").and_then(Value::as_str).map(str::to_string))
                            .or_else(|| {
                                v.get("name").and_then(Value::as_str).map(str::to_string)
                            })?;
                        Some(ModelOption::new(id, None))
                    })
                    .collect();
            }
        }
    }
    default_audiocpp_voices(model)
}

/// Known built-in speakers for the audio.cpp CustomVoice models.
fn default_audiocpp_voices(model: &str) -> Vec<ModelOption> {
    if model.contains("qwen3") {
        ["Serena", "Dylan"]
            .iter()
            .map(|v| ModelOption::new(*v, None))
            .collect()
    } else {
        Vec::new()
    }
}

/// Root directory scanned for local model files.
fn models_dir() -> PathBuf {
    std::env::var_os("AUDIOCHAT_MODELS_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Local whisper GGML models (`.bin`) under the models directory.
pub fn local_whisper_models() -> Vec<ModelOption> {
    scan_files(&models_dir(), 2, |p| {
        p.extension().and_then(|e| e.to_str()) == Some("bin")
    })
}

/// Local Piper voices (`.onnx`) under the models directory.
pub fn local_piper_voices() -> Vec<ModelOption> {
    scan_files(&models_dir(), 3, |p| {
        p.extension().and_then(|e| e.to_str()) == Some("onnx")
    })
}

fn scan_files(dir: &Path, depth: usize, pred: impl Fn(&Path) -> bool) -> Vec<ModelOption> {
    let mut out = Vec::new();
    walk(dir, depth, &pred, &mut out);
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out.dedup_by(|a, b| a.id == b.id);
    out
}

fn walk(dir: &Path, depth: usize, pred: &impl Fn(&Path) -> bool, out: &mut Vec<ModelOption>) {
    if depth == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, depth - 1, pred, out);
        } else if pred(&path) {
            out.push(ModelOption::new(path.display().to_string(), None));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_includes_detail() {
        assert_eq!(ModelOption::new("a", None).label(), "a");
        assert_eq!(
            ModelOption::new("a", Some("streaming".into())).label(),
            "a  (streaming)"
        );
        assert_eq!(ModelOption::new("a", Some(String::new())).label(), "a");
    }

    #[test]
    fn known_qwen3_voices() {
        let voices = default_audiocpp_voices("qwen3-tts");
        assert!(voices.iter().any(|v| v.id == "Serena"));
    }

    #[test]
    fn scans_files_recursively() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("audiochat-discover-{nanos}"));
        let voices = root.join("voices");
        std::fs::create_dir_all(&voices).unwrap();
        std::fs::write(root.join("ggml-tiny.en.bin"), b"x").unwrap();
        std::fs::write(voices.join("en_US-lessac-medium.onnx"), b"x").unwrap();

        let bins = scan_files(&root, 2, |p| {
            p.extension().and_then(|e| e.to_str()) == Some("bin")
        });
        assert!(bins.iter().any(|m| m.id.ends_with("ggml-tiny.en.bin")));

        let voices = scan_files(&root, 3, |p| {
            p.extension().and_then(|e| e.to_str()) == Some("onnx")
        });
        assert!(voices
            .iter()
            .any(|m| m.id.ends_with("en_US-lessac-medium.onnx")));

        let _ = std::fs::remove_dir_all(&root);
    }
}
