//! Pluggable LLM clients (Ollama, OpenAI-compatible, ...).

mod chat;
mod ollama;
mod openai;

pub use chat::DEFAULT_SYSTEM_PROMPT;
pub use ollama::Ollama;
pub use openai::OpenAiCompatible;

/// Resolve an OpenCode Go API key.
///
/// Order: `AUDIOCHAT_LLM_API_KEY`, then `OPENCODE_API_KEY`, then the
/// `opencode-go` entry in opencode's `auth.json` (`~/.local/share/opencode/`).
/// Returns `None` when no key is found.
pub fn opencode_go_api_key() -> Option<String> {
    for var in ["AUDIOCHAT_LLM_API_KEY", "OPENCODE_API_KEY"] {
        if let Ok(value) = std::env::var(var) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let path = std::path::Path::new(&home)
            .join(".local/share/opencode")
            .join("auth.json");
        if let Ok(data) = std::fs::read_to_string(path) {
            return auth_key_from_json(&data);
        }
    }
    None
}

/// Extract the `opencode-go` API key from an opencode `auth.json` document.
fn auth_key_from_json(data: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    value
        .get("opencode-go")?
        .get("key")?
        .as_str()
        .map(str::to_string)
        .filter(|k| !k.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::auth_key_from_json;

    #[test]
    fn extracts_opencode_go_key() {
        let json = r#"{"opencode-go":{"type":"api","key":"sk-test-123"}}"#;
        assert_eq!(auth_key_from_json(json).as_deref(), Some("sk-test-123"));
    }

    #[test]
    fn missing_or_empty_key_is_none() {
        assert_eq!(
            auth_key_from_json(r#"{"opencode-go":{"type":"api"}}"#),
            None
        );
        assert_eq!(auth_key_from_json(r#"{"opencode-go":{"key":""}}"#), None);
        assert_eq!(auth_key_from_json("not json"), None);
    }
}
