//! Global default configuration for the audiochat CLI.
//!
//! Connection settings that would otherwise be hardcoded — server hosts, model
//! ids, voices — live in `~/.config/audiochat/config.json` so the repository
//! carries no machine-specific values. The file uses the same JSON shape as a
//! saved template; the root can be overridden with `AUDIOCHAT_CONFIG_DIR` or
//! `XDG_CONFIG_HOME`.
//!
//! These defaults are the lowest priority layer: an explicit CLI flag, the
//! matching `AUDIOCHAT_*` environment variable, and a loaded template all win.

use std::path::{Path, PathBuf};

use crate::template::{self, Template};

/// Path to the global defaults file (`<config_dir>/config.json`).
pub fn config_path() -> PathBuf {
    template::config_dir().join("config.json")
}

/// Load the global defaults, if the file exists. A malformed file is reported
/// rather than silently ignored, so a typo doesn't fall back to built-ins.
pub fn load_defaults() -> Result<Option<Template>, String> {
    load_defaults_at(&config_path())
}

/// Load defaults from an explicit path (see [`load_defaults`]).
fn load_defaults_at(path: &Path) -> Result<Option<Template>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let mut defaults: Template = serde_json::from_str(&data)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
    // Identity fields belong to templates, not the defaults file.
    defaults.name = None;
    defaults.mode = None;
    Ok(Some(defaults))
}

/// Write `defaults` to `<config_dir>/config.json`, creating the directory if
/// needed. `name` and `mode` are omitted since they don't apply to defaults.
pub fn save_defaults(defaults: &Template) -> Result<PathBuf, String> {
    let path = config_path();
    save_defaults_at(&path, defaults)?;
    Ok(path)
}

/// Write defaults to an explicit path (see [`save_defaults`]).
fn save_defaults_at(path: &Path, defaults: &Template) -> Result<(), String> {
    let mut copy = defaults.clone();
    copy.name = None;
    copy.mode = None;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(&copy)
        .map_err(|e| format!("failed to serialize defaults: {e}"))?;
    std::fs::write(path, format!("{json}\n"))
        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("audiochat-config-{nanos}"))
    }

    #[test]
    fn missing_file_yields_none() {
        let path = temp_dir().join("config.json");
        assert!(load_defaults_at(&path).unwrap().is_none());
    }

    #[test]
    fn round_trips_defaults() {
        let path = temp_dir().join("config.json");
        let template: Template = serde_json::from_str(
            r#"{"stt":{"backend":"audiocpp","url":"http://gpu:8080"},"name":"ignored","mode":"s2s"}"#,
        )
        .unwrap();
        save_defaults_at(&path, &template).unwrap();
        let loaded = load_defaults_at(&path).unwrap().unwrap();
        assert_eq!(loaded.stt.url.as_deref(), Some("http://gpu:8080"));
        assert_eq!(loaded.name, None);
        assert_eq!(loaded.mode, None);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn malformed_file_is_an_error() {
        let path = temp_dir().join("config.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{not json").unwrap();
        assert!(load_defaults_at(&path).is_err());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
