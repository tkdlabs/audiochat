# AGENTS.md

Guidance for AI agents working in this repository.

## Workflow

- On feature completion, commit and push the finished work. Do not leave a
  completed feature uncommitted.
- Write concise commit messages in the imperative mood, matching repo style.
- Never commit secrets, private hostnames, or other machine-specific values.
  Endpoints and defaults belong in the local config file, not the source.

## Build, test, lint

```sh
cargo build --workspace
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Run all three before considering a change complete.

## Project notes

- Rust workspace; the `audiochat` CLI lives in `audiochat-cli/` (`src/main.rs`,
  plus `template.rs`, `wizard.rs`, `discover.rs`, `config.rs`).
- The `audiochat-cli` binary must be built with `--release` for usable latency.
- Machine-specific defaults (hosts, model ids, voice, device) live in
  `~/.config/audiochat/config.json` (override with `AUDIOCHAT_CONFIG_DIR` or
  `XDG_CONFIG_HOME`). Precedence, lowest to highest: built-in fallbacks → config
  file → template → `AUDIOCHAT_*` env vars → CLI flags. The wizard is seeded
  from the config file.
- Templates are JSON under `~/.config/audiochat/templates/`.
