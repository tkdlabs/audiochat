# audiochat

A low-latency, small-footprint speech-to-speech interface for LLM interaction.

Speak, get a spoken response. The LLM is connected via a pluggable API (e.g. Ollama), keeping the footprint small and response times low.

## Goals

- **Low latency** from end of speech to start of spoken reply
- **Small footprint** — minimal dependencies, lightweight audio/streaming
- **Pluggable LLM backend** — swap between Ollama, OpenAI-compatible APIs, etc.

## Architecture

```
[mic] --> [STT stream] --> [VAD] --utterance--> [LLM client]
                                                   |
                                                   v
[speaker] <-- [TTS] <-- [response stream] <--------+
```

Rust workspace. See `PLAN.md` for the detailed first-iteration plan.

## Workspace layout

- `audiochat-core` — pipeline orchestration, VAD, pluggable traits
- `audiochat-stt-whisper` — whisper.cpp STT backend
- `audiochat-tts-piper` — Piper TTS backend
- `audiochat-llm` — pluggable LLM clients (Ollama, OpenAI-compatible)
- `audiochat-cli` — end-to-end CLI binary

## Getting Started

Requires [Rust](https://rustup.rs) (stable).

```
cargo build
cargo run -p audiochat-cli
```

## Status

- [x] M0 — Workspace scaffold + CI
- [ ] M1 — Mic → text (capture + VAD + STT)
- [ ] M2 — Text → audio (TTS)
- [ ] M3 — Text → text (LLM)
- [ ] M4 — End-to-end speech-to-speech
- [ ] M5 — Hardening & latency measurement
