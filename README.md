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

Requires [Rust](https://rustup.rs) (stable), `cmake`, and `libclang` (for building
whisper.cpp bindings).

```
# 1. Download a whisper model into models/ (gitignored), e.g.:
curl -L -o models/ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin

# 2. Offline STT validation on a WAV:
cargo run -p audiochat-stt-whisper --example transcribe -- models/ggml-tiny.en.bin test.wav

# 3. Live mic -> text:
cargo run -p audiochat-cli -- models/ggml-tiny.en.bin

# List input devices, then pick one by name (case-insensitive substring):
cargo run -p audiochat-core --example audio_devices
cargo run -p audiochat-cli -- --device "C920" models/ggml-tiny.en.bin
```

## Status

- [x] M0 — Workspace scaffold + CI
- [x] M1 — Mic → text (capture + VAD + STT)
- [ ] M2 — Text → audio (TTS)
- [ ] M3 — Text → text (LLM)
- [ ] M4 — End-to-end speech-to-speech
- [ ] M5 — Hardening & latency measurement
