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

## TTS (Piper)

Piper runs as a persistent subprocess: `audiochat-tts-piper` drives a small
Python helper (via `piper-tts`) that keeps the voice model loaded and streams
raw PCM back, so the model isn't re-loaded per sentence.

```
# 1. Install piper + a voice model (models/ is gitignored):
python3 -m venv .venv
./.venv/bin/pip install piper-tts
curl -L -o models/voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

# 2. Speak text aloud:
AUDIOCHAT_PYTHON=$PWD/.venv/bin/python \
  cargo run -p audiochat-cli -- --speak "Hello world" \
  --tts-model models/voices/en_US-lessac-medium.onnx

# 3. Offline TTS -> WAV (no playback):
AUDIOCHAT_PYTHON=$PWD/.venv/bin/python \
  cargo run -p audiochat-tts-piper --example tts -- \
  models/voices/en_US-lessac-medium.onnx "Text to a wav file."
```

## LLM (Ollama)

The LLM backend is pluggable; the Ollama client talks to the streaming
`/api/chat` endpoint and keeps a bounded multi-turn conversation history, so in
`--s2s` each spoken question builds on the prior turns. History is capped (10
turns by default) with the oldest turns dropped; call `reset_conversation()` on
the client to clear it.

```
# Ask Ollama a question (it must be running on localhost:11434):
ollama pull gemma4:e2b   # or any installed model
cargo run -p audiochat-cli -- --prompt "What is 2+2?" --llm-model gemma4:e2b
```

## Speech-to-speech (end to end)

`--s2s` drives the full loop: mic -> VAD -> STT -> LLM -> Piper -> speaker.
LLM tokens stream to the console in real time while each completed sentence is
synthesized and played. Capture runs on a **dedicated background thread**: the
mic is recorded and VAD-segmented continuously and independently of the main
thread, so speech spoken while the LLM is thinking is never lost. Turn-taking
is **half-duplex** — a shared gate tells the capture thread to discard audio
while the assistant's own reply is playing (so it doesn't hear itself), with a
short cooldown after playback before listening resumes.

With `--barge-in`, the user can interrupt an in-progress reply by speaking over
it: the capture thread detects sustained speech during playback, the pipeline
stops the current TTS (and the in-flight LLM stream) immediately, and the
interruption is transcribed as the next turn. Barge-in assumes **headphones or
earbuds** so the mic doesn't pick up the assistant's own voice; without them the
assistant may interrupt itself (echo cancellation is out of scope).

Playback uses a single persistent output stream (opened once, sequential
queue) rather than opening the device per chunk, which avoids ALSA "I/O error"
churn from rapid open/close cycles.

```
AUDIOCHAT_PYTHON=$PWD/.venv/bin/python cargo run -p audiochat-cli -- --s2s \
  models/ggml-tiny.en.bin \
  --tts-model models/voices/en_US-lessac-medium.onnx \
  --llm-model gemma4:e2b
```

Add `--silent` to print replies without speaking them. Add `--verbose` to print
per-turn latency breakdown (STT → first-token → first-audio → stream end →
total) plus live capture diagnostics on stderr: when the app is listening vs.
muted, when it catches speech, when it dispatches an utterance to STT, and when
it drops a too-short utterance — useful for tuning `--vad-threshold` and
`--vad-silence` if you're being cut off. All options have `AUDIOCHAT_*` env-var fallbacks (`AUDIOCHAT_DEVICE`,
`AUDIOCHAT_TTS_MODEL`, `AUDIOCHAT_PYTHON`, `AUDIOCHAT_LLM_MODEL`,
`AUDIOCHAT_LLM_URL`); flags take precedence. `--s2s` handles Ctrl-C gracefully,
shutting down the capture thread and stopping playback before exit.

Markdown in LLM replies is stripped to plain text before synthesis, so Piper
doesn't read out `###`, `*`, backticks, etc.

If the app fires turns on brief pauses, raise the silence window with
`--vad-silence 1500` (default 600 ms). If it doesn't detect the end of speech,
raise `--vad-threshold` (e.g. `0.04`).

Turn-ending is handled by a **sentence-boundary endpointer**, not the VAD's
silence window alone: the VAD's `--vad-silence` window produces candidate
segments, and the STT worker holds any candidate whose transcript doesn't end in
`.` / `!` / `?` for up to one more window (waiting for the user to resume) before
dispatching. This means a natural mid-sentence pause longer than the window
continues the same turn instead of splitting it, while a completed thought still
dispatches quickly.

## Status

- [x] M0 — Workspace scaffold + CI
- [x] M1 — Mic → text (capture + VAD + STT)
- [x] M2 — Text → audio (TTS + playback)
- [x] M3 — Text → text (LLM client)
- [x] M4 — End-to-end speech-to-speech
- [x] M5 — Hardening & measurement
