# First Iteration Plan

A low-latency, small-footprint speech-to-speech interface for LLM interaction.

## Stack

- **Language/Runtime:** Rust (low latency, small footprint, easy cross-compilation)
- **STT:** whisper.cpp (local, streaming-capable) — with a pluggable trait for other engines
- **TTS:** Piper (lightweight local neural TTS) — pluggable for espeak-ng/system fallback
- **LLM:** pluggable HTTP API (Ollama first, with an OpenAI-compatible abstraction)

## Architecture

```
[mic] --> [STT stream] --> [VAD] --utterance--> [LLM client]
                                                   |
                                                   v
[speaker] <-- [TTS] <-- [response stream] <--------+
```

Core pipeline, all local processing passed through channels:

1. **Capture:** read mic audio (PCM 16-bit, 16 kHz mono)
2. **VAD:** voice activity detection to segment speech into utterances
3. **STT:** transcribe the utterance to text (whisper.cpp)
4. **LLM:** send text to a pluggable backend, stream tokens
5. **TTS:** synthesize spoken reply as tokens stream in (Piper)
6. **Playback:** queue synthesized audio for the speaker

Low-latency strategy: stream tokens from the LLM straight into Piper, so audio
starts playing before the full reply is generated ("streaming TTS chunking"),
rather than waiting for complete generation.

## Proposed crate layout (single repo, workspace)

- `audiochat-core` — pipeline orchestration, VAD, channels, pluggable traits
- `audiochat-stt-whisper` — whisper.cpp bindings (or subprocess wrapper)
- `audiochat-tts-piper` — Piper bindings / subprocess + PCMA input
- `audiochat-llm` — pluggable LLM client (Ollama, OpenAI-compatible)
- `audiochat-cli` — minimal CLI binary to run end-to-end

Kept small: no heavy async runtime if avoidable (channels + polling); revisit
if streaming complexity demands tokio.

## Milestones

### M0 — Skeleton & project scaffold
- Init Cargo workspace, crates above, empty placeholders
- CI config (GitHub Actions): build + clippy + fmt on `main` and PRs
- ADRs / docs notes for pluggable trait design

### M1 — STT path: mic → text  ✅
- Implement capture (PCM) + VAD segmentation + whisper transcription
- Trait: `SpeechRecognizer` with a stub/echo impl for offline testing
- Manual test: speak a phrase, print transcription to stdout

Status: `MicCapture` (cpal) resamples native device audio to 16 kHz mono i16;
`EnergyVad` (energy-based, frame-wise); `WhisperRecognizer` (whisper-rs,
`ggml-tiny.en.bin` downloaded to `models/`). CLI wires them together.

Validated with the whisper.cpp JFK sample via
`cargo run -p audiochat-stt-whisper --example transcribe -- models/ggml-tiny.en.bin <wav>` —
produces the correct transcript. Live mic confirmed on a C920 webcam
(`--device "C920"` selects a specific input device by name).

Remaining: echo/stub impl for offline testing; threading/backoff not yet needed
at this stage.

### M2 — TTS path: text → audio  ✅
- Implement Piper synthesis + playback to speaker
- Trait: `TextToSpeech`
- Manual test: feed some text, hear it spoken

Status: `audiochat-tts-piper` (subprocess wrapper around the `piper` CLI; feeds
text via stdin, reads raw 22.05 kHz PCM, resamples to 16 kHz mono); `play_pcm`
playback in core; CLI `--speak "<text>" --tts-model <voice.onnx>` mode. Tested
with `en_US-lessac-medium` voice (played aloud and to a WAV). Piper runs from a
Python venv via `PIPER_BIN` env var.

Remaining: keep the piper process alive for streaming TTS (M4), espeak/system
fallback backend.

### M3 — LLM path: text → text (no audio)  ✅
- Pluggable `Llm` trait; Ollama backend + OpenAI-compatible backend
- Manual test: prompt typed at CLI, streamed text reply printed

Status: `audiochat-llm` crate with `Ollama` backend using ureq (sync, small footprint)
against Ollama's `/api/generate` streaming endpoint. CLI `--prompt "<text>" --llm-model <name>`
mode streams the reply token-by-token. Tested with `gemma4:e2b` on localhost.

Remaining: OpenAI-compatible backend (trivial swap). Multi-turn conversation
context landed later (see M4 planning note).

### M4 — Integration: end-to-end speech-to-speech  ✅
- Wire M1+M2+M3 through the pipeline
- Streaming TTS: feed LLM token stream into Piper for low-latency audio
- First manual end-to-end run: "what's 2+2?" → heard spoken answer

Status: `Pipeline` orchestrator in core; a background `AudioSink` thread plays
each completed sentence (queued in order, so playback never blocks the LLM
token stream). Tokens print to the console in real time; sentence boundaries
(`.`/`!`/`?`) trigger synthesis + queued playback. CLI `--s2s` mode (
`--tts-model`, `--llm-model`, optional `--silent`). STT/TTS/LLM constructed from
the same backends as M1-M3.

Remaining: true token-by-token streaming TTS (piper reads all stdin to EOF, so
sentence-chunked synthesis is the current design); mic-audio validation of the
s2s loop on real hardware.

### M5 — Hardening & measurement  ✅
- Latency instrumentation per stage (capture→VAD→STT→LLM→first-audio)
- Error handling / backoff for backend unavailability
- Config via env or a small config file (backend URL, model, device)
- Clean shutdown / resource release

Status: `Pipeline` records a `TurnTiming` per spoken turn (STT, first-token,
first-audio, full-stream, total latencies) printed to stderr when `--verbose` is
on. Transient LLM/TTS failures retry with
exponential backoff (`with_retry`). CLI options read `AUDIOCHAT_*` env vars as
fallbacks (flag takes precedence). Graceful SIGINT shutdown in `--s2s` flushes
the pending VAD utterance before exiting (via the `ctrlc` crate). Backends
(Ollama/Piper/whisper) are constructed identically to M1-M3.

Remaining: mic-audio validation of the s2s loop on real hardware (headless env
can only initialize the pipeline); OpenAI-compatible LLM backend; multi-turn
context.

Multi-turn context: done — `Ollama` now uses `/api/chat` with a bounded
conversation history held internally (transparent to the `Llm` trait). The
client appends each user turn, streams the reply, and stores the assistant turn;
oldest turns beyond `max_turns` (default 10) are trimmed. Exposes
`reset_conversation()`, `set_max_turns()`, `history_len()`. Verified live on
Ollama (turn 1 introduced a fact, turn 2 recalled it) plus unit tests for
trim/reset. `--s2s` inherits this automatically since the client persists for
the session.

## Success criteria for iteration 1

- Clean end-to-end loop: spoken question → spoken answer
- Latency instrumented and reported (goal: < ~1–2 s S-to-S for short prompts)
- Pluggable SEAT/LLM backends demonstrated (both STT and LLM swap without
  changing pipeline code)
- Builds and passes `cargo clippy` + `cargo fmt --check` in CI

## Out of scope for iteration 1

- Wake-word / always-on interruption handling
- Noise suppression / echo cancellation
- Mobile/embedded deployment packaging
- Caching, buffering optimizations
