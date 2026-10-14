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

Status: `audiochat-tts-piper` (persistent subprocess wrapping the `piper` Python
library via an embedded helper; feeds length-prefixed text via stdin, reads raw
22.05 kHz PCM, resamples to 16 kHz mono); `play_pcm`/`AudioSink` playback in
core; CLI `--speak "<text>" --tts-model <voice.onnx>` mode. Tested with
`en_US-lessac-medium` voice (played aloud and to a WAV). Piper runs from a
Python venv via the `AUDIOCHAT_PYTHON` env var (default `python3`).

Remaining: espeak/system fallback backend.

### M3 — LLM path: text → text (no audio)  ✅
- Pluggable `Llm` trait; Ollama backend + OpenAI-compatible backend
- Manual test: prompt typed at CLI, streamed text reply printed

Status: `audiochat-llm` crate with `Ollama` backend using ureq (sync, small footprint)
against Ollama's `/api/generate` streaming endpoint. CLI `--prompt "<text>" --llm-model <name>`
mode streams the reply token-by-token. Tested with `gemma4:e2b` on localhost.

Remaining: ~~OpenAI-compatible backend~~ done in iteration 2 (see below).
Multi-turn conversation context landed later (see M4 planning note).

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

Status: `Pipeline` records a `TurnTiming` per spoken turn, printed to stderr when
`--verbose` is on. Timing is broken out per layer, all measured from the true end
of the user's speech: `vad_ms` (VAD silence window), `stt_ms` (recognition),
`endpoint_ms` (end of speech → dispatch), `first_token_ms` (LLM TTFT),
`tts_first_ms` (first TTS call), `first_audio_ms`, `full_stream_ms`, `total_ms`;
the same fields go to the conversation log. A mic-free benchmark
(`cargo run -p audiochat-cli --example latency -- "..."`) exercises the
LLM→TTS pipeline. Transient LLM/TTS failures retry with
exponential backoff (`with_retry`). CLI options read `AUDIOCHAT_*` env vars as
fallbacks (flag takes precedence). Graceful SIGINT shutdown in `--s2s` sets a
stop flag read by the `Session` loop; it shuts down the background capture
thread and playback before exiting (via the `ctrlc` crate). Backends
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

Barge-in (interrupting an in-progress reply by speaking) is implemented as an
opt-in `--barge-in` mode; it still assumes headphones/earbuds since echo
cancellation remains out of scope.

## Iteration 2 — OpenAI-compatible LLM + audio.cpp TTS ✅

Two opt-in backends were added without changing the pipeline or the defaults
(Ollama + Piper remain the out-of-the-box choices):

- **`OpenAiCompatible` LLM client** (`audiochat-llm`): talks to any OpenAI-style
  `/chat/completions` server, streaming SSE deltas and keeping the same bounded
  multi-turn history as `Ollama`. Reasoning models are run in non-thinking mode
  by default via `{"thinking": {"type": "disabled"}}`. A `--llm-provider
  opencode-go` preset targets the OpenCode Go subscription
  (`https://opencode.ai/zen/go/v1`, `deepseek-v4.1-flash`); the API key resolves
  from `AUDIOCHAT_LLM_API_KEY` / `OPENCODE_API_KEY` / opencode's `auth.json`.
  Shared history helpers were factored into `audiochat-llm::chat`.
- **`audiochat-tts-audiocpp`**: an `AudioCppTts` backend that POSTs to an
  [audio.cpp](https://github.com/0xShug0/audio.cpp) server's
  `/v1/audio/speech`, decodes the returned WAV, downmixes to mono and resamples
  to 16 kHz. The host is configured per machine (`--tts-url`, the config file, or
  `with_base`); defaults are model `qwen3-tts`, voice `Serena`.
- **`audiochat-stt-audiocpp`**: an `AudioCppStt` backend that uploads a completed
  utterance as a WAV to audio.cpp's OpenAI-compatible
  `/v1/audio/transcriptions` (multipart), returning the transcript. The host is
  configured per machine (`--stt-url` or the config file), offloading STT from
  local whisper to a GPU host. The server's streaming
  `/v1/audio/transcriptions/live` endpoint is available for a future streaming
  recognizer.

CLI: `--llm-provider ollama|openai|opencode-go`, `--llm-api-key`,
`--llm-thinking`; `--tts-backend piper|audiocpp`, `--tts-url`, `--tts-voice`;
`--stt-backend whisper|audiocpp`, `--stt-url`, `--stt-model`, `--stt-language`
(all with `AUDIOCHAT_*` env fallbacks, and a `~/.config/audiochat/config.json`
defaults file). Verified live against OpenCode Go (DeepSeek V4.1 Flash,
non-thinking) and an audio.cpp CUDA server on a remote GPU host.

### Turn-taking ergonomics ✅

- The endpointer hold is decoupled from the VAD candidate window: `--pause`
  (env `AUDIOCHAT_PAUSE`) sets how long a mid-sentence transcript is held,
  independent of `--vad-silence`. It is backed by an `AtomicU64` so it can be
  changed while running.
- A stdin **control console** (only when stdin is a terminal) tunes the pause
  live (`+`/`-`/`pause <ms>`) and stops playback (`stop`) without restarting.
- A spoken **stop word** (`--stop-word`, default `stop`) finalizes the current
  turn immediately instead of waiting out the pause, or submits a held question;
  the word is never sent to the LLM. A genuine question ending in the word is
  left intact. With `--barge-in`, the same word interrupts a spoken reply.

## Iteration 3 — speculative holding line ✅

Thinking models add latency before the first spoken token, leaving dead air in
the speech-to-speech loop. With `--thinking-filler` (and `--llm-thinking`), the
pipeline fires a **parallel, non-thinking** request to a fast OpenAI-compatible
model that speaks a short holding line while the main model works:

- `OpenAiCompatible` gained a **stateless** mode (`with_stateless`) that sends
  only the system prompt plus the current prompt and neither reads nor writes
  the conversation history, so a helper request never becomes a session turn.
  A dedicated `FILLER_SYSTEM_PROMPT` instructs the helper to stall, not answer.
- `Pipeline` grew a persistent `FillerWorker` thread that owns the helper client
  and streams a line per turn. `respond` reads the main reply on a worker thread
  so it can service the holding line and **pre-empt** it the instant the first
  real token arrives: the line's playback is stopped (`sink.stop`, reusing the
  barge-in path) and the helper request is cancelled. If the main model produces
  no reply at all, the in-flight line is allowed to finish.
- The helper defaults to `deepseek-v4.1-flash` for `opencode-go` and is skipped
  when replies are silent or the provider is not OpenAI-compatible.
- Templates/wizard carry `llm.thinking_filler` and `llm.filler_model`; the
  per-turn log and `--verbose` output add a `holding-first=…ms` metric
  (`filler_ms`).

## Iteration 4 — persistent memory ✅

Optional, per-profile long-term memory (`--memory`) so the assistant recalls
earlier conversations:

- `audiochat-core::memory` defines the store (`Memory`), per-category limits
  (`MemoryLimits`), Markdown/JSON rendering, and an `update` that asks a
  summarizer model for an updated JSON store. Three categories with independent
  expiry: `recent` (count-based, newest N), `fundamental` (permanent, strict
  character cap), and `todo` (time-based TTL + count). `sanitize` enforces every
  limit/expiry in code rather than trusting the model.
- Memory is keyed by an explicit **profile** (`memory.profile`, defaulting to the
  template name) and stored under `<config_dir>/memory/<profile>.json`, separate
  from the template so user edits never clobber it. Templates and the wizard
  carry the settings; flags/env (`AUDIOCHAT_MEMORY*`) override.
- At startup the store is rendered into `opts.system_prompt` (Markdown by
  default, JSON optional); at the end of an `--s2s` session the completed turns
  (collected by `Pipeline` and returned by `Session::run`) are sent to a
  stateless summarizer (`build_memory_llm`, reusing the holding-line pattern).
  `--show-memory`/`--forget-memory` inspect and clear a profile.
- Injection is best-effort: with no summarizer model the existing memory is still
  used but not rewritten; a malformed summarizer reply leaves the store intact.

## Iteration 5 — timestamp side-information ✅

Optional time awareness (`--timestamps`, template `timestamps`, env
`AUDIOCHAT_TIMESTAMPS`): each prompt is prefixed with a bracketed line giving the
local wall-clock time (with UTC offset and a rough IANA zone) and the gap since
the previous exchange, e.g. `[context: 2026-10-14 13:32:07 -06:00 (US/Mountain,
Wednesday); 3 minutes since the previous exchange]`. `audiochat-core::timefmt`
formats the local time via the C library's thread-safe `localtime_r` (Unix `libc`
dependency; the zone name comes from `TZ` or the `/etc/localtime` symlink, with a
UTC fallback). `Pipeline` tracks when the last turn finished to compute the gap
and keeps the line in the bounded history. The raw question is still what is
printed and logged, and the setting round-trips through templates,
`--save-config`, and the wizard.
