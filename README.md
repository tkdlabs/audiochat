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
- `audiochat-stt-audiocpp` — audio.cpp HTTP STT backend
- `audiochat-tts-piper` — Piper TTS backend
- `audiochat-tts-audiocpp` — audio.cpp HTTP TTS backend
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

### STT (audio.cpp, remote/fast)

Instead of local whisper, STT can run on an [audio.cpp](https://github.com/0xShug0/audio.cpp)
GPU host via its OpenAI-compatible `POST /v1/audio/transcriptions`. Select it
with `--stt-backend audiocpp`; point it at your server with `--stt-url` (or
`AUDIOCHAT_STT_URL`, or the [config file](#defaults)). The built-in fallback is
`http://localhost:8080` with model `moonshine-asr` (Moonshine Streaming Tiny —
English, and very fast). Use `--stt-model nemotron-asr` (Nemotron 3.5 ASR
Streaming 0.6B) for multilingual recognition.

```
# Live mic -> text using the remote GPU STT:
cargo run -p audiochat-cli -- --stt-backend audiocpp

# Offline STT on a WAV (no playback):
cargo run -p audiochat-stt-audiocpp --example audiocpp_transcribe -- test.wav
```

`--stt-url` (env `AUDIOCHAT_STT_URL`) sets the base URL, `--stt-model` (env
`AUDIOCHAT_STT_MODEL`) the server-side ASR model id, and `--stt-language` (env
`AUDIOCHAT_STT_LANGUAGE`, e.g. `en-US`) the recognition language (omit for
auto-detect). With this backend no local whisper model file is needed. audio.cpp
also exposes a streaming `/v1/audio/transcriptions/live` endpoint; a streaming
recognizer that transcribes while you speak is a future addition.

## Setup wizard & templates

`--wizard` walks through the STT, TTS, and LLM backends plus the system prompt,
optional [persistent memory](#persistent-memory), and
[timestamp side-information](#timestamps--time-awareness) with arrow-key menus,
then saves the result as a named template and/or launches speech-to-speech. Model fields are populated from what the backends actually
report — audio.cpp's `/v1/models` (ASR/TTS), Ollama's `/api/tags`, and
OpenAI-compatible `/v1/models` — while local whisper/Piper files are found under
`models/`; every list has a "Type manually" escape hatch:

```
cargo run --release -p audiochat-cli -- --wizard
```

Templates are JSON files under `~/.config/audiochat/templates/` (override the
root with `AUDIOCHAT_CONFIG_DIR` or `XDG_CONFIG_HOME`):

```
# Save the current flags as a reusable template (no mic is opened):
cargo run -p audiochat-cli -- --save-template fast \
  --stt-backend audiocpp --llm-provider opencode-go --tts-backend audiocpp

# Run from a saved template (explicit flags still take precedence):
cargo run --release -p audiochat-cli -- --template fast --s2s
# List / edit / delete:
cargo run -p audiochat-cli -- --list-templates
cargo run -p audiochat-cli -- --edit-template fast
cargo run -p audiochat-cli -- --delete-template fast
```

The wizard never writes the LLM API key to disk; it is read from
`AUDIOCHAT_LLM_API_KEY`/`OPENCODE_API_KEY` or opencode's `auth.json`.

Loading a template that configures STT, TTS, and an LLM enables `--s2s`
automatically, so the wizard-built and full-stack templates run the whole loop;
an explicit `--prompt`/`--speak` still overrides.

## Defaults

Machine-specific values — server hosts, model ids, voice, default device — live
in a global defaults file rather than in the source, so the repository carries
no private endpoints. It is `~/.config/audiochat/config.json` (same root as
templates; override with `AUDIOCHAT_CONFIG_DIR` or `XDG_CONFIG_HOME`) and uses
the template JSON shape:

```json
{
  "stt": { "backend": "audiocpp", "url": "http://gpu-host:8080", "model": "moonshine-asr" },
  "tts": { "backend": "audiocpp", "url": "http://gpu-host:8080", "model": "qwen3-tts", "voice": "Serena" },
  "llm": { "provider": "opencode-go", "model": "deepseek-v4.1-flash" }
}
```

Write it from the current flags/env with `--save-config`:

```
cargo run -p audiochat-cli -- --save-config \
  --stt-backend audiocpp --stt-url http://gpu-host:8080 \
  --tts-backend audiocpp --tts-url http://gpu-host:8080
```

Precedence, lowest to highest: built-in fallbacks → this config file → a loaded
template → `AUDIOCHAT_*` environment variables → explicit CLI flags. So defaults
apply automatically, and any of the more specific layers still overrides them.
The `--wizard` pre-fills its prompts from these defaults (a template being
edited still wins). The config file is read on every run but, like
`--save-template`, writing it alone does not open the mic.

## Quick start (full speech-to-speech)

Talk to it end to end using the fast remote backends: [audio.cpp](#stt-audiocpp-remotefast)
for STT, [OpenCode Go](#llm-openai-compatible--opencode-go) for the LLM, and
[audio.cpp](#tts-audiocpp-remotefast) for speech. Mic → STT → LLM → TTS →
speaker, all in one loop.

```
cargo run --release -p audiochat-cli -- --s2s \
  --stt-backend audiocpp \
  --llm-provider opencode-go \
  --tts-backend audiocpp
```

Notes:

- Use `--release`; a debug build adds noticeable latency.
- The remote STT/TTS hosts fall back to `http://localhost:8080` (set yours with
  `--stt-url`/`--tts-url`, the matching env vars, or the
  [config file](#defaults)) and the LLM uses OpenCode Go's API; no local model
  files are needed.
- The OpenCode Go key is read automatically from
  `~/.local/share/opencode/auth.json` (or set `OPENCODE_API_KEY` /
  `AUDIOCHAT_LLM_API_KEY`). It runs DeepSeek V4.1 Flash in non-thinking mode.
- Override endpoints with `--stt-url`/`--stt-model`, `--tts-url`/`--tts-model`/
  `--tts-voice`.
- To use local whisper instead, drop `--stt-backend audiocpp` and pass the model
  path as the first argument (e.g. `models/ggml-tiny.en.bin`).
- Add `--verbose` for per-turn latency, `--barge-in` to interrupt by speaking
  (headphones recommended), `--silent` to print replies without speaking, and
  `--device "NAME"` to pick a mic.
- To tolerate long thinking pauses, raise `--pause <ms>` (e.g. `--pause 2500`).
  Tune it live: type `+` / `-` / `pause <ms>` while it runs (see
  [Live controls](#live-controls--the-stop-word)).
- Say **"stop"** to finalize the current turn immediately instead of waiting out
  the pause; customize with `--stop-word` or disable with `--no-stop-word`.
- Ctrl-C stops it cleanly.

To run fully locally instead, use Ollama + Piper (see the sections below).

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

### TTS (audio.cpp, remote/fast)

For lower-latency synthesis you can run [audio.cpp](https://github.com/0xShug0/audio.cpp)
on a GPU host and point audiochat at its OpenAI-compatible speech endpoint
(`POST /v1/audio/speech`). Select it with `--tts-backend audiocpp`; point it at
your server with `--tts-url` (or `AUDIOCHAT_TTS_URL`, or the
[config file](#defaults)). The built-in fallback is `http://localhost:8080`
with model `qwen3-tts` and voice `Serena`.

```
# Speak text via a remote audio.cpp server:
cargo run -p audiochat-cli -- --tts-backend audiocpp \
  --speak "Hello world" --tts-url http://localhost:8080

# Offline audio.cpp TTS -> WAV (no playback):
cargo run -p audiochat-tts-audiocpp --example audiocpp_tts -- "Text to a wav file."
```

`--tts-url` (env `AUDIOCHAT_TTS_URL`) sets the base URL, `--tts-voice` (env
`AUDIOCHAT_TTS_VOICE`) picks the speaker, and `--tts-model` (env
`AUDIOCHAT_TTS_MODEL`) is the server-side model id. The server returns 24 kHz
WAV; audiochat decodes it and resamples to the pipeline's 16 kHz mono.

Pass `--tts-instruct "Speak softly and a little slower"` (env
`AUDIOCHAT_TTS_INSTRUCT`) to steer style, emotion, and pacing. It is sent as the
`instructions` field of `POST /v1/audio/speech`, which audio.cpp maps to the
model's `instruction` request option; the Qwen3-TTS CustomVoice and VoiceDesign
variants honor it, other families ignore it:

## LLM (Ollama)

The LLM backend is pluggable; the Ollama client talks to the streaming
`/api/chat` endpoint and keeps a bounded multi-turn conversation history, so in
`--s2s` each spoken question builds on the prior turns. History is capped (10
turns by default) with the oldest turns dropped; call `reset_conversation()` on
the client to clear it.

```
# Ask Ollama a question (defaults to localhost:11434):
ollama pull gemma4:e2b   # or any installed model
cargo run -p audiochat-cli -- --prompt "What is 2+2?" --llm-model gemma4:e2b

# Point to a remote Ollama server:
cargo run -p audiochat-cli -- --prompt "What is 2+2?" \
  --llm-model gemma4:e2b --llm-url http://192.168.1.50:11434
```

`--llm-url` (or env `AUDIOCHAT_LLM_URL`) sets the Ollama base URL. Use this
when Ollama runs on a different machine or a non-default port.

Sampling is configurable per request: `--llm-temperature 0.7` (env
`AUDIOCHAT_LLM_TEMPERATURE`) sets the temperature and `--llm-num-ctx 4096` (env
`AUDIOCHAT_LLM_NUM_CTX`) sets the context window in tokens; both fall back to
the model's defaults. A lower temperature makes replies more deterministic (a
gemma-default 0.8 can feel rambling); a larger `num_ctx` helps long
conversations.

### LLM (OpenAI-compatible / OpenCode Go)

Set `--llm-provider openai` to talk to any OpenAI-compatible `/chat/completions`
server, or `--llm-provider opencode-go` for the [OpenCode Go](https://opencode.ai/docs/go)
subscription (base URL `https://opencode.ai/zen/go/v1`, default model
`deepseek-v4.1-flash`). Replies stream token-by-token and keep the same bounded
multi-turn history as the Ollama backend.

```
# OpenCode Go (key from AUDIOCHAT_LLM_API_KEY, OPENCODE_API_KEY, or opencode's auth.json):
cargo run -p audiochat-cli -- --llm-provider opencode-go \
  --prompt "What is 2+2?"

# Any OpenAI-compatible server:
cargo run -p audiochat-cli -- --llm-provider openai \
  --llm-url https://api.example.com/v1 --llm-model some-model \
  --llm-api-key "$MY_KEY" --prompt "Hello"
```

`--llm-api-key` (env `AUDIOCHAT_LLM_API_KEY`) sets the bearer token; for
`opencode-go` it falls back to `OPENCODE_API_KEY` and then the `opencode-go`
entry in `~/.local/share/opencode/auth.json`. Reasoning models like DeepSeek
V4.1 Flash think by default; audiochat runs them in **non-thinking mode** unless
you pass `--llm-thinking` (env `AUDIOCHAT_LLM_THINKING=1`).

### Speculative holding line

Thinking models add latency before the first spoken token. With
`--thinking-filler` (env `AUDIOCHAT_THINKING_FILLER=1`) and `--llm-thinking`,
audiochat fires a **parallel, non-thinking** request to a fast model that speaks
a short holding line ("Good question, let me think about that…") while the main
model works. The first real reply token **pre-empts** the holding line, cutting
its playback and cancelling the helper request, so the user never hears both.

```
cargo run -p audiochat-cli -- --s2s --llm-thinking --thinking-filler \
  --llm-provider opencode-go --tts-backend audiocpp
```

The holding-line model defaults to `deepseek-v4.1-flash` for `opencode-go`; set
`--filler-model NAME` (env `AUDIOCHAT_FILLER_MODEL`) for other OpenAI-compatible
servers. The helper client is stateless, so its request never becomes part of
the conversation history. This needs an OpenAI-compatible provider (`openai` or
`opencode-go`); it is ignored otherwise. With `--verbose` the holding-line
latency is reported as `holding-first=…ms`.

### Persistent memory

With `--memory` (env `AUDIOCHAT_MEMORY=1`), audiochat keeps a small memory per
**profile** and folds it into the assistant's system prompt, so it recalls
earlier conversations. Memory is written at the **end of an `--s2s` session**,
when a fast, stateless summarizer model rewrites the store from that session's
turns — no per-turn latency is added. Templates carry the settings
(`memory.enabled`, `memory.profile`, `memory.format`, limits); the mutable store
lives outside the template under `<config_dir>/memory/<profile>.json` (profile
defaults to the template name, then `default`).

Three categories are kept, each with its own expiry:

| Category | Content | Expiry | Default |
|---|---|---|---|
| `recent` | one-line summary per past conversation | count-based (newest N) | 8 |
| `fundamental` | durable facts, opinions, preferences | permanent, strict length | 1200 chars |
| `todo` | follow-ups to raise next time | time-based (TTL) + count | 14 days / 10 |

```
# Recall past conversations and update memory when you quit:
cargo run --release -p audiochat-cli -- --s2s --memory \
  --llm-provider opencode-go --tts-backend audiocpp

# Inspect or clear a profile (no mic is opened):
cargo run -p audiochat-cli -- --show-memory --memory-profile work
cargo run -p audiochat-cli -- --forget-memory --memory-profile work
```

The summarizer is a separate **stateless** call (like the holding line), so it
never pollutes the conversation history. It uses `--memory-model` (env
`AUDIOCHAT_MEMORY_MODEL`), falling back to `--filler-model` and then the
provider default. If no suitable model is available the existing memory is still
injected, it just isn't updated at exit. Tune the categories with
`--memory-recent`, `--memory-fundamental-chars`, `--memory-todo-ttl-days`, and
`--memory-todo-max`; choose how memory is injected with `--memory-format md`
(default) or `json`, and name a profile with `--memory-profile`. `--no-memory`
overrides a template that enables it.

Memory stores excerpts of your conversations on disk; keep profiles separate for
different personas or contexts and delete them with `--forget-memory`.

### Timestamps & time awareness

With `--timestamps` (env `AUDIOCHAT_TIMESTAMPS=1`), each prompt is prefixed with
a bracketed side-information line giving the local wall-clock time (with UTC
offset and a rough IANA time zone) and the time since the previous exchange:

```
[context: 2026-10-14 13:32:07 -06:00 (US/Mountain, Wednesday); 3 minutes since the previous exchange]
What's the weather?
```

The raw question is still what's printed and written to the conversation log; the
line is added only for the model, and stays in the bounded conversation history so
the model can see when each turn happened. The local offset comes from the OS
(`localtime_r`); the zone name is taken from `TZ` or the `/etc/localtime` symlink,
and the timestamp falls back to UTC where the offset isn't available. It pairs
well with persistent memory (both help the model reason about time). The wizard
exposes the same toggle, and templates store it as `timestamps`.

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

Barge-in requires the user to talk noticeably louder than ordinary speech
(default `--barge-threshold 0.06`, vs the normal `--vad-threshold 0.02`) for a
sustained ~300 ms before it fires, to avoid cutting off the reply on faint echo
or noise. If the assistant interrupts itself, raise `--barge-threshold` (e.g.
`0.10`); if a deliberate "stop" is missed, lower it (e.g. `0.04`).

Playback uses a single persistent output stream (opened once, sequential
queue) rather than opening the device per chunk, which avoids ALSA "I/O error"
churn from rapid open/close cycles.

```
AUDIOCHAT_PYTHON=$PWD/.venv/bin/python cargo run -p audiochat-cli -- --s2s \
  models/ggml-tiny.en.bin \
  --tts-model models/voices/en_US-lessac-medium.onnx \
  --llm-model gemma4:e2b

# Fast remote setup: OpenCode Go LLM + audio.cpp TTS:
cargo run -p audiochat-cli -- --s2s \
  models/ggml-tiny.en.bin \
  --llm-provider opencode-go \
  --tts-backend audiocpp
```

Add `--silent` to print replies without speaking them. Add `--verbose` to print a
per-turn latency breakdown on stderr, broken out by layer:

```
└ latency: endpoint=660ms (vad=600 stt=25) first-token=1520ms tts-first=210ms first-audio=1760ms stream=2100ms total=2600ms
```

- `endpoint` — end of speech → turn dispatched, split into `vad` (the VAD's
  trailing-silence window) and `stt` (speech-recognition inference).
- `first-token` — end of speech → first LLM token (includes the endpoint wait).
- `tts-first` — duration of the first TTS synthesis call.
- `first-audio` — end of speech → first audio enqueued.
- `stream` / `total` — end of speech → last LLM token / last audio enqueued.

`--verbose` also prints live capture diagnostics: when the app is listening vs.
muted, when it catches speech, when it dispatches an utterance to STT, and when
it drops a too-short utterance — useful for tuning `--vad-threshold` and
`--vad-silence` if you're being cut off. To measure the LLM→TTS pipeline without
a mic, run `cargo run --release -p audiochat-cli --example latency -- "a question"`.
All options have `AUDIOCHAT_*` env-var fallbacks (`AUDIOCHAT_DEVICE`,
`AUDIOCHAT_STT_BACKEND`, `AUDIOCHAT_STT_URL`, `AUDIOCHAT_STT_MODEL`,
`AUDIOCHAT_STT_LANGUAGE`, `AUDIOCHAT_TTS_BACKEND`, `AUDIOCHAT_TTS_MODEL`,
`AUDIOCHAT_TTS_URL`, `AUDIOCHAT_TTS_VOICE`, `AUDIOCHAT_PYTHON`,
`AUDIOCHAT_LLM_PROVIDER`, `AUDIOCHAT_LLM_MODEL`, `AUDIOCHAT_LLM_URL`,
`AUDIOCHAT_LLM_API_KEY`, `AUDIOCHAT_PAUSE`, `AUDIOCHAT_STOP_WORD`,
`AUDIOCHAT_MEMORY`, `AUDIOCHAT_MEMORY_PROFILE`, `AUDIOCHAT_MEMORY_FORMAT`,
`AUDIOCHAT_MEMORY_MODEL`, `AUDIOCHAT_MEMORY_RECENT`,
`AUDIOCHAT_MEMORY_FUNDAMENTAL_CHARS`, `AUDIOCHAT_MEMORY_TODO_TTL_DAYS`,
`AUDIOCHAT_MEMORY_TODO_MAX`, `AUDIOCHAT_TIMESTAMPS`); flags take precedence. `--s2s` handles Ctrl-C gracefully,
shutting down the capture thread and stopping playback before exit.

`--conversation-log conversations.jsonl` (env `AUDIOCHAT_CONVERSATION_LOG`) is a
structured conversation log: after every completed turn the question, reply,
model, and timing breakdown are appended as a single JSON line (JSONL). Each
record looks like:

```json
{"timestamp":1786326794,"turn":0,"model":"gemma4:e2b","user":"What time is it?","assistant":"It's about 3:20 PM.","timing":{"vad_ms":600,"stt_ms":25,"endpoint_ms":660,"first_token_ms":1520,"tts_first_ms":210,"first_audio_ms":1760,"stream_ms":2100,"total_ms":2600}}
```

Markdown in LLM replies is stripped to plain text before synthesis, so Piper
doesn't read out `###`, `*`, backticks, etc.

The VAD adapts to the room's background noise by default: it tracks the recent
noise floor and classifies speech only above `max(vad-threshold, noise * ratio)`
(`--vad-threshold 0.02`, `--noise-ratio 2.0` by default). A noisy room (a fan,
blower, etc.) automatically raises the effective threshold, so you generally
don't need to hand-tune it per session. Pass `--no-adaptive` to disable this and
use `--vad-threshold` as a fixed absolute threshold instead. If it doesn't
detect the end of speech, raise `--vad-threshold` (e.g. `0.04`) or lower
`--noise-ratio`.

Turn-ending is handled by a **sentence-boundary endpointer**, not the VAD's
silence window alone: the VAD's `--vad-silence` window produces candidate
segments, and the STT worker holds any candidate whose transcript doesn't end in
`.` / `!` / `?` for up to `--pause` ms (waiting for the user to resume) before
dispatching. This means a natural mid-sentence pause longer than the window
continues the same turn instead of splitting it, while a completed thought still
dispatches quickly.

`--vad-silence` sets both the VAD's candidate window and the endpointer hold by
default; `--pause` overrides just the hold, so you can cut candidates promptly
(for responsive stop-word handling) while still tolerating long thinking pauses.

### Live controls & the stop word

While `--s2s` runs interactively, a control console reads commands from stdin
(press Enter after each; only active when stdin is a terminal):

| Input | Effect |
|---|---|
| `+` / `longer` | increase the pause window by 250 ms |
| `-` / `shorter` | decrease it (floor 100 ms) |
| `pause` / `?` | print the current pause window |
| `pause <ms>` | set it exactly |
| `stop` | interrupt the reply currently playing |
| `quit` / `exit` | stop the session |

This makes the pause window tunable without restarting — the biggest win when a
room or speaking style needs a different wait.

A spoken **stop word** (default `stop`, `--stop-word <word>`,
`--no-stop-word` to disable) finalizes the current turn immediately instead of
waiting out the pause:

- Say `stop` on its own to submit a question the endpointer is holding.
- Say `<question>. stop` to submit the question right away.
- A question that merely ends in the word (e.g. "how do I make it stop") is left
  alone, so it still reaches the LLM.

To interrupt the assistant's *reply* by voice, enable `--barge-in` (headphones
required, since echo cancellation is out of scope) and say `stop`; the reply
stops and the word is not sent to the LLM.

## Status

- [x] M0 — Workspace scaffold + CI
- [x] M1 — Mic → text (capture + VAD + STT)
- [x] M2 — Text → audio (TTS + playback)
- [x] M3 — Text → text (LLM client)
- [x] M4 — End-to-end speech-to-speech
- [x] M5 — Hardening & measurement
