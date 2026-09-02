//! Turn-taking session: a background capture thread plus a main loop that
//! processes captured utterances one at a time.
//!
//! The capture thread continuously runs mic audio through an `EnergyVad` and
//! pushes completed utterances onto a channel, **independent of what the main
//! thread is doing**. This means speech is never lost while the LLM is
//! generating a (potentially slow) reply. A shared gate flag tells the capture
//! thread to discard audio while the assistant is speaking (half-duplex), so
//! the assistant's own voice doesn't become a new turn.

use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::capture::MicCapture;
use crate::pipeline::Pipeline;
use crate::traits::SpeechRecognizer;
use crate::vad::EnergyVad;
use crate::DEFAULT_SAMPLE_RATE;

/// Minimum utterance length (samples at 16 kHz) before it is transcribed.
/// ~120 ms filters out noise blips that would otherwise trigger a wasteful,
/// high-latency LLM call.
const MIN_UTTERANCE_SAMPLES: usize = 1920;
/// Cooldown after playback ends before the capture thread accepts audio again,
/// letting the assistant's audio tail clear.
const POST_PLAYBACK_COOLDOWN_MS: u64 = 250;
/// Maximum audio (samples at 16 kHz) the endpointer holds while waiting for a
/// sentence boundary, before it flushes anyway. ~30 s.
const MAX_TURN_SAMPLES: usize = 30 * DEFAULT_SAMPLE_RATE as usize;

/// Shared controls handed to the background capture thread.
struct CaptureCtrl {
    /// True while the assistant is speaking (playback active).
    gate: Arc<AtomicBool>,
    /// Set to true to ask the capture thread to stop (e.g. on Ctrl-C).
    stop: Arc<AtomicBool>,
}

/// A completed utterance plus the instant the VAD detected end of speech, so
/// the pipeline can measure latency against the true end of the user's turn.
struct Utterance {
    pcm: Vec<i16>,
    completed_at: Instant,
}

/// A transcribed utterance ready for the LLM, plus its timing provenance.
struct SttResult {
    question: String,
    completed_at: Instant,
    stt_ms: u64,
}

/// A transcription that ended mid-sentence. The endpointer holds it and waits
/// for more audio (in case the user is only pausing) before dispatching, so a
/// natural mid-thought pause doesn't split one turn into several.
struct PendingTurn {
    /// Accumulated PCM for the turn so far.
    audio: Vec<i16>,
    /// End-of-speech instant of the most recent candidate segment, used for
    /// latency provenance on the eventual `SttResult`.
    completed_at: Instant,
    /// When we started holding (after STT found no sentence boundary), used to
    /// time out the hold if the user never resumes.
    hold_since: Instant,
    /// Most recent transcription of `audio`, so a timeout flush doesn't need to
    /// re-run STT.
    text: String,
}

/// Whether a transcript ends in sentence-ending punctuation, indicating the user
/// finished a complete thought rather than pausing mid-sentence.
fn ends_with_sentence_boundary(text: &str) -> bool {
    text.ends_with('.') || text.ends_with('!') || text.ends_with('?')
}

/// Send a finalized turn to the main loop, recording its STT latency.
fn dispatch(
    result_tx: &mpsc::SyncSender<SttResult>,
    question: String,
    completed_at: Instant,
    verbose: bool,
) {
    let stt_ms = completed_at.elapsed().as_millis() as u64;
    if verbose {
        log("stt", &format!("transcribed ({stt_ms} ms): {question}"));
    }
    let _ = result_tx.try_send(SttResult {
        question,
        completed_at,
        stt_ms,
    });
}

/// Runs the capture thread for the lifetime of `mic`, sending each completed
/// utterance to `utt_tx`.
fn capture_loop(
    vad: EnergyVad,
    ctrl: CaptureCtrl,
    mic: MicCapture,
    utt_tx: mpsc::SyncSender<Utterance>,
    verbose: bool,
) {
    let mut vad = vad;
    let mut prev_gate = false;
    let mut listening = true;
    let mut prev_speech = false;
    let mut cooldown_until = Instant::now() - Duration::from_secs(1);

    loop {
        if ctrl.stop.load(Ordering::Relaxed) {
            break;
        }
        let pcm = match mic.rx.recv_timeout(Duration::from_millis(50)) {
            Ok(pcm) => pcm,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        let gate = ctrl.gate.load(Ordering::Acquire);

        if gate != prev_gate {
            // Either playback started or stopped: drop any partially-built
            // utterance and (on stop) ignore the assistant's audio tail.
            let _ = vad.flush();
            prev_gate = gate;
            if verbose {
                log(
                    state_label(listening),
                    if gate {
                        "muted (assistant speaking)"
                    } else {
                        "playback done, pausing"
                    },
                );
            }
            if gate {
                listening = false;
            } else {
                cooldown_until = Instant::now() + Duration::from_millis(POST_PLAYBACK_COOLDOWN_MS);
            }
            continue;
        }

        let in_cooldown = Instant::now() < cooldown_until;
        if gate || in_cooldown {
            continue;
        }
        if !listening {
            listening = true;
            if verbose {
                log(state_label(listening), "listening for input...");
            }
        }

        let mut dispatched = false;
        for utterance in vad.feed(&pcm) {
            let ms = utterance.len() * 1000 / DEFAULT_SAMPLE_RATE as usize;
            if utterance.len() >= MIN_UTTERANCE_SAMPLES {
                dispatched = true;
                if verbose {
                    log(
                        state_label(listening),
                        &format!("dispatched utterance ({ms} ms) -> STT"),
                    );
                }
                let _ = utt_tx.send(Utterance {
                    pcm: utterance,
                    completed_at: Instant::now(),
                });
            } else if verbose {
                log(
                    state_label(listening),
                    &format!(
                        "dropped short utterance ({ms} ms < {} ms)",
                        MIN_UTTERANCE_SAMPLES * 1000 / DEFAULT_SAMPLE_RATE as usize
                    ),
                );
            }
        }

        let speech_now = vad.speech_in_last_frame();
        if !dispatched && speech_now && !prev_speech && verbose {
            log(state_label(listening), "caught speech, listening...");
        }
        prev_speech = speech_now;
    }

    if let Some(utt) = vad.flush() {
        if utt.len() >= MIN_UTTERANCE_SAMPLES {
            let _ = utt_tx.send(Utterance {
                pcm: utt,
                completed_at: Instant::now(),
            });
        }
    }
}

/// Runs the STT worker for the lifetime of the session, transcribing each
/// captured utterance and pushing the result to `result_tx`. This runs on its
/// own thread so the next utterance's transcription overlaps the LLM/TTS work
/// the main thread is doing for the current turn.
///
/// The worker also acts as the turn endpointer: instead of trusting the VAD's
/// fixed silence window blindly, it accumulates candidate segments and only
/// finalizes a turn once the transcript ends in sentence-ending punctuation, or
/// the user has stayed silent for another full window after a mid-sentence
/// transcription (or the audio has grown too long). This keeps a natural pause
/// from splitting one thought into several turns without making the assistant
/// wait on a hand-tuned silence threshold.
fn stt_loop(
    mut recognizer: Box<dyn SpeechRecognizer>,
    utt_rx: mpsc::Receiver<Utterance>,
    result_tx: mpsc::SyncSender<SttResult>,
    stop: Arc<AtomicBool>,
    vad_max_silence_ms: u64,
    verbose: bool,
) {
    let mut pending: Option<PendingTurn> = None;

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // Flush a held (mid-sentence) turn once the user has stayed silent long
        // enough, or the audio has grown unboundedly long, so we never wait
        // forever for a sentence boundary that never arrives.
        if let Some(p) = &pending {
            let idle_ms = p.hold_since.elapsed().as_millis() as u64;
            if idle_ms >= vad_max_silence_ms || p.audio.len() >= MAX_TURN_SAMPLES {
                let p = pending.take().expect("pending checked above");
                dispatch(&result_tx, p.text, p.completed_at, verbose);
                continue;
            }
        }

        let utterance = match utt_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(u) => u,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        // Accumulate the new candidate onto any held audio: a mid-sentence
        // pause continues the same turn rather than starting a new one.
        let (mut audio, last_text) = match pending.take() {
            Some(p) => (p.audio, Some(p.text)),
            None => (Vec::new(), None),
        };
        audio.extend_from_slice(&utterance.pcm);
        let completed_at = utterance.completed_at;

        match recognizer.transcribe(&audio) {
            Ok(question) => {
                let question = question.trim().to_string();
                if question.is_empty() {
                    // The candidate was noise; keep holding any earlier speech.
                    if let Some(text) = last_text {
                        pending = Some(PendingTurn {
                            audio,
                            completed_at,
                            hold_since: Instant::now(),
                            text,
                        });
                    }
                    continue;
                }
                if ends_with_sentence_boundary(&question) {
                    dispatch(&result_tx, question, completed_at, verbose);
                } else {
                    if verbose {
                        log("stt", &format!("holding mid-sentence: {question}"));
                    }
                    pending = Some(PendingTurn {
                        audio,
                        completed_at,
                        hold_since: Instant::now(),
                        text: question,
                    });
                }
            }
            Err(e) => {
                eprintln!("audiochat: STT failed: {e}");
                // Preserve any held speech; it will flush on timeout.
                if let Some(text) = last_text {
                    pending = Some(PendingTurn {
                        audio,
                        completed_at,
                        hold_since: Instant::now(),
                        text,
                    });
                }
            }
        }
    }

    // Flush a held turn on shutdown so it isn't silently dropped.
    if let Some(p) = pending {
        dispatch(&result_tx, p.text, p.completed_at, verbose);
    }
}

fn state_label(listening: bool) -> &'static str {
    if listening {
        "talk"
    } else {
        "muted"
    }
}
fn log(state: &str, msg: &str) {
    eprintln!("[{state}] {msg}");
}

/// The main turn-taking session. Owns the pipeline/mic and drives the capture
/// thread + processing loop.
pub struct Session {
    /// RMS threshold (0..1) above which a frame counts as speech.
    pub vad_threshold: f32,
    /// Trailing silence (ms) that ends a candidate speech segment. The STT
    /// endpointer may hold a mid-sentence segment for up to one more window.
    pub vad_max_silence_ms: u64,
    /// Whether to log listening/muting activity to stderr.
    pub verbose: bool,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            vad_threshold: 0.02,
            vad_max_silence_ms: 600,
            verbose: false,
        }
    }
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the VAD RMS threshold.
    pub fn with_vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the trailing silence that ends a candidate speech segment. The STT
    /// endpointer may hold a mid-sentence segment for up to one more window.
    pub fn with_vad_max_silence(mut self, ms: u64) -> Self {
        self.vad_max_silence_ms = ms;
        self
    }

    /// Enable/disable verbose listening/muting logging.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Run the speech-to-speech session to completion.
    ///
    /// `mic` is moved into a background capture thread; `stt` is moved into a
    /// background transcription thread; `pipeline` (LLM + TTS) runs on the main
    /// thread and processes transcribed turns one at a time. Captured utterances
    /// are transcribed in parallel with the current turn's LLM/TTS work, so a
    /// queued turn is ready by the time the main loop reaches it. If `stop` is
    /// provided, the loop polls it and returns cleanly when it becomes `true`
    /// (e.g. on Ctrl-C).
    pub fn run(
        self,
        mic: MicCapture,
        stt: Box<dyn SpeechRecognizer>,
        mut pipeline: Pipeline,
        stop: Option<Arc<AtomicBool>>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let gate = Arc::new(AtomicBool::new(false));
        pipeline = pipeline.with_gate(Some(Arc::clone(&gate)));

        let stop_flag = stop.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

        let (utt_tx, utt_rx) = mpsc::sync_channel::<Utterance>(16);
        let (result_tx, result_rx) = mpsc::sync_channel::<SttResult>(8);

        let vad = EnergyVad::new(DEFAULT_SAMPLE_RATE)
            .with_threshold(self.vad_threshold)
            .with_max_silence(self.vad_max_silence_ms);
        let ctrl = CaptureCtrl {
            gate,
            stop: Arc::clone(&stop_flag),
        };
        let verbose = self.verbose;

        let capture_handle = std::thread::spawn(move || {
            capture_loop(vad, ctrl, mic, utt_tx, verbose);
        });
        let stt_stop = Arc::clone(&stop_flag);
        let stt_handle = std::thread::spawn(move || {
            stt_loop(
                stt,
                utt_rx,
                result_tx,
                stt_stop,
                self.vad_max_silence_ms,
                verbose,
            );
        });

        if verbose {
            log("talk", "session started, listening for input...");
        }

        loop {
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }
            match result_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(result) => {
                    pipeline.respond(&result.question, result.completed_at, result.stt_ms)?;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        stt_handle.join().map_err(|_| "STT thread panicked")?;
        capture_handle
            .join()
            .map_err(|_| "capture thread panicked")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_sentence_boundaries() {
        assert!(ends_with_sentence_boundary("What's the weather?"));
        assert!(ends_with_sentence_boundary("Hello world."));
        assert!(ends_with_sentence_boundary("Stop!"));
        assert!(!ends_with_sentence_boundary("I think"));
        assert!(!ends_with_sentence_boundary("Hello world,"));
        assert!(!ends_with_sentence_boundary(""));
    }

    /// A `SpeechRecognizer` that returns a scripted sequence of transcripts.
    struct ScriptedRecognizer {
        transcriptions: std::vec::IntoIter<String>,
    }

    impl SpeechRecognizer for ScriptedRecognizer {
        fn transcribe(&mut self, _pcm: &[i16]) -> Result<String, Box<dyn Error + Send + Sync>> {
            Ok(self.transcriptions.next().unwrap_or_default())
        }
    }

    #[test]
    fn endpointer_holds_mid_sentence_then_combines() {
        let (utt_tx, utt_rx) = mpsc::sync_channel::<Utterance>(4);
        let (result_tx, result_rx) = mpsc::sync_channel::<SttResult>(4);
        let stop = Arc::new(AtomicBool::new(false));

        // First candidate transcribes without a boundary; the next (combined)
        // candidate resolves it.
        let recognizer = ScriptedRecognizer {
            transcriptions: vec!["I think".to_string(), "I think it's sunny".to_string()]
                .into_iter(),
        };
        let handle = std::thread::spawn(move || {
            stt_loop(Box::new(recognizer), utt_rx, result_tx, stop, 600, false);
        });

        utt_tx
            .send(Utterance {
                pcm: vec![1, 2, 3],
                completed_at: Instant::now(),
            })
            .unwrap();
        // A mid-sentence result must be held, not dispatched immediately.
        assert!(result_rx.recv_timeout(Duration::from_millis(100)).is_err());

        utt_tx
            .send(Utterance {
                pcm: vec![4, 5, 6],
                completed_at: Instant::now(),
            })
            .unwrap();
        // Still no sentence boundary, so it stays held.
        assert!(result_rx.recv_timeout(Duration::from_millis(100)).is_err());

        drop(utt_tx);
        // Shutdown flushes the held turn.
        let result = result_rx.recv_timeout(Duration::from_millis(200)).unwrap();
        assert_eq!(result.question, "I think it's sunny");
        handle.join().unwrap();
    }

    #[test]
    fn endpointer_dispatches_on_boundary() {
        let (utt_tx, utt_rx) = mpsc::sync_channel::<Utterance>(4);
        let (result_tx, result_rx) = mpsc::sync_channel::<SttResult>(4);
        let stop = Arc::new(AtomicBool::new(false));

        let recognizer = ScriptedRecognizer {
            transcriptions: vec!["What's the weather?".to_string()].into_iter(),
        };
        let handle = std::thread::spawn(move || {
            stt_loop(Box::new(recognizer), utt_rx, result_tx, stop, 600, false);
        });

        utt_tx
            .send(Utterance {
                pcm: vec![1, 2, 3],
                completed_at: Instant::now(),
            })
            .unwrap();
        let result = result_rx.recv_timeout(Duration::from_millis(200)).unwrap();
        assert_eq!(result.question, "What's the weather?");
        drop(utt_tx);
        handle.join().unwrap();
    }
}
