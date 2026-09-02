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
use crate::vad::EnergyVad;
use crate::DEFAULT_SAMPLE_RATE;

/// Minimum utterance length (samples at 16 kHz) before it is transcribed.
/// ~120 ms filters out noise blips that would otherwise trigger a wasteful,
/// high-latency LLM call.
const MIN_UTTERANCE_SAMPLES: usize = 1920;
/// Cooldown after playback ends before the capture thread accepts audio again,
/// letting the assistant's audio tail clear.
const POST_PLAYBACK_COOLDOWN_MS: u64 = 250;

/// Shared controls handed to the background capture thread.
struct CaptureCtrl {
    /// True while the assistant is speaking (playback active).
    gate: Arc<AtomicBool>,
}

/// Runs the capture thread for the lifetime of `mic`, sending each completed
/// utterance to `utt_tx`.
fn capture_loop(
    vad: EnergyVad,
    ctrl: CaptureCtrl,
    mic: MicCapture,
    utt_tx: mpsc::Sender<Vec<i16>>,
) {
    let mut vad = vad;
    let mut prev_gate = false;
    let mut cooldown_until = Instant::now() - Duration::from_secs(1);

    while let Ok(pcm) = mic.rx.recv() {
        let gate = ctrl.gate.load(Ordering::Acquire);

        if gate != prev_gate {
            // Either playback started or stopped: drop any partially-built
            // utterance and (on stop) ignore the assistant's audio tail.
            let _ = vad.flush();
            prev_gate = gate;
            if !gate {
                cooldown_until = Instant::now() + Duration::from_millis(POST_PLAYBACK_COOLDOWN_MS);
            }
            continue;
        }

        if gate || Instant::now() < cooldown_until {
            continue;
        }

        for utterance in vad.feed(&pcm) {
            if utterance.len() >= MIN_UTTERANCE_SAMPLES {
                let _ = utt_tx.send(utterance);
            }
        }
    }

    if let Some(utt) = vad.flush() {
        if utt.len() >= MIN_UTTERANCE_SAMPLES {
            let _ = utt_tx.send(utt);
        }
    }
}

/// The main turn-taking session. Owns the pipeline/mic and drives the capture
/// thread + processing loop.
pub struct Session {
    /// RMS threshold (0..1) above which a frame counts as speech.
    pub vad_threshold: f32,
    /// Trailing silence (ms) that ends a spoken utterance.
    pub vad_max_silence_ms: u64,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            vad_threshold: 0.02,
            vad_max_silence_ms: 1200,
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

    /// Set the trailing silence that ends an utterance.
    pub fn with_vad_max_silence(mut self, ms: u64) -> Self {
        self.vad_max_silence_ms = ms;
        self
    }

    /// Run the speech-to-speech session to completion.
    ///
    /// `mic` is moved into a background capture thread; `pipeline` is moved
    /// onto the main thread and processes turns one at a time. If `stop` is
    /// provided, the loop polls it and returns cleanly when it becomes `true`
    /// (e.g. on Ctrl-C).
    pub fn run(
        self,
        mic: MicCapture,
        mut pipeline: Pipeline,
        stop: Option<Arc<AtomicBool>>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let gate = Arc::new(AtomicBool::new(false));
        pipeline = pipeline.with_gate(Some(Arc::clone(&gate)));

        let (utt_tx, utt_rx) = mpsc::channel::<Vec<i16>>();
        let vad = EnergyVad::new(DEFAULT_SAMPLE_RATE)
            .with_threshold(self.vad_threshold)
            .with_max_silence(self.vad_max_silence_ms);
        let ctrl = CaptureCtrl { gate };
        let handle = std::thread::spawn(move || {
            capture_loop(vad, ctrl, mic, utt_tx);
        });

        loop {
            if let Some(s) = &stop {
                if s.load(Ordering::Relaxed) {
                    break;
                }
            }
            match utt_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(utterance) => {
                    pipeline.prompt(&utterance)?;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        handle.join().map_err(|_| "capture thread panicked")?;
        Ok(())
    }
}
