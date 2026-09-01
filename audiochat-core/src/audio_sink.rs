//! Sequential audio output queue that plays PCM chunks on a background thread.

use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::config::AudioConfig;
use crate::playback::play_pcm;

/// Plays a sequence of 16 kHz mono i16 PCM chunks in order on a background
/// thread, letting callers enqueue audio without blocking on playback.
pub struct AudioSink {
    tx: Option<mpsc::Sender<Vec<i16>>>,
    handle: Option<JoinHandle<()>>,
}

impl AudioSink {
    pub fn new(cfg: AudioConfig) -> Self {
        let (tx, rx) = mpsc::channel::<Vec<i16>>();
        let handle = std::thread::spawn(move || {
            while let Ok(pcm) = rx.recv() {
                if let Err(e) = play_pcm(&pcm, cfg) {
                    eprintln!("playback error: {e}");
                }
            }
        });
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Enqueue a PCM chunk for playback (non-blocking).
    ///
    /// Returns `false` if the sink has been dropped/closed.
    pub fn push(&self, pcm: Vec<i16>) -> bool {
        match &self.tx {
            Some(tx) => tx.send(pcm).is_ok(),
            None => false,
        }
    }
}

impl Drop for AudioSink {
    fn drop(&mut self) {
        // Dropping the sender closes the channel; the worker drains remaining
        // queued chunks, then exits when the channel is empty.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
