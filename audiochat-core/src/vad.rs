//! Voice activity detection: segments raw PCM into speech utterances.

/// Simple energy-based voice activity detector.
///
/// Works on 16 kHz mono `i16` PCM. Segments continuous speech into utterances
/// separated by silence longer than `max_silence_ms`.
#[derive(Debug)]
pub struct EnergyVad {
    /// RMS threshold (0..1) below which a frame is "silence".
    threshold: f32,
    /// Frame length in samples (default 30 ms).
    frame_len: usize,
    /// Silence (in ms) that ends an utterance.
    max_silence_ms: u64,
    frame_ms: u64,
    /// Leftover input samples not yet consumed by a full frame.
    leftover: Vec<i16>,
    /// Accumulated speech samples for the current utterance.
    utterance: Vec<i16>,
    /// Consecutive silent frames since last speech.
    silence_frames: u64,
    /// Whether any speech has been seen since the last flush.
    armed: bool,
    /// Whether the most recently fed frame was classified as speech.
    last_frame_speech: bool,
    /// Cumulative samples of speech seen so far in the current utterance.
    speech_samples: usize,
}

impl EnergyVad {
    pub fn new(sample_rate: u32) -> Self {
        let frame_len = (sample_rate as usize * 30) / 1000; // 30 ms
        Self {
            threshold: 0.01,
            frame_len,
            max_silence_ms: 600,
            frame_ms: 30,
            leftover: Vec::new(),
            utterance: Vec::new(),
            silence_frames: 0,
            armed: false,
            last_frame_speech: false,
            speech_samples: 0,
        }
    }

    /// Configure the RMS threshold (0..1) above which audio is speech.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Configure how many ms of trailing silence ends an utterance.
    pub fn with_max_silence(mut self, ms: u64) -> Self {
        self.max_silence_ms = ms;
        self
    }

    /// Feed a block of PCM, returning any completed utterances.
    pub fn feed(&mut self, pcm: &[i16]) -> Vec<Vec<i16>> {
        self.leftover.extend_from_slice(pcm);
        let mut completed = Vec::new();

        while self.leftover.len() >= self.frame_len {
            let frame: Vec<i16> = self.leftover.drain(..self.frame_len).collect();
            let speech = rms(&frame) >= self.threshold;
            self.last_frame_speech = speech;

            if speech {
                self.armed = true;
                self.silence_frames = 0;
                self.speech_samples += frame.len();
                self.utterance.extend_from_slice(&frame);
            } else if self.armed {
                self.silence_frames += 1;
                if self.silence_frames * self.frame_ms >= self.max_silence_ms {
                    completed.push(std::mem::take(&mut self.utterance));
                    self.armed = false;
                    self.speech_samples = 0;
                    self.silence_frames = 0;
                } else {
                    self.utterance.extend_from_slice(&frame);
                }
            }
        }

        completed
    }

    /// Whether the most recently fed frame was classified as speech.
    pub fn speech_in_last_frame(&self) -> bool {
        self.last_frame_speech
    }

    /// Total speech samples accumulated for the in-progress utterance.
    pub fn speech_samples(&self) -> usize {
        self.speech_samples
    }

    /// Force-flush any in-progress utterance.
    pub fn flush(&mut self) -> Option<Vec<i16>> {
        let out = if self.armed {
            Some(std::mem::take(&mut self.utterance))
        } else {
            None
        };
        self.armed = false;
        self.silence_frames = 0;
        self.speech_samples = 0;
        self.leftover.clear();
        out
    }
}

fn rms(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: i64 = frame.iter().map(|&s| (s as i64) * (s as i64)).sum();
    let mean = sum as f32 / frame.len() as f32;
    (mean / (i16::MAX as f32 * i16::MAX as f32)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(len: usize) -> Vec<i16> {
        vec![0; len]
    }

    fn speech(len: usize) -> Vec<i16> {
        (0..len)
            .map(|i| if i % 3 == 0 { 4000 } else { 0 })
            .collect()
    }

    #[test]
    fn segments_single_utterance() {
        let mut vad = EnergyVad::new(16_000).with_max_silence(600);
        let frame = 480; // 30 ms
        let mut out = Vec::new();
        out.extend(vad.feed(&speech(frame * 5)));
        out.extend(vad.feed(&silence(frame * 30))); // 900 ms silence -> flush
        assert_eq!(out.len(), 1);
        assert!(!out[0].is_empty());
    }

    #[test]
    fn discards_leading_silence() {
        let mut vad = EnergyVad::new(16_000).with_max_silence(600);
        let frame = 480;
        let out = vad.feed(&silence(frame * 5));
        assert!(out.is_empty());
    }

    #[test]
    fn flush_returns_partial_utterance() {
        let mut vad = EnergyVad::new(16_000).with_max_silence(600);
        let frame = 480;
        vad.feed(&speech(frame * 3));
        let out = vad.flush();
        assert!(out.is_some());
        assert!(!out.unwrap().is_empty());
    }
}
