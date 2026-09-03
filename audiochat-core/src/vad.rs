//! Voice activity detection: segments raw PCM into speech utterances.

use std::collections::VecDeque;

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
    /// RMS (0..1) of the most recently fed frame, for callers that want to
    /// apply a different threshold than `threshold` (e.g. barge-in detection).
    last_rms: f32,
    /// Cumulative samples of speech seen so far in the current utterance.
    speech_samples: usize,
    /// Rolling buffer of recent silence, prepended to each utterance so the
    /// first phoneme isn't clipped by frame quantization.
    pre_roll: VecDeque<i16>,
    /// Maximum pre-roll length in samples.
    pre_roll_len: usize,
    /// Whether to estimate a background noise floor and raise the effective
    /// threshold above `threshold` as the room gets noisier.
    adaptive: bool,
    /// Rolling window of recent frame RMS values; its minimum is the noise floor.
    noise_window: VecDeque<f32>,
    /// Maximum number of frames kept in `noise_window` (~2 s).
    noise_window_len: usize,
    /// Multiplier applied to the noise floor when `adaptive` is enabled.
    noise_ratio: f32,
}

impl EnergyVad {
    pub fn new(sample_rate: u32) -> Self {
        let frame_len = (sample_rate as usize * 30) / 1000; // 30 ms
                                                            // ~2 s of 30 ms frames, used to estimate the background noise floor.
        let noise_window_len = (2 * sample_rate as usize) / frame_len;
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
            last_rms: 0.0,
            speech_samples: 0,
            pre_roll: VecDeque::new(),
            pre_roll_len: (sample_rate as usize * 200) / 1000, // 200 ms
            adaptive: true,
            noise_window: VecDeque::new(),
            noise_window_len,
            noise_ratio: 2.0,
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

    /// Configure how much leading silence (ms) is prepended to each utterance.
    pub fn with_pre_roll_ms(mut self, ms: u64) -> Self {
        self.pre_roll_len = self.frame_len * ms as usize / self.frame_ms as usize;
        self
    }

    /// Enable or disable adaptive noise-floor tracking. When enabled (default),
    /// the effective threshold is `max(threshold, noise_floor * noise_ratio)`,
    /// so a noisier room automatically raises the threshold.
    pub fn with_adaptive_noise(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Set the multiplier applied to the estimated noise floor when adaptive
    /// noise tracking is enabled (default 2.0).
    pub fn with_noise_ratio(mut self, ratio: f32) -> Self {
        self.noise_ratio = ratio;
        self
    }

    /// Estimated background noise floor: the minimum frame RMS seen in the
    /// recent noise window (0.0 before any audio has been fed).
    pub fn noise_floor(&self) -> f32 {
        self.noise_window
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(0.0)
    }

    /// The RMS threshold currently used to classify speech, after applying
    /// adaptive noise-floor tracking (if enabled).
    pub fn current_threshold(&self) -> f32 {
        if self.adaptive {
            self.threshold.max(self.noise_floor() * self.noise_ratio)
        } else {
            self.threshold
        }
    }

    /// Feed a block of PCM, returning any completed utterances.
    pub fn feed(&mut self, pcm: &[i16]) -> Vec<Vec<i16>> {
        self.leftover.extend_from_slice(pcm);
        let mut completed = Vec::new();

        while self.leftover.len() >= self.frame_len {
            let frame: Vec<i16> = self.leftover.drain(..self.frame_len).collect();
            let frame_rms = rms(&frame);
            let speech = frame_rms >= self.current_threshold();
            self.last_rms = frame_rms;
            self.last_frame_speech = speech;

            self.noise_window.push_back(frame_rms);
            if self.noise_window.len() > self.noise_window_len {
                self.noise_window.pop_front();
            }

            if speech {
                if !self.armed {
                    // Speech onset: prepend the captured pre-speech audio so the
                    // first phoneme isn't clipped by frame quantization.
                    self.armed = true;
                    self.utterance.extend(self.pre_roll.iter().copied());
                }
                self.pre_roll.clear();
                self.silence_frames = 0;
                self.speech_samples += frame.len();
                self.utterance.extend_from_slice(&frame);
            } else {
                self.pre_roll.extend(frame.iter().copied());
                let excess = self.pre_roll.len().saturating_sub(self.pre_roll_len);
                if excess > 0 {
                    self.pre_roll.drain(..excess);
                }
                if self.armed {
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
        }

        completed
    }

    /// Whether the most recently fed frame was classified as speech.
    pub fn speech_in_last_frame(&self) -> bool {
        self.last_frame_speech
    }

    /// RMS (0..1) of the most recently fed frame, independent of `threshold`.
    pub fn last_frame_rms(&self) -> f32 {
        self.last_rms
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
        self.pre_roll.clear();
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

    #[test]
    fn prepends_pre_roll_before_speech() {
        let mut vad = EnergyVad::new(16_000);
        let frame = 480; // 30 ms
        vad.feed(&silence(frame)); // 1 silent frame -> pre-roll
        vad.feed(&speech(frame)); // onset: prepends the silent pre-roll
        let utt = vad.flush().unwrap();
        assert_eq!(utt.len(), frame * 2);
        assert!(utt[..frame].iter().all(|&s| s == 0));
    }

    #[test]
    fn reports_last_frame_rms() {
        let mut vad = EnergyVad::new(16_000);
        let frame = 480;
        vad.feed(&speech(frame));
        assert!(vad.last_frame_rms() > 0.0);
        vad.feed(&silence(frame));
        assert_eq!(vad.last_frame_rms(), 0.0);
    }

    #[test]
    fn adaptive_noise_raises_threshold() {
        let mut vad = EnergyVad::new(16_000)
            .with_adaptive_noise(true)
            .with_threshold(0.01);
        let frame = 480;
        // Background noise at RMS ~0.02.
        vad.feed(&vec![655; frame]);
        // The noise floor raises the threshold above the noise level.
        assert!(vad.current_threshold() > 0.02);
        // A frame just above the noise but below the raised threshold is silence.
        vad.feed(&vec![983; frame]); // RMS ~0.03
        assert!(!vad.speech_in_last_frame());
    }

    #[test]
    fn adaptive_noise_tracks_rising_noise() {
        let mut vad = EnergyVad::new(16_000)
            .with_adaptive_noise(true)
            .with_threshold(0.01);
        let frame = 480;
        // Quiet room: threshold stays at the floor.
        for _ in 0..70 {
            vad.feed(&vec![1; frame]);
        }
        assert_eq!(vad.current_threshold(), 0.01);
        // Sustained loud background noise (~2 s) pushes the floor up.
        let noise = vec![1966; frame]; // RMS ~0.06
        for _ in 0..70 {
            vad.feed(&noise);
        }
        assert!(vad.current_threshold() > 0.06);
    }

    #[test]
    fn non_adaptive_uses_fixed_threshold() {
        let mut vad = EnergyVad::new(16_000)
            .with_adaptive_noise(false)
            .with_threshold(0.01);
        let frame = 480;
        for _ in 0..70 {
            vad.feed(&vec![655; frame]); // RMS ~0.02 noise
        }
        // Without adaptation the threshold is untouched by the noise.
        assert_eq!(vad.current_threshold(), 0.01);
    }
}
