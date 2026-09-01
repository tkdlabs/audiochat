//! End-to-end speech-to-speech pipeline orchestration.

use std::error::Error;
use std::time::Instant;

use crate::audio_sink::AudioSink;
use crate::config::AudioConfig;
use crate::traits::{Llm, SpeechRecognizer, TextToSpeech};
use crate::vad::EnergyVad;

/// Sentence-ending punctuation used to chunk streamed LLM tokens for TTS.
fn is_sentence_boundary(segment: &str) -> bool {
    segment.ends_with('.') || segment.ends_with('!') || segment.ends_with('?')
}

/// Run a fallible closure with a fixed number of retries and exponential
/// backoff between attempts. Use for transient backend failures.
fn with_retry<T>(
    what: &str,
    attempts: u32,
    base_delay_ms: u64,
    mut f: impl FnMut() -> Result<T, Box<dyn Error + Send + Sync>>,
) -> Result<T, Box<dyn Error + Send + Sync>> {
    let mut delay = base_delay_ms;
    let mut last_msg: Option<String> = None;
    for attempt in 0..attempts {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                let msg = e.to_string();
                if attempt + 1 < attempts {
                    eprintln!("audiochat: {what} failed (retry {}): {msg}", attempt + 1);
                    std::thread::sleep(std::time::Duration::from_millis(delay));
                    delay *= 2;
                }
                last_msg = Some(msg);
            }
        }
    }
    Err(last_msg.unwrap_or_else(|| format!("{what} failed")).into())
}

/// Per-turn latency metrics, in milliseconds, for a spoken round-trip.
#[derive(Debug, Clone, Copy, Default)]
pub struct TurnTiming {
    /// End of speech (utterance handed to the pipeline) -> STT result.
    pub stt_ms: u64,
    /// End of speech -> first LLM token received.
    pub first_token_ms: u64,
    /// End of speech -> first synthesized audio enqueued for playback.
    pub first_audio_ms: u64,
    /// End of speech -> last token received (full reply streamed).
    pub full_stream_ms: u64,
    /// End of speech -> last audio enqueued (whole reply spoken/printed).
    pub total_ms: u64,
}

impl TurnTiming {
    fn print(&self) {
        eprintln!(
            "  └ latency: stt={}ms first-token={}ms first-audio={}ms stream={}ms total={}ms",
            self.stt_ms,
            self.first_token_ms,
            self.first_audio_ms,
            self.full_stream_ms,
            self.total_ms
        );
    }
}

/// Orchestrates mic -> VAD -> STT -> LLM -> TTS -> playback.
pub struct Pipeline {
    vad: EnergyVad,
    stt: Box<dyn SpeechRecognizer>,
    pub llm: Box<dyn Llm>,
    tts: Box<dyn TextToSpeech>,
    cfg: AudioConfig,
    sink: Option<AudioSink>,
    /// If `true`, spoken replies are played aloud; otherwise only printed.
    pub speak_replies: bool,
    /// Whether to print per-turn latency metrics to stderr.
    pub verbose: bool,
    /// RMS threshold (0..1) above which a frame counts as speech.
    vad_threshold: f32,
    /// Trailing silence (ms) that ends an utterance.
    vad_max_silence_ms: u64,
}

impl Pipeline {
    pub fn new(
        stt: Box<dyn SpeechRecognizer>,
        llm: Box<dyn Llm>,
        tts: Box<dyn TextToSpeech>,
    ) -> Self {
        Self {
            vad: EnergyVad::new(crate::DEFAULT_SAMPLE_RATE),
            stt,
            llm,
            tts,
            cfg: AudioConfig::default(),
            sink: None,
            speak_replies: true,
            verbose: false,
            vad_threshold: 0.02,
            vad_max_silence_ms: 1200,
        }
    }

    /// Set the VAD RMS threshold (0..1) above which audio counts as speech.
    ///
    /// A higher value treats signal more leniently as silence, so a lower mic
    /// gain / quiet room is less likely to be mistaken for continuous speech.
    pub fn with_vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold.clamp(0.0, 1.0);
        self.vad = self.vad.with_threshold(self.vad_threshold);
        self
    }

    /// Set the trailing silence (ms) that ends a spoken utterance.
    pub fn with_vad_max_silence(mut self, ms: u64) -> Self {
        self.vad_max_silence_ms = ms;
        self.vad = self.vad.with_max_silence(ms);
        self
    }

    /// Process a block of raw mic PCM, returning any transcripts produced.
    pub fn feed(&mut self, pcm: &[i16]) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let mut out = Vec::new();
        for utterance in self.vad.feed(pcm) {
            if let Some(reply) = self.process_utterance(&utterance)? {
                out.push(reply);
            }
        }
        Ok(out)
    }

    /// Transcribe `utterance`, ask the LLM, and speak the streamed reply.
    fn process_utterance(
        &mut self,
        utterance: &[i16],
    ) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
        let start = Instant::now();
        let mut timing = TurnTiming::default();

        let question = self.stt.transcribe(utterance)?;
        timing.stt_ms = start.elapsed().as_millis() as u64;
        if question.trim().is_empty() {
            return Ok(None);
        }
        println!("you:  {question}");

        println!("ai:   ",);
        // Retry generating the response; the reply could stream gradually, so
        // only the initial call is retried (streams resume is not possible).
        let mut resp = with_retry("llm generate", 3, 300, || self.llm.generate(&question))?;
        let mut reply = String::new();
        let mut segment = String::new();
        let mut saw_first_token = false;
        let mut saw_first_audio = false;

        if let Some(stream) = resp.stream.take() {
            for item in stream {
                let token = item?;
                if !saw_first_token {
                    timing.first_token_ms = start.elapsed().as_millis() as u64;
                    saw_first_token = true;
                }
                reply.push_str(&token);
                segment.push_str(&token);
                print!("{token}");
                use std::io::Write;
                let _ = std::io::stdout().flush();

                if is_sentence_boundary(segment.trim_end()) {
                    self.synthesize_and_play(&segment)?;
                    if !saw_first_audio {
                        timing.first_audio_ms = start.elapsed().as_millis() as u64;
                        saw_first_audio = true;
                    }
                    segment.clear();
                }
            }
        }
        timing.full_stream_ms = start.elapsed().as_millis() as u64;

        if !segment.trim().is_empty() {
            self.synthesize_and_play(&segment)?;
            if !saw_first_audio {
                timing.first_audio_ms = start.elapsed().as_millis() as u64;
            }
        }
        if !reply.is_empty() {
            println!();
        }
        if self.verbose && !reply.is_empty() {
            timing.total_ms = start.elapsed().as_millis() as u64;
            timing.print();
        }
        Ok((!reply.is_empty()).then_some(reply))
    }

    /// Play `text` by synthesizing it and enqueuing to the audio sink.
    fn synthesize_and_play(&mut self, text: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        // LLM replies are often Markdown; Piper would read the markup symbols
        // aloud, so speak a plain-text version (the terminal keeps the original).
        let plain = crate::markdown::strip_markdown(text);
        let plain = plain.trim();
        if plain.is_empty() {
            return Ok(());
        }
        self.ensure_sink()?;
        let pcm = with_retry("tts synthesize", 2, 200, || self.tts.synthesize(plain))?;
        if self.speak_replies && !pcm.is_empty() {
            if let Some(sink) = &self.sink {
                sink.push(pcm);
            }
        }
        Ok(())
    }

    /// Lazily open the audio sink if replies are to be spoken.
    fn ensure_sink(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if self.speak_replies && self.sink.is_none() {
            self.sink = Some(AudioSink::new(self.cfg)?);
        }
        Ok(())
    }

    /// Block until all synthesized audio for the current reply has finished
    /// playing. Used for half-duplex turn-taking: the pipeline should not
    /// listen for the next utterance until the previous reply has completed.
    pub fn wait_playback_done(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if self.speak_replies {
            self.ensure_sink()?;
        }
        if let Some(sink) = &self.sink {
            sink.drain();
        }
        Ok(())
    }

    /// Force-flush any pending VAD utterance (e.g. on shutdown).
    pub fn flush(&mut self) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let mut out = Vec::new();
        if let Some(utterance) = self.vad.flush() {
            if let Some(reply) = self.process_utterance(&utterance)? {
                out.push(reply);
            }
        }
        Ok(out)
    }
}
