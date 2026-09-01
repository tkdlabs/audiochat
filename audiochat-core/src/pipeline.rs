//! End-to-end speech-to-speech pipeline orchestration.

use std::error::Error;

use crate::audio_sink::AudioSink;
use crate::config::AudioConfig;
use crate::traits::{Llm, SpeechRecognizer, TextToSpeech};
use crate::vad::EnergyVad;

/// Sentence-ending punctuation used to chunk streamed LLM tokens for TTS.
fn is_sentence_boundary(segment: &str) -> bool {
    segment.ends_with('.') || segment.ends_with('!') || segment.ends_with('?')
}

/// Orchestrates mic -> VAD -> STT -> LLM -> TTS -> playback.
pub struct Pipeline {
    vad: EnergyVad,
    pub stt: Box<dyn SpeechRecognizer>,
    pub llm: Box<dyn Llm>,
    tts: Box<dyn TextToSpeech>,
    cfg: AudioConfig,
    sink: Option<AudioSink>,
    /// If `true`, spoken replies are played aloud; otherwise only printed.
    pub speak_replies: bool,
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
        }
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
        let question = self.stt.transcribe(utterance)?;
        if question.trim().is_empty() {
            return Ok(None);
        }
        println!("you:  {question}");

        println!("ai:   ",);
        let mut resp = self.llm.generate(&question)?;
        let mut reply = String::new();
        let mut segment = String::new();

        if let Some(stream) = resp.stream.take() {
            for item in stream {
                let token = item?;
                reply.push_str(&token);
                segment.push_str(&token);
                print!("{token}");
                use std::io::Write;
                let _ = std::io::stdout().flush();

                if is_sentence_boundary(segment.trim_end()) {
                    self.synthesize_and_play(&segment)?;
                    segment.clear();
                }
            }
        }

        if !segment.trim().is_empty() {
            self.synthesize_and_play(&segment)?;
        }
        if !reply.is_empty() {
            println!();
        }
        Ok((!reply.is_empty()).then_some(reply))
    }

    /// Play `text` by synthesizing it and enqueuing to the audio sink.
    fn synthesize_and_play(&mut self, text: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        if self.speak_replies && self.sink.is_none() {
            self.sink = Some(AudioSink::new(self.cfg));
        }
        let pcm = self.tts.synthesize(text)?;
        if self.speak_replies && !pcm.is_empty() {
            if let Some(sink) = &self.sink {
                sink.push(pcm);
            }
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
