//! Latency benchmark: run one turn through the real LLM -> TTS pipeline (no mic)
//! and print the per-stage timing that `--s2s --verbose` reports.
//!
//! Usage:
//!   cargo run --release -p audiochat-cli --example latency -- "What is 2+2?"
//! Env: the usual AUDIOCHAT_LLM_* / AUDIOCHAT_TTS_* variables.

use std::error::Error;
use std::time::Instant;

use audiochat_core::{Llm, Pipeline, SpeechTiming, TextToSpeech};
use audiochat_llm::{opencode_go_api_key, OpenAiCompatible};
use audiochat_tts_audiocpp::AudioCppTts;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let question = if args.is_empty() {
        "What is the capital of France?".to_string()
    } else {
        args.join(" ")
    };

    let model =
        std::env::var("AUDIOCHAT_LLM_MODEL").unwrap_or_else(|_| "deepseek-v4.1-flash".to_string());
    let mut llm = OpenAiCompatible::new(model).with_thinking(Some(false));
    if let Some(key) = opencode_go_api_key() {
        llm = llm.with_api_key(key);
    }

    let base = std::env::var("AUDIOCHAT_TTS_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());
    let tts_model =
        std::env::var("AUDIOCHAT_TTS_MODEL").unwrap_or_else(|_| "qwen3-tts".to_string());
    let voice = std::env::var("AUDIOCHAT_TTS_VOICE").unwrap_or_else(|_| "Serena".to_string());
    let instruct = std::env::var("AUDIOCHAT_TTS_INSTRUCT").ok();
    let tts: Box<dyn TextToSpeech> = Box::new(
        AudioCppTts::with_base(base, tts_model)
            .with_voice(voice)
            .with_instruct(instruct),
    );

    let mut pipeline = Pipeline::new(Box::new(llm) as Box<dyn Llm>, tts);
    pipeline.speak_replies = false;
    pipeline.verbose = true;

    eprintln!("benchmarking: {question}");
    let start = Instant::now();
    pipeline.respond(&question, start, SpeechTiming::default())?;
    Ok(())
}
