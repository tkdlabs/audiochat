//! Persistent playback engine that plays a queue of PCM buffers in order on a
//! single cpal output stream.
//!
//! Keeping one output stream alive (rather than open/close per chunk) avoids
//! ALSA "I/O error" churn from rapidly opening/closing the device. Callers
//! enqueue 16 kHz mono i16 buffers with `push` and block until everything
//! has finished with `drain`, enabling half-duplex turn-taking.

use std::collections::VecDeque;
use std::error::Error;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::config::AudioConfig;
use crate::resample::LinearResampler;

/// Shared state read/written by the cpal output callback.
///
/// Buffers are stored already converted to the device's native rate and
/// channel layout (done at `push` time), so the callback just copies samples
/// into the output linearly.
struct OutState {
    /// Format-converted, interleaved buffers waiting to play, in order.
    queue: VecDeque<Vec<i16>>,
    /// Buffer currently being emitted.
    cur: Vec<i16>,
    /// Next sample index in `cur`.
    pos: usize,
}

impl OutState {
    fn is_playing(&self) -> bool {
        !self.queue.is_empty() || self.pos < self.cur.len()
    }
}

/// A background playback engine. Sounds pushed via `push` play sequentially in
/// order through a single persistent output stream.
pub struct AudioSink {
    handle: Option<JoinHandle<()>>,
    shared: Arc<Mutex<OutState>>,
    shutdown_tx: Option<mpsc::Sender<()>>,
    native_rate: u32,
    native_channels: usize,
    in_rate: u32,
}

impl AudioSink {
    /// Start a background playback engine, opening the default output device.
    pub fn new(cfg: AudioConfig) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("no default output device")?;
        let supported = device.default_output_config()?;
        let sample_format = supported.sample_format();
        let native_rate = supported.sample_rate();
        let native_channels = supported.channels() as usize;
        let config: cpal::StreamConfig = supported.into();

        let shared: Arc<Mutex<OutState>> = Arc::new(Mutex::new(OutState {
            queue: VecDeque::new(),
            cur: Vec::new(),
            pos: 0,
        }));
        let stream = build_stream(&device, config, sample_format, Arc::clone(&shared))?;
        stream.play()?;

        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            // The callback runs on its own thread; owning the stream here
            // keeps it alive. Block until shutdown; dropping `_stream` then
            // closes the audio device.
            let _stream = stream;
            let _ = shutdown_rx.recv();
        });

        Ok(Self {
            handle: Some(handle),
            shared,
            native_rate,
            native_channels,
            in_rate: cfg.sample_rate,
            shutdown_tx: Some(shutdown_tx),
        })
    }

    /// Enqueue a 16 kHz mono i16 PCM buffer for playback.
    pub fn push(&self, pcm: Vec<i16>) {
        if pcm.is_empty() {
            return;
        }
        let converted =
            convert_to_native(&pcm, self.in_rate, self.native_rate, self.native_channels);
        let mut s = self.shared.lock().unwrap();
        s.queue.push_back(converted);
    }

    /// Block until all audio pushed so far has finished playing.
    pub fn drain(&self) {
        loop {
            {
                let s = self.shared.lock().unwrap();
                if !s.is_playing() {
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Whether the queue is empty and nothing is currently playing.
    pub fn is_idle(&self) -> bool {
        let s = self.shared.lock().unwrap();
        !s.is_playing()
    }

    /// Immediately stop playback: drop all queued audio and silence the output.
    /// Used for barge-in, where the assistant should fall quiet at once.
    pub fn stop(&self) {
        let mut s = self.shared.lock().unwrap();
        s.queue.clear();
        s.cur.clear();
        s.pos = 0;
    }
}

/// Build the output stream whose callback drains `state` into `data`, writing
/// silence when nothing remains.
fn build_stream(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    format: cpal::SampleFormat,
    shared: Arc<Mutex<OutState>>,
) -> Result<cpal::Stream, Box<dyn Error + Send + Sync>> {
    let err_fn = |err| eprintln!("audio stream error: {err}");
    match format {
        cpal::SampleFormat::F32 => {
            let state = Arc::clone(&shared);
            device
                .build_output_stream(
                    config,
                    move |data: &mut [f32], _| {
                        emit(state.lock().unwrap(), data, |v| v as f32 / i16::MAX as f32);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| Box::<dyn Error + Send + Sync>::from(e.to_string()))
        }
        cpal::SampleFormat::I16 => {
            let state = Arc::clone(&shared);
            device
                .build_output_stream(
                    config,
                    move |data: &mut [i16], _| {
                        emit(state.lock().unwrap(), data, |v| v);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| Box::<dyn Error + Send + Sync>::from(e.to_string()))
        }
        cpal::SampleFormat::U8 => {
            let state = Arc::clone(&shared);
            device
                .build_output_stream(
                    config,
                    move |data: &mut [u8], _| {
                        emit(state.lock().unwrap(), data, |v| (v >> 8) as u8 ^ 0x80);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| Box::<dyn Error + Send + Sync>::from(e.to_string()))
        }
        other => Err(format!("unsupported output sample format: {other}").into()),
    }
}

/// Copy samples from `state` into the (interleaved) output buffer.
fn emit<T, F>(mut s: std::sync::MutexGuard<'_, OutState>, data: &mut [T], mut convert: F)
where
    F: FnMut(i16) -> T,
{
    for slot in data.iter_mut() {
        *slot = convert(next_sample(&mut s));
    }
}

/// Return the next sample to output, advancing through queued buffers.
fn next_sample(s: &mut OutState) -> i16 {
    loop {
        if s.pos < s.cur.len() {
            let v = s.cur[s.pos];
            s.pos += 1;
            return v;
        }
        match s.queue.pop_front() {
            Some(next) => {
                s.cur = next;
                s.pos = 0;
            }
            None => return 0, // silence
        }
    }
}

/// Resample mono 16 kHz -> native rate and expand to the device channel count
/// (interleaved). Channels are filled by repeating each mono sample.
fn convert_to_native(pcm: &[i16], in_rate: u32, out_rate: u32, channels: usize) -> Vec<i16> {
    let mono: Vec<i16> = if in_rate == out_rate {
        pcm.to_vec()
    } else {
        let mut up = LinearResampler::new(in_rate, out_rate);
        let in_f: Vec<f32> = pcm.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
        up.process(&in_f)
            .into_iter()
            .map(|v| (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect()
    };

    if channels <= 1 {
        return mono;
    }
    let mut out = Vec::with_capacity(mono.len() * channels);
    for &s in &mono {
        for _ in 0..channels {
            out.push(s);
        }
    }
    out
}

impl Drop for AudioSink {
    fn drop(&mut self) {
        // Dropping the sender unblocks the worker's `recv`, causing it to exit
        // and drop the stream (closing the audio device). Then join.
        self.shutdown_tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
