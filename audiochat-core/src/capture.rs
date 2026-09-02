//! Microphone capture via cpal, exposed as a blocking channel.
//!
//! cpal delivers audio through a callback; we translate that into a channel so
//! the rest of the pipeline can consume it sequentially. The native device
//! format (typically 48 kHz stereo F32) is converted to 16 kHz mono i16 using
//! linear resampling and channel averaging.

use std::error::Error;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::config::AudioConfig;
use crate::resample::LinearResampler;

/// A running microphone capture. Read audio blocks from `rx`.
pub struct MicCapture {
    /// Receiving end of the captured, resampled 16 kHz mono i16 frames.
    pub rx: mpsc::Receiver<Vec<i16>>,
    _stream: cpal::Stream,
}

/// Selects an input device by case-insensitive substring match on its name.
fn select_input_device(
    host: &cpal::Host,
    name: &str,
) -> Result<cpal::Device, Box<dyn Error + Send + Sync>> {
    let needle = name.to_lowercase();
    let mut best: Option<cpal::Device> = None;
    for device in host
        .input_devices()
        .map_err(|e| format!("failed to enumerate input devices: {e}"))?
    {
        let d = device.to_string();
        if d.to_lowercase().contains(&needle) {
            best = Some(device);
        }
    }
    best.ok_or_else(|| {
        {
            let names: Vec<String> = host
                .input_devices()
                .ok()
                .into_iter()
                .flatten()
                .map(|d| d.to_string())
                .collect();
            format!(
                "no input device matching '{name}'. Available inputs: {}",
                if names.is_empty() {
                    "<none>".to_string()
                } else {
                    names.join(", ")
                }
            )
        }
        .into()
    })
}

/// Shared state used by the capture callback: a resampler plus the output
/// channel. Only the resampler is mutated.
struct SharedOutput {
    resampler: Mutex<LinearResampler>,
    tx: mpsc::SyncSender<Vec<i16>>,
}

impl SharedOutput {
    fn push(&self, mono_f32: Vec<f32>) {
        let mut r = self.resampler.lock().unwrap();
        let out = r.process(&mono_f32);
        drop(r);
        let pcm: Vec<i16> = out
            .into_iter()
            .map(|v| (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        // `try_send` never blocks the real-time audio callback; if the bounded
        // channel is full we drop the frame rather than stall the stream.
        let _ = self.tx.try_send(pcm);
    }
}

impl MicCapture {
    /// Start capturing from the default input device, resampled to `cfg`.
    pub fn start(cfg: AudioConfig) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Self::start_with_device(cfg, None)
    }

    /// Start capturing from an input device selected by name, resampled to `cfg`.
    ///
    /// When `device_name` is `None`, the default input device is used. The name
    /// is matched as a case-insensitive substring against available input devices.
    pub fn start_with_device(
        cfg: AudioConfig,
        device_name: Option<&str>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => select_input_device(&host, name)?,
            None => host
                .default_input_device()
                .ok_or("no default input device")?,
        };
        let device_desc = device.to_string();
        let supported = device.default_input_config()?;
        let sample_format = supported.sample_format();
        let native_rate = supported.sample_rate();
        let native_channels = supported.channels() as usize;
        let config: cpal::StreamConfig = supported.into();
        eprintln!("audiochat: using input device '{device_desc}' ({native_rate} Hz, {native_channels} ch)");

        let (tx, rx) = mpsc::sync_channel::<Vec<i16>>(64);
        let shared = Arc::new(SharedOutput {
            resampler: Mutex::new(LinearResampler::new(native_rate, cfg.sample_rate)),
            tx,
        });

        let err_fn = |err| eprintln!("audio stream error: {err}");
        let to_mono = |channels: usize| {
            move |frames: &[f32]| -> Vec<f32> {
                frames
                    .chunks_exact(channels)
                    .map(|ch| ch.iter().sum::<f32>() / channels as f32)
                    .collect()
            }
        };

        let stream = match sample_format {
            cpal::SampleFormat::I16 => {
                let shared = Arc::clone(&shared);
                let to_mono = to_mono(native_channels);
                device.build_input_stream(
                    config,
                    move |data: &[i16], _| {
                        let mono: Vec<f32> =
                            data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        shared.push(to_mono(&mono));
                    },
                    err_fn,
                    None,
                )?
            }
            cpal::SampleFormat::U8 => {
                let shared = Arc::clone(&shared);
                let to_mono = to_mono(native_channels);
                device.build_input_stream(
                    config,
                    move |data: &[u8], _| {
                        let frames: Vec<f32> =
                            data.iter().map(|&b| (b as f32 / 128.0) - 1.0).collect();
                        shared.push(to_mono(&frames));
                    },
                    err_fn,
                    None,
                )?
            }
            cpal::SampleFormat::F32 => {
                let shared = Arc::clone(&shared);
                let to_mono = to_mono(native_channels);
                device.build_input_stream(
                    config,
                    move |data: &[f32], _| {
                        shared.push(to_mono(data));
                    },
                    err_fn,
                    None,
                )?
            }
            other => return Err(format!("unsupported sample format: {other}").into()),
        };

        stream.play()?;
        Ok(Self {
            rx,
            _stream: stream,
        })
    }
}
