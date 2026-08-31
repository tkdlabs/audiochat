//! Microphone capture via cpal, exposed as a blocking channel.
//!
//! cpal delivers audio through a callback; we translate that into a channel so
//! the rest of the pipeline can consume it sequentially.

use std::error::Error;
use std::sync::mpsc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::config::AudioConfig;

/// A running microphone capture. Read audio blocks from `rx`.
pub struct MicCapture {
    /// Receiving end of the captured audio frames.
    pub rx: mpsc::Receiver<Vec<i16>>,
    _stream: cpal::Stream,
}

impl MicCapture {
    /// Start capturing from the default input device at `cfg`.
    pub fn start(cfg: AudioConfig) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("no default input device")?;
        let supported = device.default_input_config()?;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();

        let (tx, rx) = mpsc::channel::<Vec<i16>>();
        let sample_rate = cfg.sample_rate;
        let channels = cfg.channels;

        let err_fn = |err| eprintln!("audio stream error: {err}");
        let _ = channels;

        let stream = match sample_format {
            cpal::SampleFormat::I16 => device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    let _ = tx.send(data.to_vec());
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::U8 => device.build_input_stream(
                config,
                move |data: &[u8], _| {
                    let frame: Vec<i16> = data.iter().map(|&b| (b as i16 - 128) << 8).collect();
                    let _ = tx.send(frame);
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::F32 => device.build_input_stream(
                config,
                move |data: &[f32], _| {
                    let frame: Vec<i16> = data
                        .iter()
                        .map(|&v| (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                        .collect();
                    let _ = tx.send(frame);
                },
                err_fn,
                None,
            )?,
            other => return Err(format!("unsupported sample format: {other}").into()),
        };

        // Note: device native config rarely matches 16 kHz mono exactly. For M1 we
        // capture at the device's native rate and treat it as 16 kHz; resampling is
        // deferred to a later milestone. `sample_rate` is kept for future resampling.
        let _ = sample_rate;

        stream.play()?;
        Ok(Self {
            rx,
            _stream: stream,
        })
    }
}
