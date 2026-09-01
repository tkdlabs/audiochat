//! Simple audio playback to the default output device via cpal.

use std::error::Error;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::config::AudioConfig;

/// Shared playback state: the prepared buffer and the next sample index.
struct PlaybackState {
    pcm: Vec<i16>,
    pos: usize,
}

/// Play 16 kHz mono i16 PCM to the default output device, blocking until the
/// buffer has been fully written to the device.
pub fn play_pcm(pcm: &[i16], cfg: AudioConfig) -> Result<(), Box<dyn Error + Send + Sync>> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no default output device")?;
    let supported = device.default_output_config()?;
    let sample_format = supported.sample_format();
    let native_rate = supported.sample_rate();
    let native_channels = supported.channels() as usize;
    let config: cpal::StreamConfig = supported.into();

    // Prepare a 16k mono buffer resampled/expanded to the device native format.
    let prepared = resample_to_native(pcm, cfg.sample_rate, native_rate, native_channels);
    let state: Arc<Mutex<PlaybackState>> = Arc::new(Mutex::new(PlaybackState {
        pcm: prepared,
        pos: 0,
    }));
    let (done_tx, done_rx) = mpsc::channel::<()>();

    let err_fn = |err| eprintln!("audio stream error: {err}");

    let stream = match sample_format {
        cpal::SampleFormat::I16 => device.build_output_stream(
            config,
            {
                let state = Arc::clone(&state);
                let done_tx = done_tx.clone();
                move |data: &mut [i16], _| fill(&state, data, &done_tx, |s| s)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U8 => device.build_output_stream(
            config,
            {
                let state = Arc::clone(&state);
                let done_tx = done_tx.clone();
                move |data: &mut [u8], _| fill(&state, data, &done_tx, |s| ((s >> 8) as u8) ^ 0x80)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_output_stream(
            config,
            {
                let state = Arc::clone(&state);
                let done_tx = done_tx.clone();
                move |data: &mut [f32], _| {
                    fill(&state, data, &done_tx, |s| s as f32 / i16::MAX as f32)
                }
            },
            err_fn,
            None,
        )?,
        other => return Err(format!("unsupported output sample format: {other}").into()),
    };

    stream.play()?;

    // Wait for playback to finish (best-effort with a generous timeout).
    let _ = done_rx.recv_timeout(std::time::Duration::from_secs(120));
    drop(stream);
    Ok(())
}

/// Advance through the shared buffer, converting each i16 sample with `convert`.
/// Signals `done` when the buffer is exhausted.
fn fill<T, F>(
    state: &Arc<Mutex<PlaybackState>>,
    data: &mut [T],
    done: &mpsc::Sender<()>,
    mut convert: F,
) where
    F: FnMut(i16) -> T,
{
    let mut s = state.lock().unwrap();
    for frame in data.iter_mut() {
        let v = s.pcm.get(s.pos).copied().unwrap_or(0);
        s.pos += 1;
        *frame = convert(v);
    }
    if s.pos >= s.pcm.len() {
        let _ = done.send(());
    }
}

/// Expand 16k mono to the device channel count; linearly resample if the rate
/// differs (e.g. 16 kHz -> 48 kHz).
fn resample_to_native(pcm: &[i16], in_rate: u32, out_rate: u32, channels: usize) -> Vec<i16> {
    let mono: Vec<i16> = if in_rate == out_rate {
        pcm.to_vec()
    } else {
        let out_len = pcm.len() * out_rate as usize / in_rate as usize;
        let mut mono = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f64 * in_rate as f64 / out_rate as f64;
            let idx = src.floor() as usize;
            let frac = src - idx as f64;
            let a = pcm.get(idx).copied().unwrap_or(0) as f64;
            let b = pcm.get(idx + 1).copied().unwrap_or(0) as f64;
            mono.push((a + (b - a) * frac) as i16);
        }
        mono
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
