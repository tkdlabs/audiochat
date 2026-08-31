//! Shared configuration types.

/// Audio capture configuration.
#[derive(Debug, Clone, Copy)]
pub struct AudioConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
}

impl AudioConfig {
    /// Default configuration: 16 kHz mono.
    pub fn default_16khz_mono() -> Self {
        Self {
            sample_rate: crate::DEFAULT_SAMPLE_RATE,
            channels: crate::CHANNELS,
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self::default_16khz_mono()
    }
}
