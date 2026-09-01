//! Simple linear resampling between audio rates.

/// Resamples a mono `f32` stream from one rate to another using linear
/// interpolation, accumulating a fractional position across calls so that
/// arbitrary-length buffers can be fed incrementally.
#[derive(Debug)]
pub struct LinearResampler {
    pos: f64,
    step: f64,
}

impl LinearResampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        Self {
            pos: 0.0,
            step: in_rate as f64 / out_rate as f64,
        }
    }

    /// Feed mono `f32` samples; returns resampled samples at `out_rate`.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        while self.pos < input.len() as f64 {
            let idx = self.pos.floor() as usize;
            let frac = self.pos - idx as f64;
            let a = input[idx];
            let b = input.get(idx + 1).copied().unwrap_or(a);
            out.push(a + (b - a) * frac as f32);
            self.pos += self.step;
        }
        self.pos -= input.len() as f64;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::LinearResampler;

    #[test]
    fn down_samples_by_integer_factor() {
        // 48 kHz -> 16 kHz: constant signal preserved, length 1/3.
        let mut r = LinearResampler::new(48_000, 16_000);
        let out = r.process(&vec![1.0; 48_000]);
        assert_eq!(out.len(), 16_000);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn preserves_length_proportion() {
        let mut r = LinearResampler::new(48_000, 16_000);
        let input: Vec<f32> = (0..48_000).map(|i| i as f32 / 48_000.0).collect();
        let out = r.process(&input);
        assert_eq!(out.len(), 16_000);
    }
}
