//! Audio feature extraction.

//! - write_wav: 16-bit PCM WAV writer (used for analysis dumps and debugging)
//! - resample_linear: linear resampling
//! - log_mel_fbank: log-mel Fbank with 25 ms frame length / 10 ms hop / 80 mel bins

use ndarray::{Array1, Array2};

pub const TARGET_SAMPLE_RATE: u32 = 16000;
pub const MEL_BINS: usize = 80;

// -- WAV writing --

/// Write mono f32 samples -> 16-bit PCM WAV (saves analysis recordings for replaying and reproducing recognition issues)
pub fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w =
        hound::WavWriter::create(path, spec).map_err(|e| format!("failed to create the WAV {path}: {e}"))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        w.write_sample(v)
            .map_err(|e| format!("failed to write the WAV: {e}"))?;
    }
    w.finalize().map_err(|e| format!("failed to close the WAV: {e}"))
}

// -- Linear resampling --

pub fn resample_linear(samples: &[f32], src_hz: u32, dst_hz: u32) -> Vec<f32> {
    if src_hz == dst_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = dst_hz as f64 / src_hz as f64;
    let target_len = ((samples.len() as f64 * ratio).round() as usize).max(1);
    let mut out = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let idx = i as f64 / ratio;
        let left = idx.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (idx - left as f64) as f32;
        out.push(samples[left] * (1.0 - frac) + samples[right] * frac);
    }
    out
}

// -- Window function and mel filterbank --

/// Hamming window.
pub fn hamming_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| {
            0.54 - 0.46
                * (2.0 * std::f32::consts::PI * n as f32 / (size as f32 - 1.0).max(1.0)).cos()
        })
        .collect()
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// Mel filterbank, shape [num_bins, fft_size/2 + 1].
pub fn mel_filterbank(sample_rate: u32, fft_size: usize, num_bins: usize) -> Array2<f32> {
    let spectrum_bins = fft_size / 2 + 1;
    let mut filters = Array2::<f32>::zeros((num_bins, spectrum_bins));
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate as f64 / 2.0);
    let mel_points: Vec<f64> = (0..num_bins + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (num_bins + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((fft_size + 1) as f64 * hz / sample_rate as f64) as usize)
        .collect();

    for m in 1..=num_bins {
        let left = fft_bins[m - 1].min(spectrum_bins - 1);
        let center = fft_bins[m].min(spectrum_bins - 1);
        let right = fft_bins[m + 1].min(spectrum_bins - 1);
        if center > left {
            for k in left..center {
                filters[[m - 1, k]] = (k - left) as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for k in center..right {
                filters[[m - 1, k]] = (right - k) as f32 / (right - center) as f32;
            }
        }
    }
    filters
}

// ── log-mel Fbank ────────────────────────────────────

struct FbankPlan {
    filters: Array2<f32>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

fn fbank_plan(
    sample_rate: u32,
    fft_size: usize,
    num_bins: usize,
    frame_len: usize,
) -> std::sync::Arc<FbankPlan> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<(u32, usize, usize), std::sync::Arc<FbankPlan>>>,
    > = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    cache
        .lock()
        .unwrap()
        .entry((sample_rate, fft_size, num_bins))
        .or_insert_with(|| {
            std::sync::Arc::new(FbankPlan {
                filters: mel_filterbank(sample_rate, fft_size, num_bins),
                window: hamming_window(frame_len),
                fft: {
                    let mut planner = rustfft::FftPlanner::<f32>::new();
                    planner.plan_fft_forward(fft_size)
                },
            })
        })
        .clone()
}

pub fn log_mel_fbank(samples: &[f32], sample_rate: u32, num_bins: usize) -> Array2<f32> {
    let frame_len = (sample_rate as f32 * 0.025) as usize;
    let hop_len = (sample_rate as f32 * 0.010) as usize;
    let fft_size = ((frame_len - 1).max(1)).next_power_of_two().max(512);

    if samples.len() < frame_len || frame_len == 0 || hop_len == 0 {
        return Array2::zeros((0, num_bins));
    }

    let plan = fbank_plan(sample_rate, fft_size, num_bins, frame_len);
    let spectrum_bins = fft_size / 2 + 1;

    let num_frames = (samples.len() - frame_len) / hop_len + 1;
    let mut frames = Array2::<f32>::zeros((num_frames, num_bins));
    let mut spectrum = vec![rustfft::num_complex::Complex::new(0.0f32, 0.0); fft_size];

    for i in 0..num_frames {
        let start = i * hop_len;
        for k in 0..frame_len {
            spectrum[k] =
                rustfft::num_complex::Complex::new(samples[start + k] * plan.window[k], 0.0);
        }
        for k in frame_len..fft_size {
            spectrum[k] = rustfft::num_complex::Complex::new(0.0, 0.0);
        }
        plan.fft.process(&mut spectrum);
        let mut power = Array1::<f32>::zeros(spectrum_bins);
        for k in 0..spectrum_bins {
            power[k] = spectrum[k].re * spectrum[k].re + spectrum[k].im * spectrum[k].im;
        }
        // mel filtering + log
        let mel_energy = plan.filters.dot(&power);
        for b in 0..num_bins {
            frames[[i, b]] = mel_energy[b].max(1e-10).ln();
        }
    }
    frames
}

// -- Tests: values and shapes --

#[cfg(test)]
mod tests {
    use super::*;

    /// Fbank sanity: a deterministic synthetic signal -> check shape, range and static frames.
    /// No external audio files required.
    #[test]
    fn fbank_deterministic_and_shape() {
        // 1 s of a 1 kHz sine at 16 kHz
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin())
            .collect();
        let feats = log_mel_fbank(&samples, TARGET_SAMPLE_RATE, MEL_BINS);
        assert_eq!(feats.nrows(), 98);
        assert_eq!(feats.ncols(), MEL_BINS);
        // Frame values sit in a plausible range (log-mel energy between -23 and +10)
        for v in feats.iter() {
            assert!(*v > -23.1 && *v < 10.0, "feature value out of range: {v}");
        }
        // Deterministic: two runs are bit-identical
        let feats2 = log_mel_fbank(&samples, TARGET_SAMPLE_RATE, MEL_BINS);
        assert_eq!(feats, feats2);
        // Silence -> every frame -23.02585 (log 1e-10)
        let silence = vec![0.0f32; 16000];
        let feats_s = log_mel_fbank(&silence, TARGET_SAMPLE_RATE, MEL_BINS);
        assert!((feats_s[[0, 0]] - (-23.02585)).abs() < 1e-3);
    }

    #[test]
    fn resample_linear_ratio() {
        let src: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let out = resample_linear(&src, 44100, 16000);
        assert_eq!(out.len(), 36); // round(100 * 16000/44100) = 36
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[35] - 0.964687).abs() < 1e-3);
    }
}
