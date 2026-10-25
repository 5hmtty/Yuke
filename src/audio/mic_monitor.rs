//! Always-on microphone monitor: continuous capture, never written to disk; feeds the live level meter in the UI.

use crate::features::{TARGET_SAMPLE_RATE, resample_linear};

/// Open an input device: by name or system default, and validate the f32 format.
/// Returns (device, StreamConfig, channels, sample rate).
pub fn open_input(
    device: Option<&str>,
) -> anyhow::Result<(cpal::Device, cpal::StreamConfig, usize, u32)> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let dev = match device {
        Some(name) => host
            .input_devices()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("input device not found: {name}"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("no input device (microphone) found"))?,
    };
    let config = dev.default_input_config()?;
    if config.sample_format() != cpal::SampleFormat::F32 {
        anyhow::bail!("only the f32 sample format is supported");
    }
    let channels = config.channels() as usize;
    let sr = config.sample_rate().0;
    Ok((dev, config.into(), channels, sr))
}

/// Always-on monitor: the callback keeps appending audio to a buffer (16 kHz mono). Dropping it stops capture.
pub struct MicMonitor {
    _stream: cpal::Stream,
    buf: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
}

impl MicMonitor {
    /// Open the device and capture continuously (device = None uses the system default)
    pub fn start(device: Option<&str>) -> anyhow::Result<Self> {
        use cpal::traits::{DeviceTrait, StreamTrait};
        let (dev, config, channels, sr) = open_input(device)?;
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
        let buf2 = buf.clone();
        let max_samples = (TARGET_SAMPLE_RATE * 30) as usize;
        // WASAPI callback blocks vary in length: resampling per callback would misalign the timeline at block boundaries.
        // The callback only accumulates raw samples and, once a full 50 ms block is available, mixes to mono and resamples -> continuous timeline.
        let mut pending: Vec<f32> = Vec::new();
        let block = (sr as f32 * 0.05) as usize;
        let stream = dev.build_input_stream(
            &config.into(),
            move |data: &[f32], _| {
                pending.extend_from_slice(data);
                while pending.len() >= block * channels {
                    let mut mono = Vec::with_capacity(block);
                    let drained: Vec<f32> = pending.drain(..block * channels).collect();
                    for ch in drained.chunks(channels) {
                        if ch.len() == channels {
                            mono.push(ch.iter().sum::<f32>() / channels as f32);
                        }
                    }
                    let resampled = resample_linear(&mono, sr, TARGET_SAMPLE_RATE);
                    let mut b = buf2.lock().unwrap();
                    b.extend_from_slice(&resampled);
                    if b.len() > max_samples {
                        let drop = b.len() - max_samples;
                        b.drain(..drop);
                    }
                }
            },
            |e| eprintln!("capture error: {e}"),
            None,
        )?;
        stream.play()?;
        Ok(MicMonitor {
            _stream: stream,
            buf,
        })
    }

    /// Current buffer length (16 kHz mono sample count)
    pub fn len(&self) -> usize {
        self.buf.lock().unwrap().len()
    }

    /// Take the audio from [start..] (16 kHz mono); an out-of-range start returns everything
    pub fn snapshot(&self, start: usize) -> Vec<f32> {
        let b = self.buf.lock().unwrap();
        b[start.min(b.len())..].to_vec()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buf.lock().unwrap().clear();
    }
}
