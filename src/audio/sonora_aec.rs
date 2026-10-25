//! Failure handling: same template as start_comm_capture -- loopback unavailable or a runtime error -> log it and
//! fall back to cpal direct capture via tx_fallback (the UI already has a consumer and builds the fallback stream), never dropping a feature silently.

use anyhow::Context;
use sonora_common_audio::push_resampler::PushResampler;
use windows::Win32::Media::Audio::{
    AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, EDataFlow, ERole, IAudioCaptureClient,
    IAudioClient, MMDeviceEnumerator,
};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
};

/// RPC_E_CHANGED_MODE: the thread already initialized COM with another apartment model (continue, do not re-init)
const RPC_E_CHANGED_MODE: i32 = 0x8001_0106u32 as i32;

/// AEC3 processing rate: 16 kHz full band (num_bands=1). See the module header for the rationale.
const AEC_RATE: usize = 16000;
/// AEC3 10 ms frame at 16 kHz (sonora-aec3 src/common.rs FRAME_SIZE)
const AEC_FRAME: usize = 160;

fn com_init() -> anyhow::Result<()> {
    unsafe {
        let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
        if hr.is_err() && hr.0 != RPC_E_CHANGED_MODE {
            anyhow::bail!("CoInitializeEx failed: {hr:?}");
        }
    }
    Ok(())
}

fn open_wasapi_stream(
    loopback: bool,
    device_name: Option<&str>,
) -> anyhow::Result<(IAudioClient, IAudioCaptureClient, usize, u32, String)> {
    com_init()?;
    unsafe {
        let enumr: windows::Win32::Media::Audio::IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .context("failed to create MMDeviceEnumerator")?;
        // EDataFlow(0)=eRender, EDataFlow(1)=eCapture; ERole(0)=eConsole (default device)
        // mic side flags=0: default stream category (deliberately no SetClientProperties(Communications), which
        // would stack on top of the system APO while AEC3 takes full responsibility for echo cancellation)
        let (client, capture, ch, sr, label): (
            IAudioClient,
            IAudioCaptureClient,
            usize,
            u32,
            String,
        ) = if loopback {
            let dev = enumr
                .GetDefaultAudioEndpoint(EDataFlow(0), ERole(0))
                .context("failed to get the default render device (loopback reference)")?;
            let label = format!("{} (default render device)", crate::comm::device_name_of(&dev));
            open_on(
                &dev,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                "default render device (loopback reference)",
                label,
            )?
        } else {
            let (dev, label) = crate::comm::pick_capture_device(&enumr, device_name)?;
            open_on(&dev, 0u32, "capture device (microphone)", label)?
        };
        Ok((client, capture, ch, sr, label))
    }
}

/// Open a shared-mode capture stream on the chosen endpoint (Initialize + Start + obtain IAudioCaptureClient).
/// Returns (client, capture, channels, mix rate, label).
fn open_on(
    dev: &windows::Win32::Media::Audio::IMMDevice,
    flags: u32,
    what: &str,
    label: String,
) -> anyhow::Result<(IAudioClient, IAudioCaptureClient, usize, u32, String)> {
    unsafe {
        let client: IAudioClient = dev
            .Activate(CLSCTX_ALL, None)
            .context("failed to activate IAudioClient")?;
        let fmt = client.GetMixFormat().context("GetMixFormat failed")?;
        let (ch, sr) = ((*fmt).nChannels as usize, (*fmt).nSamplesPerSec);
        client
            .Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, fmt, None)
            .context(format!("Initialize({what}) failed"))?;
        CoTaskMemFree(Some(fmt.cast()));
        let capture = client
            .GetService::<IAudioCaptureClient>()
            .context("failed to get IAudioCaptureClient")?;
        client.Start().context("Start failed")?;
        Ok((client, capture, ch.max(1), sr, label))
    }
}

/// Drain every accumulated packet from the WASAPI capture client -> mix to mono f32 (at the raw mix rate).
/// Mirrors comm.rs CommCapture::poll: silent packets (AUDCLNT_BUFFERFLAGS_SILENT=0x2) are treated as zeros.
fn poll_mono(capture: &IAudioCaptureClient, channels: usize) -> anyhow::Result<Vec<f32>> {
    let mut out = Vec::new();
    unsafe {
        loop {
            let n = capture.GetNextPacketSize().unwrap_or(0);
            if n == 0 {
                break;
            }
            let mut data: *mut u8 = std::ptr::null_mut();
            let mut frames: u32 = 0;
            let mut flags: u32 = 0;
            capture.GetBuffer(&mut data, &mut frames, &mut flags, None, None)?;
            let frames = frames as usize;
            if frames > 0 && !data.is_null() {
                if flags & 0x2 == 0 {
                    let f32slice =
                        std::slice::from_raw_parts(data as *const f32, frames * channels);
                    for fr in 0..frames {
                        let mut acc = 0.0f32;
                        for c in 0..channels {
                            acc += f32slice[fr * channels + c];
                        }
                        out.push(acc / channels as f32);
                    }
                } else {
                    out.extend(std::iter::repeat(0.0f32).take(frames));
                }
            }
            capture.ReleaseBuffer(frames as u32)?;
        }
    }
    Ok(out)
}

/// Microphone capture stream (**the device configured in settings**; default category -- **not** Communications, see the module header).
pub struct MicCapture {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    channels: usize,
    sample_rate: u32,
    label: String,
}

impl MicCapture {
    /// `device_name`: the configured microphone friendly name (None/empty = system default).
    /// When nothing matches it falls back to the default endpoint, and `device_label()` records why.
    pub fn open(device_name: Option<&str>) -> anyhow::Result<Self> {
        let (client, capture, channels, sample_rate, label) =
            open_wasapi_stream(false, device_name)?;
        Ok(MicCapture {
            client,
            capture,
            channels,
            sample_rate,
            label,
        })
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    /// Label of the device actually used (logs show at a glance whether the setting took effect or a fallback happened).
    pub fn device_label(&self) -> &str {
        &self.label
    }
    /// Drain every accumulated packet -> mix to mono f32 (raw mix rate). Empty when there is nothing.
    pub fn poll(&mut self) -> anyhow::Result<Vec<f32>> {
        poll_mono(&self.capture, self.channels)
    }
    pub fn stop(&mut self) {
        unsafe {
            let _ = self.client.Stop();
        }
    }
}

pub struct LoopbackCapture {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    channels: usize,
    sample_rate: u32,
    label: String,
}

impl LoopbackCapture {
    pub fn open() -> anyhow::Result<Self> {
        let (client, capture, channels, sample_rate, label) = open_wasapi_stream(true, None)?;
        Ok(LoopbackCapture {
            client,
            capture,
            channels,
            sample_rate,
            label,
        })
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    /// Label of the device actually used (for logs).
    pub fn device_label(&self) -> &str {
        &self.label
    }
    pub fn poll(&mut self) -> anyhow::Result<Vec<f32>> {
        poll_mono(&self.capture, self.channels)
    }
    pub fn stop(&mut self) {
        unsafe {
            let _ = self.client.Stop();
        }
    }
}

// -- Dual-path 10 ms frame alignment (pure functions, unit-tested) --

/// Alignment verdict.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AlignAction {
    /// Both paths yield one 10 ms frame, processed as a pair
    Pair,
    /// Microphone frame plus a zero reference frame (the reference stalled -- render silence with no packets is a normal boundary)
    MicWithZeroRef,
    RefWithZeroMic,
    /// Wait (neither path has a full frame, or the late path is still inside the 40 ms tolerance window)
    Wait,
}

/// Late-arrival tolerance: only when either path backs up by more than 4 ten-millisecond frames (40 ms) do we fill with zeros --
/// smaller arrival jitter is absorbed by waiting, which avoids zero-frame churn.
const STALL_FRAMES: usize = 4;

pub(crate) fn align_action(
    mic_len: usize,
    mic_need: usize,
    ref_len: usize,
    ref_need: usize,
) -> AlignAction {
    let mic_ready = mic_len >= mic_need;
    let ref_ready = ref_len >= ref_need;
    if mic_ready && ref_ready {
        AlignAction::Pair
    } else if mic_ready && mic_len > mic_need * STALL_FRAMES {
        // The reference stalled beyond the tolerance window (render silence / no packets): fill zeros and keep consuming the microphone
        AlignAction::MicWithZeroRef
    } else if ref_ready && ref_len > ref_need * STALL_FRAMES {
        AlignAction::RefWithZeroMic
    } else {
        AlignAction::Wait
    }
}

// -- AEC3 wrapper --

/// sonora-aec3 wrapper: dual-path input at native rates (feed, any block size) -> echo-cancelled 16 kHz mono output.

/// Internal chain: input accumulator -> 10 ms frame alignment (align_action: pair when both are ready / zero-fill a stalled path /
/// wait inside the tolerance window) -> PushResampler sinc resampling (**each path at its own native rate**, its 10 ms frame ->
/// 160 samples at 16 kHz; identical rates take the identity-copy fast path) -> the sonora 0.2.0 public facade `AudioProcessing`
/// (upstream authoritative driver loop: SwapQueue bridges render and capture, render before capture;
/// **APM is fixed at 16 kHz** -- upstream bug: with 48 kHz capture, audio_buffer.rs:265 copy_from_float goes out of bounds and
/// panics, as shown by tests/aec3_crash_probe.rs) -> a 16 kHz output accumulator drains by arrival. See the module header for the full-band choice.
pub struct SonoraAec {
    apm: sonora::AudioProcessing,
    /// 10 ms frame length on the mic side at the mic's native rate (160@16k / 441@44.1k / 480@48k)
    mic_frame: usize,
    ref_frame: usize,
    mic_buf: Vec<f32>,
    ref_buf: Vec<f32>,
    rs_mic: PushResampler<f32>,
    rs_ref: PushResampler<f32>,
    out: Vec<f32>,
}

impl SonoraAec {
    /// `mic_sr` / `ref_sr` are the native mix-format rates of the microphone and the reference (render loopback).
    /// **They may differ** (each WASAPI endpoint has its own mix format: driverless USB mics are often 16 kHz,
    /// onboard output 48 kHz). Each must be within 8000-48000 and divisible by 100 (10 ms frame samples = sr/100 must be an integer).
    /// APM is fixed at 16 kHz internally.
    pub fn new(mic_sr: u32, ref_sr: u32) -> anyhow::Result<Self> {
        let frame_of = |sr: u32, what: &str| -> anyhow::Result<usize> {
            if !(8000..=48000).contains(&sr) || sr % 100 != 0 {
                anyhow::bail!("unsupported {what} sample rate {sr} (must be 8000-48000 and divisible by 100)");
            }
            Ok(sr as usize / 100)
        };
        let mic_frame = frame_of(mic_sr, "microphone")?;
        let ref_frame = frame_of(ref_sr, "reference (loopback)")?;
        let config = sonora::config::Config {
            echo_canceller: Some(sonora::config::EchoCanceller::default()),
            ..Default::default()
        };
        let apm = sonora::AudioProcessing::builder()
            .config(config)
            .capture_config(sonora::StreamConfig::new(AEC_RATE as u32, 1))
            .render_config(sonora::StreamConfig::new(AEC_RATE as u32, 1))
            .build();
        Ok(SonoraAec {
            apm,
            mic_frame,
            ref_frame,
            mic_buf: Vec::new(),
            ref_buf: Vec::new(),
            rs_mic: PushResampler::new(mic_frame, AEC_FRAME, 1),
            rs_ref: PushResampler::new(ref_frame, AEC_FRAME, 1),
            out: Vec::new(),
        })
    }

    /// Feed dual-path samples at native rates (as much as poll returned; the two block sizes need not match): internal 10 ms frame alignment,
    /// sinc resampling and AEC3, returning the accumulated ready 16 kHz cleaned samples.
    /// APM statistics snapshot: (ERLE dB, ERL dB, median delay ms). ERLE < 10 dB ≈ not converged;
    /// delay should stay between tens of ms and 200 ms. For telemetry (most sonora facade fields are Option).
    pub fn stats_snapshot(&self) -> (Option<f64>, Option<f64>, Option<i32>) {
        let st = self.apm.statistics();
        (
            st.echo_return_loss_enhancement,
            st.echo_return_loss,
            st.delay_median_ms,
        )
    }

    pub fn feed(&mut self, mic: &[f32], reference: &[f32]) -> Vec<f32> {
        self.mic_buf.extend_from_slice(mic);
        self.ref_buf.extend_from_slice(reference);
        let (fm, fr) = (self.mic_frame, self.ref_frame);
        loop {
            match align_action(self.mic_buf.len(), fm, self.ref_buf.len(), fr) {
                AlignAction::Pair => {
                    let m = drain_frame(&mut self.mic_buf, fm);
                    let r = drain_frame(&mut self.ref_buf, fr);
                    let m16 = resample_to_16k(&mut self.rs_mic, &m);
                    let r16 = resample_to_16k(&mut self.rs_ref, &r);
                    self.process_pair(&m16, &r16);
                }
                AlignAction::MicWithZeroRef => {
                    // The reference stalled beyond the tolerance window (render silence or no packets is a normal boundary): keep consuming the microphone with a zero reference.
                    // Zero-filling is harmless: the number of frames is driven by the **microphone clock**, exactly equal to elapsed real time, so the timeline does not drift.
                    let m = drain_frame(&mut self.mic_buf, fm);
                    let m16 = resample_to_16k(&mut self.rs_mic, &m);
                    let z16 = vec![0.0f32; AEC_FRAME];
                    self.process_pair(&m16, &z16);
                }
                AlignAction::RefWithZeroMic => {
                    let r = drain_frame(&mut self.ref_buf, fr);
                    let mut r16 = resample_to_16k(&mut self.rs_ref, &r);
                    for v in r16.iter_mut() {
                        *v += if *v >= 0.0 { 1e-9 } else { -1e-9 }; // denormal dither (issue #34)
                    }
                    let mut r_out = r16.clone();
                    self.apm.process_render_f32(&[&r16], &mut [&mut r_out]).ok();
                }
                AlignAction::Wait => break,
            }
        }
        std::mem::take(&mut self.out)
    }

    /// One pair of 16 kHz 10 ms frames through the APM: render is processed first (into the render buffer), capture second
    /// (it consumes render internally for cancellation). The render dest is a dummy output (process_render always writes back a frame of the same length).
    /// A -180 dBFS dither is mixed into the render input (sonora issue #34 workaround): an exactly-zero render segment
    /// decays the adaptive filter coefficients into denormals and costs up to 15 ms per frame, so the dither prevents exact zeros.
    fn process_pair(&mut self, mic16: &[f32], ref16: &[f32]) {
        const DITHER: f32 = 1e-9;
        let mut r_in = ref16.to_vec();
        for v in r_in.iter_mut() {
            *v += if *v >= 0.0 { DITHER } else { -DITHER };
        }
        let mut r_out = r_in.clone();
        self.apm
            .process_render_f32(&[&r_in], &mut [&mut r_out])
            .ok();
        let mut dest = mic16.to_vec();
        self.apm
            .process_capture_f32(&[mic16], &mut [&mut dest])
            .ok();
        self.out.extend_from_slice(&dest);
    }
}

/// Take exactly one frame of `need` samples.
fn drain_frame(acc: &mut Vec<f32>, need: usize) -> Vec<f32> {
    acc.drain(..need).collect()
}

/// One 10 ms frame at a native rate -> 160 samples at 16 kHz (sinc resampling; identical rates take the copy fast path).
fn resample_to_16k(rs: &mut PushResampler<f32>, frame: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; AEC_FRAME];
    rs.resample_mono(frame, &mut out);
    out
}

/// RMS (for the smoke output).
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|&s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_action_covers_pair_wait_zero_fill() {
        let need = 480usize; // 10ms @48k
        assert_eq!(align_action(0, need, 0, need), AlignAction::Wait);
        assert_eq!(align_action(need, need, need, need), AlignAction::Pair);
        assert_eq!(align_action(need + 5, need, need, need), AlignAction::Pair);
        // The reference stalled but the microphone backlog is within the 40 ms tolerance window -> wait
        assert_eq!(align_action(need * 3, need, 0, need), AlignAction::Wait);
        // The reference stalled beyond the tolerance window -> microphone frame plus a zero reference
        assert_eq!(
            align_action(need * 4 + 1, need, 0, need),
            AlignAction::MicWithZeroRef
        );
        assert_eq!(
            align_action(0, need, need * 4 + 1, need),
            AlignAction::RefWithZeroMic
        );
        // Both paths have data but the reference is backed up -> still pair them (pairing takes priority)
        assert_eq!(align_action(need, need, need * 5, need), AlignAction::Pair);
    }

    /// Unequal rates: the two paths have different sample counts for "one 10 ms frame", each judged against its own need.
    /// The old signature shared one need (slicing the reference into thirds); this test locks the fix in.
    #[test]
    fn align_action_supports_different_frame_sizes() {
        let (fm, fr) = (160usize, 480usize); // mic 16k / reference 48k
        assert_eq!(align_action(fm, fm, fr, fr), AlignAction::Pair);
        // The reference has 300 samples (fewer than 480, i.e. not a full frame) while the mic has 3 frames queued -> wait (inside the tolerance window)
        assert_eq!(align_action(fm * 3, fm, 300, fr), AlignAction::Wait);
        // The reference still has under one frame while the mic is more than 4 frames behind -> mic frame plus a zero reference
        assert_eq!(align_action(fm * 5, fm, 0, fr), AlignAction::MicWithZeroRef);
        assert_eq!(align_action(0, fm, fr * 5, fr), AlignAction::RefWithZeroMic);
    }

    #[test]
    fn sonora_aec_rejects_bad_rates() {
        assert!(SonoraAec::new(16000, 16000).is_ok());
        assert!(
            SonoraAec::new(48000, 48000).is_ok(),
            "48k = the real-device mix-format main path"
        );
        assert!(SonoraAec::new(44100, 44100).is_ok());
        assert!(SonoraAec::new(16000, 48000).is_ok(), "mismatched rates must be accepted");
        assert!(SonoraAec::new(48000, 16000).is_ok(), "the reverse mismatch is accepted too");
        assert!(
            SonoraAec::new(44101, 48000).is_err(),
            "a non-integer 10 ms mic frame must be rejected"
        );
        assert!(
            SonoraAec::new(48000, 44101).is_err(),
            "a non-integer 10 ms reference frame must be rejected"
        );
        assert!(SonoraAec::new(96000, 48000).is_err(), "out of range must be rejected");
        assert!(SonoraAec::new(4000, 16000).is_err(), "below 8 kHz must be rejected");
    }

    /// Synthetic echo cancellation: mic = 0.5 × reference (true buffer delay 80 samples at 16 kHz = 5 ms) -> once settled the cleaned RMS
    /// is clearly below the mic RMS (the core capability of AEC3 on a static echo path).
    /// Excitation = deterministic pseudo-random white noise (AEC3 converges fastest on broadband excitation; a pure sine is degenerate).
    #[test]
    fn sonora_aec_suppresses_synthetic_echo() {
        const FRAMES: usize = 500; // 5s
        const FRAME16: usize = 160;
        const DELAY: usize = 80; // 5ms @16k
        let ref_all: Vec<f32> = (0..FRAMES * FRAME16)
            .map(|n| {
                let mut x = (n as u64) * 2654435761u64;
                x ^= x >> 15;
                x = x.wrapping_mul(2246822519u64);
                x ^= x >> 13;
                (((x >> 32) as f32) / u32::MAX as f32 - 0.5) * 0.6
            })
            .collect();
        let mut aec = SonoraAec::new(16000, 16000).unwrap();
        let (mut mic_sq, mut clean_sq, mut clean_n) = (0.0f64, 0.0f64, 0usize);
        for step in 0..FRAMES {
            let base = step * FRAME16;
            let ref16 = &ref_all[base..base + FRAME16];
            let mic16: Vec<f32> = (0..FRAME16)
                .map(|i| {
                    let n = base + i;
                    if n >= DELAY {
                        0.5 * ref_all[n - DELAY]
                    } else {
                        0.0
                    }
                })
                .collect();
            let cleaned = aec.feed(&mic16, ref16);
            if step >= 100 {
                for &s in &cleaned {
                    clean_sq += (s as f64) * (s as f64);
                    clean_n += 1;
                }
                for &s in &mic16 {
                    mic_sq += (s as f64) * (s as f64);
                }
            }
        }
        let mic_rms = (mic_sq / clean_n.max(1) as f64).sqrt();
        let clean_rms = (clean_sq / clean_n.max(1) as f64).sqrt();
        eprintln!(
            "[sonora_aec] after synthetic-echo convergence: mic_rms={mic_rms:.5} clean_rms={clean_rms:.5} (ratio {:.3})",
            clean_rms / mic_rms
        );
        assert!(
            clean_rms < mic_rms * 0.5,
            "cleaned must be well below mic (echo cancelled)"
        );
    }

    #[test]
    fn sonora_aec_48k_frontend_suppresses_synthetic_echo() {
        const FRAMES: usize = 500; // 5 s (10 ms frames)
        const FRAME48: usize = 480;
        const DELAY: usize = 240; // 5ms @48k
        let ref_all: Vec<f32> = (0..FRAMES * FRAME48)
            .map(|n| {
                let mut x = (n as u64) * 2654435761u64;
                x ^= x >> 15;
                x = x.wrapping_mul(2246822519u64);
                x ^= x >> 13;
                (((x >> 32) as f32) / u32::MAX as f32 - 0.5) * 0.6
            })
            .collect();
        let mut aec = SonoraAec::new(48000, 48000).unwrap();
        let (mut mic_sq, mut clean_sq, mut clean_n) = (0.0f64, 0.0f64, 0usize);
        for step in 0..FRAMES {
            let base = step * FRAME48;
            let r48 = &ref_all[base..base + FRAME48];
            let m48: Vec<f32> = (0..FRAME48)
                .map(|i| {
                    let n = base + i;
                    if n >= DELAY {
                        0.5 * ref_all[n - DELAY]
                    } else {
                        0.0
                    }
                })
                .collect();
            // Split the reference across two feeds (mic 480 / ref 240+240, unequal block sizes)
            let _ = aec.feed(&[], &r48[..240]);
            let cleaned = aec.feed(&m48, &r48[240..]);
            if step >= 100 {
                for &s in &cleaned {
                    clean_sq += (s as f64) * (s as f64);
                    clean_n += 1;
                }
                for &s in &m48 {
                    mic_sq += (s as f64) * (s as f64);
                }
            }
        }
        let mic_rms = (mic_sq / clean_n.max(1) as f64).sqrt();
        let clean_rms = (clean_sq / clean_n.max(1) as f64).sqrt();
        eprintln!(
            "[sonora_aec] 48k front-end after synthetic-echo convergence: mic_rms={mic_rms:.5} clean_rms={clean_rms:.5} (ratio {:.3})",
            clean_rms / mic_rms
        );
        assert!(
            clean_rms < mic_rms * 0.6,
            "the 48k front-end (alignment + resampling) must keep the echo-cancellation ability"
        );
    }

    #[test]
    fn sonora_aec_mixed_rate_frontend_suppresses_synthetic_echo() {
        const STEPS: usize = 500; // 5 s (10 ms steps)
        const F16: usize = 160;
        const F48: usize = 480;
        const DELAY: usize = 80; // 5ms @16k
        // Excitation: deterministic pseudo-random white noise at 16 kHz (the same LCG as the other tests)
        let src16: Vec<f32> = (0..STEPS * F16)
            .map(|n| {
                let mut x = (n as u64) * 2654435761u64;
                x ^= x >> 15;
                x = x.wrapping_mul(2246822519u64);
                x ^= x >> 13;
                (((x >> 32) as f32) / u32::MAX as f32 - 0.5) * 0.6
            })
            .collect();
        // Reference: linearly interpolated up to 48 kHz
        let ref48: Vec<f32> = (0..STEPS * F48)
            .map(|n| {
                let idx = n / 3;
                let frac = (n % 3) as f32 / 3.0;
                let a = src16[idx];
                let b = *src16.get(idx + 1).unwrap_or(&a);
                a + (b - a) * frac
            })
            .collect();
        let mut aec = SonoraAec::new(16000, 48000).unwrap();
        let (mut mic_sq, mut clean_sq, mut clean_n) = (0.0f64, 0.0f64, 0usize);
        for step in 0..STEPS {
            let base = step * F16;
            let mic16: Vec<f32> = (0..F16)
                .map(|i| {
                    let n = base + i;
                    if n >= DELAY {
                        0.5 * src16[n - DELAY]
                    } else {
                        0.0
                    }
                })
                .collect();
            let rb = step * F48;
            let cleaned = aec.feed(&mic16, &ref48[rb..rb + F48]);
            if step >= 100 {
                for &s in &cleaned {
                    clean_sq += (s as f64) * (s as f64);
                    clean_n += 1;
                }
                for &s in &mic16 {
                    mic_sq += (s as f64) * (s as f64);
                }
            }
        }
        let mic_rms = (mic_sq / clean_n.max(1) as f64).sqrt();
        let clean_rms = (clean_sq / clean_n.max(1) as f64).sqrt();
        eprintln!(
            "[sonora_aec] mismatched rates (mic 16k / ref 48k) after synthetic-echo convergence: mic_rms={mic_rms:.5} clean_rms={clean_rms:.5} (ratio {:.3})",
            clean_rms / mic_rms
        );
        assert!(
            clean_rms < mic_rms * 0.7,
            "the mismatched-rate front-end must keep echo cancellation (ratio was ~1 before the fix)"
        );
    }

}
