// sonora APM crash-boundary probe (standalone integration test; public crates only).
//
// Pins down three facts:
//   - `split_at_mut(0)` on the render input/output slices yields empty src + full dest;
//     a length mismatch triggers the audio_buffer.rs:265 out-of-bounds panic;
//   - 48 kHz APM with exact 10 ms frames (480/480), 200 clean pairs -> no panic;
//   - the real crash condition is **input frame length != the configured 10 ms frame (sr/100)**;
//     feeding any WASAPI poll block size straight into the facade hits it.
// The production wrapper (src/sonora_aec.rs) therefore has a 10 ms frame-aligning accumulator:
// arbitrary block sizes enter the accumulator and APM is fed whole frames.
use sonora::AudioProcessingBuilder;
use sonora::StreamConfig;
use sonora::config::{Config, EchoCanceller};

fn apm(sr: u32) -> sonora::AudioProcessing {
    AudioProcessingBuilder::default()
        .config(Config {
            echo_canceller: Some(EchoCanceller::default()),
            ..Default::default()
        })
        .capture_config(StreamConfig::new(sr, 1))
        .render_config(StreamConfig::new(sr, 1))
        .build()
}

/// Deterministic pseudo-random samples (no external dependency).
fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as f32 / 65536.0) - 0.5
        })
        .collect()
}

/// Fact 1: 16 kHz APM + exact 10 ms frames (160/160) -> 200 clean pairs pass.
#[test]
fn aec3_16k_minimal_ok() {
    let mut apm = apm(16000);
    let all = noise(200 * 160, 42);
    for k in 0..200 {
        let frame = &all[k * 160..(k + 1) * 160];
        let mut r_out = vec![0.0f32; 160];
        apm.process_render_f32(&[frame], &mut [&mut r_out]).unwrap();
        let mut dest = vec![0.0f32; 160];
        apm.process_capture_f32(&[frame], &mut [&mut dest]).unwrap();
    }
}

/// Fact 2 (misdiagnosis correction): 48 kHz APM + exact 10 ms frames (480/480) -> also clean; 48 kHz itself is fine.
#[test]
fn aec3_48k_minimal_ok() {
    let mut apm = apm(48000);
    let all = noise(200 * 480, 42);
    for k in 0..200 {
        let frame = &all[k * 480..(k + 1) * 480];
        let mut r_out = vec![0.0f32; 480];
        apm.process_render_f32(&[frame], &mut [&mut r_out]).unwrap();
        let mut dest = vec![0.0f32; 480];
        apm.process_capture_f32(&[frame], &mut [&mut dest]).unwrap();
    }
}

/// Fact 3 (real crash condition): input frame length != the configured 10 ms frame (160 at 48 kHz) ->
/// audio_buffer.rs:265 `range end index 480 out of range for slice of length 160`.
/// `#[should_panic]` pins this: the upstream facade has zero tolerance for block size, so production
/// must align to whole frames before feeding.
#[test]
#[should_panic(expected = "range end index")]
fn aec3_wrong_frame_len_panics() {
    let mut apm = apm(48000);
    let all = noise(10 * 160, 42);
    for k in 0..10 {
        let frame = &all[k * 160..(k + 1) * 160];
        let mut r_out = vec![0.0f32; 160];
        let _ = apm.process_render_f32(&[frame], &mut [&mut r_out]);
        let mut dest = vec![0.0f32; 160];
        let _ = apm.process_capture_f32(&[frame], &mut [&mut dest]);
    }
}
