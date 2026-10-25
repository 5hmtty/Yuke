//! AEC availability criterion (identical to the official AECCapture.cpp, no effect GUID needed):
//! `audioClient.GetService::<IAcousticEchoCancellationControl>()` succeeding means the stream's
//! effect chain contains AEC (that control interface is only obtainable when an AEC effect exists; E_NOINTERFACE = none).
//! `IAudioEffectsManager::GetAudioEffects` is used for diagnostic enumeration (Windows 11 build 22000+).

//! Capture: shared mode + mix format (f32 multi-channel, usually 48 kHz). This module averages channels down to mono;
//! resampling to 16 kHz is left to the caller (the capture thread in live.rs, using the same resample_linear as the cpal/AEC path).

use anyhow::Context;
use windows::Win32::Devices::Properties::DEVPKEY_Device_FriendlyName;
use windows::Win32::Media::Audio::{
    AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMOPTIONS, AudioCategory_Communications,
    AudioClientProperties, DEVICE_STATE_ACTIVE, EDataFlow, ERole, IAcousticEchoCancellationControl,
    IAudioCaptureClient, IAudioClient, IAudioClient2, IAudioEffectsManager, MMDeviceEnumerator,
};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree, STGM_READ,
};
use windows::Win32::UI::Shell::PropertiesSystem::PROPERTYKEY;
use windows::core::Interface;

/// RPC_E_CHANGED_MODE: the thread already initialized COM with another apartment model (safe to continue, do not re-init)
const RPC_E_CHANGED_MODE: i32 = 0x8001_0106u32 as i32;

/// Probe result (drives whether the echo-cancellation button in Settings is enabled).
#[derive(Clone, Default)]
pub struct CommProbe {
    /// Device friendly name (kept for diagnostics)
    pub device_name: String,
    /// Endpoint selection label (produced by `pick_capture_device`; when unspecified or unmatched it says "fell back to default",
    /// so "the configured microphone was not found" is never swallowed silently)
    pub device_label: String,
    /// Capture device ID (kept for diagnostics)
    pub device_id: String,
    /// Communications-mode client created successfully
    pub comms_ok: bool,
    /// **Whether an AEC effect exists** (authoritative criterion: `IAudioEffectsManager::GetAudioEffects` enumerating
    /// `AUDIO_EFFECT_TYPE_ACOUSTIC_ECHO_CANCELLATION`). `None` = **cannot be determined** on this machine
    /// (enumeration unavailable: Windows 11 build < 22000, or GetService(IAudioEffectsManager) failed).
    pub aec_effect: Option<bool>,
    /// **Reference endpoint controllable**: `GetService(IAcousticEchoCancellationControl)` succeeded (Windows 11 22621+).
    /// Official wording: success means the capture endpoint supports AEC **and** allows specifying the reference stream; failure (E_NOINTERFACE)
    /// means there is no AEC **or** AEC does not allow controlling the reference endpoint -- so failure must NOT be read as "no AEC".
    pub aec_control: bool,
    /// Effect GUIDs enumerated on the stream (diagnostics)
    pub effect_guids: Vec<String>,
    /// Why enumeration failed or came back empty (distinguishes "enumeration unavailable" from "genuinely no effects")
    pub effects_note: Option<String>,
    pub error: Option<String>,
}

/// System AEC availability verdict (three states -- "unavailable" needs evidence; "cannot detect" is not "absent").
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AecVerdict {
    /// AEC present (effect enumeration hit, or the reference endpoint is controllable -- the latter is a sufficient condition per official docs)
    Present,
    /// Confirmed absent: enumeration succeeded without AEC **and** reference-endpoint control is unsupported (two independent negative proofs)
    Absent,
    /// Undetermined: enumeration is limited on this machine (e.g. Windows 10) but there is no negative evidence either -> allow it and say so in the UI
    Unknown,
}

impl CommProbe {
    pub fn aec_verdict(&self) -> AecVerdict {
        if self.aec_control || self.aec_effect == Some(true) {
            AecVerdict::Present
        } else if self.aec_effect == Some(false) {
            AecVerdict::Absent
        } else {
            AecVerdict::Unknown
        }
    }
    pub fn aec_selectable(&self) -> bool {
        self.aec_verdict() != AecVerdict::Absent
    }
}

fn com_init() -> anyhow::Result<()> {
    unsafe {
        let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
        if hr.is_err() && hr.0 != RPC_E_CHANGED_MODE {
            anyhow::bail!("CoInitializeEx failed: {hr:?}");
        }
    }
    Ok(())
}

pub(crate) fn pick_capture_device(
    enumr: &windows::Win32::Media::Audio::IMMDeviceEnumerator,
    want: Option<&str>,
) -> anyhow::Result<(windows::Win32::Media::Audio::IMMDevice, String)> {
    let default_dev = || -> anyhow::Result<(windows::Win32::Media::Audio::IMMDevice, String)> {
        unsafe {
            let d = enumr
                .GetDefaultAudioEndpoint(EDataFlow(1), ERole(0))
                .context("failed to get the default capture device (microphone)")?;
            let n = device_name_of(&d);
            Ok((d, n))
        }
    };
    let Some(want) = want.filter(|s| !s.is_empty()) else {
        return default_dev();
    };
    unsafe {
        let coll = enumr
            .EnumAudioEndpoints(EDataFlow(1), DEVICE_STATE_ACTIVE)
            .context("failed to enumerate capture endpoints")?;
        let count = coll.GetCount().unwrap_or(0);
        for i in 0..count {
            let Ok(d) = coll.Item(i) else { continue };
            if device_name_of(&d) == want {
                return Ok((d, want.to_string()));
            }
        }
    }
    let (d, name) = default_dev()?;
    Ok((d, format!("{name} (the configured \"{want}\" was not found; using the default)")))
}

/// Default **render** endpoint ID (used as the AEC reference stream). None = unavailable (no output device etc.).
pub(crate) fn default_render_endpoint_id(
    enumr: &windows::Win32::Media::Audio::IMMDeviceEnumerator,
) -> Option<String> {
    unsafe {
        let dev = enumr.GetDefaultAudioEndpoint(EDataFlow(0), ERole(0)).ok()?;
        dev.GetId().ok()?.to_string().ok()
    }
}

pub fn probe(device_name: Option<&str>) -> CommProbe {
    com_init().unwrap_or_else(|e| {
        eprintln!("[comm] COM initialisation failed: {e:#}");
    });
    unsafe {
        let enumr: windows::Win32::Media::Audio::IMMDeviceEnumerator =
            match CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) {
                Ok(e) => e,
                Err(err) => {
                    let mut r = CommProbe::default();
                    r.error = Some(format!("failed to create MMDeviceEnumerator: {err}"));
                    return r;
                }
            };
        match pick_capture_device(&enumr, device_name) {
            Ok((dev, label)) => {
                let mut r = probe_device(&dev);
                r.device_label = label;
                r
            }
            Err(e) => {
                let mut r = CommProbe::default();
                r.error = Some(format!("{e:#}"));
                r
            }
        }
    }
}

pub(crate) fn device_name_of(dev: &windows::Win32::Media::Audio::IMMDevice) -> String {
    let id_fallback = || -> String {
        unsafe {
            dev.GetId()
                .ok()
                .and_then(|pw| pw.to_string().ok())
                .unwrap_or_default()
        }
    };
    unsafe {
        let Ok(ps) = dev.OpenPropertyStore(STGM_READ) else {
            return id_fallback();
        };
        let key = PROPERTYKEY {
            fmtid: DEVPKEY_Device_FriendlyName.fmtid,
            pid: DEVPKEY_Device_FriendlyName.pid,
        };
        // GetValue returns a PROPVARIANT (Display comes back as BSTR; VT_LPWSTR is the friendly name)
        match ps.GetValue(&key) {
            Ok(v) if !v.is_empty() => v.to_string(),
            _ => id_fallback(),
        }
    }
}

/// AEC effect GUID. In windows 0.58 this constant lives under `Win32_Media_KernelStreaming` (that feature is not enabled here),
/// so the value is taken from the crate source at `KernelStreaming/mod.rs`, `AUDIO_EFFECT_TYPE_ACOUSTIC_ECHO_CANCELLATION`
/// (= official header ksmedia.h / driver docs AUDIO_EFFECT_TYPE_ACOUSTIC_ECHO_CANCELLATION).
const AUDIO_EFFECT_TYPE_ACOUSTIC_ECHO_CANCELLATION: windows::core::GUID =
    windows::core::GUID::from_u128(0x6f64adbe_8211_11e2_8c70_2c27d7f001fa);

/// Probe a specific capture endpoint for AEC in Communications mode.
fn probe_device(dev: &windows::Win32::Media::Audio::IMMDevice) -> CommProbe {
    let name = device_name_of(dev);
    let mut r = CommProbe {
        device_name: name.clone(),
        // Defaults to the endpoint's own friendly name; `probe` overrides it with
        // the label from `pick_capture_device`, which states "fell back to default" when nothing matched.
        device_label: name,
        device_id: String::new(),
        comms_ok: false,
        aec_effect: None,
        aec_control: false,
        effect_guids: Vec::new(),
        effects_note: None,
        error: None,
    };
    let mut inner = || -> anyhow::Result<()> {
        com_init()?;
        unsafe {
            r.device_id = dev
                .GetId()
                .ok()
                .and_then(|pw| pw.to_string().ok())
                .unwrap_or_default();
            let client: IAudioClient = dev
                .Activate(CLSCTX_ALL, None)
                .context("failed to activate IAudioClient")?;
            let client2 = client
                .cast::<IAudioClient2>()
                .context("IAudioClient2 unavailable (Windows 7?)")?;
            let mut props = AudioClientProperties::default();
            props.cbSize = std::mem::size_of::<AudioClientProperties>() as u32;
            props.eCategory = AudioCategory_Communications;
            props.Options = AUDCLNT_STREAMOPTIONS(0); // Do not request RAW -- RAW bypasses every system effect
            client2
                .SetClientProperties(&props)
                .context("SetClientProperties(Communications) failed")?;
            let fmt = client.GetMixFormat().context("GetMixFormat failed")?;
            // Required by the API: **Initialize before GetService** (otherwise the interface cannot be obtained)
            client
                .Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 0, 0, fmt, None)
                .context("Initialize(Communications) failed")?;
            r.comms_ok = true;

            match client2.GetService::<IAudioEffectsManager>() {
                Ok(mgr) => {
                    let mut list: *mut windows::Win32::Media::Audio::AUDIO_EFFECT =
                        std::ptr::null_mut();
                    let mut n: u32 = 0;
                    match mgr.GetAudioEffects(&mut list, &mut n) {
                        Ok(()) => {
                            let mut has_aec = false;
                            if !list.is_null() && n > 0 {
                                let slice = std::slice::from_raw_parts(list, n as usize);
                                for e in slice {
                                    if e.id == AUDIO_EFFECT_TYPE_ACOUSTIC_ECHO_CANCELLATION {
                                        has_aec = true;
                                    }
                                    r.effect_guids.push(format!("{:?}", e.id));
                                }
                            }
                            r.aec_effect = Some(has_aec);
                            if n == 0 {
                                r.effects_note = Some(
                                    "no effects were enumerated on the stream (outside RAW mode this means there is no system APO)".into(),
                                );
                            }
                            CoTaskMemFree(Some(list.cast()));
                        }
                        Err(e) => {
                            r.effects_note = Some(format!("GetAudioEffects failed: {e}"));
                        }
                    }
                }
                Err(e) => {
                    r.effects_note = Some(format!("effect enumeration unavailable (needs Windows 11 build 22000+): {e}"));
                }
            }

            r.aec_control = client2
                .GetService::<IAcousticEchoCancellationControl>()
                .is_ok();
            CoTaskMemFree(Some(fmt.cast()));
        }
        Ok(())
    };
    match inner() {
        Ok(()) => {}
        Err(e) => r.error = Some(format!("{e:#}")),
    }
    r
}

/// Communications-mode capture session (same-thread COM init -> open -> start -> poll).
pub struct CommCapture {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    channels: usize,
    sample_rate: u32,
    label: String,
    /// Whether the AEC reference stream has been explicitly pointed at the system default render endpoint (see the note inside open)
    ref_endpoint_set: bool,
}

impl CommCapture {
    /// Opens a Communications-mode stream on **the capture device configured in Settings** (None/empty = system default)
    /// (no AEC availability check -- probe first). Falls back to the default when nothing matches; `device_label()` says so.
    pub fn open(device_name: Option<&str>) -> anyhow::Result<Self> {
        com_init()?;
        unsafe {
            let enumr: windows::Win32::Media::Audio::IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                    .context("failed to create MMDeviceEnumerator")?;
            let (dev, label) = pick_capture_device(&enumr, device_name)?;
            let client: IAudioClient = dev
                .Activate(CLSCTX_ALL, None)
                .context("failed to activate IAudioClient")?;
            let client2 = client.cast::<IAudioClient2>()?;
            let mut props = AudioClientProperties::default();
            props.cbSize = std::mem::size_of::<AudioClientProperties>() as u32;
            props.eCategory = AudioCategory_Communications;
            props.Options = AUDCLNT_STREAMOPTIONS(0);
            client2
                .SetClientProperties(&props)
                .context("SetClientProperties(Communications) failed")?;
            let fmt = client.GetMixFormat().context("GetMixFormat failed")?;
            let (ch, sr) = ((*fmt).nChannels as usize, (*fmt).nSamplesPerSec);
            // Required by the API: Initialize before GetService
            client
                .Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 0, 0, fmt, None)
                .context("Initialize(Communications) failed")?;
            CoTaskMemFree(Some(fmt.cast()));

            // Explicitly set the AEC reference stream to the system default render endpoint (not fatal on failure: log it and let the system default rule apply)
            let mut ref_endpoint_set = false;
            if let Ok(ctrl) = client2.GetService::<IAcousticEchoCancellationControl>() {
                if let Some(render_id) = default_render_endpoint_id(&enumr) {
                    let id_w = windows::core::HSTRING::from(render_id.as_str());
                    match ctrl.SetEchoCancellationRenderEndpoint(&id_w) {
                        Ok(()) => ref_endpoint_set = true,
                        Err(e) => eprintln!("[comm] SetEchoCancellationRenderEndpoint failed: {e}"),
                    }
                }
            }

            let capture = client
                .GetService::<IAudioCaptureClient>()
                .context("failed to get IAudioCaptureClient")?;
            client.Start().context("Start failed")?;
            Ok(CommCapture {
                client,
                capture,
                channels: ch.max(1),
                sample_rate: sr,
                label,
                ref_endpoint_set,
            })
        }
    }

    /// Label of the capture device actually used (logs show at a glance whether the setting took effect or a fallback happened).
    pub fn device_label(&self) -> &str {
        &self.label
    }

    /// Whether the AEC reference stream was explicitly pointed at the system default render endpoint.
    pub fn ref_endpoint_set(&self) -> bool {
        self.ref_endpoint_set
    }

    /// Sample rate of the mix format (used by the caller when resampling to 16 kHz).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Drain every accumulated packet -> mix to mono f32 (at the raw mix sample rate). Returns empty when there is nothing.
    pub fn poll(&mut self) -> anyhow::Result<Vec<f32>> {
        let mut out = Vec::new();
        unsafe {
            loop {
                let n = self.capture.GetNextPacketSize().unwrap_or(0);
                if n == 0 {
                    break;
                }
                let mut data: *mut u8 = std::ptr::null_mut();
                let mut frames: u32 = 0;
                let mut flags: u32 = 0;
                self.capture
                    .GetBuffer(&mut data, &mut frames, &mut flags, None, None)?;
                let frames = frames as usize;
                if frames > 0 && !data.is_null() {
                    // Silent packets (AUDCLNT_BUFFERFLAGS_SILENT=0x2) are treated as zeros
                    if flags & 0x2 == 0 {
                        let f32slice =
                            std::slice::from_raw_parts(data as *const f32, frames * self.channels);
                        // Average channels down to mono
                        for fr in 0..frames {
                            let mut acc = 0.0f32;
                            for c in 0..self.channels {
                                acc += f32slice[fr * self.channels + c];
                            }
                            out.push(acc / self.channels as f32);
                        }
                    } else {
                        out.extend(std::iter::repeat(0.0f32).take(frames));
                    }
                }
                self.capture.ReleaseBuffer(frames as u32)?;
            }
        }
        Ok(out)
    }

    /// Stop the stream (dropping it releases IAudioClient as a backstop; the explicit call only makes the intent clear).
    pub fn stop(&mut self) {
        unsafe {
            let _ = self.client.Stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aec_verdict_three_state_semantics() {
        let mk = |effect: Option<bool>, control: bool| CommProbe {
            aec_effect: effect,
            aec_control: control,
            ..Default::default()
        };
        // Either piece of evidence alone is enough to conclude "AEC present"
        assert_eq!(
            mk(Some(true), false).aec_verdict(),
            AecVerdict::Present,
            "effect enumeration hit"
        );
        assert_eq!(
            mk(None, true).aec_verdict(),
            AecVerdict::Present,
            "the reference endpoint is controllable (sufficient condition)"
        );
        // When the two disagree, trust "reference endpoint controllable" (official docs: success means AEC is supported and the reference endpoint is controllable)
        assert_eq!(mk(Some(false), true).aec_verdict(), AecVerdict::Present);
        // Only "enumeration succeeded with no AEC effect" **and** "reference-endpoint control unsupported" counts as **confirmed absent**
        assert_eq!(mk(Some(false), false).aec_verdict(), AecVerdict::Absent);
        // Enumeration unavailable (below Windows 11 22000) -> undetermined, and must **not** be treated as absent
        assert_eq!(mk(None, false).aec_verdict(), AecVerdict::Unknown);
        assert!(!mk(Some(false), false).aec_selectable(), "confirmed absent -> block");
        assert!(mk(None, false).aec_selectable(), "undeterminable -> allow");
        assert!(mk(Some(true), false).aec_selectable());
    }
}
