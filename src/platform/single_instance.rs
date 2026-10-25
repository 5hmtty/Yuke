use std::sync::OnceLock;
use windows::Win32::Foundation::{ERROR_ALREADY_EXISTS, GetLastError, HANDLE, WAIT_OBJECT_0};
use windows::Win32::System::Threading::{
    CreateEventW, CreateMutexW, EVENT_MODIFY_STATE, OpenEventW, SetEvent, WaitForSingleObject,
};
use windows::Win32::UI::WindowsAndMessaging::{ASFW_ANY, AllowSetForegroundWindow};
use windows::core::{PCWSTR, w};

const MUTEX_NAME: PCWSTR = w!("Local\\Yuke.SingleInstance.Mutex");
const EVENT_NAME: PCWSTR = w!("Local\\Yuke.SingleInstance.Show");

/// Mutex plus activation-event handle (alive for the process lifetime -- handles close automatically on exit).
/// A windows HANDLE is a raw pointer and is neither Send nor Sync; kernel sync objects are process-scoped anyway and
/// are only used on the UI thread, so declaring the wrapper thread-safe by hand is sound.
#[derive(Clone, Copy)]
struct SyncHandle(HANDLE);
unsafe impl Send for SyncHandle {}
unsafe impl Sync for SyncHandle {}

static HANDLES: OnceLock<(SyncHandle, SyncHandle)> = OnceLock::new();

pub enum Startup {
    /// This is the primary instance (mutex and event acquired); start normally.
    Primary,
    /// Another instance is running: the activation signal has been sent, so the caller should exit immediately.
    AlreadyRunning,
}

pub fn acquire_or_signal() -> Startup {
    unsafe {
        let mutex = match CreateMutexW(None, false, MUTEX_NAME) {
            Ok(h) => h,
            Err(e) => {
                // After the primary instance elevated, a non-elevated second instance gets ACCESS_DENIED from CreateMutex --
                // treat that as "already running" and exit (better not to start than to run twice with double keying)
                eprintln!("[single-instance] CreateMutexW failed; exiting as if another instance is running: {e}");
                return Startup::AlreadyRunning;
            }
        };
        if GetLastError() == ERROR_ALREADY_EXISTS {
            let _ = AllowSetForegroundWindow(ASFW_ANY);
            // Send the activation signal (the primary instance polls the event every frame and raises its window)
            match OpenEventW(EVENT_MODIFY_STATE, false, EVENT_NAME) {
                Ok(ev) => {
                    let _ = SetEvent(ev);
                }
                Err(e) => eprintln!("[single-instance] OpenEventW failed (only the old window cannot be raised): {e}"),
            }
            return Startup::AlreadyRunning;
        }
        let ev = match CreateEventW(None, false, false, EVENT_NAME) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[single-instance] CreateEventW failed (a repeated launch will not raise the old window): {e}");
                HANDLE::default()
            }
        };
        let _ = HANDLES.set((SyncHandle(mutex), SyncHandle(ev)));
        Startup::Primary
    }
}

/// The primary instance polls every frame: whether the activation event fired (a second instance or a repeated launch raises the window).
pub fn poll_show_request() -> bool {
    let Some((_, ev)) = HANDLES.get() else {
        return false;
    };
    if ev.0.is_invalid() {
        return false;
    }
    unsafe { WaitForSingleObject(ev.0, 0) == WAIT_OBJECT_0 }
}
