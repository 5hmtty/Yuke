//! Windows key injection.

//! Input modes (switchable in Settings, applied globally):
//! - Standard (default): SendInput with both wVk and wScan filled
//!   (standard keyboard events); if a target game window is configured, it is brought to the foreground first,
//!   (SetWindowPos(HWND_TOP) + SetForegroundWindow) --
//!   because SendInput injects only into the foreground window and input is lost when the game loses focus
//! - ScanCode: SendInput with raw scan codes only (wVk=0) -- the compatibility workaround for some Unity games
//! - VirtualKey: SendInput with virtual key codes -- the standard application case
//! - Background: background window messages (PostMessage WM_KEYDOWN/KEYUP straight to the target window) --
//!   the game needs no foreground focus and no administrator rights;
//!   only works for games that handle window messages (e.g. Unity), not for games reading DirectInput
//! - Legacy: keybd_event -- the old API, a fallback for some legacy environments

use crate::i18n_keys::k;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

/// Current input mode (switched on the settings page, applies globally)
static INPUT_MODE: AtomicU8 = AtomicU8::new(0);

/// Target game window title keyword (configured on the settings page; empty = no foreground enforcement and no background mode)
static TARGET_WINDOW: Mutex<String> = Mutex::new(String::new());

/// Set the target game window title keyword (matched with FindWindow)
pub fn set_target_window(name: &str) {
    if let Ok(mut v) = TARGET_WINDOW.lock() {
        *v = name.to_string();
    }
}

fn target_window() -> String {
    TARGET_WINDOW.lock().map(|v| v.clone()).unwrap_or_default()
}

/// SendInput failure / interception records (the UI has no console, so eprintln is invisible; the UI drains these every frame)
static SEND_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Drain pending key errors for display (called from the UI trigger-log area)
pub fn drain_send_log() -> Vec<String> {
    if let Ok(mut v) = SEND_LOG.lock() {
        std::mem::take(&mut *v)
    } else {
        Vec::new()
    }
}

fn log_send(msg: &str) {
    eprintln!("{msg}");
    if let Ok(mut v) = SEND_LOG.lock() {
        v.push(msg.to_string());
    }
}

pub const MODE_STANDARD: u8 = 0; // SendInput with both VK and scan code (default)
pub const MODE_SCANCODE: u8 = 1; // SendInput with raw scan codes
pub const MODE_VIRTUALKEY: u8 = 2; // SendInput with virtual key codes
pub const MODE_BACKGROUND: u8 = 3; // Background window messages (PostMessage)
pub const MODE_LEGACY: u8 = 4; // keybd_event, the legacy API

/// Set the key input mode (keys::MODE_*)
pub fn set_input_mode(mode: u8) {
    INPUT_MODE.store(mode, Ordering::Relaxed);
}

pub fn input_mode() -> u8 {
    INPUT_MODE.load(Ordering::Relaxed)
}

const MAPVK_VK_TO_VSC: u32 = 0;
const INPUT_KEYBOARD: u32 = 1;
const KEYEVENTF_KEYUP: u32 = 0x0002;
const KEYEVENTF_SCANCODE: u32 = 0x0008;
const MOUSEEVENTF_LEFTDOWN: u32 = 0x0002;
const MOUSEEVENTF_LEFTUP: u32 = 0x0004;
const MOUSEEVENTF_RIGHTDOWN: u32 = 0x0008;
const MOUSEEVENTF_RIGHTUP: u32 = 0x0010;
const MOUSEEVENTF_MIDDLEDOWN: u32 = 0x0020;
const MOUSEEVENTF_MIDDLEUP: u32 = 0x0040;
const MOUSEEVENTF_XDOWN: u32 = 0x0080;
const MOUSEEVENTF_XUP: u32 = 0x0100;
const XBUTTON1: u32 = 0x0001;
const XBUTTON2: u32 = 0x0002;
const INPUT_MOUSE: u32 = 0;
const KEYEVENTF_EXTENDEDKEY: u32 = 0x0001;
const WM_ACTIVATE: u32 = 0x0006;
const WA_ACTIVE: u32 = 1;
const WM_KEYDOWN: u32 = 0x0100;
const WM_KEYUP: u32 = 0x0101;
const SWP_NOSIZE: u32 = 0x0001;
const SWP_NOMOVE: u32 = 0x0002;
const SWP_SHOWWINDOW: u32 = 0x0040;
const HWND_TOP: *mut c_void = 0isize as *mut c_void;

type HWND = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct KEYBDINPUT {
    w_vk: u16,
    w_scan: u16,
    dw_flags: u32,
    time: u32,
    dw_extra_info: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MOUSEINPUT {
    dx: i32,
    dy: i32,
    mouse_data: u32,
    dw_flags: u32,
    time: u32,
    dw_extra_info: usize,
}

#[repr(C)]
union InputUnion {
    // The mi placeholder is mandatory: the union is sized by its largest member (32 B after alignment), which is what makes
    // INPUT the 40 B the system expects -- otherwise SendInput rejects the cbSize and returns 0 (a silent no-op)
    mi: MOUSEINPUT,
    ki: KEYBDINPUT,
}

#[repr(C)]
struct Input {
    type_: u32,
    u: InputUnion,
}

#[link(name = "user32")]
unsafe extern "system" {
    fn MapVirtualKeyW(u_code: u32, u_map_type: u32) -> u32;
    fn SendInput(c_inputs: u32, p_inputs: *const Input, cb_size: i32) -> u32;
    fn mouse_event(dw_flags: u32, dx: u32, dy: u32, dw_data: u32, dw_extra_info: usize);
    fn keybd_event(b_vk: u8, b_scan: u8, dw_flags: u32, dw_extra_info: usize);
    fn EnumWindows(
        lp_enum_func: Option<unsafe extern "system" fn(HWND, *mut c_void) -> i32>,
        l_param: *mut c_void,
    ) -> i32;
    fn GetWindowTextW(h_wnd: HWND, lp_string: *mut u16, n_max_count: i32) -> i32;
    fn IsWindowVisible(h_wnd: HWND) -> i32;
    fn GetWindowThreadProcessId(h_wnd: HWND, process_id: *mut u32) -> u32;
    fn SetWindowPos(
        h_wnd: HWND,
        h_wnd_insert_after: HWND,
        x: i32,
        y: i32,
        cx: i32,
        cy: i32,
        u_flags: u32,
    ) -> i32;
    fn SetForegroundWindow(h_wnd: HWND) -> i32;
    fn GetForegroundWindow() -> HWND;
    fn PostMessageW(h_wnd: HWND, msg: u32, w_param: usize, l_param: isize) -> i32;
}

/// Find a visible top-level window by title keyword (title contains the keyword; the own process window is excluded)
fn find_target_hwnd() -> Option<HWND> {
    let kw = target_window();
    if kw.is_empty() {
        return None;
    }
    struct Ctx {
        kw: String,
        found: Option<HWND>,
    }
    let mut ctx = Ctx { kw, found: None };
    unsafe extern "system" fn cb(hwnd: HWND, lparam: *mut c_void) -> i32 {
        unsafe {
            let ctx = &mut *(lparam as *mut Ctx);
            if IsWindowVisible(hwnd) == 0 {
                return 1;
            }
            let mut buf = [0u16; 256];
            let n = GetWindowTextW(hwnd, buf.as_mut_ptr(), 256);
            if n <= 0 {
                return 1;
            }
            let title = String::from_utf16_lossy(&buf[..n as usize]);
            // Exclude the own process window
            let mut pid: u32 = 0;
            GetWindowThreadProcessId(hwnd, &mut pid);
            if pid == std::process::id() {
                return 1;
            }
            if title.contains(ctx.kw.as_str()) {
                ctx.found = Some(hwnd);
                return 0; // Stop enumeration
            }
            1
        }
    }
    unsafe {
        EnumWindows(Some(cb), &mut ctx as *mut Ctx as *mut c_void);
    }
    ctx.found
}

/// Enumerate all visible top-level window titles (settings-page dropdown; the own process is excluded)
pub fn list_window_titles() -> Vec<String> {
    struct Ctx {
        out: Vec<String>,
    }
    let mut ctx = Ctx { out: Vec::new() };
    unsafe extern "system" fn cb(hwnd: HWND, lparam: *mut c_void) -> i32 {
        unsafe {
            let ctx = &mut *(lparam as *mut Ctx);
            if IsWindowVisible(hwnd) == 0 {
                return 1;
            }
            let mut buf = [0u16; 256];
            let n = GetWindowTextW(hwnd, buf.as_mut_ptr(), 256);
            if n <= 0 {
                return 1;
            }
            let title = String::from_utf16_lossy(&buf[..n as usize]);
            let mut pid: u32 = 0;
            GetWindowThreadProcessId(hwnd, &mut pid);
            if pid == std::process::id() {
                return 1;
            }
            if !ctx.out.contains(&title) {
                ctx.out.push(title);
            }
            1
        }
    }
    unsafe {
        EnumWindows(Some(cb), &mut ctx as *mut Ctx as *mut c_void);
    }
    ctx.out.sort();
    ctx.out
}

/// Bring the target window to the foreground before sending:
/// SetWindowPos(HWND_TOP) + SetForegroundWindow, retrying once on failure.
/// SendInput only injects into the foreground window -- input is lost when the game loses focus.
fn ensure_foreground(hwnd: HWND) {
    unsafe {
        if hwnd == GetForegroundWindow() {
            return;
        }
        SetWindowPos(
            hwnd,
            HWND_TOP,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
        SetForegroundWindow(hwnd);
        std::thread::sleep(std::time::Duration::from_millis(10));
        if hwnd != GetForegroundWindow() {
            SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

/// Background keying: PostMessage WM_KEYDOWN/KEYUP straight to the target window.
/// WM_ACTIVATE is sent first so the window appears active (engines may ignore input while inactive).
/// Only effective for games that handle window messages (e.g. Unity); games reading DirectInput or GetAsyncKeyState ignore it.
fn send_background_key(vk: u32, keyup: bool) {
    let Some(hwnd) = find_target_hwnd() else {
        log_send(&crate::lang::t(k::MSG_SEND_NO_TARGET));
        return;
    };
    unsafe {
        PostMessageW(hwnd, WM_ACTIVATE, WA_ACTIVE as usize, 0);
        let scan = MapVirtualKeyW(vk, MAPVK_VK_TO_VSC);
        // lParam construction:
        // bits 0-15 repeat=1, bits 16-23 scan code, bit 30 previous state, bit 31 transition state
        let lparam: isize = if keyup {
            (1 | ((scan as isize) << 16) | (1 << 30) | (1 << 31)) as isize
        } else {
            (1 | ((scan as isize) << 16)) as isize
        };
        PostMessageW(
            hwnd,
            if keyup { WM_KEYUP } else { WM_KEYDOWN },
            vk as usize,
            lparam,
        );
    }
}

/// Current foreground window title (diagnostic: when SendInput fails, shows where the keys may have gone)
fn foreground_window_title() -> String {
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return "(no foreground window)".to_string();
        }
        let mut buf = [0u16; 256];
        let n = GetWindowTextW(hwnd, buf.as_mut_ptr(), 256);
        if n <= 0 {
            "(untitled)".to_string()
        } else {
            String::from_utf16_lossy(&buf[..n as usize])
        }
    }
}

fn vk_code(key: &str) -> u32 {
    match key {
        // Modifier and control keys
        "SPACE" => 0x20,
        "SHIFT" => 0x10,
        "ENTER" => 0x0D,
        "TAB" => 0x09,
        "CTRL" => 0x11,
        "ALT" => 0x12,
        "ESC" | "ESCAPE" => 0x1B,
        "BACKSPACE" => 0x08,
        "DELETE" => 0x2E,
        "INSERT" => 0x2D,
        "HOME" => 0x24,
        "END" => 0x23,
        "PAGEUP" => 0x21,
        "PAGEDOWN" => 0x22,
        // Arrow keys (short names = what UI capture emits; long names = legacy config aliases)
        "UP" | "ARROWUP" => 0x26,
        "DOWN" | "ARROWDOWN" => 0x28,
        "LEFT" | "ARROWLEFT" => 0x25,
        "RIGHT" | "ARROWRIGHT" => 0x27,
        // Main-block punctuation (OEM keys; PLUS = the "=" key 0xBB, the same position as Code::Equal in the UI's parse_hotkey)
        "MINUS" => 0xBD,
        "PLUS" => 0xBB,
        "COMMA" => 0xBC,
        "PERIOD" => 0xBE,
        "SLASH" => 0xBF,
        "BACKSLASH" => 0xDC,
        "OPENBRACKET" => 0xDB,
        "CLOSEBRACKET" => 0xDD,
        // F1-F12 (starting at 0x70)
        "F1" => 0x70,
        "F2" => 0x71,
        "F3" => 0x72,
        "F4" => 0x73,
        "F5" => 0x74,
        "F6" => 0x75,
        "F7" => 0x76,
        "F8" => 0x77,
        "F9" => 0x78,
        "F10" => 0x79,
        "F11" => 0x7A,
        "F12" => 0x7B,
        _ => key
            .chars()
            .next()
            .map(|c| c.to_ascii_uppercase() as u32)
            .unwrap_or(0),
    }
}

/// Send a keyboard event (per the current input mode):
/// - ScanCode mode: wVk=0, wScan = the real scan code (best game compatibility)
/// - VirtualKey mode: wVk = virtual key code, wScan=0
/// - Standard mode: both wVk and wScan, and the target window is brought forward first if one is configured
/// - ScanCode mode: wVk=0 + KEYEVENTF_SCANCODE
/// - VirtualKey mode: wVk = virtual key code, wScan=0
/// - Background mode: PostMessage window messages (no foreground needed)
/// - Legacy API mode: keybd_event
fn send_key(vk: u32, keyup: bool) {
    let mode = input_mode();
    let flags_keyup = if keyup { KEYEVENTF_KEYUP } else { 0 };
    let extended = matches!(vk, 0x21..=0x28 | 0x2D | 0x2E); // PGUP PGDN HOME END arrows INS DEL
    let flags_keyup = if extended {
        flags_keyup | KEYEVENTF_EXTENDEDKEY
    } else {
        flags_keyup
    };
    match mode {
        MODE_STANDARD => {
            // Both wVk and wScan filled (no SCANCODE flag)
            // -> the system produces a standard WM_KEYDOWN (wParam=VK, lParam carries the scan code) and
            // GetAsyncKeyState/GetKeyState update in sync, which gives the widest compatibility
            if let Some(hwnd) = find_target_hwnd() {
                ensure_foreground(hwnd);
            }
            let scan = unsafe { MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) };
            let input = Input {
                type_: INPUT_KEYBOARD,
                u: InputUnion {
                    ki: KEYBDINPUT {
                        w_vk: vk as u16,
                        w_scan: scan as u16,
                        dw_flags: flags_keyup,
                        time: 0,
                        dw_extra_info: 0,
                    },
                },
            };
            let sent = unsafe { SendInput(1, &input, size_of::<Input>() as i32) };
            if sent != 1 {
                let title = foreground_window_title();
                log_send(&crate::tfmt!(k::MSG_SEND_FAILED, sent, vk, keyup, title));
            }
        }
        MODE_SCANCODE => {
            let scan = unsafe { MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) };
            let input = Input {
                type_: INPUT_KEYBOARD,
                u: InputUnion {
                    ki: KEYBDINPUT {
                        w_vk: 0,
                        w_scan: scan as u16,
                        dw_flags: KEYEVENTF_SCANCODE | flags_keyup,
                        time: 0,
                        dw_extra_info: 0,
                    },
                },
            };
            let sent = unsafe { SendInput(1, &input, size_of::<Input>() as i32) };
            if sent != 1 {
                let title = foreground_window_title();
                log_send(&crate::tfmt!(k::MSG_SEND_FAILED, sent, vk, keyup, title));
            }
        }
        MODE_VIRTUALKEY => {
            let input = Input {
                type_: INPUT_KEYBOARD,
                u: InputUnion {
                    ki: KEYBDINPUT {
                        w_vk: vk as u16,
                        w_scan: 0,
                        dw_flags: flags_keyup,
                        time: 0,
                        dw_extra_info: 0,
                    },
                },
            };
            let sent = unsafe { SendInput(1, &input, size_of::<Input>() as i32) };
            if sent != 1 {
                let title = foreground_window_title();
                log_send(&crate::tfmt!(k::MSG_SEND_FAILED_VK, sent, vk, keyup, title));
            }
        }
        MODE_BACKGROUND => {
            // Background window messages: never enter the system input queue; no foreground or administrator rights needed
            send_background_key(vk, keyup);
        }
        _ => {
            // keybd_event: wScan=0 (legacy scheme; some games ignore virtual key codes)
            unsafe { keybd_event(vk as u8, 0, flags_keyup, 0) };
        }
    }
}

/// Parse a key sequence: "CTRL+E" -> ["CTRL","E"]; a single key "E" -> ["E"].
/// Chord semantics: press in order (modifier down -> main key tap -> modifier up).
fn parse_key_seq(key: &str) -> Vec<String> {
    key.split('+')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn send_down(k: &str) {
    match k {
        "LBUTTON" => unsafe { mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0) },
        "RBUTTON" => send_mouse(MOUSEEVENTF_RIGHTDOWN, 0),
        "MBUTTON" => send_mouse(MOUSEEVENTF_MIDDLEDOWN, 0),
        "MOUSE4" => send_mouse(MOUSEEVENTF_XDOWN, XBUTTON1),
        "MOUSE5" => send_mouse(MOUSEEVENTF_XDOWN, XBUTTON2),
        _ => send_key(vk_code(k), false),
    }
}

fn send_up(k: &str) {
    match k {
        "LBUTTON" => unsafe { mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0) },
        "RBUTTON" => send_mouse(MOUSEEVENTF_RIGHTUP, 0),
        "MBUTTON" => send_mouse(MOUSEEVENTF_MIDDLEUP, 0),
        "MOUSE4" => send_mouse(MOUSEEVENTF_XUP, XBUTTON1),
        "MOUSE5" => send_mouse(MOUSEEVENTF_XUP, XBUTTON2),
        _ => send_key(vk_code(k), true),
    }
}

/// Mouse button injection (SendInput MOUSEINPUT; the two X buttons are told apart by mouse_data 1/2)
fn send_mouse(flags: u32, mouse_data: u32) {
    unsafe {
        SendInput(
            1,
            [Input {
                type_: INPUT_MOUSE,
                u: InputUnion {
                    mi: MOUSEINPUT {
                        dx: 0,
                        dy: 0,
                        mouse_data,
                        dw_flags: flags,
                        time: 0,
                        dw_extra_info: 0,
                    },
                },
            }]
            .as_ptr(),
            std::mem::size_of::<Input>() as i32,
        );
    }
}

/// Simulate pressing (without releasing) a key. Supports keyboard letters and named keys (SPACE/SHIFT etc.),
/// the left mouse button (LBUTTON), and chord sequences (CTRL+E: CTRL down, then E)
pub fn key_down(key: &str) {
    for k in parse_key_seq(key) {
        send_down(&k);
    }
}

/// Simulate releasing a key (chords release in reverse: main key first, then the modifiers)
pub fn key_up(key: &str) {
    let seq = parse_key_seq(key);
    for k in seq.iter().rev() {
        send_up(k);
    }
}

pub fn message_beep(kind: u32) {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn MessageBeep(utype: u32) -> i32;
    }
    unsafe { MessageBeep(kind) };
}

pub fn press_key(key: &str) {
    press_key_hold(key, 5);
}

pub fn press_key_hold(key: &str, hold_ms: u64) {
    let seq = parse_key_seq(key);
    if seq.is_empty() {
        return;
    }
    if seq.len() == 1 {
        send_down(&seq[0]);
        std::thread::sleep(std::time::Duration::from_millis(hold_ms));
        send_up(&seq[0]);
        return;
    }
    // Chord: modifiers first
    for k in &seq[..seq.len() - 1] {
        send_down(k);
    }
    let main = seq.last().unwrap();
    send_down(main);
    std::thread::sleep(std::time::Duration::from_millis(hold_ms));
    send_up(main);
    for k in seq[..seq.len() - 1].iter().rev() {
        send_up(k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vk_code_short_names_match_ui() {
        // Arrow keys (WinUser.h: LEFT 0x25 / UP 0x26 / RIGHT 0x27 / DOWN 0x28)
        assert_eq!(vk_code("UP"), 0x26);
        assert_eq!(vk_code("DOWN"), 0x28);
        assert_eq!(vk_code("LEFT"), 0x25);
        assert_eq!(vk_code("RIGHT"), 0x27);
        // Control keys
        assert_eq!(vk_code("SPACE"), 0x20);
        assert_eq!(vk_code("ENTER"), 0x0D);
        assert_eq!(vk_code("TAB"), 0x09);
        assert_eq!(vk_code("ESC"), 0x1B);
        assert_eq!(vk_code("SHIFT"), 0x10);
        assert_eq!(vk_code("CTRL"), 0x11);
        assert_eq!(vk_code("ALT"), 0x12);
        assert_eq!(vk_code("BACKSPACE"), 0x08);
        assert_eq!(vk_code("DELETE"), 0x2E);
        assert_eq!(vk_code("INSERT"), 0x2D);
        assert_eq!(vk_code("HOME"), 0x24);
        assert_eq!(vk_code("END"), 0x23);
        assert_eq!(vk_code("PAGEUP"), 0x21);
        assert_eq!(vk_code("PAGEDOWN"), 0x22);
        // Punctuation (OEM keys)
        assert_eq!(vk_code("MINUS"), 0xBD);
        assert_eq!(vk_code("PLUS"), 0xBB);
        assert_eq!(vk_code("COMMA"), 0xBC);
        assert_eq!(vk_code("PERIOD"), 0xBE);
        assert_eq!(vk_code("SLASH"), 0xBF);
        assert_eq!(vk_code("BACKSLASH"), 0xDC);
        assert_eq!(vk_code("OPENBRACKET"), 0xDB);
        assert_eq!(vk_code("CLOSEBRACKET"), 0xDD);
        // F keys
        assert_eq!(vk_code("F1"), 0x70);
        assert_eq!(vk_code("F8"), 0x77);
        assert_eq!(vk_code("F12"), 0x7B);
        // Single letters and digits still go through the fallback (uppercase ASCII = the VK code)
        assert_eq!(vk_code("A"), 0x41);
        assert_eq!(vk_code("z"), 0x5A);
        assert_eq!(vk_code("5"), 0x35);
    }

    #[test]
    fn vk_code_legacy_long_aliases() {
        assert_eq!(vk_code("ESCAPE"), vk_code("ESC"));
        assert_eq!(vk_code("ARROWUP"), 0x26);
        assert_eq!(vk_code("ARROWDOWN"), 0x28);
        assert_eq!(vk_code("ARROWLEFT"), 0x25);
        assert_eq!(vk_code("ARROWRIGHT"), 0x27);
    }

    /// Chord parsing: modifiers first, main key last (CTRL+E -> ["CTRL","E"])
    #[test]
    fn parse_key_seq_combo_order() {
        assert_eq!(
            parse_key_seq("CTRL+E"),
            vec!["CTRL".to_string(), "E".to_string()]
        );
        assert_eq!(parse_key_seq("E"), vec!["E".to_string()]);
        assert_eq!(
            parse_key_seq("CTRL+F8"),
            vec!["CTRL".to_string(), "F8".to_string()]
        );
        assert!(parse_key_seq("  ").is_empty());
    }
}
