//! Windows administrator elevation.

//! Games often run with administrator privileges (anti-cheat, launchers), and SendInput from a normal-privilege
//! process is blocked by UIPI (it returns 0 and fails silently -- seen as "keys do not reach the game"). In UI mode,
//! a non-elevated process restarts itself through runas (one UAC prompt) and the current process exits.
//! CLI subcommands (replay/analyze etc.) never elevate.

use std::ffi::OsStr;
use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::ptr;

#[link(name = "advapi32")]
#[link(name = "shell32")]
unsafe extern "system" {
    fn OpenProcessToken(process: *mut c_void, access: u32, token: *mut *mut c_void) -> i32;
    fn GetTokenInformation(
        token: *mut c_void,
        class: u32,
        info: *mut c_void,
        len: u32,
        ret: *mut u32,
    ) -> i32;
    fn CloseHandle(h: *mut c_void) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
    fn ShellExecuteW(
        hwnd: *mut c_void,
        op: *const u16,
        file: *const u16,
        params: *const u16,
        dir: *const u16,
        show: i32,
    ) -> isize;
}

const TOKEN_QUERY: u32 = 0x0008;
const TOKEN_ELEVATION: u32 = 20;

/// Whether the current process runs elevated (TokenElevation check)
fn is_admin() -> bool {
    unsafe {
        let mut token: *mut c_void = ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return false;
        }
        let mut elev: u32 = 0;
        let mut ret: u32 = 0;
        let ok = GetTokenInformation(
            token,
            TOKEN_ELEVATION,
            &mut elev as *mut u32 as *mut c_void,
            std::mem::size_of::<u32>() as u32,
            &mut ret,
        );
        CloseHandle(token);
        ok != 0 && elev != 0
    }
}

pub fn elevate_if_needed() {
    if is_admin() {
        return;
    }
    let Some(exe) = std::env::current_exe().ok() else {
        return;
    };
    let params = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let exe_w: Vec<u16> = OsStr::new(exe.as_os_str())
        .encode_wide()
        .chain(Some(0))
        .collect();
    let op_w: Vec<u16> = OsStr::new("runas").encode_wide().chain(Some(0)).collect();
    let params_w: Vec<u16> = OsStr::new(&params).encode_wide().chain(Some(0)).collect();
    unsafe {
        let r = ShellExecuteW(
            ptr::null_mut(),
            op_w.as_ptr(),
            exe_w.as_ptr(),
            params_w.as_ptr(),
            ptr::null(),
            1,
        );
        if r as isize > 32 {
            // The elevated process has started; exit the current non-elevated one
            std::process::exit(0);
        }
        eprintln!("elevation failed (r={r}); continuing without administrator rights");
    }
}
