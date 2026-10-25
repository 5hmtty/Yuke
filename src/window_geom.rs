//! **Why adaptive sizing**: DPI scaling brings in two coordinate systems -- physical pixels and logical points --
//! and egui/`ViewportBuilder` work in **logical points**. A hardcoded logical size hits the "small logical
//! resolution" wall: 1920x1080@150% has only ~680pt of usable height and 1366x768@125% only ~574pt, so the 700pt

//! **Pure logic only**: `default_size` / `min_size` / `saved_ok` / `clamp_into` / `parse_saved`
//! make no system calls and are directly unit-testable; OS probing is confined to `primary_work_area`.

//! Probing uses only ancient Win32 APIs (`SPI_GETWORKAREA` + `GetDeviceCaps(LOGPIXELSX)`) -- **not**
//! `GetDpiForSystem` (Windows 10 1607+ only): statically importing a missing export makes the exe fail at load,
//! a failure class already recorded in this repository's Cargo.toml for bcryptprimitives/SystemPrng. Note also that
//! this function runs **before** winit creates the event loop, so the process may still be DPI-unaware and the
//! system returns coordinates **virtualized to 96 dpi** -- which happen to equal logical points, so the result
//! remains usable (ppp reports 1.0, so the ppp check uses real values on the first frame; see `first_frame_window_check` in ui.rs).

use serde_json::Value;

/// Monitor work area (logical points, taskbar already excluded) plus the scale factor at that moment (ppp = dpi / 96)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkArea {
    pub left: f32,
    pub top: f32,
    pub w: f32,
    pub h: f32,
    pub ppp: f32,
}

/// Conservative fallback when probing fails (assume 1600×900@100%): better one size smaller than off-screen.
pub const WORK_AREA_FALLBACK: WorkArea = WorkArea {
    left: 0.0,
    top: 0.0,
    w: 1600.0,
    h: 900.0,
    ppp: 1.0,
};

/// Share of the screen the default size may occupy (width / height): used **only when the baseline does not fit**, to scale down proportionally.
const DEFAULT_W_RATIO: f32 = 0.75;
const DEFAULT_H_RATIO: f32 = 0.80;
const REF_W: f32 = 720.0;
const REF_H: f32 = 710.0;
/// Scale ceiling = **1.0: never enlarge on big screens**, only scale down when the screen cannot fit the baseline.
/// Rationale: enlarging the window without scaling the content just adds whitespace, so a slightly smaller default is better.
const MAX_SCALE: f32 = 1.0;
/// Absolute ceiling for the minimum size (logical points): the window may be shrunk to this, but does not need to be larger
const MIN_W: f32 = 720.0;
const MIN_H: f32 = 520.0;
/// Minimum visible amount required for "still on screen" (logical points): enough to expose the title bar for grabbing
const MIN_VISIBLE: f32 = 120.0;

/// Remembered window geometry (settings.json `window_rect`). Position is the **outer top-left corner**, size is the **inner box**.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavedGeometry {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Native ppp at save time (a scale change invalidates the record)
    pub ppp: f32,
}

impl SavedGeometry {
    pub fn to_json(self) -> Value {
        serde_json::json!({ "x": self.x, "y": self.y, "w": self.w, "h": self.h, "ppp": self.ppp })
    }
}

/// Parse `window_rect`: missing field / wrong type / non-finite -> None (lenient: a bad value counts as absent).
pub fn parse_saved(raw: &Value) -> Option<SavedGeometry> {
    let o = raw.get("window_rect")?;
    let f = |k: &str| {
        o.get(k)
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .filter(|v| v.is_finite())
    };
    Some(SavedGeometry {
        x: f("x")?,
        y: f("y")?,
        w: f("w")?,
        h: f("h")?,
        ppp: f("ppp")?,
    })
}

/// Primary monitor work area plus DPI scale probing (logical points). API failure -> None (the caller applies the fallback).
pub fn primary_work_area() -> Option<WorkArea> {
    #[repr(C)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }
    type Handle = *mut core::ffi::c_void;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn SystemParametersInfoW(
            action: u32,
            param: u32,
            data: *mut core::ffi::c_void,
            win_ini: u32,
        ) -> i32;
        fn GetDC(hwnd: Handle) -> Handle;
        fn ReleaseDC(hwnd: Handle, hdc: Handle) -> i32;
    }
    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn GetDeviceCaps(hdc: Handle, index: i32) -> i32;
    }
    const SPI_GETWORKAREA: u32 = 0x0030;
    const LOGPIXELSX: i32 = 88;

    let mut rc = Rect {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };
    // SAFETY: process-level query with no window handle; rc is a fixed-size stack struct matching the API contract
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETWORKAREA,
            0,
            &mut rc as *mut Rect as *mut core::ffi::c_void,
            0,
        )
    };
    if ok == 0 {
        return None;
    }
    let (pw, ph) = ((rc.right - rc.left) as f32, (rc.bottom - rc.top) as f32);
    if pw <= 0.0 || ph <= 0.0 {
        return None;
    }
    // SAFETY: GetDC(NULL) takes the screen DC and is released immediately; a 0 result is handled by the caller
    let dpi = unsafe {
        let hdc = GetDC(std::ptr::null_mut());
        if hdc.is_null() {
            0
        } else {
            let d = GetDeviceCaps(hdc, LOGPIXELSX);
            ReleaseDC(std::ptr::null_mut(), hdc);
            d
        }
    };
    let ppp = if dpi > 0 { dpi as f32 / 96.0 } else { 1.0 };
    Some(WorkArea {
        left: rc.left as f32 / ppp,
        top: rc.top as f32 / ppp,
        w: pw / ppp,
        h: ph / ppp,
        ppp,
    })
}

/// Default window size: **baseline 760×700 scaled proportionally, shrink-only** (`MAX_SCALE = 1.0`).

/// Three constraints tighten the scale factor: 1) width ≤ 75% of the work area; 2) height ≤ 80%; 3) ≤ `MAX_SCALE`.
/// If the screen fits -> the baseline itself; if not (small logical screen) -> scale down proportionally and still fit.
/// Do **not** clamp width and height independently -- that bends the aspect ratio.
pub fn default_size(a: &WorkArea) -> (f32, f32) {
    let fit = (a.w * DEFAULT_W_RATIO / REF_W).min(a.h * DEFAULT_H_RATIO / REF_H);
    let s = fit.min(MAX_SCALE);
    ((REF_W * s).max(1.0), (REF_H * s).max(1.0))
}

/// Minimum size: **must not exceed the default** -- otherwise a small logical screen makes "the default size itself invalid" (winit rejects it).
pub fn min_size(def: (f32, f32)) -> (f32, f32) {
    (def.0.min(MIN_W), def.1.min(MIN_H))
}

/// Whether the remembered geometry is still usable: finite values + not smaller than the minimum + the rectangle still **mostly inside** the work area.
/// If any of these fails -> invalid (the caller discards it and resets to defaults, satisfying "reset automatically on surprises").
pub fn saved_ok(g: &SavedGeometry, a: &WorkArea, min: (f32, f32)) -> bool {
    let finite = g.x.is_finite() && g.y.is_finite() && g.w.is_finite() && g.h.is_finite();
    if !finite || g.w < min.0 || g.h < min.1 {
        return false;
    }
    // Top edge above the work area (title bar unreachable) or below it -> invalid
    if g.y < a.top - 1.0 || g.y > a.top + a.h - MIN_VISIBLE {
        return false;
    }
    if g.x > a.left + a.w - MIN_VISIBLE || g.x + g.w < a.left + MIN_VISIBLE {
        return false;
    }
    // At least half the area must be visible, so the window is never "a corner peeking out"
    let ix = (g.x + g.w).min(a.left + a.w) - g.x.max(a.left);
    let iy = (g.y + g.h).min(a.top + a.h) - g.y.max(a.top);
    ix > 0.0 && iy > 0.0 && ix * iy >= 0.5 * g.w * g.h
}

/// Clamp the geometry into the work area: size no smaller than min and no larger than the work area, position keeps the whole window visible.
pub fn clamp_into(mut g: SavedGeometry, a: &WorkArea, min: (f32, f32)) -> SavedGeometry {
    // Note: the lower clamp bound must be min(upper bound) first, otherwise lo > hi on a tiny work area panics
    g.w = g.w.clamp(min.0.min(a.w), a.w);
    g.h = g.h.clamp(min.1.min(a.h), a.h);
    g.x = g.x.clamp(a.left, (a.left + a.w - g.w).max(a.left));
    g.y = g.y.clamp(a.top, (a.top + a.h - g.h).max(a.top));
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(w: f32, h: f32) -> WorkArea {
        WorkArea {
            left: 0.0,
            top: 0.0,
            w,
            h,
            ppp: 1.0,
        }
    }

    #[test]
    fn default_size_keeps_reference_aspect_ratio() {
        let want = REF_W / REF_H;
        for a in [
            area(1093.0, 574.0),
            area(1366.0, 728.0),
            area(1600.0, 900.0),
            area(1920.0, 1040.0),
            area(3840.0, 2080.0),
        ] {
            let (w, h) = default_size(&a);
            assert!(
                (w / h - want).abs() < 0.01,
                "{a:?} -> {w}x{h} aspect drifted (expected {want})"
            );
            assert!(w <= a.w && h <= a.h, "{a:?} -> {w}x{h} must fit");
        }
    }

    /// The default size must fit on small logical resolutions (the core bug this solves); on large screens the ceiling applies
    #[test]
    fn default_size_fits_small_logical_screens() {
        // 1366×768@125% -> logical ~1093×574 (taskbar excluded)
        let (w, h) = default_size(&area(1093.0, 574.0));
        assert!(h <= 574.0 && w <= 1093.0, "the default size must fit on a small screen: {w}x{h}");
        // Large screen: held by MAX_SCALE -- 900×660 × 1.2 = 1080×792
        let big = default_size(&area(3840.0, 2080.0));
        assert!(
            (big.0 - REF_W * MAX_SCALE).abs() < 0.01 && (big.1 - REF_H * MAX_SCALE).abs() < 0.01,
            "a large screen must be capped by MAX_SCALE: {big:?}"
        );
        // Medium screen fits the baseline -> use the baseline as-is (no enlargement)
        let mid = default_size(&area(1600.0, 900.0));
        assert!(
            (mid.0 - REF_W * MAX_SCALE).abs() < 0.01 && (mid.1 - REF_H * MAX_SCALE).abs() < 0.01,
            "when it fits the result should equal the base size (no upscale): {mid:?}"
        );
    }

    #[test]
    fn saved_ok_rejects_offscreen_and_oversize() {
        let a = area(1093.0, 574.0);
        let min = min_size(default_size(&a));
        let g = |x: f32, y: f32, w: f32, h: f32| SavedGeometry {
            x,
            y,
            w,
            h,
            ppp: 1.0,
        };
        // Sample sizes must be >= min (otherwise they fail early on "smaller than minimum" and hide the geometry being tested)
        assert!(min.1 <= 540.0, "the sample height 540 must not fall below the minimum {min:?}");
        assert!(
            saved_ok(&g(100.0, 60.0, 800.0, 540.0), &a, min),
            "a normal value should hit the clamp"
        );
        // After unplugging an external monitor: the saved position falls in a region that no longer exists
        assert!(
            !saved_ok(&g(3000.0, 200.0, 800.0, 540.0), &a, min),
            "entirely off-screen to the right"
        );
        assert!(
            !saved_ok(&g(100.0, -300.0, 800.0, 540.0), &a, min),
            "top edge above the screen (title bar unreachable)"
        );
        assert!(
            !saved_ok(&g(100.0, 800.0, 800.0, 540.0), &a, min),
            "top edge below the screen"
        );
        assert!(
            !saved_ok(&g(-980.0, 60.0, 800.0, 540.0), &a, min),
            "only one corner visible"
        );
        // Size below the minimum
        assert!(
            !saved_ok(&g(10.0, 10.0, 300.0, 200.0), &a, min),
            "smaller than the minimum size"
        );
        // An oversized value left over from a high scale factor (a common symptom after changing scaling)
        assert!(
            !saved_ok(&g(0.0, 0.0, 1500.0, 900.0), &a, min),
            "larger than the work area -> less than half visible"
        );
        assert!(
            !saved_ok(&g(f32::NAN, 0.0, 800.0, 500.0), &a, min),
            "a non-finite value"
        );
    }

    /// Clamping: keeps the whole window visible and no smaller than min (a small work area must not panic)
    #[test]
    fn clamp_into_keeps_window_visible() {
        let a = area(1093.0, 574.0);
        let min = (720.0, 520.0);
        let c = clamp_into(
            SavedGeometry {
                x: 900.0,
                y: 500.0,
                w: 800.0,
                h: 500.0,
                ppp: 1.0,
            },
            &a,
            min,
        );
        assert!(
            c.x + c.w <= a.w + 0.01 && c.y + c.h <= a.h + 0.01,
            "the bottom-right must not exceed the screen: {c:?}"
        );
        assert!(c.w >= min.0 && c.h >= min.1, "must not be smaller than the minimum: {c:?}");
        // Work area smaller than the minimum size -> squeeze to the work area without panicking
        let tiny = clamp_into(
            SavedGeometry {
                x: 0.0,
                y: 0.0,
                w: 800.0,
                h: 600.0,
                ppp: 1.0,
            },
            &area(600.0, 400.0),
            min,
        );
        assert!(tiny.w <= 600.0 && tiny.h <= 400.0, "{tiny:?}");
    }

    /// `window_rect` parsing: bad values count as absent (no panic, no half-parsed data)
    #[test]
    fn parse_saved_is_lenient() {
        assert!(parse_saved(&serde_json::json!({})).is_none());
        assert!(parse_saved(&serde_json::json!({"window_rect": 3})).is_none());
        assert!(
            parse_saved(&serde_json::json!({"window_rect": {"x":1,"y":2,"w":3}})).is_none(),
            "a missing field"
        );
        assert!(
            parse_saved(&serde_json::json!({"window_rect": {"x":"a","y":2,"w":3,"h":4,"ppp":1}}))
                .is_none(),
            "a wrong-typed field"
        );
        let g = parse_saved(
            &serde_json::json!({"window_rect": {"x":1.0,"y":2.0,"w":300.0,"h":400.0,"ppp":1.5}}),
        )
        .expect("a normal value should parse");
        assert_eq!(
            g,
            SavedGeometry {
                x: 1.0,
                y: 2.0,
                w: 300.0,
                h: 400.0,
                ppp: 1.5
            }
        );
        // Round-trip: to_json can be parsed back by ourselves
        assert_eq!(
            parse_saved(&serde_json::json!({"window_rect": g.to_json()})),
            Some(g)
        );
    }

}
