//! Automatic activity-gate adjustment: estimates the noise floor from a low percentile of the band-passed envelope over the last few seconds and follows it.
//!
//! Decision rule:
//! ```text
//! noise floor nf = P20(F1 band-pass envelope in dB over the last <=25 s)
//! gate           = clamp(nf + 10 dB, -70, -25)
//! ```

//! Re-evaluated once per second ([`CHUNK`]) and constant in between; each step is limited to +/-[`MAX_STEP`] dB,
//! moves smaller than [`DEADZONE`] are ignored, and the first adjustment of a session (`aggressive`) skips the step clamp.

pub const CHUNK: usize = 100;
/// Analysis window cap: 2500 frames (25 s); the envelope ring keeps 3000 frames (30 s) for 5 s of headroom.
/// Longer windows steady P20 (speech duty cycle is diluted); with the ±3 dB/s step clamp a step change still converges in about 10 s.
pub const LOOKBACK_MAX: usize = 2500;
/// Minimum frame count: no decision below this (the gate stays put for the first 2 s of a session).
pub const MIN_FRAMES: usize = 200;
const MARGIN: f64 = 10.0;
/// Step clamp of ±3 dB per tick (= 3 dB/s): a physical upper bound on churn.
const MAX_STEP: f64 = 3.0;
const DEADZONE: f64 = 1.0;
const FLOOR: f64 = -70.0;
const CEILING: f64 = -25.0;

/// Result of one tick.
pub struct Decision {
    /// Some = rewrite the gate to this value; None = keep the current gate (no adjustment).
    pub new_gate: Option<f64>,
    pub nf_est: Option<f64>,
    /// Why the decision was made (used by diagnostic lines and test assertions).
    pub reason: String,
}

/// Nearest-rank percentile (sort and index, no interpolation; constant-run test vectors stay hand-verifiable).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((q / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Three narrowing steps, in this order so the dead zone compares the value that is actually written:
/// 1) hard bounds [FLOOR, CEILING] -> 2) per-tick step clamp (skipped when aggressive=true, which takes a session's first
/// adjustment straight to target and removes the cold-start window) -> 3) dead zone, no move.
fn decide(cand: f64, gate: f64, nf_est: f64, reason: String, aggressive: bool) -> Decision {
    let cand = cand.clamp(FLOOR, CEILING);
    if aggressive {
        if (cand - gate).abs() < DEADZONE {
            return Decision {
                new_gate: None,
                nf_est: Some(nf_est),
                reason: format!("{reason}; candidate inside the dead zone"),
            };
        }
        return Decision {
            new_gate: Some(cand),
            nf_est: Some(nf_est),
            reason,
        };
    }
    let cand = cand.clamp(gate - MAX_STEP, gate + MAX_STEP);
    if (cand - gate).abs() < DEADZONE {
        return Decision {
            new_gate: None,
            nf_est: Some(nf_est),
            reason: format!("{reason}; candidate inside the dead zone"),
        };
    }
    Decision {
        new_gate: Some(cand),
        nf_est: Some(nf_est),
        reason,
    }
}

/// One automatic gate decision.
pub fn tick(env_tail: &[f64], gate: f64, aggressive: bool) -> Decision {
    if env_tail.len() < MIN_FRAMES {
        return Decision {
            new_gate: None,
            nf_est: None,
            reason: "not enough samples".into(),
        };
    }
    let take = env_tail.len().min(LOOKBACK_MAX);
    let win = &env_tail[env_tail.len() - take..];
    let mut sorted = win.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p20 = percentile(&sorted, 20.0);
    let cand = p20 + MARGIN;
    decide(
        cand,
        gate,
        p20,
        format!("P20 window {:.1}s", take as f64 / 100.0),
        aggressive,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test window = 5 s (500 frames): must be >= MIN_FRAMES (200) and long enough for the low percentile to land in silence.
    const W: usize = 500;

    /// Noise floor -50, gate -30 -> candidate -40; the aggressive first adjustment goes straight to -40.
    #[test]
    fn steady_noise_converges_then_freezes() {
        let env = vec![-50.0; W];
        let d = tick(&env, -30.0, true);
        assert_eq!(
            d.new_gate,
            Some(-40.0),
            "the first tick should jump straight to P20+10, reason={}",
            d.reason
        );
        assert_eq!(d.nf_est, Some(-50.0));
        let d2 = tick(&env, d.new_gate.unwrap(), false);
        assert_eq!(d2.new_gate, None, "already in place: frozen, reason={}", d2.reason);
    }

    #[test]
    fn p20_ignores_speech_duty() {
        // 16% speech (-25 dB) + 84% silence (-50 dB) with the gate already at -40 (= silence + 10) -> must not move
        let mut env = Vec::new();
        for i in 0..W {
            env.push(if i % 25 < 4 { -25.0 } else { -50.0 });
        }
        let d = tick(&env, -40.0, false);
        assert_eq!(
            d.nf_est,
            Some(-50.0),
            "P20 must land in silence (-50), got {:?}",
            d.nf_est
        );
        assert_eq!(d.new_gate, None, "speech must not push the gate up, reason={}", d.reason);
    }

    #[test]
    fn p20_follows_when_speech_dominates() {
        let mut env = Vec::new();
        for i in 0..W {
            env.push(if i % 10 < 9 { -30.0 } else { -60.0 });
        }
        let d = tick(&env, -40.0, true);
        // P20 = -30 (90% of frames at -30) -> candidate -20 -> clamped to the -25 ceiling
        assert_eq!(d.new_gate, Some(-25.0), "reason={}", d.reason);
    }

    /// Step clamp: a 30 dB drop in the noise floor moves the gate by at most -3 dB per tick (non-aggressive path).
    #[test]
    fn step_clamp_limits_rate() {
        let env = vec![-80.0; W];
        let mut gate = -40.0;
        for k in 1..=3 {
            let d = tick(&env, gate, false);
            assert_eq!(d.new_gate, Some(gate - 3.0), "tick {k} should drop exactly 3 dB");
            gate = d.new_gate.unwrap();
        }
        // Aggressive first adjustment: straight to floor (candidate -70 is inside the hard bounds)
        let d = tick(&env, -40.0, true);
        assert_eq!(d.new_gate, Some(-70.0), "reason={}", d.reason);
    }

    /// Hard bounds: very quiet floor -> -70; very loud steady noise -> -25.
    #[test]
    fn hard_bounds_apply() {
        let d = tick(&vec![-95.0; W], -40.0, true);
        assert_eq!(d.new_gate, Some(-70.0), "reason={}", d.reason);
        let d2 = tick(&vec![-20.0; LOOKBACK_MAX], -40.0, true);
        assert_eq!(d2.new_gate, Some(-25.0), "reason={}", d2.reason);
    }


}
