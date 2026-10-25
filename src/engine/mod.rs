//! Engine group: the decision core.
//!
//! - `udkws_v2`: streaming decision engine (activity gate / syllable nuclei / identity matching / emission)
//! - `gate`: real-time gating processor (level meter, trigger consumption and dispatch, veto layer, macro execution)
//! - `auto_gate`: low-percentile estimator that lets the activity gate follow the noise floor
//! - `features`: log-mel features, wav I/O, resampling
//! - `nuclei_cache`: syllable-nucleus template cache (startup acceleration)
//! - `proto_meta`: template input fingerprint (decides whether to rebuild at startup)

pub mod auto_gate;
pub mod features;
pub mod gate;
pub mod nuclei_cache;
pub mod proto_meta;
pub mod udkws_v2;
