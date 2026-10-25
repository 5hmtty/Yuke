//! Reuses the project's existing ort stack (Session::builder + run, same as Encoder256) and loads
//! `silero_vad.onnx`.
//! Inputs `input [B,N] fp32` + `state [2,B,128]` + `sr`; outputs `output [B,1]` speech probability
//! plus `stateN` (streamed state back to the caller).
use anyhow::Context;

pub struct SileroVad {
    sess: ort::session::Session,
    /// Actual inference input per block = 512 (current block) + 64 (context tail of the previous block) = 576 @16 kHz
    input_len: usize,
    /// Samples per current block (512 @16 kHz)
    chunk: usize,
    /// Context length (64 @16 kHz)
    context_len: usize,
    sample_rate: i64,
}

impl SileroVad {
    /// Load silero_vad.onnx; plaintext or VIQENC-encrypted both work (crypto detects it).
    pub fn load(path: &std::path::Path, sample_rate: i64) -> anyhow::Result<Self> {
        use ort::session::builder::GraphOptimizationLevel;
        let mut b = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("silero session builder failed: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| anyhow::anyhow!("silero optimisation level failed: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("silero thread setup failed: {e}"))?;
        let raw = crate::pak::read("assets/silero_vad.onnx")
            .or_else(|| std::fs::read(path).ok())
            .with_context(|| {
                format!(
                    "failed to read the silero onnx (present in neither the pack nor {}): {}",
                    path.display(),
                    crate::pak::PACK_NAME
                )
            })?;
        let sess = b
            .commit_from_memory(&raw)
            .with_context(|| "silero in-memory session failed")?;
        Ok(SileroVad {
            sess,
            chunk: 512,
            context_len: 64,
            input_len: 512 + 64,
            sample_rate,
        })
    }

    /// Start of a new audio stream: clear the state.
    pub fn new_state(&self) -> Vec<Vec<f32>> {
        vec![vec![0.0f32; 128], vec![0.0f32; 128]] // [2][1][128] flattened into two parts
    }

    /// One-shot judgement of a whole clip (for breakpoints / evaluation): returns the speech probability of each chunk.
    /// `samples`: raw 16 kHz mono f32 samples.
    /// Official convention (utils_vad.OnnxWrapper.__call__): feed each block with [context of the previous block's last 64 samples + current 512]
    /// = 576 samples; state starts at all zeros and stateN is fed back. Omitting the context concatenation makes the output collapse toward 0.
    pub fn prob_sequence(&mut self, samples: &[f32]) -> anyhow::Result<Vec<f32>> {
        let mut out = Vec::new();
        let mut state = self.new_state();
        let context_len = self.context_len; // 64
        let chunk = self.chunk; // 512
        let input_len = self.input_len; // 576
        let mut context = vec![0.0f32; context_len]; // Tail 64 samples of the previous chunk
        let mut pos = 0usize;
        while pos < samples.len() {
            let end = (pos + chunk).min(samples.len());
            let seg = &samples[pos..end];
            let mut cur = vec![0.0f32; chunk];
            cur[..seg.len()].copy_from_slice(seg);
            // Concatenate [context 64 + cur 512] -> 576
            let mut inp = vec![0.0f32; input_len];
            inp[..context_len].copy_from_slice(&context);
            inp[context_len..].copy_from_slice(&cur);
            let (p, st) = self.run_chunk(&inp, &state)?;
            state = st;
            out.push(p);
            // Update context = the tail 64 samples of the current input (official x[..., -context_size:])
            let n = inp.len();
            context.copy_from_slice(&inp[n - context_len..]);
            pos = end;
        }
        Ok(out)
    }

    /// Single-block inference: `input` has length [input_len=576]; `state` is [2][128]. Returns (prob, new_state).
    fn run_chunk(
        &mut self,
        input: &[f32],
        state: &[Vec<f32>],
    ) -> anyhow::Result<(f32, Vec<Vec<f32>>)> {
        use ndarray::Array;
        let x = Array::from_shape_vec((1, self.input_len), input.to_vec())?;
        let state_arr = Array::from_shape_vec(
            (2, 1, 128),
            state.iter().flat_map(|v| v.iter().copied()).collect(),
        )?;
        let sr = Array::from_elem((), self.sample_rate);
        let feeds: Vec<(String, ort::session::SessionInputValue)> = vec![
            ("input".into(), ort::value::Value::from_array(x)?.into()),
            (
                "state".into(),
                ort::value::Value::from_array(state_arr)?.into(),
            ),
            ("sr".into(), ort::value::Value::from_array(sr)?.into()),
        ];
        let outputs = self.sess.run(feeds).with_context(|| "silero inference failed")?;
        let prob_arr = outputs[0].try_extract_array::<f32>()?;
        // Output [1,1] scalar -> take the single element
        let prob = *prob_arr.iter().next().unwrap_or(&0.0f32);
        // outputs[1] = stateN [2,1,128] <- used as the next frame's state
        let st = outputs[1].try_extract_array::<f32>()?;
        let flat: Vec<f32> = st.iter().copied().collect();
        let new_state = vec![flat[0..128].to_vec(), flat[128..256].to_vec()];
        Ok((prob, new_state))
    }
}
