# E3-A pilot — frozen-backbone VIB on BoolQ

Implementation: `src/w2c/train_e3.py`
Config: `configs/e3a_boolq.json` (+ refined beta grid `configs/e3a_boolq_refine.json`)

Checkpoints (validation-selected E2, `last-L19`, 100% BoolQ):

| Backbone | E2 seed | Checkpoint |
|---|---:|---|
| encoder | 1 | `runs/e2_boolq_68m_ckpt/encoder/checkpoints/encoder__last-L19__N100__s1.pt` |
| decoder | 2 | `runs/e2_boolq_68m_ckpt/decoder/checkpoints/decoder__last-L19__N100__s2.pt` |

VIB: linear μ/logvar, latent 32 primary, 128 control, MC 16 train / 32 eval,
β grid plus refined intermediate values, 3 VIB seeds.

## d32 rate–distortion, mean over 3 VIB seeds

| β | Encoder R | Encoder expCE | Decoder R | Decoder expCE |
|---:|---:|---:|---:|---:|
| 0 | 243.4 | 0.667 | 31.0 | 0.848 |
| 0.01 | 14.75 | 0.685 | 10.97 | 0.851 |
| 0.1 | 2.25 | 0.765 | 2.99 | 0.841 |
| 0.2 | 1.47 | 0.830 | 2.32 | 0.885 |
| 0.3 | 0.829 | 0.790 | 1.66 | 0.920 |
| 0.5 | 0.406 | 0.860 | 0.841 | 0.948 |
| 0.7 | 0.092 | 0.951 | 0.074 | 0.955 |
| 1.0 | 0.061 | 0.957 | 0.002 | 0.957 |
| 10 | 0.042 | 0.957 | 0.002 | 0.957 |

Deterministic no-bottleneck controls: encoder CE **0.633**, decoder CE **0.747**.
Prior baseline: CE **0.9568**.

## Sanity checks

- **Zero-rate:** passes. β=1/10 give near-zero rate and CE ≈ prior 0.957.
- **High-rate:** partially fails. β=0 stochastic VIB does not recover the
  deterministic control (encoder 0.667 vs 0.633; decoder 0.848 vs 0.747).
  Latent 128 was worse for the decoder (β=0 CE 0.965), so simply increasing
  dimension did not fix it.
- **Monotonicity:** unstable. Encoder β=0.3 is better than β=0.2, and the
  decoder curve has a large jump at β=0.5.

## Preliminary reading

- At matched nonzero rates, the **encoder currently dominates**:
  ~2.25 bits: encoder 0.765 vs decoder 0.841;
  ~0.4–0.8 bits: encoder 0.79–0.86 vs decoder 0.92–0.95.
- But the pilot is **not yet a valid rate–distortion conclusion** because:
  high-rate recovery fails, the β→rate mapping is unstable, and only one E2
  checkpoint per backbone was used.
- Expected stochastic CE and MC-averaged CE differ substantially for the
  decoder, indicating high posterior variance/noise.

## Next steps before the full sweep

1. Fix high-rate recovery: initialize/clamp logvar, tune MC/variance, or use a
   deterministic bottleneck+head as the formal high-rate control.
2. Refine β around the transition and repeat with 3 VIB seeds.
3. Add the 5 E2 backbone seeds for the final E3-A sweep.
