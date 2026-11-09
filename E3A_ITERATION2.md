# E3-A iteration 2 — high-rate diagnosis and stabilized pilot

## High-rate diagnosis (β=0, decoder first)

Main issue: **stochastic inference noise**, not missing representational capacity.

| Decoder β=0 | Test CE |
|---|---:|
| stochastic, expected CE (old E3) | 0.848 |
| mean-only inference | 0.744 |
| deterministic no-bottleneck control | 0.747 |
| stochastic, full-batch + logvar-init −4 | 0.752 |

Encoder:

| Encoder β=0 | Test CE |
|---|---:|
| stochastic expected | 0.667 |
| mean-only | 0.631 |
| deterministic control | 0.633 |
| full-batch + logvar-init −4 | 0.633 |

So mean-only inference recovers the deterministic baseline for both models.
The old failure was caused by unconstrained posterior variance under
minibatched β=0 training. Full-batch training with low log-variance
initialization fixes the β=0 stochastic point for the decoder too.

However, logvar-init −4 applied to all β breaks the KL/zero-rate regime:
β=10 no longer reaches zero rate. Therefore the high-rate control should be
the mean-only/deterministic bottleneck, while finite-rate stochastic runs use
logvar-init 0 and full-batch training.

## Stabilized full-batch, logvar-init 0 pilot

Outputs: `runs/e3a_boolq_v2/{encoder,decoder}`

| β | Encoder R | Encoder mean CE | Decoder R | Decoder mean CE |
|---:|---:|---:|---:|---:|
| 0 | 252.2 | 0.634 | 19.2 | 0.750 |
| 0.01 | 25.3 | 0.635 | 8.32 | 0.748 |
| 0.05 | 4.86 | 0.634 | 4.54 | 0.749 |
| 0.1 | 3.54 | 0.651 | 3.36 | 0.752 |
| 0.15 | 1.75 | 0.667 | 2.85 | 0.766 |
| 0.2 | 1.46 | 0.685 | 2.51 | 0.781 |
| 0.3 | 1.12 | 0.738 | 2.16 | 0.812 |
| 0.5 | 0.733 | 0.817 | 1.58 | 0.849 |
| 0.7 | 0.728 | 0.898 | 0.697 | 0.904 |
| 1.0 | 0.080 | 0.957 | 0.628 | 0.943 |
| 10 | 0.069 | 0.958 | 0.0095 | 0.959 |

Deterministic controls: encoder 0.633, decoder 0.747. Prior 0.9568.

## Reading

- Mean-only high-rate recovery passes for both models.
- Encoder dominates the decoder over the useful rate interval:
  at R≈4.5–5 bits both are near baseline, but encoder stays near baseline down
  to R≈1.7, while decoder is already at 0.766.
- Rate monotonicity still fails in places: encoder β=0.5/0.7 have nearly the
  same measured rate; decoder β=0.7/1.0 also overlap.
- The zero-rate check passes for logvar-init 0 (β=10 → prior CE), but not for
  logvar-init −4.

## Next actions

1. Use deterministic/mean-only control for high-rate sanity; report stochastic
   expected CE as the primary finite-rate distortion.
2. Rerun the transition region with matched measured rates, more seeds, and a
   per-β variance schedule or deterministic warm start.
3. Only then repeat across all five E2 checkpoints.
