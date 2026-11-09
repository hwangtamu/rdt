# E3-A iteration 6 — zero-rate diagnostics and target-rate training

## Zero-rate diagnostics (β=10)

Runs: `runs/e3a_zerorate_diag/{encoder,decoder}`
Per-epoch logs include train CE, train KL, objective, val CE, val KL, grad norm,
best epoch. Forced prior control: CE **0.9568**, acc **0.6218**.

Encoder: 5/5 seeds reached near-zero rate and prior CE.

| Seed | Best epoch | Test rate | Test CE | Mean-only CE |
|---:|---:|---:|---:|---:|
| 0 | 249 | 0.055 | 0.960 | 0.960 |
| 1 | 243 | 0.059 | 0.957 | 0.957 |
| 2 | 145 | 0.094 | 0.957 | 0.957 |
| 3 | 141 | 0.102 | 0.957 | 0.957 |
| 4 | 249 | 0.062 | 0.957 | 0.957 |

Decoder: 4/5 seeds pass; seed 1 fails.

| Seed | Best epoch | Test rate | Test CE | Mean-only CE |
|---:|---:|---:|---:|---:|
| 0 | 249 | 0.007 | 0.961 | 0.961 |
| 1 | 0 | **5.80** | **1.037** | 0.947 |
| 2 | 145 | 0.020 | 0.957 | 0.957 |
| 3 | 114 | 0.027 | 0.958 | 0.957 |
| 4 | 249 | 0.005 | 0.958 | 0.958 |

Increasing β to 100 or 1000 did not fix decoder seed 1; it still selected epoch 0
and stayed at rate 5.8. The failure is a training/initialization failure, not a
β-too-small failure.

## Target-rate training

Runs: `runs/e3a_target/{encoder,decoder}`
Adaptive dual update on measured training KL, 5 seeds.

Encoder controller diverged: β hit the upper clamp, measured rates stayed ~92
bits for every target, and CE degraded. Decoder controller was stable only for
targets 5/4/3:

| Target | Encoder rate / mean CE | Decoder rate / mean CE |
|---:|---|---|
| 5 | 92.8 / 0.803 | 3.27 / 0.744 |
| 4 | 92.3 / 0.869 | 2.83 / 0.742 |
| 3 | 92.2 / 0.870 | 2.13 / 0.744 |
| 2 | 92.0 / 0.873 | 2.34 / 0.774 |
| 1 | 91.8 / 0.937 | 2.56 / 0.799 |
| 0.5 | 91.8 / 0.937 | 2.54 / 0.803 |

The encoder's dual update is unstable; decoder cannot reach <2 bits with this
controller. Therefore target-rate training is **rejected in its current form**.

## Status

- Zero-rate: encoder passes; decoder has one reproducible failed seed.
- Target-rate transition: unstable; not usable for matched-rate comparison.
- P1 five-E2-seed validation: **not started**.

## Next steps

1. Replace the dual update with a slower log-space controller or run β sweeps and
   interpolate at matched measured rates; do not use the current target controller.
2. For decoder seed 1, either use a deterministic prior/zero-information control
   or restart from a different initialization; log it as a failed seed.
3. Only after zero-rate and matched-rate points are stable, run P1.
