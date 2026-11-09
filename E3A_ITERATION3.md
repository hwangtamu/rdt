# E3-A iteration 3 — corrected measurement and stabilization attempts

## 1. Corrected measurement (MC=64)

Runs: `runs/e3a_boolq_v2_mc64/{encoder,decoder}`
Both Expected Stochastic CE and Mean-Only CE are recorded for every run.

| Model | β | Rate | Expected CE | Mean-only CE |
|---|---:|---:|---:|---:|
| encoder | 0 | 252.0 | 0.642 | 0.634 |
| encoder | 0.05 | 5.11 | 0.712 | 0.635 |
| encoder | 0.10 | 3.78 | 0.754 | 0.658 |
| encoder | 0.15 | 1.80 | 0.793 | 0.668 |
| encoder | 0.20 | 1.51 | 0.822 | 0.692 |
| encoder | 0.30 | 0.94 | 0.869 | 0.728 |
| encoder | 0.50 | 0.84 | 0.928 | 0.814 |
| encoder | 1.0 | 0.074 | 0.958 | 0.958 |
| decoder | 0 | 19.2 | 0.787 | 0.750 |
| decoder | 0.05 | 4.93 | 0.821 | 0.749 |
| decoder | 0.10 | 3.46 | 0.837 | 0.752 |
| decoder | 0.15 | 2.95 | 0.859 | 0.766 |
| decoder | 0.20 | 2.62 | 0.879 | 0.782 |
| decoder | 0.30 | 2.27 | 0.916 | 0.812 |
| decoder | 0.50 | 1.46 | 0.967 | 0.844 |
| decoder | 1.0 | 0.63 | 0.984 | 0.943 |

The gap between expected stochastic CE and mean-only CE is large in the
transition region (~0.07–0.10 bits), so both must be reported.

## 2. High-rate recovery

Decoder β=0:

| Variant | Test CE |
|---|---:|
| stochastic, minibatch | 0.848 |
| mean-only | 0.744 |
| deterministic no-bottleneck | 0.747 |
| stochastic, full-batch + logvar-init −4 | 0.752 |

Encoder β=0:

| Variant | Test CE |
|---|---:|
| stochastic, minibatch | 0.667 |
| mean-only | 0.631 |
| deterministic no-bottleneck | 0.633 |
| stochastic, full-batch + logvar-init −4 | 0.633 |

So high-rate recovery is achievable with deterministic/mean-only evaluation
and with full-batch + low-variance initialization for β=0.

But logvar-init −4 applied to all β breaks the zero-rate regime: β≥1 no longer
reaches the prior CE. Therefore:
- high-rate sanity control: deterministic/mean-only bottleneck;
- finite-rate stochastic runs: full-batch, logvar-init 0, no strong warm start.

## 3. Warm-start attempt

A deterministic warm start (30 epochs) followed by logvar-init −6 and a 30-epoch
KL warmup was tried in `runs/e3a_boolq_v4`.

It failed rate control: β=10 still had large rate and non-prior CE
(encoder R≈136, CE≈0.724; decoder R≈37, CE≈0.895). This failure is retained,
not smoothed. The warm-start recipe is rejected in its current form.

## 4. Current status against the action items

- P0 measurement: **done** for the current pilot (MC=64, expected + mean-only).
- P0 high-rate recovery: **diagnosed and controlled**; mean-only/deterministic
  control recovers baseline within ~0.02; stochastic expected CE remains noisy.
- P0 transition stabilization: **not solved**. Rate monotonicity still fails in
  places and the warm-start variant was rejected.
- P1 five-backbone-seed validation: **not started**.

## Next iteration

1. Use a per-β variance schedule rather than one global logvar init:
   low variance for high-rate points, logvar-init 0 for β≥1.
2. Drop the current warm-start; try deterministic initialization of μ/logvar
   from a trained linear head, without a long KL warmup.
3. Rerun matched-rate points in 0.5–5 bits with ≥5 VIB seeds and keep raw
   failed points.
