# E3-A iteration 5 — refined transition grid and zero-rate check

Configs:
- `configs/e3a_boolq_v6.json` — refined transition, 5 VIB seeds, full batch, MC=64
- `configs/e3a_zerorate_lr5.json` — zero-rate diagnostic

Runs:
- `runs/e3a_boolq_v6/{encoder,decoder}`
- `runs/e3a_zerorate_lr5/{encoder,decoder}`

## Refined transition (mean over 5 seeds)

| Model | β | Rate | Expected CE | Mean-only CE |
|---|---:|---:|---:|---:|
| encoder | 0.05 | 5.58 | 0.709 | 0.638 |
| encoder | 0.08 | 3.78 | 0.736 | 0.648 |
| encoder | 0.10 | 3.51 | 0.753 | 0.657 |
| encoder | 0.13 | 2.76 | 0.776 | 0.666 |
| encoder | 0.15 | 2.32 | 0.790 | 0.675 |
| encoder | 0.20 | 1.63 | 0.819 | 0.692 |
| encoder | 0.30 | 1.16 | 0.866 | 0.730 |
| encoder | 0.50 | 1.23 | 0.926 | 0.818 |
| decoder | 0.05 | 4.66 | 0.822 | 0.750 |
| decoder | 0.08 | 3.73 | 0.831 | 0.751 |
| decoder | 0.10 | 3.37 | 0.836 | 0.753 |
| decoder | 0.13 | 2.98 | 0.849 | 0.761 |
| decoder | 0.15 | 2.86 | 0.857 | 0.768 |
| decoder | 0.20 | 2.52 | 0.877 | 0.783 |
| decoder | 0.30 | 2.16 | 0.914 | 0.813 |
| decoder | 0.50 | 1.59 | 0.962 | 0.849 |

The 3–5 bit region is now sampled. The transition curves are much smoother than
before, but expected/mean-only CE still differ substantially because of
posterior variance.

## Zero-rate check

`e3a_zerorate_lr5` (β=10, lr=5e-4) still has unstable seeds:

- encoder rates: 0.056, 0.057, 3.06, 4.74, 57.5
- decoder rates: 0.004, 0.005, 0.005, 4.57

Mean-only CE stays near the prior 0.9568 even when the measured rate is
non-zero, so prediction-prior behavior is mostly satisfied while rate control
is not reproducible across seeds.

## Status

- P0 measurement: completed.
- P0 high-rate recovery: completed via deterministic/mean-only control.
- P0 transition: substantially improved, but still has individual unstable
  seeds; raw failures retained.
- P1 five-E2-seed validation: not started.
