# E3-A iteration 4 — per-β schedule + 5 VIB seeds

Config: `configs/e3a_boolq_v5.json`
Runs: `runs/e3a_boolq_v5/{encoder,decoder}`
5 VIB seeds, full-batch, MC=64, per-β logvar init, deterministic warm start
only for β ≤ 0.1, no KL warmup.

## Mean over 5 VIB seeds

| Model | β | Rate | Expected CE | Mean-Only CE | Acc |
|---|---:|---:|---:|---:|---:|
| encoder | 0.05 | 58.89 | 0.716 | 0.666 | 0.803 |
| encoder | 0.08 | 59.21 | 0.720 | 0.670 | 0.804 |
| encoder | 0.10 | 59.16 | 0.724 | 0.672 | 0.803 |
| encoder | 0.15 | 2.32 | 0.790 | 0.675 | 0.806 |
| encoder | 0.20 | 1.63 | 0.819 | 0.692 | 0.802 |
| encoder | 0.30 | 1.16 | 0.866 | 0.730 | 0.799 |
| encoder | 0.50 | 1.23 | 0.926 | 0.818 | 0.766 |
| encoder | 1.0 | 0.41 | 0.963 | 0.946 | 0.636 |
| encoder | 10 | 0.069 | 0.958 | 0.958 | 0.622 |
| decoder | 0.05 | 15.61 | 0.782 | 0.746 | 0.755 |
| decoder | 0.08 | 13.89 | 0.794 | 0.751 | 0.750 |
| decoder | 0.10 | 11.85 | 0.803 | 0.752 | 0.751 |
| decoder | 0.15 | 2.86 | 0.857 | 0.768 | 0.744 |
| decoder | 0.20 | 2.52 | 0.877 | 0.783 | 0.744 |
| decoder | 0.30 | 2.16 | 0.914 | 0.813 | 0.740 |
| decoder | 0.50 | 1.59 | 0.963 | 0.849 | 0.731 |
| decoder | 1.0 | 0.38 | 0.974 | 0.948 | 0.633 |
| decoder | 10 | 1.17 | 0.974 | 0.956 | 0.623 |

## Findings

- High-rate: mean-only CE at β=0.05 is 0.666 encoder / 0.746 decoder,
  close to deterministic baselines 0.633 / 0.747.
- Expected stochastic CE is much higher in the transition region; posterior
  variance remains the dominant gap.
- The per-β warm start created a rate gap: β=0.1 is still high-rate (59/12),
  while β=0.15 drops to 2.3/2.9. The 3–5 bit region is under-sampled.
- Zero-rate: encoder β=10 passes; decoder β=10 has one failed seed with rate
  5.8 and best epoch 0. Raw failure retained.
- Rate monotonicity still fails: encoder β=0.5 rate 1.23 > β=0.3 1.16.

## Next

1. Remove warm start for β ≥ 0.1; add β≈0.10–0.14 points to cover 3–5 bits.
2. Fix the single decoder β=10 collapse with a fallback/no-warmstart run.
3. Do not start the five-E2-checkpoint P1 until these two are resolved.
