# E3-A iteration 7 — matched-rate comparison and zero-rate status

## Matched-rate results from the β sweep

Because the target-rate controller was unstable, matched-rate comparisons use
interpolation on the existing `runs/e3a_boolq_v6` β sweep, with 5 VIB seeds.
No extrapolation is done outside per-seed observed rate ranges.

Expected Stochastic CE (primary):

| Target R | Seeds | Encoder CE | Decoder CE | ΔD = dec − enc [95% CI] |
|---:|---:|---:|---:|---|
| 1.60 | 4 | 0.8798 | 0.9537 | +0.0739 [−0.0446, +0.1923] |
| 2.00 | 5 | 0.7983 | 0.9329 | **+0.1345** [+0.0620, +0.2071] |
| 2.50 | 5 | 0.8244 | 0.8795 | +0.0551 [−0.0826, +0.1928] |
| 3.00 | 5 | 0.7993 | 0.8488 | +0.0495 [−0.0729, +0.1719] |
| 3.50 | 5 | 0.7735 | 0.8346 | +0.0611 [−0.0454, +0.1677] |
| 4.00 | 4 | 0.7746 | 0.8273 | +0.0527 [−0.0572, +0.1627] |
| 4.50 | 4 | 0.7564 | 0.8223 | +0.0659 [−0.0040, +0.1357] |

Mean-Only CE (diagnostic):

| Target R | Seeds | Encoder CE | Decoder CE | ΔD = dec − enc [95% CI] |
|---:|---:|---:|---:|---|
| 1.60 | 4 | 0.7720 | 0.8550 | +0.0829 [−0.0473, +0.2131] |
| 2.00 | 5 | 0.6825 | 0.8288 | **+0.1463** [+0.0669, +0.2257] |
| 2.50 | 5 | 0.7174 | 0.7835 | +0.0662 [−0.0751, +0.2075] |
| 3.00 | 5 | 0.6998 | 0.7599 | +0.0601 [−0.0615, +0.1817] |
| 3.50 | 5 | 0.6806 | 0.7517 | +0.0711 [−0.0318, +0.1739] |
| 4.00 | 4 | 0.6826 | 0.7506 | +0.0679 [−0.0358, +0.1717] |
| 4.50 | 4 | 0.6682 | 0.7498 | **+0.0815** [+0.0179, +0.1452] |

The sign consistently favors the encoder, but with 5 VIB seeds the confidence
intervals are wide; only some points exclude zero.

## Zero-rate and forced control

- Forced prior control: CE **0.9568**, accuracy **0.6218**.
- Encoder β=10: 5/5 seeds near-zero rate and prior CE.
- Decoder β=10: 4/5 seeds pass; seed 1 fails reproducibly
  (best epoch 0, rate 5.8, expected CE 1.037, mean-only CE 0.947).
- Increasing β to 100/1000 did not fix decoder seed 1.

## Target-rate training

The adaptive dual controller was rejected: encoder β diverged to the clamp and
measured rates stayed around 92 bits for every target; decoder could not reach
below ~2 bits. Target-rate results should not be used.

## Status

- Zero-rate: encoder passes; decoder has one documented failed seed.
- Transition: matched-rate interpolation is usable for analysis.
- P1 five-E2-seed validation: still not started; current pilot uses one E2
  checkpoint per architecture.
