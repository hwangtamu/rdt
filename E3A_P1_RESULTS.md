# E3-A P1 — five E2 backbone seeds

Runs: `runs/e3a_boolq_p1/{encoder,decoder}/s{0..4}`
Config: `configs/e3a_boolq_p1.json`
5 E2 checkpoints × 3 VIB seeds × 10 β values, full-batch, MC=64.
Matched-rate interpolation is per E2 seed, then averaged across E2 seeds.

## Zero-rate status

Forced prior control: CE **0.9568**, acc **0.6218**.

- Encoder β=10: **7 / 15** checkpoint×VIB runs pass the zero-rate criterion.
- Decoder β=10: **13 / 15** pass.
- Failures include encoder checkpoint 0 (rates up to 508), checkpoint 2
  (rates up to 98), and two decoder runs with rates 5.8 and 4.0.

So the stochastic zero-rate control is still not reliable across E2 seeds.
The deterministic prior control remains valid; the trained VIB zero-rate point
does not.

## Matched-rate comparison — Expected Stochastic CE

| Target R | Seeds | Encoder CE | Decoder CE | ΔD = dec − enc [95% CI] |
|---:|---:|---:|---:|---|
| 1.6 | 3 | 0.9213 | 0.9271 | +0.0213 [−0.2427, +0.2852] |
| 2.0 | 4 | 0.9143 | 0.9093 | +0.0049 [−0.1361, +0.1460] |
| 2.5 | 4 | 0.8787 | 0.8563 | −0.0175 [−0.1391, +0.1041] |
| 3.0 | 4 | 0.8571 | 0.8362 | −0.0082 [−0.0836, +0.0672] |
| 3.5 | 3 | 0.8225 | 0.8272 | +0.0133 [−0.0844, +0.1110] |
| 4.0 | 2 | 0.8258 | 0.8211 | +0.0394 [−0.3910, +0.4699] |
| 4.5 | 1 | 0.7941 | 0.8255 | +0.0070 (single seed) |

## Matched-rate comparison — Mean-Only CE

| Target R | Seeds | Encoder CE | Decoder CE | ΔD = dec − enc [95% CI] |
|---:|---:|---:|---:|---|
| 1.6 | 3 | 0.8237 | 0.8232 | +0.0148 [−0.3274, +0.3570] |
| 2.0 | 4 | 0.8197 | 0.8246 | +0.0136 [−0.2022, +0.2294] |
| 2.5 | 4 | 0.7766 | 0.7588 | −0.0182 [−0.1758, +0.1394] |
| 3.0 | 4 | 0.7558 | 0.7467 | +0.0004 [−0.0901, +0.0909] |
| 3.5 | 3 | 0.7258 | 0.7446 | +0.0220 [−0.1048, +0.1488] |
| 4.0 | 2 | 0.7359 | 0.7442 | +0.0470 [−0.4612, +0.5552] |

## Conclusion

- The single-checkpoint pilot's encoder advantage **does not replicate** across
  the five E2 backbone seeds.
- At matched rates with acceptable seed coverage (2.5–3.5 bits), the
  encoder–decoder difference is near zero and all confidence intervals include
  zero.
- High-rate matched points have poor coverage: only 2–3 E2 seeds overlap at
  R≥3.5, and only one at R=4.5.
- Zero-rate failures prevent a final clean rate–distortion claim.

This is closer to **Outcome B** (similar compression behaviour), but the
zero-rate instability means it is not yet a definitive result.
