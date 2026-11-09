> **Superseded by `E3A_P2_RESULTS.md`.** See that report for the corrected P2 run.

# E3-A final status after five-backbone-seed P1

## P1 matched-rate result

Five E2 checkpoints × 3 VIB seeds × β sweep, full-batch, MC=64.
Coverage ≥4 E2 seeds is only available for R≈2.0–3.5 bits.

Expected Stochastic CE, ΔD = decoder − encoder:

| Target R | Seeds | ΔD | 95% CI |
|---:|---:|---:|---|
| 2.0 | 4 | +0.0049 | [−0.1361, +0.1460] |
| 2.5 | 4 | −0.0175 | [−0.1391, +0.1041] |
| 3.0 | 4 | −0.0082 | [−0.0836, +0.0672] |
| 3.5 | 3 | +0.0133 | [−0.0844, +0.1110] |

Mean-Only CE gives the same qualitative result.

## Zero-rate

- Forced prior control: CE 0.9568, acc 0.6218.
- Trained VIB β=10 across P1: encoder 7/15 pass, decoder 13/15 pass.
- Adding a 50-epoch KL warmup did not fix encoder/decoder zero-rate failures.
- Increasing β to 100/1000 did not fix the known decoder failure.
- The trained-VIB zero-rate point is therefore rejected as a reliable control.
  The forced prior predictor is the valid zero-information control.

## Target-rate controller

Rejected: encoder dual update diverges, decoder cannot reach <2 bits.

## Conclusion

E3-A does **not** support a robust encoder information-efficiency advantage.
The single-checkpoint pilot advantage does not replicate across the five E2
backbone seeds. At matched rates with adequate coverage, the difference is
statistically indistinguishable from zero. The result is closer to Outcome B
(similar compression behaviour), with zero-rate VIB instability preventing a
fully definitive claim.

## Decision

Do not proceed to E3-B or claim an encoder rate–distortion advantage. Either:
1. report E3-A as negative/inconclusive with the forced-prior zero-rate control, or
2. redesign the zero-rate objective and rerun P1.
