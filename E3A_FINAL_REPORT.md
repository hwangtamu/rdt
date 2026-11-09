> **Superseded by `E3A_P2_RESULTS.md`.** The P1 run used unnormalized features,
> selected checkpoints by validation CE only, and averaged rates across VIB
> seeds before interpolation. See the P2 report for the corrected result.

# E3-A final report — negative / inconclusive

## Scope

Frozen-backbone VIB after the E2 `last-L19` decision vector, BoolQ, Ettin-68M
encoder and decoder. Five E2 backbone checkpoints, three VIB seeds, full-batch,
MC=64. Primary distortion: Expected Stochastic CE. Mean-Only CE is diagnostic.

## Zero-rate status

Forced prior control: CE **0.9568**, accuracy **0.6218**.

Trained VIB β=10 was not reliable across all E2 checkpoints:
- encoder: 7/15 checkpoint×VIB runs pass;
- decoder: 13/15 pass;
- adding KL warmup or increasing β did not fix the failures.

The trained-VIB zero-rate point is therefore rejected as a control. The forced
prior predictor is the valid zero-information control.

## Matched-rate comparison

Coverage ≥4 E2 seeds is available only around R≈2.0–3.5 bits. No extrapolation.

Expected Stochastic CE, ΔD = decoder − encoder:

| Target R | E2 seeds | ΔD | 95% CI |
|---:|---:|---:|---|
| 2.0 | 4 | +0.0049 | [−0.1361, +0.1460] |
| 2.5 | 4 | −0.0175 | [−0.1391, +0.1041] |
| 3.0 | 4 | −0.0082 | [−0.0836, +0.0672] |
| 3.5 | 3 | +0.0133 | [−0.0844, +0.1110] |

Mean-Only CE gives the same qualitative result. **No matched-rate difference is
detected.** This is not proof of equivalence; the intervals are wide and
high-rate coverage is limited.

## Compression penalty

Penalty = E3 CE at matched rate − deterministic no-bottleneck control CE for the
same E2 checkpoint. Δpenalty = decoder penalty − encoder penalty; negative means
the encoder has the larger compression penalty.

Expected CE:

| Target R | E2 seeds | Δpenalty | 95% CI |
|---:|---:|---:|---|
| 2.0 | 4 | −0.0365 | [−0.1957, +0.1227] |
| 2.5 | 4 | −0.0590 | [−0.1974, +0.0794] |
| 3.0 | 4 | −0.0498 | [−0.1372, +0.0377] |
| 3.5 | 3 | −0.0193 | [−0.0828, +0.0441] |

Mean-Only CE:

| Target R | E2 seeds | Δpenalty | 95% CI |
|---:|---:|---:|---|
| 2.5 | 4 | −0.0597 | [−0.2278, +0.1083] |
| 3.0 | 4 | −0.0412 | [−0.1268, +0.0443] |

Descriptively, the encoder often has the larger compression penalty relative to
its own deterministic baseline, but every CI includes zero. This does not
establish that the encoder is more compression-sensitive.

## Bottleneck-position assessment

The current E3-A bottleneck sits after the final pooled decision vector. For a
binary task like BoolQ, that vector is already close to the output; the
experiment therefore mostly measures compressibility of the final decision
readout, not token-level information organization.

A pre-pooling/token-level bottleneck would answer a different question. It
requires matched token-level VIB + pooling architectures, length-dependent rate
accounting, and a separate validation protocol. It should be treated as a new
E3-B experiment, not as another point on the E3-A curve.

## Decision

E3-A is **negative/inconclusive**:
- no robust encoder information-efficiency advantage;
- zero-rate VIB training is not fully stable;
- compression-penalty differences are not statistically supported;
- post-pooling bottleneck is likely too late for token-level claims.

Do not proceed to a strong rate–distortion claim or to E3-B without redesigning
the zero-rate control and pre-pooling experiment.
