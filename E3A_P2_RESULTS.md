# E3-A P2 — corrected 68M BoolQ rate–distortion run

Run: `runs/e3a_boolq_p2/{encoder,decoder}/s{0..4}`  
Config: `configs/e3a_boolq_p2.json`  
Analysis: `runs/e3a_boolq_p2/analysis/`  
New analysis code: `src/w2c/compare_e3_p2.py`

## What was fixed relative to the old E3-A P1 run

1. **Feature scaling.** The old run used raw last-token features. Their global
   scale differs strongly between backbones (encoder train norm ≈ 44–109,
   decoder ≈ 15–20), which confounds the KL rate. P2 uses train-only **global
   scalar scaling** (`standardize: "global"`), which removes the overall scale
   without z-scoring individual noisy dimensions. Per-dimension z-scoring was
   also tested and degraded the linear readout, so it was not used.
2. **Checkpoint selection.** VIB checkpoints are now selected on the validation
   objective `CE + beta * R`, not validation CE alone. This prevents high-beta
   runs from selecting an informative high-rate model.
3. **High-beta patience.** `patience_by_beta` gives β=1.0 and β=10.0 longer
   training (100 / 150 epochs), so the posterior can actually collapse.
4. **High-rate coverage.** Added β=0 and small β values (0.002, 0.005, 0.01,
   0.02). β=0 now recovers the deterministic control.
5. **Capacity-matched control.** Added a deterministic bottleneck control
   (`z = mu`, no KL, latent 32) in addition to the full 512-d linear head.
6. **Matched-rate analysis.** The new analysis matches on **validation rate**,
   interpolates each `(E2 seed, VIB seed)` curve separately, and bootstraps over
   E2 seeds, VIB seeds, and test examples. It no longer averages rates across
   VIB seeds before interpolation.
7. **Saved diagnostics.** Every run now stores per-epoch history, validation
   rate/objective, and per-example expected/mean CE predictions.

## Sanity checks

Deterministic controls (test CE bits):

| E2 seed | Encoder full linear | Encoder det-bottleneck d32 | Decoder full linear | Decoder det-bottleneck d32 |
|---:|---:|---:|---:|---:|
| 0 | 0.6657 | 0.6665 | 0.8009 | 0.7965 |
| 1 | 0.6306 | 0.6306 | 0.7464 | 0.7463 |
| 2 | 0.7295 | 0.7319 | 0.7502 | 0.7520 |
| 3 | 0.7373 | 0.7358 | 0.7442 | 0.7526 |
| 4 | 0.6819 | 0.6846 | 0.7589 | 0.7610 |

**High-rate recovery:** β=0 expected CE is within ~0.01 bits of the
deterministic control for every encoder/decoder seed.

**Zero-rate:** all 30 β=10 runs pass. Max test rate is 0.0153 bits (encoder) and
0.0495 bits (decoder); max test CE is 0.9608 / 0.9617 versus the prior 0.9568.

## Matched-rate result — expected stochastic CE (primary)

Matched on validation rate. Delta D = decoder − encoder; positive favors the
encoder. 95% CI: hierarchical bootstrap over E2 seeds, VIB seeds, and test
examples.

| Target R | E2 seeds | Cells | Encoder CE | Decoder CE | Delta D | Delta 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| 1.00 | 5 | 15 | 0.8347 | 0.9195 | +0.0848 | [+0.0512, +0.1202] |
| 1.50 | 5 | 15 | 0.8085 | 0.8876 | +0.0791 | [+0.0481, +0.1184] |
| 2.00 | 5 | 15 | 0.7927 | 0.8661 | +0.0734 | [+0.0371, +0.1110] |
| 2.50 | 5 | 15 | 0.7818 | 0.8518 | +0.0701 | [+0.0335, +0.1049] |
| 3.00 | 5 | 15 | 0.7730 | 0.8417 | +0.0688 | [+0.0314, +0.1048] |
| 3.50 | 5 | 15 | 0.7667 | 0.8334 | +0.0667 | [+0.0272, +0.1071] |
| 4.00 | 5 | 15 | 0.7632 | 0.8311 | +0.0679 | [+0.0248, +0.1056] |
| 4.50 | 5 | 15 | 0.7593 | 0.8297 | +0.0704 | [+0.0269, +0.1102] |
| 5.00 | 5 | 15 | 0.7550 | 0.8285 | +0.0735 | [+0.0326, +0.1154] |
| 6.00 | 5 | 15 | 0.7471 | 0.8271 | +0.0801 | [+0.0398, +0.1239] |
| 8.00 | 5 | 15 | 0.7372 | 0.8351 | +0.0979 | [+0.0472, +0.1449] |

The encoder has lower absolute CE at every matched rate, with CIs excluding zero.

## Compression penalty — expected CE

Penalty = matched-rate CE − that seed’s deterministic no-bottleneck CE.
Delta penalty = decoder penalty − encoder penalty; negative means the encoder
has the larger compression penalty.

| Target R | Encoder penalty | Decoder penalty | Delta penalty | Delta penalty 95% CI |
|---:|---:|---:|---:|---|
| 1.00 | +0.1457 | +0.1594 | +0.0137 | [−0.0412, +0.0541] |
| 1.50 | +0.1195 | +0.1275 | +0.0080 | [−0.0362, +0.0415] |
| 2.00 | +0.1037 | +0.1059 | +0.0022 | [−0.0331, +0.0297] |
| 2.50 | +0.0928 | +0.0917 | −0.0011 | [−0.0262, +0.0245] |
| 3.00 | +0.0840 | +0.0816 | −0.0024 | [−0.0252, +0.0190] |
| 3.50 | +0.0777 | +0.0733 | −0.0044 | [−0.0243, +0.0147] |
| 4.00 | +0.0742 | +0.0709 | −0.0033 | [−0.0234, +0.0165] |
| 4.50 | +0.0703 | +0.0695 | −0.0007 | [−0.0232, +0.0196] |
| 5.00 | +0.0661 | +0.0684 | +0.0024 | [−0.0218, +0.0246] |
| 6.00 | +0.0581 | +0.0670 | +0.0089 | [−0.0148, +0.0344] |
| 8.00 | +0.0482 | +0.0750 | +0.0268 | [−0.0053, +0.0588] |

Every delta-penalty CI includes zero. The encoder’s absolute matched-rate
advantage is therefore approximately its uncompressed E2 advantage (~0.071
bits), not evidence of a smaller compression penalty.

## Mean-only CE (diagnostic)

| Target R | Encoder CE | Decoder CE | Delta D | Delta 95% CI |
|---:|---:|---:|---:|---|
| 1.00 | 0.7204 | 0.7506 | +0.0302 | [−0.0147, +0.0742] |
| 2.00 | 0.7047 | 0.7483 | +0.0436 | [−0.0005, +0.0862] |
| 3.00 | 0.6971 | 0.7491 | +0.0520 | [+0.0078, +0.0977] |
| 4.00 | 0.6941 | 0.7539 | +0.0598 | [+0.0128, +0.1030] |
| 5.00 | 0.6926 | 0.7614 | +0.0688 | [+0.0224, +0.1188] |

Mean-only CE also favors the encoder, but the expected-vs-mean gap is large at
low rates, especially for the decoder. This is posterior-variance behavior, not
a calibration failure. Expected stochastic CE remains the primary distortion, as
specified in the plan.

## Interpretation

The corrected E3-A result is **not** the old P1 null result. With comparable
feature scale, objective-based selection, full zero/high-rate coverage, and
per-seed matched-rate analysis:

1. The **encoder has lower absolute distortion at matched rate** across the
   entire shared 1–8 bit region.
2. That advantage is **approximately the uncompressed baseline difference**.
   Compression-penalty differences are not statistically distinguishable from
   zero.
3. Therefore the 68M encoder’s full-data E2 advantage looks more like a better
   uncompressed decision representation / readout compatibility than a
   lower rate–distortion frontier.
4. This is closer to **Outcome B** (similar compression behavior, different
   absolute quality), not Outcome A (encoder-specific compression efficiency).

## Caveats

- BoolQ only; binary task.
- One readout (final last-token vector); no attention-L18 robustness run.
- Latent 32 only; no latent-128 control in P2.
- Expected stochastic CE is sensitive to posterior variance; mean-only CE is
  reported as a diagnostic.
- The VIB rate is a variational upper bound on `I(X;Z)`, not exact mutual
  information.
- Global scalar scaling removes the overall encoder/decoder scale mismatch but
  is not the same as per-dimension standardization; the per-dimension ablation
  was not run at full scale because it degraded the linear readout.
