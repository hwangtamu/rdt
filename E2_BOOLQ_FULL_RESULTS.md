# E2-BoolQ full data-scaling run — 5 seeds

Run: `runs/e2_boolq_full/{encoder,decoder}`
Config: `configs/e2_boolq_full.json`
Data: official BoolQ, split as in `data/boolq_e0`
Readouts: encoder-primary `lasth-L17`; decoder-primary `last-L22`; both run on both backbones.
Seeds: 0–4. Fractions: 5 %, 10 %, 25 %, 50 %, 100 %.
Aggregate table: `runs/e2_boolq_full/summary.csv`.

## Mean test CE (bits) and accuracy, 5 seeds

| N | Encoder `lasth-L17` | Encoder `last-L22` | Decoder `lasth-L17` | Decoder `last-L22` |
|---:|---|---|---|---|
| 5 % | 1.954 ± 1.863 (med 1.154) / 0.580 | **0.957 ± 0.003** / 0.622 | 1.024 ± 0.085 / 0.580 | **0.974 ± 0.013** / 0.605 |
| 10 % | 1.077 ± 0.099 / 0.599 | **0.958 ± 0.003** / 0.622 | 0.995 ± 0.017 / 0.622 | **0.946 ± 0.035** / 0.632 |
| 25 % | 1.034 ± 0.062 / 0.557 | **0.928 ± 0.049** / 0.645 | 0.937 ± 0.050 / 0.638 | **0.936 ± 0.018** / 0.636 |
| 50 % | 0.956 ± 0.135 / 0.640 | 0.931 ± 0.029 / 0.634 | 0.838 ± 0.044 / 0.709 | **0.797 ± 0.054** / 0.723 |
| 100 % | **0.740 ± 0.134** / 0.768 | 0.834 ± 0.145 / 0.693 | 0.776 ± 0.062 / 0.747 | **0.728 ± 0.021** / 0.767 |

Format: mean CE ± seed SD / mean accuracy. Bold marks the lower-CE readout at that N for that backbone.

## One unstable run

Encoder `lasth-L17` at 5 % contains one diverged seed:

```
seed 4: val CE 5.244, test CE 5.282, train loss spikes 235 → 86 → 32
```

Even excluding it, the other four seeds are CE 1.04–1.25, so the low-data
encoder-L17 result is genuinely worse than `last-L22`, not just an outlier
artifact. It should be rerun with a more stable optimization recipe.

## Encoder vs decoder at 100 % (paired test bootstrap)

Delta = CE_decoder − CE_encoder; negative favors decoder.

| Readout | Encoder CE | Decoder CE | ΔD [95 % CI] | p |
|---|---:|---:|---|---:|
| `last-L22` | 0.834 | 0.728 | **−0.1055** [−0.1387, −0.0712] | <0.001 |
| `lasth-L17` | 0.740 | 0.776 | **+0.0359** [+0.0096, +0.0627] | 0.006 |

So the encoder/decoder winner at full data **depends on the readout**:

- With `last-L22`, the **decoder wins**.
- With `lasth-L17`, the **encoder wins**, but only at 100 %; the decoder wins at 5–50 %.


## Primary matched-readout scaling comparison: `last-L22`

`last-L22` is the robust matched readout for the encoder-vs-decoder `D(N)` comparison:
it is stable at low N, comparable across both architectures, and has the cleanest
decoder scaling curve. `lasth-L17` is retained as the encoder's full-data readout,
not as the primary matched readout.

| N | Encoder CE | Decoder CE | ΔD = dec − enc [95 % CI] | Enc Acc | Dec Acc |
|---:|---:|---:|---|---:|---:|
| 5 % | 0.957 | 0.974 | +0.0170 [+0.0022, +0.0308] | 0.622 | 0.605 |
| 10 % | 0.958 | 0.946 | −0.0116 [−0.0310, +0.0073] | 0.622 | 0.632 |
| 25 % | 0.927 | 0.936 | +0.0089 [−0.0106, +0.0277] | 0.645 | 0.636 |
| 50 % | 0.931 | 0.797 | **−0.1341** [−0.1585, −0.1088] | 0.634 | 0.723 |
| 100 % | 0.834 | 0.728 | **−0.1055** [−0.1387, −0.0712] | 0.693 | 0.767 |

Reading: encoder and decoder are indistinguishable at 10–25 %, then the
**decoder pulls clearly ahead at 50–100 %**. At full data, `lasth-L17` gives the
encoder its best cell (0.740 CE, 0.768 acc), so the final report should show both
the matched-readout result and the encoder-full-data readout.


## Stabilization of the 5 % encoder `lasth-L17` cell

The original 5 % cell had one diverged seed (test CE 5.28). Reruns with fp32,
lower LR and longer warmup still diverged for seed 4:

| Setting | Seed-4 test CE | Other seeds | Verdict |
|---|---:|---|---|
| fp32, lr 1e-5, warmup 0.10 | 8.08 | 1.11–1.32 | still diverges |
| fp32, lr 2e-5, warmup 0.10 | 9.58 | 1.03–1.18 | still diverges |
| fp32, lr 2e-5, warmup 0.10, standardization, grad-clip 0.1 | **1.03** | 1.02–1.07 | **stable** |

Stable 5-seed result for encoder `lasth-L17` at 5 %:

- **CE = 1.035 ± 0.019** (range 1.016–1.067)
- **accuracy = 0.561**

So the instability was fixed by standardizing the pre-norm L17 feature and
tightening gradient clipping. Even stabilized, it remains worse than encoder
`last-L22` at 5 % (0.957 CE, 0.622 acc). This supports using `last-L22` as the
matched readout at low N and `lasth-L17` as the encoder full-data readout.

## Reading

1. **The decoder has the cleaner scaling curve.** `last-L22` improves
   monotonically: 0.974 → 0.946 → 0.936 → 0.797 → 0.728.
2. **The encoder is not monotonic.** `last-L22` is flat ~0.93–0.96 until 100 %
   (0.834); `lasth-L17` is worse than `last-L22` until 100 %, then becomes the
   best cell at 0.740.
3. **Low-data training is unstable.** The 5 % encoder-L17 run diverges for one
   seed; several low-N runs have very wide seed variance.
4. **Readout × backbone interaction remains the main story.** There is no single
   readout under which one backbone uniformly wins across N.
5. **Do not fit `D(N) = D∞ + aN^-α` yet.** The encoder curve is too non-monotonic
   and the 5 % cell needs a stable rerun.

## Recommended next steps

1. Rerun the unstable 5 % encoder-L17 cell with fp32 / lower LR / longer warmup.
2. Use `last-L22` as the robust matched readout for the encoder-vs-decoder
   scaling comparison; keep `lasth-L17` as the encoder’s full-data readout.
3. Only after stabilizing low-N, fit the scaling curve.
4. The remote L22 pre/post-norm/standardization/MLP ablation should explain why
   the encoder’s pre-norm L17 readout only wins at full data.
