# E1 on BoolQ — full Ettin family (24 models)

Frozen-backbone probing on BoolQ, run through the existing E1 pipeline
(`w2c.train_e1`). Last-token pooling at the end of the prompt, linear + MLP
probes (768→256, GELU, dropout 0.1), 3 seeds, standardized features. The
reported layer is the best **validation** CE (mean over seeds); test = 1,634
held-out BoolQ examples (chance CE = 1.0 bit, majority-class acc = 0.622).

## Setup / files

- Configs: `configs/e1_boolq_family{,_g1,_g2}.json`
- Runs: `runs/e1_boolq_family_g1/` (encoders: scratch + from-decoder) and
  `runs/e1_boolq_family_g2/` (decoders: scratch + from-encoder)
- Feature cache: `runs/features_boolq_family/` (~9 GB, shared across shards)
- Aggregation: `src/w2c/compare_e1_family.py`
- Outputs: `runs/e1_boolq_family/{family_summary.csv, family_layers.csv,
  family_deltas.csv, family_report.md, family_ce_vs_size.svg,
  family_depth_{1b,150m}.svg}`

Naming: `enc`/`dec` = trained from scratch; `encfd` = encoder initialized from
decoder; `decfe` = decoder initialized from encoder.

## Headline: best validation-layer test results (MLP probe; CE / accuracy)

| Size | encoder (scratch) | encoder←decoder | decoder (scratch) | decoder←encoder |
|------|------------------|-----------------|-------------------|-----------------|
| 17m  | 0.927 / .631 | 0.937 / .629 | 0.937 / .643 | **0.906 / .646** |
| 32m  | **0.905 / .654** | 0.916 / .646 | 0.919 / .656 | 0.967 / .636 |
| 68m  | **0.906 / .654** | 0.907 / .657 | 0.914 / .666 | 0.913 / .653 |
| 150m | 0.919 / .658 | 0.880 / .670 | 0.902 / .669 | **0.877 / .685** |
| 400m | 0.865 / .681 | **0.829 / .701** | **0.801 / .724** | 0.806 / .732 |
| 1b   | 0.788 / .732 | 0.749 / .747 | 0.787 / .740 | **0.676 / .790** |

## Reading

1. **Capacity dominates over architecture.** The strongest driver of frozen
   transfer is model size: best CE falls monotonically-ish from ~0.93 (17m) to
   ~0.68 (1b), and accuracy rises from ~0.63 to ~0.79. The 1b models all
   clearly beat every ≤150m model regardless of architecture.

2. **The "from-decoder" initialization is the best 1b encoder, and the
   "from-encoder" decoder is the best overall model.** At 1b, MLP CE:
   `decfe-1b` 0.676 (acc .790) beats every other model in the family; the
   encoder family winner is `encfd-1b` (0.749, CE from-decoder > from-scratch
   `enc-1b` 0.788). So cross-initialization, not the from-scratch training
   target, yields the most linearly/MLP-decodable BoolQ decision information at
   the large scale.

3. **No reliable scratch encoder-vs-decoder advantage (E1's H1 again).**
   Matched-depth deltas between `enc` and `dec` are mostly non-significant; the
   only significant scratch pair at 1b favours the **encoder** (+0.054 [0.014,
   0.107] linear), consistent with the earlier E2 finding that the encoder wins
   at full data. Mid sizes flip around and stay inside the CI.

4. **The decoder←encoder crossover flips with scale.** At 400m the scratch
   decoder (0.801) beats the from-encoder decoder (0.806) — and matched-depth
   deltas show the scratch decoder significantly better (+0.086/+0.095). At 1b
   the opposite is strong: `decfe-1b` beats `dec-1b` by −0.107/−0.077
   (matched-depth, significant). Pretraining-scale interacts with the
   initialization target.

5. **1b `enc` vs `decfe` is the single largest within-family gap.** Matched at
   depth 27, `decfe-1b` beats `enc-1b` by −0.103 [−0.146, −0.057] (MLP) —
   larger than any encoder-vs-decoder scratch difference anywhere in the
   family.

## Caveats

- Frozen linear/MLP probing measures **accessibility**, not `I(Y;Z)`; the MLP
  gap and the layer-selection-on-validation protocol bound claims accordingly
  (rate–distortion is E3/E4).
- Validation-selected layer; an oracle (best test layer) column in
  `family_summary.csv` shows selection barely moves the ranking.
- The from-decoder/from-encoder variants share the same tokenizer and BoolQ
  split, so deltas are paired and matched in depth at the reference model's
  best-validation layer.
