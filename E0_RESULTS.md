# E0 — BoolQ binary frozen-probe sanity check

Question: do the two frozen 150M backbones carry a *linearly accessible*
binary decision signal on a clean, well-known task, before blaming LevBench
difficulty on capacity?

**Answer: yes, but weakly.** Both backbones beat the label prior on BoolQ; the
best decoder probe resolves only ~5.7 % of the test label entropy.

This run *is* the E1 protocol on BoolQ: it uses the same `train_e1` layer sweep,
`linear`/`MLP` probes, grouped splits, and `compare_e1` encoder-vs-decoder
comparison. `E0` is only the experiment label used to distinguish this binary
sanity check from the LevBench E1-v2 run. The generated E1 report is
`runs/e0_boolq/REPORT.md`; the encoder-vs-decoder table is
`runs/e0_boolq/comparison.md`.

## Data

Official `google/boolq` parquet files were downloaded to
`data/boolq_raw/data/`. Build:

```bash
PYTHONPATH=src python -m w2c.build_boolq_e0 \
    --train-parquet data/boolq_raw/data/train-00000-of-00001.parquet \
    --val-parquet data/boolq_raw/data/validation-00000-of-00001.parquet \
    --out-dir data/boolq_e0
```

- train: official BoolQ train, n = 9,427
- val/test: official validation split once, stratified alternating halves,
  n = 1,636 / 1,634 (BoolQ has no public labelled test set)
- prompt: `Passage:\n...\n\nQuestion: ...\n\nA. No\nB. Yes\n\nAnswer:`
- `max_length = 512`; test truncation rate = 0.6 %

Test prior: `P(No) = 0.378`, `P(Yes) = 0.622`, majority accuracy = **0.6218**,
label entropy `H(Y) = 0.9568` bits.

## Runs

Config: `configs/e0_boolq.json`
Outputs: `runs/e0_boolq/{manifest.json,runs.jsonl,preds/,comparison.md,...}`
Runs: 2 backbones × 23 layers × {linear, MLP} × 3 seeds = **276**.

Probe selection uses mean **validation** CE over seeds; test is held out.
Δ is paired bootstrap on per-example CE against the train-prior predictor
(one-sided p for Δ > 0).

## Selected configuration per backbone / probe

| Backbone | Probe | Layer | Val CE | Test CE | Acc | Macro-F1 | Balanced acc | ECE | Δ vs prior [95 % CI] | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| decoder | MLP | 14 | 0.8864 | **0.9023** | **0.6691** | **0.6144** | 0.6140 | 0.0383 | **+0.0545** [+0.0145, +0.0925] | 0.005 |
| decoder | linear | 18 | 0.8945 | 0.9174 | 0.6720 | 0.6259 | 0.6235 | 0.0234 | +0.0393 [−0.0044, +0.0785] | 0.038 |
| encoder | MLP | 17 | 0.9162 | 0.9169 | 0.6501 | 0.5830 | 0.5877 | 0.0395 | +0.0399 [+0.0184, +0.0637] | <0.001 |
| encoder | linear | 16 | 0.9335 | 0.9302 | 0.6457 | 0.5931 | 0.5934 | 0.0312 | +0.0266 [+0.0004, +0.0507] | 0.022 |

Layer 0 is at the prior (test CE ≈ 0.957 for both). Performance rises with
depth; the decoder is generally better in late layers (e.g. L16 MLP
`ΔD = CE_dec − CE_enc = −0.0329`, L18 MLP `−0.0322`).

## Reading

1. **Not a pure capacity failure.** With the same frozen 150M Ettin
   backbones, a linear/MLP probe on BoolQ reaches ~0.65–0.67 accuracy against
   a 0.622 majority baseline and a 0.9568-bit prior. Both models carry usable
   binary decision information.
2. **The signal is small.** Best probe extracts 0.055 of 0.957 bits
   (≈ 5.7 % of `H(Y)`); balanced accuracy is only ~0.62. Large headroom remains.
3. **Decoder ≥ encoder at depth, on this task.** Early-layer differences are
   small; late decoder layers are consistently a little better.
4. **This does not rescue E1.5-v2 by itself.** BoolQ is binary and the readout
   is still a single pooled prompt vector. LevBench is 6-way, multi-source, and
   needs option comparison. E0 says the backbones are not degenerate; it does
   not say the E1.5 readout is adequate for LevBench.

## Caveats

- The “test” split is half of the official BoolQ validation split; no public
  BoolQ test labels exist.
- Frozen probing measures accessibility with one pooled vector, not `I(Y;Z)`
  and not fine-tuned capability.
- BoolQ has known lexical/annotation shortcuts; beating the prior may partly
  reflect surface cues. That is still the relevant E0 question — whether the
  representation exposes a usable signal at all.
- Results are on Ettin-150M encoder/decoder only; not a general 150M claim.

## Reproduce

```bash
PYTHONPATH=src $VENV/bin/python -m w2c.build_boolq_e0
PYTHONPATH=src $VENV/bin/python -m w2c.train_e1 --config configs/e0_boolq.json
PYTHONPATH=src $VENV/bin/python -m w2c.compare_e1 --run-dir runs/e0_boolq
```
