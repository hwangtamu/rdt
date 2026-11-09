# E1 results — frozen representation probing

Primary run: `runs/e1_w2c_pref/` (`configs/e1_w2c_pref.json`)
Sensitivity run (SFT+pref merged, prior-shifted): `runs/e1_w2c/`

## Setup

| Item | Value |
|---|---|
| Backbones | Ettin Encoder 150M (ModernBertModel), Ettin Decoder 150M (ModernBertDecoderModel), frozen |
| Train pool | When2Call `train_pref` **chosen** responses |
| Test | Official `when2call_test_mcq` (3,652), untouched |
| Classes | `tool_call`, `request_for_info`, `cannot_answer` (3-way) |
| Splits | train 7,532 / val 1,329 / test 3,652; grouped by normalized question |
| Position | last non-pad token of `"Decision:"`, both models |
| Layers | 0, 6, 11, 12, 17, 22 (of 22 blocks + embeddings) |
| Probes | linear; MLP 768→256→k, GELU, dropout 0.1 |
| Seeds | 3; validation-only early stopping; paired bootstrap (2,000) over test examples |
| Compute | 72 runs, 5m42s on one RTX 6000 Ada |

## Why 3-way, not 4-way

The release has no usable `direct` class:

- SFT split: `tool_call` = **0** of 15,000 (cannot 7,561 / request 7,337 / direct 102).
- Preference split chosen responses: tool_call 3,000 / cannot 2,991 / request 2,963 / **direct 46**.
- MCQ test: tool_call 1,295 / cannot 1,295 / request 1,062 / **direct 0**.

Audited response rules found only **6** genuine direct answers in 24,000 responses. E1
therefore trains and evaluates the three supported classes. This is a data finding, not
a modelling choice: a four-way claim cannot be supported by this release.

## Labels (Phase 0)

Preference chosen responses were labelled by ordered, audited rules:
`<TOOLCALL>`/JSON `name` → tool_call; inability/refusal phrase → cannot_answer;
clarification cue or trailing `?` → request_for_info; otherwise direct.
Counts before class filtering: 3,000 / 10,528 / 10,460 / 6 over 24,000 rows.
25 seeded random samples per class (`--audit-seed 0`) are written to
`data/when2call_pref/audit_samples.jsonl`.
2 rows were dropped for normalized-question overlap with the test set, 135 duplicate
(question, label) rows, and 2 excluded `direct` rows. The one preference
refusal+clarification row is logged and adjudicated ambiguous: it both declines and
asks for the missing parameter.

## Headline results (mean over 3 seeds)

Best linear probe per backbone (selected by CE):

| Backbone | Layer | CE (bits) | Accuracy | Macro-F1 | ECE | Per-class F1 (tool / request / cannot) |
|---|---:|---:|---:|---:|---:|---|
| Decoder | 6 | **1.5476** | **0.4936** | **0.4956** | 0.163 | 0.462 / 0.497 / 0.528 |
| Encoder | 12 | 1.5525 | 0.4575 | 0.4584 | **0.125** | 0.437 / 0.440 / 0.498 |

Paired ΔD = D_decoder − D_encoder (negative favors encoder):

| Layer | Linear ΔD [95% CI] | MLP ΔD [95% CI] |
|---:|---|---|
| 0 | +0.000 [−0.000, +0.001] | +0.000 [−0.000, +0.000] |
| 6 | **−0.090** [−0.133, −0.048] | +0.062 [−0.010, +0.129] |
| 11 | **+0.070** [+0.013, +0.129] | **+0.309** [+0.229, +0.389] |
| 12 | **+0.181** [+0.121, +0.246] | **+0.425** [+0.338, +0.514] |
| 17 | **−0.200** [−0.262, −0.142] | +0.083 [+0.000, +0.164] |
| 22 | **−0.133** [−0.197, −0.073] | +0.085 [+0.003, +0.163] |

## What this says

1. **No uniform encoder advantage.** The sign flips with depth: decoder is better at
   layers 6, 17, 22 (linear), encoder is better at 11–12. H1 is not supported as a
   global statement by E1.
2. **A mid-depth encoder advantage exists but does not survive to the output.** Encoder
   layers 11–12 are the most linearly decodable; the decoder's late layers are better
   overall. This is exactly the kind of layer-specific effect that E2–E4 must test,
   not average away.
3. **CE and accuracy disagree.** The encoder at layer 11 has lower CE than the decoder
   at 11 but lower accuracy: the decoder is more accurate but more overconfident, so
   its CE is worse. Distortion and thresholded accuracy rank models differently.
4. **Absolute transfer is weak.** Best CE ≈ 1.55 bits against ln(3) = 1.585 bits, and
   accuracy ≈ 0.50. Frozen pretrained features carry little directly transferable
   When2Call decision information across the pref→MCQ distribution shift.
5. **Layer 0 is uninformative by construction.** It is the embedding of the shared
   final `:` token, identical for every input; its CE is the class prior.

## Threats to validity (specific to this run)

- **Distribution shift:** training prompts come from preference/SFT API-benchmark
  questions; the test is the MCQ benchmark. Absolute scores conflate this shift with
  representation quality.
- **Prior shift avoided, then reintroduced:** pref-only training is balanced
  (33/33/33) like the test (35/29/35). The merged SFT run is 14/42/44 and its confusion
  matrices are dominated by request_for_info over-prediction; it is retained only as a
  sensitivity check.
- **Rule-based labels — audited:** the seeded stratified sample (25/class) was
  75/75 correct; all 2,968 tool-call rows parse and name only offered tools; the
  replay from source matches `raw.jsonl` exactly. Two residual ambiguity sources:
  one refusal+clarification row retained as `cannot_answer` (both labels defensible),
  and 70 rows (35 question pairs with identical tools but opposing chosen decisions),
  ≈0.8% of training. The official test set itself has 27 same-tool conflicting gold
  pairs and 258 `cannot_answer` rows with no tools. Full report:
  `data/when2call_pref/AUDIT.md`; per-row verdicts: `audit_review.jsonl`.
- **Frozen probing ≠ information content:** a linear/MLP gap does not bound `I(Y;Z)`;
  rate–distortion claims belong to E3/E4.
- **No direct class:** four-way accuracy/macro-F1 cannot be reported honestly.

## Reproduce

```bash
VENV=/mnt/local-disk/decision-models/.venv-ouro-hf
PYTHONPATH=src $VENV/bin/python -m w2c.build_w2c_e1 \
    --out-dir data/when2call_pref --no-sft --drop-class direct --audit-seed 0
PYTHONPATH=src $VENV/bin/python -m w2c.audit_labels \
    --when2call-dir /mnt/local-disk/bench/when2call --out-dir data/when2call_pref
PYTHONPATH=src $VENV/bin/python -m w2c.train_e1 --config configs/e1_w2c_pref.json
PYTHONPATH=src $VENV/bin/python -m w2c.compare_e1 --run-dir runs/e1_w2c_pref
PYTHONPATH=src $VENV/bin/python -m w2c.report_e1 --run-dir runs/e1_w2c_pref
```

Outputs: `runs/e1_w2c_pref/{manifest.json, runs.jsonl, preds/, comparison.csv,
comparison.md, REPORT.md}` and `data/when2call_pref/{audit_samples.jsonl,
audit_review.jsonl, audit_report.json, AUDIT.md}`.
