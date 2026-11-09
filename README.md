# E1 — Frozen representation probing (When2Call)

Runnable scaffold for **Phase 1 / Experiment E1** of `RDT.md`: freeze the Ettin
Encoder 150M and Ettin Decoder 150M backbones, extract the decision-position
hidden state, and fit matched linear and MLP probes. It also implements the
paired bootstrap comparison of decision cross-entropy in bits.

```
config → grouped/stratified splits → cached hidden states → linear & MLP probes
      → metrics + bootstrap CIs → paired encoder-vs-decoder ΔD
```

## Important scope note

E1 measures **accessibility** of decision information to a lightweight head. A
frozen linear-probe advantage does **not** establish higher `I(Y;Z)` for the
encoder; an MLP probe may recover information the linear probe misses. Report E1
as a baseline, not as a rate–distortion result. Rate–distortion belongs to E3/E4.

## Environment and weights (this machine)

Use the existing CUDA venv rather than creating one:

```bash
VENV=/mnt/local-disk/decision-models/.venv-ouro-hf
$VENV/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.15.0+cu132 True   (transformers 4.54.1)
```

The checkpoints are **local directories**, not huggingface.co cache entries:

```
/mnt/local-disk/decision-models/tiny-decision-model/out/ettin-encoder-150m
/mnt/local-disk/decision-models/tiny-decision-model/out/ettin-decoder-150m
```

> Note: `HF_HOME=/local_disk/` contains `models--jhu-clsp--ettin-*` directories
> with only `refs/main` — no `snapshots/` or `blobs/`. Loading by hub id will
> therefore try the network and fail. `configs/e1.json` sets the optional
> `path` field per backbone; `features.model_ref()` prefers it over `name`.

Verify both load and expose matched layers (22 blocks + embeddings = 23 hidden
states, hidden size 768):

```bash
PYTHONPATH=src $VENV/bin/python -c "
from transformers import AutoModel
for p in ['/mnt/local-disk/decision-models/tiny-decision-model/out/ettin-encoder-150m',
          '/mnt/local-disk/decision-models/tiny-decision-model/out/ettin-decoder-150m']:
    m = AutoModel.from_pretrained(p); print(type(m).__name__, sum(x.numel() for x in m.parameters())/1e6)
"
```

`configs/e1.json` uses `cuda:1` (RTX 6000 Ada, 49 GB) and `max_length: 2048`.

Data lives at `/mnt/local-disk/bench/when2call/`:
`train/when2call_train_sft.jsonl` (15k), `train/when2call_train_pref.jsonl`
(9k), `test/when2call_test_mcq.jsonl` (3,652 labelled MCQ). The MCQ file has a
`correct_answer` field usable for auditing/held-out evaluation, but it must stay
held out of probe training.

## Install

```bash
$VENV/bin/pip install -e .          # numpy/torch/transformers already present
$VENV/bin/pip install pytest        # tests only
```

Requires `numpy`, `torch`, `transformers`. No sklearn/YAML dependency (configs
are JSON deliberately).

## Data

JSONL, one record per line:

```json
{"id": "ex-001",
 "question": "Book a flight from Dallas to NYC on Friday.",
 "tools": [{"name": "book_flight",
            "description": "Book a flight.",
            "parameters": {"origin": "string (required)",
                           "destination": "string (required)",
                           "date": "string (required)"}}],
 "decision": "tool_call",
 "group_id": "booking-family-17"}
```

* Backbone configs accept an optional `path` (local checkpoint dir), which takes
  precedence over the hub `name` for loading and caching.
* `decision` accepts canonical names (`direct`, `tool_call`,
  `request_for_info`, `cannot_answer`) or letters `A`–`D`. Aliases live in
  `src/w2c/labels.py`; unrecognised/ambiguous values are **dropped and counted**,
  never guessed. Audit the drop counts before training.
* `data.classes` selects and orders the label space, e.g.
  `["tool_call", "request_for_info", "cannot_answer"]`; rows outside it are
  dropped and counted. Probe output width follows this list.
* `group_id` is the leakage barrier: all members of a group land in the same
  split. If absent, a hash of the normalized question is used, which catches
  exact duplicates but not paraphrases — supply real family IDs where possible.
* Field names are configurable under `data.*`.
* Never put `correct_answer`, `target_tool`, or gold assistant text in any
  field that reaches the template.

**Splits.** Either point `data.train/val/test` at three files (recommended; keep
the official test set intact), or set `data.raw` and let the scaffold split it
with group-level, class-stratified sampling. Splitting fails loudly if a group
appears in two splits.

## Phase 0 — build When2Call labels

```bash
VENV=/mnt/local-disk/decision-models/.venv-ouro-hf
PYTHONPATH=src $VENV/bin/python -m w2c.build_w2c_e1 \
    --out-dir data/when2call_pref --no-sft --drop-class direct
```

This reads `/mnt/local-disk/bench/when2call`, labels the preference-flavour
`chosen_response` with audited rules, drops normalized-question overlaps with
the MCQ test, and writes `raw.jsonl` / `test.jsonl` / `stats.json` /
`audit_samples.jsonl` (25 seeded random rows per class; override with
`--samples-per-class` / `--audit-seed`).

To run the reproducible label audit (mechanical checks plus the manual verdicts
recorded in `audit_review.jsonl`):

```bash
PYTHONPATH=src $VENV/bin/python -m w2c.audit_labels \
    --when2call-dir /mnt/local-disk/bench/when2call \
    --out-dir data/when2call_pref
```

It writes `audit_report.json` and `AUDIT.md`. Current result: 75/75 sampled rows
correct, 2,968/2,968 tool calls structurally valid, one ambiguity from a
refusal+clarification row, and 35 same-tool conflicting question pairs (70 rows)
in training — an irreducible noise floor. The official test set has 27 such pairs.

Data reality check: the release has **no usable `direct` class** (0 in SFT,
46/9,000 in preference, 0/3,652 in test; only 6 genuine direct answers in 24,000
audited responses). The real E1 therefore uses three classes. The SFT split has
zero tool calls and is only useful as a prior-shifted sensitivity check.

## Run

Smoke test with a deterministic dummy backbone (no downloads, CPU, ~seconds):

```bash
PYTHONPATH=src python -m w2c.make_dummy_data --out data/dummy/raw.jsonl
PYTHONPATH=src python -m w2c.train_e1 --config configs/e1_dummy.json
PYTHONPATH=src python -m w2c.compare_e1 --run-dir runs/dummy/e1
```

Real E1 (preference-only, class-balanced, 3-way; results in `E1_RESULTS.md`):

```bash
PYTHONPATH=src $VENV/bin/python -m w2c.train_e1 --config configs/e1_w2c_pref.json
PYTHONPATH=src $VENV/bin/python -m w2c.compare_e1 --run-dir runs/e1_w2c_pref
PYTHONPATH=src $VENV/bin/python -m w2c.report_e1 --run-dir runs/e1_w2c_pref
```

`configs/e1_w2c.json` is the same run with SFT merged in (prior-shifted
sensitivity check). `configs/e1.json` remains the generic template and defaults
to all four canonical classes; set `data.classes` to the classes you can
actually support.

Useful overrides: `--models encoder,decoder`, `--probes linear,mlp`,
`--layers 0,6,11,12,17,22`, `--bootstrap 500` for a quick pass.

## E1.5 — readout comparison on the frozen backbone

E1 fixes the readout to the last token of `"Decision:"`. E1.5 keeps both
backbones frozen and asks the orthogonal question: how much does the readout
matter? It sweeps four readouts on the same grouped splits and test set:

| Readout | Definition | Extra params |
|---|---|---|
| `last` | last non-pad token (E1 baseline) | 0 |
| `mean` | masked mean over all prompt tokens | 0 |
| `attn` | learned single-query attention over tokens, trained with the probe | `hidden` (768) |
| `concat` | decision-position states concatenated over `e15.concat.layer_sets` | 0 |

`last`, `mean` and `attn` are swept over every `features.layers` entry; `concat`
over the configured layer sets. Each (backbone, readout, probe) configuration is
selected by **mean validation CE over seeds**; the official test set is never
used for selection.

```bash
PYTHONPATH=src $VENV/bin/python -m w2c.train_e15 --config configs/e15_w2c_pref.json
PYTHONPATH=src $VENV/bin/python -m w2c.compare_e15 --run-dir runs/e15_w2c_pref
```

Run IDs are `{backbone}__{readout}__[L{layer}|concat-...]__{probe}__s{seed}`.
`compare_e15` writes the selected-configuration table, a paired ΔD of each
readout against `last`, an encoder-vs-decoder table at matched
(readout, probe, config), and `all_configs.csv` with the full layer curves.

Attention pooling needs token-level hidden states. They are cached ragged (no
padding) as float16 in `features.token_cache_dir`, ≈6 GB per layer per backbone
on this dataset. Dummy smoke config: `configs/e15_dummy.json`.

Caveat: `attn` and `concat` are not parameter-matched to `last`. Attention adds a
query vector trained jointly with the probe, so E1.5 compares *readouts*, not
pure representations.

## Outputs

```
runs/e1/
  config.json          exact config used
  manifest.json        split sizes, label counts, groups, dropped labels
  runs.jsonl           one record per (backbone, layer, probe, seed), deduped on read
  runs/<run_id>.json   same records, one file each
  preds/<run_id>.npz   logits, labels, ids, per-example CE on test
  comparison.csv       ΔD per (layer, probe) with CI and p
  comparison.md       human-readable table
  REPORT.md            generated summary (metrics, CIs, confusion, caveats)
```

```
runs/e15/
  config.json / manifest.json
  runs.jsonl           one record per (backbone, readout, config, probe, seed)
  runs/<run_id>.json   same records, one file each
  preds/<run_id>.npz   logits, labels, ids, per-example CE on test
  all_configs.csv      every readout configuration, averaged over seeds
  comparison.csv       selected config per (backbone, readout, probe), ΔD vs last
  comparison.md        selected table + encoder-vs-decoder + guardrails
  encoder_vs_decoder.csv
```

`ΔD = D_decoder − D_encoder`, restricted to the same test examples and the same
layer index/probe type. Negative ΔD favors the encoder. CIs are percentile
bootstraps over test examples on seed-averaged per-example CE; `p_two_sided` is
a bootstrap tail probability, not a parametric test.

## What is cached, and why

Hidden states are cached per `(backbone, split, template, max_length, pooling,
example IDs)` in `features.cache_dir`, so changing probe hyperparameters or seeds
does not re-run the backbone. Cache size for 15k examples, all 23 layers, 768-d,
float16 ≈ 0.5 GB per split per model; restrict `features.layers` if disk-bound.

The decision position is the last non-pad token of the template (`"Decision:"`),
computed as `argmax(position * attention_mask)` so it works with either padding
side. Encoder and decoder therefore read the same token index in the same
prompt. Truncation is audited per split: token lengths are measured without
truncation and `full_length >= max_length` is recorded; `train_e1` prints the
test truncation rate. Any non-trivial rate invalidates the input-visibility
assumption and must be fixed (longer `max_length` or shorter tool rendering).

## Matched conditions enforced by the scaffold

* identical prompt text, split files, labels, and test example order;
* identical probe architecture/width for both backbones (MLP width is fixed by
  `probes.mlp_hidden`);
* identical optimizer, epochs, early-stopping rule, and feature standardization;
* per-seed seeding of torch/numpy;
* MLP-vs-linear gap is directly comparable because both are reported per layer.

Not enforced automatically — you must check:

* equal hyperparameter-search budget per backbone (the scaffold does one fixed
  recipe; if you tune, log the search budget);
* the decoder and encoder use different tokenizers, so sequence length differs
  even with identical text; the truncation audit is the guard;
* training steps vs FLOPs vs wall-clock are different fairness criteria.

## Tests

```bash
pytest -q
```

The smoke test runs the full pipeline on synthetic data with the dummy backbone
and checks split leakage, run counts, and comparison output.

`pytest` is not installed in `.venv-ouro-hf`; either install it there or run the
tests with system Python (`PYTHONPATH=src python -m pytest -q`).
