# E3-B GH200 corrected-pilot results

**Status:** preliminary engineering pilot, not a final E3-B rate–distortion
result. The rejected `E3B_BUDGET_*` curves remain rejected. This report records
the first corrected-code run on the remote GH200 and the protocol gates that
still fail.

## Environment

- Host: `phyvran10prpl02` (`105.165.111.166`), NVIDIA GH200 480GB.
- The GPU is shared with a vLLM process; E3-B used approximately 36 GB free.
- Python: `~/llm/bin/python`, torch 2.11.0+cu130, transformers 5.9.0.
- Workspace: `~/rdt-e3b`.
- Data: `data/boolq_e0` (9,427 train / 1,636 validation / 1,634 test).
- Backbones: paired E2 `last-L19`, `N100`, 68M encoder/decoder checkpoints.
- Pilot used E2 seed 0 only, VIB seed 0, latent 32, `last` readout.

## Implementation revision

Code revision `10922fd` before this report adds:

- portable backbone-path resolution for remote checkpoints;
- train-only global scalar token standardization;
- example-weighted CE and total-rate evaluation;
- checkpoint selection by validation `CE + beta * R_total`;
- restoration of every trained component, including `last_gate`;
- fixed-support attention with prior samples at padding positions;
- forced-prior evaluation with zero measured rate;
- GH200 smoke/pilot configs and E3-B protocol sanity tests.

## Pilot configuration

| Setting | Value |
|---|---:|
| E2 checkpoint seed | 0 (paired encoder/decoder) |
| VIB seeds | 0 |
| Latent dimension | 32 |
| Readout | `last` |
| Max length | 512 |
| Batch size | 2048 |
| MC train / eval | 4 / 64 |
| Epoch cap / patience | 20 / 5 |
| Standardization | train-only global scalar |
| Beta grid | 3e-4, 1e-3, 3e-3, 1e-2, 3e-2 |

Mean token length was 164.7 train / 162.6 validation / 163.8 test. Token-length
truncation is not yet recorded as an artifact, so this pilot does not certify a
zero truncation rate.

## Total-rate pilot

Expected stochastic test CE in bits; total rate in bits/example.

| Backbone | Beta | Test rate | Test CE | Test accuracy | Best epoch |
|---|---:|---:|---:|---:|---:|
| encoder | 3e-4 | 789.97 | 0.7859 | 0.7876 | 19 |
| encoder | 1e-3 | 589.12 | 0.7413 | 0.7840 | 19 |
| encoder | 3e-3 | 492.38 | 0.6968 | 0.7882 | 19 |
| encoder | 1e-2 | 453.55 | 0.6945 | 0.7882 | 19 |
| encoder | 3e-2 | 439.57 | 0.7693 | 0.7840 | 19 |
| decoder | 3e-4 | 403.77 | 0.8327 | 0.7118 | 19 |
| decoder | 1e-3 | 259.56 | 0.8362 | 0.7142 | 19 |
| decoder | 3e-3 | 193.09 | 0.8469 | 0.7222 | 19 |
| decoder | 1e-2 | 170.77 | 0.8775 | 0.7197 | 19 |
| decoder | 3e-2 | 160.74 | 0.9262 | 0.6977 | 19 |

Observations:

- A shared rate region exists near **400–440 bits/example**, but it is narrow.
- Every run selected its final epoch (19), so the optimization budget is the
  binding constraint. These points are not converged.
- The encoder has lower CE at every rate in this small pilot, but the overlap is
  too narrow and seed coverage too small to support a matched-rate claim.
- Epoch histories show total rates starting near 10,000–13,000 bits/example and
  falling throughout training. Extending the run and the beta grid is required
  before interpolation.

## High-rate recovery control

Beta = 0, 100-epoch cap, same preprocessing and `last` readout.

| Backbone | Test rate | Stochastic CE | Matched deterministic CE | Gap |
|---|---:|---:|---:|---:|
| encoder | 14,122.43 | 0.66896 | 0.633 | +0.036 |
| decoder | 14,924.52 | 0.81730 | 0.747 | +0.070 |

Training stopped at epoch 3 (encoder) and epoch 10 (decoder) by validation-CE
patience. Both gaps exceed the required 0.02 bits, so **gate 6 fails** for this
recipe. The deterministic references are from the earlier E3-B diagnostic and
were not rerun in this pilot.

## Forced-prior control

Forced-prior evaluation reports exactly zero measured rate and is invariant to
input permutation in the synthetic protocol test. However, the expected CE of
the learned head under forced prior samples is **1.08–1.19 bits**, versus the
train-prior CE of **0.9568 bits**. Therefore the current forced-prior control
does **not** satisfy the required label-prior CE match. A prior-only output head
or a separately fitted label-prior predictor is needed before this gate can
pass.

## Tests

`tests/test_e3b.py` contains five protocol sanity tests:

1. analytic diagonal-Gaussian KL and zero padding contribution;
2. total rate is a sum, not a per-token mean;
3. identity deterministic bottleneck matches a raw last-token linear head;
4. fixed-support pooling has zero forced-prior rate and no mask dependence;
5. forced-prior outputs are input-independent for a fixed shape.

All five passed locally and on the GH200 with the `~/llm` environment.

## Current acceptance status

| Gate | Status |
|---|---|
| Analytic KL | pass |
| Sum-not-mean | pass |
| Identity path | pass |
| Checkpoint restoration | gate-restore code fixed; no end-to-end restore test yet |
| Zero-rate input independence | pass in synthetic test |
| Zero-rate label-prior CE | **fail** |
| High-rate recovery within 0.02 bits | **fail** |
| Rate accounting | partial: sum test passes; batch-partition test not run |
| Coverage across paired seeds | not run |
| No test-set selection | pass by construction |

## Next steps

1. Increase the epoch cap/patience and extend beta both below 3e-4 for the
   decoder and above 3e-2 for the encoder to widen the shared rate region.
2. Fix high-rate recovery: verify the deterministic control under identical
   preprocessing, and either initialize/constrain the posterior variance more
   tightly or train beta = 0 longer with a matched objective.
3. Implement a true forced-prior label predictor so the zero-rate CE gate is
   meaningful.
4. Add the fixed-support attention arm and rerun the beta pilot.
5. Only then expand to 5 paired E2 seeds × 3 VIB seeds × 2 latent dimensions.
6. Add the truncation/cache audit and matched-rate interpolation artifacts
   required by `E3B_CORRECTED_SETUP.md`.

## Artifacts

Remote run records:

- `~/rdt-e3b/runs/e3b_pilot_gh200/{encoder,decoder}/runs.jsonl`
- `~/rdt-e3b/runs/e3b_highrate100_gh200/{encoder,decoder}/runs.jsonl`
- `~/rdt-e3b/runs/e3b_smoke_gh200/`
- checkpoints under each run's `checkpoints/` directory

These are preliminary engineering artifacts, not final analysis inputs.
