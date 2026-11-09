# E1 ProofWriter — frozen probes with controlled reasoning depth

Data builder: `src/w2c/build_proofwriter_e1.py`
Data: `data/proofwriter_e1/` (mixed depth, within-depth subsets, extrapolation)
Configs: `configs/e1_proofwriter.json`, `configs/e1_proofwriter_extrap.json`
Runs: `runs/e1_proofwriter/`, `runs/e1_proofwriter_extrap/`

Primary depths: `depth-0`, `depth-1`, `depth-2`, `depth-3`, `depth-5`.
`allProofs` is excluded. Splits are theory-disjoint; the same theory never crosses
train/val/test.

## Data

Mixed-depth pool:

| Split | Rows | Groups | Depth balance |
|---|---:|---:|---|
| train | 20,020 | 2,440 | 4,000 per depth |
| val | 4,024 | 488 | ~800 per depth |
| test | 4,018 | 496 | ~800 per depth |

Test label prior: False 27.7 %, True 27.7 %, Unknown 44.6 %.
Prior CE `H(Y) = 1.545` bits; majority accuracy = 0.446.

## E1 mixed-depth frozen probe

Selected by validation CE, 3 seeds:

| Backbone | Probe | Layer | Val CE | Test CE | Acc | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| decoder | MLP | 17 | **0.9691** | **0.9697** | **0.6607** | **0.6644** |
| decoder | linear | 17 | 0.9956 | 0.9986 | 0.6616 | 0.6644 |
| encoder | MLP | 22 | 1.1755 | 1.1979 | 0.5738 | 0.5719 |
| encoder | linear | 22 | 1.2226 | 1.2368 | 0.5658 | 0.5642 |

The decoder resolves `1.545 − 0.970 = 0.575` bits ≈ 37 % of `H(Y)`.
The encoder resolves `1.545 − 1.198 = 0.347` bits ≈ 22 %.

## Accuracy vs nominal depth

Matched `MLP / L17` frozen probe:

| Depth | Decoder CE | Decoder Acc | Encoder CE | Encoder Acc |
|---|---:|---:|---:|---:|
| depth-0 | **0.780** | **0.740** | 1.110 | 0.609 |
| depth-1 | **0.929** | **0.679** | 1.239 | 0.569 |
| depth-2 | **0.981** | **0.684** | 1.230 | 0.577 |
| depth-3 | **1.043** | **0.634** | 1.300 | 0.528 |
| depth-5 | **1.116** | **0.566** | 1.219 | 0.530 |

## Accuracy vs actual `QDep` (decoder L17 linear)

| QDep | n | CE | Acc |
|---:|---:|---:|---:|
| 0 | 2,092 | 0.792 | 0.761 |
| 1 | 1,071 | 1.204 | 0.559 |
| 2 | 442 | 1.158 | 0.609 |
| 3 | 231 | 1.330 | 0.519 |
| 4 | 101 | 1.178 | 0.505 |
| 5 | 80 | 1.587 | 0.412 |

This is the clean complexity curve: frozen accessibility drops sharply from
direct-fact lookup (`QDep=0`) to multi-hop reasoning (`QDep≥3`).

## Depth extrapolation

Train on depth-0/1/2, validate on held-out depth-2 theories, test on depth-3/5.
Theory-disjoint.

| Backbone | Probe | Layer | Test CE | Acc | Macro-F1 |
|---|---|---:|---:|---:|---:|
| decoder | linear | 17 | **1.1525** | **0.5929** | **0.5950** |
| decoder | MLP | 17 | 1.1846 | 0.5682 | 0.5650 |
| encoder | MLP | 22 | 1.2892 | 0.5094 | 0.5050 |
| encoder | linear | 22 | 1.3127 | 0.5168 | 0.5074 |

Depth extrapolation is degraded relative to within-depth:
decoder depth-3 1.101 / 0.627 and depth-5 1.204 / 0.559, versus within-depth
1.043 / 0.634 and 1.116 / 0.566. The gap is real but not catastrophic; the
decoder keeps its advantage.

## Reading

1. **ProofWriter gives a usable complexity axis.** Accuracy falls monotonically
   with `QDep`; the effect is much clearer than a single aggregate score.
2. **Decoder features are much more accessible than encoder features on this
   task** under the E1 last-token readout: CE 0.970 vs 1.198.
3. **Depth extrapolation hurts, but only modestly.** Training on depth 0–2 still
   gives above-prior performance at depth 3/5.
4. **E1 is readout-confounded.** It uses the post-final-norm last-token readout.
   The BoolQ E1.5 result showed the encoder can prefer pre-final-norm L17; the
   same check must be run here before concluding encoder vs decoder.
5. **Unknown-heavy labels matter.** Majority accuracy is only 0.446; macro-F1
   and per-class F1 should accompany accuracy.

## Next steps

1. Run E1.5 on ProofWriter (`last`, `mean`, `attn`, `concat`, plus `lasth-L17`)
   to check whether the encoder’s deficit is a readout/final-norm artifact.
2. Run within-depth E1 probes per depth (separate probe per `depth-*` split) to
   remove the mixed-depth training confound.
3. Run E2 on ProofWriter with the same data-scaling design, evaluating by
   `QDep` and `depth-*`.
4. Add a binary True/False closed-world control (drop Unknown) to separate
   open-world abstention from depth-dependent reasoning.
