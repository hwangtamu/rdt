# E1 report — frozen representation probing

Run directory: `runs/e1_w2c_pref`

- Classes (3): tool_call, request_for_info, cannot_answer
- Splits: train=7532, val=1329, test=3652
- Layers probed: [0, 6, 11, 12, 17, 22]
- Probes: ['linear', 'mlp']; seeds: [0, 1, 2]
- Target: last non-pad token of `"Decision:"`
- Test truncation rate: encoder=0.219%, decoder=0.219%

## Label distribution

| Split | tool_call | request_for_info | cannot_answer |
|---|---:|---:|---:|
| train | 2534 | 2482 | 2516 |
| val | 434 | 438 | 457 |
| test | 1295 | 1062 | 1295 |

Dropped during loading: `{'kept': 8861}`

## Mean test metrics by layer (averaged over seeds)

### linear

| Backbone | Layer | CE (bits) | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 0 | 1.5852 | 0.3545 | 0.1757 | 0.0136 |
| decoder | 6 | 1.5476 | 0.4936 | 0.4956 | 0.1628 |
| decoder | 11 | 1.6491 | 0.5358 | 0.5377 | 0.1768 |
| decoder | 12 | 1.7337 | 0.5209 | 0.5200 | 0.2018 |
| decoder | 17 | 1.6960 | 0.5104 | 0.5121 | 0.2074 |
| decoder | 22 | 1.7265 | 0.4966 | 0.4989 | 0.2255 |
| encoder | 0 | 1.5848 | 0.3546 | 0.1745 | 0.0132 |
| encoder | 6 | 1.6379 | 0.4643 | 0.4616 | 0.1528 |
| encoder | 11 | 1.5795 | 0.4451 | 0.4439 | 0.1454 |
| encoder | 12 | 1.5525 | 0.4575 | 0.4584 | 0.1253 |
| encoder | 17 | 1.8962 | 0.4163 | 0.4100 | 0.2702 |
| encoder | 22 | 1.8594 | 0.4392 | 0.4370 | 0.2465 |

### mlp

| Backbone | Layer | CE (bits) | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 0 | 1.5849 | 0.3541 | 0.1750 | 0.0164 |
| decoder | 6 | 1.9679 | 0.4982 | 0.4992 | 0.2516 |
| decoder | 11 | 2.0086 | 0.5284 | 0.5302 | 0.2316 |
| decoder | 12 | 2.1123 | 0.5217 | 0.5235 | 0.2387 |
| decoder | 17 | 2.1207 | 0.5231 | 0.5261 | 0.2491 |
| decoder | 22 | 2.0817 | 0.5087 | 0.5110 | 0.2527 |
| encoder | 0 | 1.5848 | 0.3546 | 0.1745 | 0.0169 |
| encoder | 6 | 1.9056 | 0.4554 | 0.4519 | 0.2382 |
| encoder | 11 | 1.6999 | 0.4566 | 0.4582 | 0.1906 |
| encoder | 12 | 1.6869 | 0.4711 | 0.4706 | 0.1763 |
| encoder | 17 | 2.0380 | 0.4379 | 0.4324 | 0.2757 |
| encoder | 22 | 1.9969 | 0.4537 | 0.4535 | 0.2570 |

## Paired encoder-vs-decoder comparison

Delta D = D_decoder − D_encoder (negative favors the encoder).

| Layer | Probe | Delta D | 95% CI | p |
|---:|---|---:|---:|---:|
| 0 | linear | +0.0003 | [-0.0003, +0.0011] | 0.328 |
| 0 | mlp | +0.0001 | [-0.0000, +0.0003] | 0.054 |
| 6 | linear | -0.0903 | [-0.1334, -0.0475] | 0.001 |
| 6 | mlp | +0.0623 | [-0.0100, +0.1293] | 0.095 |
| 11 | linear | +0.0696 | [+0.0126, +0.1293] | 0.014 |
| 11 | mlp | +0.3087 | [+0.2293, +0.3891] | 0.000 |
| 12 | linear | +0.1812 | [+0.1207, +0.2457] | 0.000 |
| 12 | mlp | +0.4254 | [+0.3380, +0.5144] | 0.000 |
| 17 | linear | -0.2002 | [-0.2619, -0.1415] | 0.000 |
| 17 | mlp | +0.0827 | [+0.0004, +0.1641] | 0.049 |
| 22 | linear | -0.1329 | [-0.1965, -0.0729] | 0.000 |
| 22 | mlp | +0.0848 | [+0.0029, +0.1630] | 0.045 |

## Best linear probe per backbone (by mean CE)

| Backbone | Layer | CE | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 6 | 1.5476 | 0.4936 | 0.4956 | 0.1628 |
| encoder | 12 | 1.5525 | 0.4575 | 0.4584 | 0.1253 |

## Confusion matrix, best linear config (seed 0)

Rows = true label, columns = predicted; order = ['tool_call', 'request_for_info', 'cannot_answer'].

**decoder, layer 6**

```
  631   472   192
  333   636    93
  360   353   582
```

**encoder, layer 12**

```
  575   461   259
  384   533   145
  353   373   569
```

## Interpretation guardrails

* Frozen probing measures **linear/MLP accessibility** of decision information, not differences in `I(Y;Z)` (plan section 6).
* The official MCQ test set is a different distribution from the preference/SFT training pool; prior shift and domain shift both contribute to absolute scores.
* Layer 0 is the embedding of the shared final token (`:`) and is therefore input-invariant; its probes can only learn the class prior.
* Differences that flip sign across layers/probes are not evidence for a uniform architecture advantage. Report the layer and probe with the comparison.
* This release has no usable `direct` examples (5/24,000 responses; 0 in test), so E1 is run over the three supported classes unless stated otherwise.

