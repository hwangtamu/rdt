# E1 report — frozen representation probing

Run directory: `runs/e1_proofwriter_extrap`

- Classes (3): false, true, unknown
- Splits: train=12012, val=802, test=1604
- Layers probed: [0, 6, 11, 12, 17, 22]
- Probes: ['linear', 'mlp']; seeds: [0, 1, 2]
- Target: last non-pad token of `"Decision:"`
- Test truncation rate: encoder=0.000%, decoder=0.000%

## Label distribution

| Split | false | true | unknown |
|---|---:|---:|---:|
| train | 3216 | 3216 | 5580 |
| val | 222 | 222 | 358 |
| test | 458 | 458 | 688 |

## Mean test metrics by layer (averaged over seeds)

### linear

| Backbone | Layer | CE (bits) | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 0 | 1.5572 | 0.4289 | 0.2001 | 0.0162 |
| decoder | 6 | 1.1516 | 0.5987 | 0.6050 | 0.0283 |
| decoder | 11 | 1.1397 | 0.5800 | 0.5869 | 0.0521 |
| decoder | 12 | 1.1479 | 0.5777 | 0.5838 | 0.0589 |
| decoder | 17 | 1.1525 | 0.5929 | 0.5950 | 0.0704 |
| decoder | 22 | 1.1457 | 0.5800 | 0.5824 | 0.0530 |
| encoder | 0 | 1.5572 | 0.4289 | 0.2001 | 0.0162 |
| encoder | 6 | 1.2982 | 0.5278 | 0.5255 | 0.0340 |
| encoder | 11 | 1.3306 | 0.5094 | 0.4912 | 0.0456 |
| encoder | 12 | 1.2646 | 0.5316 | 0.5292 | 0.0392 |
| encoder | 17 | 1.3192 | 0.5200 | 0.5113 | 0.0641 |
| encoder | 22 | 1.3127 | 0.5168 | 0.5074 | 0.0796 |

### mlp

| Backbone | Layer | CE (bits) | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 0 | 1.5575 | 0.4289 | 0.2001 | 0.0171 |
| decoder | 6 | 1.1326 | 0.5792 | 0.5783 | 0.0474 |
| decoder | 11 | 1.1555 | 0.5665 | 0.5688 | 0.0910 |
| decoder | 12 | 1.1500 | 0.5707 | 0.5724 | 0.0883 |
| decoder | 17 | 1.1846 | 0.5682 | 0.5650 | 0.1239 |
| decoder | 22 | 1.2108 | 0.5626 | 0.5573 | 0.1105 |
| encoder | 0 | 1.5575 | 0.4289 | 0.2001 | 0.0171 |
| encoder | 6 | 1.2469 | 0.5391 | 0.5441 | 0.0687 |
| encoder | 11 | 1.2836 | 0.5160 | 0.5092 | 0.0687 |
| encoder | 12 | 1.2455 | 0.5231 | 0.5213 | 0.0912 |
| encoder | 17 | 1.3030 | 0.5035 | 0.4998 | 0.1037 |
| encoder | 22 | 1.2892 | 0.5094 | 0.5050 | 0.1051 |

## Paired encoder-vs-decoder comparison

Delta D = D_decoder − D_encoder (negative favors the encoder).

| Layer | Probe | Delta D | 95% CI | p |
|---:|---|---:|---:|---:|
| 0 | linear | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| 0 | mlp | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| 6 | linear | -0.1466 | [-0.1767, -0.1163] | 0.000 |
| 6 | mlp | -0.1143 | [-0.1495, -0.0775] | 0.000 |
| 11 | linear | -0.1908 | [-0.2301, -0.1533] | 0.000 |
| 11 | mlp | -0.1281 | [-0.1725, -0.0844] | 0.000 |
| 12 | linear | -0.1167 | [-0.1534, -0.0790] | 0.000 |
| 12 | mlp | -0.0956 | [-0.1377, -0.0522] | 0.000 |
| 17 | linear | -0.1667 | [-0.2119, -0.1201] | 0.000 |
| 17 | mlp | -0.1184 | [-0.1704, -0.0611] | 0.000 |
| 22 | linear | -0.1670 | [-0.2081, -0.1266] | 0.000 |
| 22 | mlp | -0.0783 | [-0.1253, -0.0314] | 0.001 |

## Best linear probe per backbone (by mean CE)

| Backbone | Layer | CE | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 11 | 1.1397 | 0.5800 | 0.5869 | 0.0521 |
| encoder | 12 | 1.2646 | 0.5316 | 0.5292 | 0.0392 |

## Confusion matrix, best linear config (seed 0)

Rows = true label, columns = predicted; order = ['false', 'true', 'unknown'].

**decoder, layer 11**

```
  266    18   174
   12   284   162
  151   156   381
```

**encoder, layer 12**

```
  205    17   236
   17   240   201
  125   147   416
```

## Interpretation guardrails

* Frozen probing measures **linear/MLP accessibility** of decision information, not differences in `I(Y;Z)` (plan section 6).
* The official MCQ test set is a different distribution from the preference/SFT training pool; prior shift and domain shift both contribute to absolute scores.
* Layer 0 is the embedding of the shared final token (`:`) and is therefore input-invariant; its probes can only learn the class prior.
* Differences that flip sign across layers/probes are not evidence for a uniform architecture advantage. Report the layer and probe with the comparison.
* This release has no usable `direct` examples (5/24,000 responses; 0 in test), so E1 is run over the three supported classes unless stated otherwise.

