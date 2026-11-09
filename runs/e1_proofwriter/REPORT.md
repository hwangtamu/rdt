# E1 report — frozen representation probing

Run directory: `runs/e1_proofwriter`

- Classes (3): false, true, unknown
- Splits: train=20020, val=4024, test=4018
- Layers probed: [0, 6, 11, 12, 17, 22]
- Probes: ['linear', 'mlp']; seeds: [0, 1, 2]
- Target: last non-pad token of `"Decision:"`
- Test truncation rate: encoder=0.000%, decoder=0.000%

## Label distribution

| Split | false | true | unknown |
|---|---:|---:|---:|
| train | 5454 | 5454 | 9112 |
| val | 1100 | 1100 | 1824 |
| test | 1112 | 1112 | 1794 |

## Mean test metrics by layer (averaged over seeds)

### linear

| Backbone | Layer | CE (bits) | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 0 | 1.5453 | 0.4465 | 0.2058 | 0.0051 |
| decoder | 6 | 1.1450 | 0.6022 | 0.6072 | 0.0200 |
| decoder | 11 | 1.0587 | 0.6252 | 0.6300 | 0.0331 |
| decoder | 12 | 1.0496 | 0.6311 | 0.6360 | 0.0321 |
| decoder | 17 | 0.9986 | 0.6616 | 0.6644 | 0.0167 |
| decoder | 22 | 1.0497 | 0.6296 | 0.6334 | 0.0321 |
| encoder | 0 | 1.5453 | 0.4465 | 0.2058 | 0.0051 |
| encoder | 6 | 1.3039 | 0.5276 | 0.5261 | 0.0364 |
| encoder | 11 | 1.2869 | 0.5494 | 0.5432 | 0.0342 |
| encoder | 12 | 1.2341 | 0.5577 | 0.5572 | 0.0281 |
| encoder | 17 | 1.2587 | 0.5605 | 0.5576 | 0.0495 |
| encoder | 22 | 1.2368 | 0.5658 | 0.5642 | 0.0588 |

### mlp

| Backbone | Layer | CE (bits) | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 0 | 1.5454 | 0.4465 | 0.2058 | 0.0064 |
| decoder | 6 | 1.1034 | 0.6053 | 0.6098 | 0.0459 |
| decoder | 11 | 1.0411 | 0.6324 | 0.6370 | 0.0382 |
| decoder | 12 | 1.0376 | 0.6316 | 0.6367 | 0.0428 |
| decoder | 17 | 0.9697 | 0.6607 | 0.6644 | 0.0420 |
| decoder | 22 | 1.0641 | 0.6219 | 0.6230 | 0.0523 |
| encoder | 0 | 1.5454 | 0.4465 | 0.2058 | 0.0064 |
| encoder | 6 | 1.2485 | 0.5411 | 0.5448 | 0.0652 |
| encoder | 11 | 1.2375 | 0.5529 | 0.5532 | 0.0679 |
| encoder | 12 | 1.1940 | 0.5694 | 0.5689 | 0.0603 |
| encoder | 17 | 1.2197 | 0.5626 | 0.5620 | 0.0677 |
| encoder | 22 | 1.1979 | 0.5738 | 0.5719 | 0.0764 |

## Paired encoder-vs-decoder comparison

Delta D = D_decoder − D_encoder (negative favors the encoder).

| Layer | Probe | Delta D | 95% CI | p |
|---:|---|---:|---:|---:|
| 0 | linear | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| 0 | mlp | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| 6 | linear | -0.1589 | [-0.1795, -0.1389] | 0.000 |
| 6 | mlp | -0.1452 | [-0.1666, -0.1237] | 0.000 |
| 11 | linear | -0.2282 | [-0.2511, -0.2043] | 0.000 |
| 11 | mlp | -0.1964 | [-0.2213, -0.1715] | 0.000 |
| 12 | linear | -0.1845 | [-0.2068, -0.1632] | 0.000 |
| 12 | mlp | -0.1564 | [-0.1819, -0.1322] | 0.000 |
| 17 | linear | -0.2601 | [-0.2864, -0.2338] | 0.000 |
| 17 | mlp | -0.2499 | [-0.2791, -0.2215] | 0.000 |
| 22 | linear | -0.1871 | [-0.2118, -0.1627] | 0.000 |
| 22 | mlp | -0.1338 | [-0.1614, -0.1081] | 0.000 |

## Best linear probe per backbone (by mean CE)

| Backbone | Layer | CE | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| decoder | 17 | 0.9986 | 0.6616 | 0.6644 | 0.0167 |
| encoder | 12 | 1.2341 | 0.5577 | 0.5572 | 0.0281 |

## Confusion matrix, best linear config (seed 0)

Rows = true label, columns = predicted; order = ['false', 'true', 'unknown'].

**decoder, layer 17**

```
  730    62   320
   65   808   239
  370   304  1120
```

**encoder, layer 12**

```
  561    57   494
   64   620   428
  381   356  1057
```

## Interpretation guardrails

* Frozen probing measures **linear/MLP accessibility** of decision information, not differences in `I(Y;Z)` (plan section 6).
* The official MCQ test set is a different distribution from the preference/SFT training pool; prior shift and domain shift both contribute to absolute scores.
* Layer 0 is the embedding of the shared final token (`:`) and is therefore input-invariant; its probes can only learn the class prior.
* Differences that flip sign across layers/probes are not evidence for a uniform architecture advantage. Report the layer and probe with the comparison.
* This release has no usable `direct` examples (5/24,000 responses; 0 in test), so E1 is run over the three supported classes unless stated otherwise.

