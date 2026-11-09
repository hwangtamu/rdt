# E3-B pilot — pre-pooling token VIB

Setup: frozen E2 `last-L19` checkpoints, BoolQ. Per-token Gaussian VIB
(latent 32) followed by attention pooling and a linear head. Rate is total
KL bits/example summed over valid tokens. Encoder checkpoint = E2 seed 1;
decoder checkpoint = E2 seed 2. 3 VIB seeds.

## Token VIB sweep

| Model | β | Rate bits/example | Expected CE | Mean-only CE |
|---|---:|---:|---:|---:|
| encoder | 0 | 1765 | 0.710 | 0.709 |
| encoder | 1e-5 | 336 | 0.700 | 0.700 |
| encoder | 1e-4 | 115 | 0.702 | 0.702 |
| encoder | 1e-3 | 32.0 | 0.697 | 0.697 |
| encoder | 1e-2 | 10.1 | 0.714 | 0.714 |
| encoder | 0.1 | 2.75 | 0.689 | 0.691 |
| decoder | 0 | 118 | **1.002** | **1.002** |
| decoder | 1e-5 | 82.2 | 1.002 | 1.002 |
| decoder | 1e-4 | 43.8 | 1.001 | 0.998 |
| decoder | 1e-3 | 21.9 | 0.994 | 0.991 |
| decoder | 1e-2 | 11.0 | 0.967 | 0.959 |
| decoder | 0.1 | 2.98 | 0.795 | 0.792 |

Prior CE = 0.9568.

## Deterministic token-level control

No sampling; same attention pooling + head:

| Model | Test CE | Acc |
|---|---:|---:|
| encoder | 0.710 | 0.798 |
| decoder | **0.992** | 0.753 |

## Reading

- Encoder token VIB works: high-rate CE ≈0.71, and the deterministic control
  matches it.
- Decoder token VIB **fails high-rate recovery**: CE ≈1.00, worse than the
  prior, even with deterministic inference and deterministic token control.
  This is not a sampling-noise artifact.
- The decoder failure is therefore at the token-level bottleneck/pooling/head
  stage, not just the VIB posterior.
- Total rate scales differ strongly between models (encoder thousands of bits,
  decoder tens of bits), so matched-rate comparison is not yet meaningful.

## Decision

E3-B pilot is **not valid yet**. Before matched-rate analysis:
1. Diagnose decoder token-level pooling: compare attention pooling against
   mean/last-token pooling in the deterministic token control.
2. Calibrate β to a common total-rate range without relying on the current
   decoder posterior variance.
3. Recheck high-rate recovery before running five E2 seeds.
