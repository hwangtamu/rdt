# E3-B P0 diagnostics — readout and last-token-preserving pooling

## No-bottleneck token readouts

Same frozen E2 checkpoint, raw token states, linear head; validation-selected.
Temperature is fitted on validation only.

Decoder (E2 seed 2):

| Readout | Test CE | Acc | ECE | Temp-scaled CE |
|---|---:|---:|---:|---:|
| last | **0.7468** | 0.7466 | 0.023 | 0.7472 |
| mean | 0.8184 | 0.7111 | 0.033 | 0.8138 |
| attention | 0.8097 | 0.7191 | 0.051 | 0.7888 |

Encoder (E2 seed 1):

| Readout | Test CE | Acc | ECE | Temp-scaled CE |
|---|---:|---:|---:|---:|
| last | **0.6334** | 0.8017 | 0.023 | 0.6350 |
| mean | 0.7097 | 0.7760 | 0.022 | 0.7052 |
| attention | 0.7048 | 0.7974 | 0.076 | 0.6422 |

Decoder failure is not calibration: last-token is well calibrated and best.
Mean/attention are worse, and temperature scaling does not fix them.

## Deterministic token-level controls

Same attention/last pooling but with a token-level bottleneck (z=μ, no sampling).

Decoder:

| Pooling | Latent | Test CE | Acc |
|---|---:|---:|---:|
| attention | 32 | 0.992 | 0.753 |
| last | 32 | 0.958 | 0.761 |
| last_gate | 32 | 0.948 | 0.760 |
| last_gate | 128 | 0.891 (seed 0) | 0.764 |
| last | 512 | 0.898 (seed 0) | 0.758 |

Encoder:

| Pooling | Latent | Test CE | Acc |
|---|---:|---:|---:|
| attention | 32 | 0.710 | 0.798 |
| last | 32 | 0.716 | 0.803 |
| last_gate | 32 | 0.709 | 0.805 |

## Reading

1. Decoder high-rate failure is a **readout/information-path** problem, not a
   temperature/calibration problem.
2. Raw last-token is the correct decoder readout; attention and mean pooling
   lose information.
3. A last-token-preserving token bottleneck still fails to recover the decoder
   baseline (best observed ~0.89–0.95 vs 0.747), even at latent 512 with last
   pooling. The token VIB/head training itself is losing information or failing
   to optimize.
4. Encoder token controls are closer to baseline (~0.71 vs 0.633), but still
   worse than raw last-token.

## Decision

P0.2 target (both deterministic token controls near their E2 baselines) is **not
met**. Do not run the matched-rate E3-B pilot or five-seed expansion yet.

## Identity-projection / frozen-μ controls

To test whether initialization was the problem, we initialized the token
projection to identity (latent 512) and froze μ. This should make `last`
pooling equivalent to a linear head on the raw last token.

Decoder results:

| Pooling | Init | μ | Test CE |
|---|---|---|---:|
| last | identity | trainable | 0.898–0.907 |
| last | identity | frozen | **0.978** |

Encoder:

| Pooling | Init | μ | Test CE |
|---|---|---|---:|
| last | identity | trainable | 0.681–0.710 |

The frozen-identity decoder control still fails to recover the 0.747 baseline,
so the problem is not merely μ initialization or overfitting of the projection.
There is a remaining implementation/optimization discrepancy between the
no-bottleneck last-token head and the token-VIB last-token path that must be
found before E3-B is valid.

## Debug resolution: minibatch size was the culprit

A unit check showed the TokenVIB identity + last-pooling forward pass is exactly
equivalent to a raw linear last-token head (max logit difference = 0.0).
The earlier discrepancy came from the training recipe, not the forward path.

With **batch size 2048** and identity/frozen μ, deterministic token controls
recover the E2 baselines:

| Model | Raw last CE | Token last CE | Token last_gate CE |
|---|---:|---:|---:|
| encoder | 0.633 | 0.634–0.646 | 0.653–0.663 |
| decoder | 0.747 | 0.750–0.753 | 0.748–0.762 |

So P0.2 high-rate recovery is now **passed**.

## E3-B finite-rate pilot

Token VIB, last-gate pooling, batch 2048, trainable μ/logvar:

| Model | β | Total rate | Test CE |
|---|---:|---:|---:|
| encoder | 0.15 | 7.79 | 0.643 |
| encoder | 0.2 | 5.69 | 0.648 |
| encoder | 0.3 | 4.11 | 0.659 |
| encoder | 0.5 | 1.67 | 0.684 |
| encoder | 1.0 | 1.14 | 0.728 |
| decoder | 0.3 | 54.0 | 0.751 |
| decoder | 0.5 | 50.5 | 0.751 |
| decoder | 1.0 | 43.6 | 0.759 |
| decoder | 2.0 | 43.5 | 0.796 |
| decoder | 5.0 | 38.9 | 0.863 |

The encoder reaches 1–8 bits/example. The decoder does not go below roughly
39 bits/example even at β=5, so there is no shared 1–5 bit region for matched-rate
P1. The decoder token VIB rate control still fails after high-rate recovery.

## Decision

- High-rate deterministic control: **passed**.
- Matched-rate E3-B pilot: **blocked** by decoder rate coverage.
- Do not run the five-E2-seed E3-B expansion yet.

## Critical correction: rate was per-token average, not total

The E3-B `rate` initially returned the **masked mean KL per token**, not the
total bits/example required by the E3-B design. This has been fixed:
`rate = sum_t KL_t` (bits/example). All earlier E3-B "rate" values in this file
must be interpreted as average bits/token and are not the E3-B total-rate
metric.

After the fix, calibrating β on the total rate gives:

| Model | β | Total rate bits/example | Test CE |
|---|---:|---:|---:|
| encoder | 10 | 4646.6 | 0.937 |
| encoder | 100 | 4646.6 | 0.939 |
| encoder | 1000 | 4646.6 | 0.954 |
| encoder | 10000 | 1788.0 | 1.009 |
| decoder | 10 | 6856.6 | 1.003 |
| decoder | 100 | 6856.6 | 1.005 |
| decoder | 1000 | 6856.6 | 1.010 |
| decoder | 10000 | 5303.5 | 1.010 |

The current token VIB cannot reach the requested 1–5 bits/example total rate;
the rate remains thousands of bits even at β=10000, while CE collapses to or
above the prior. This is a rate-control failure, not a matched-rate result.

## Updated decision

- Last-token deterministic high-rate control: passes at batch 2048.
- Total-rate E3-B finite-rate sweep: no shared 1–5 bits/example coverage.
- P1 matched-rate pilot and five-seed expansion remain blocked.
