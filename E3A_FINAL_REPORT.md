# E3-A final report — corrected 68M BoolQ rate–distortion analysis

**Status:** complete for the primary last-token protocol.  
**Run:** `runs/e3a_boolq_p2/{encoder,decoder}/s{0..4}`  
**Config:** `configs/e3a_boolq_p2.json`  
**Analysis:** `runs/e3a_boolq_p2/analysis/`  
**Figures:** `figures/e3a_absolute_rd.svg`, `figures/e3a_delta.svg`, `figures/e3a_penalty.svg`

## Executive summary

With corrected feature scaling, objective-based checkpoint selection, full
zero/high-rate coverage, and per-seed matched-rate analysis:

1. **Zero-rate passes** for all runs: the bottleneck collapses to the prior.
2. **High-rate recovery passes**: β=0 matches the deterministic controls.
3. **The encoder has lower absolute CE at every matched rate** in 1–8 bits.
4. **Compression-penalty differences are not statistically distinguishable
   from zero** in the primary latent-32 configuration.
5. The encoder advantage is approximately the **uncompressed E2 advantage**
   (~0.071 bits), so the result is closer to **Outcome B**: similar compression
   behaviour, different absolute quality.
6. The conclusion is stable to five VIB seeds, leave-one-E2-seed-out analysis,
   and a near-zero rate interaction. Latent-128 and normalization ablations
   change the magnitude and, in one cell, the penalty interpretation; they are
   reported as sensitivity checks, not as the primary result.

**Claim:** On BoolQ with the final last-token readout, the 68M encoder has a
better absolute decision representation at matched rate, but the data do **not**
support a robust claim that it has a lower rate–distortion frontier.

---

## Protocol and fixes relative to P1

P1’s null result was not a clean test. The corrected P2 protocol differs in
these ways:

1. **Global scalar feature scaling**, fitted on the VIB training split only.
   Raw encoder features have norms roughly 5–7× larger than decoder features;
   without scaling, the KL rate is not comparable. Per-dimension z-scoring was
   tested as an ablation and degraded the linear readout.
2. **VIB checkpoint selection on validation `CE + β·R`**, not validation CE
   alone. This prevents high-β runs from selecting informative high-rate models.
3. **Longer patience for high β** (`β=1`: 100 epochs; `β=10`: 150 epochs).
4. **High-rate coverage** with β=0 and small β values.
5. **Capacity-matched deterministic control**: a 32-d deterministic bottleneck
   plus linear head, in addition to the full 512-d linear control.
6. **Validation-rate matching** and per-`(E2 seed, VIB seed)` interpolation.
7. **Hierarchical uncertainty** over E2 seeds, VIB seeds, and test examples.
8. Per-epoch training history and per-example expected/mean CE are saved.

---

## Sanity checks

| Backbone | Deterministic CE | β=0 expected CE | β=10 max rate | β=10 max CE |
|---|---:|---:|---:|---:|
| encoder | 0.6890 | 0.6918 | 0.0153 | 0.9608 |
| decoder | 0.7601 | 0.7635 | 0.0495 | 0.9617 |

Prior CE is 0.9568. High-rate recovery is within ~0.003 bits of the
deterministic control; zero-rate runs are at the prior.

---

## Main result — expected stochastic CE at matched validation rate

ΔD = decoder − encoder CE; positive favors the encoder. 95% CI is hierarchical
over E2 seeds, VIB seeds, and test examples.

| Target R | Encoder CE | Decoder CE | ΔD | 95% CI |
|---:|---:|---:|---:|---|
| 1.0 | 0.8347 | 0.9195 | +0.0848 | [+0.0512, +0.1202] |
| 1.5 | 0.8085 | 0.8876 | +0.0791 | [+0.0481, +0.1184] |
| 2.0 | 0.7927 | 0.8661 | +0.0734 | [+0.0371, +0.1110] |
| 2.5 | 0.7818 | 0.8518 | +0.0701 | [+0.0335, +0.1049] |
| 3.0 | 0.7730 | 0.8417 | +0.0688 | [+0.0314, +0.1048] |
| 3.5 | 0.7667 | 0.8334 | +0.0667 | [+0.0272, +0.1071] |
| 4.0 | 0.7632 | 0.8311 | +0.0679 | [+0.0248, +0.1056] |
| 4.5 | 0.7593 | 0.8297 | +0.0704 | [+0.0269, +0.1102] |
| 5.0 | 0.7550 | 0.8285 | +0.0735 | [+0.0326, +0.1154] |
| 6.0 | 0.7471 | 0.8271 | +0.0801 | [+0.0398, +0.1239] |
| 8.0 | 0.7372 | 0.8351 | +0.0979 | [+0.0472, +0.1449] |

The encoder advantage is present at every target, and every CI excludes zero.

---

## Compression penalty — expected CE

Penalty = matched-rate CE − that seed’s deterministic control CE.
Δpenalty = decoder penalty − encoder penalty; negative means the encoder has the
larger penalty.

| Target R | Encoder penalty | Decoder penalty | Δpenalty | Δpenalty 95% CI |
|---:|---:|---:|---:|---|
| 1.0 | +0.1457 | +0.1594 | +0.0137 | [−0.0412, +0.0541] |
| 2.0 | +0.1037 | +0.1059 | +0.0022 | [−0.0331, +0.0297] |
| 3.0 | +0.0840 | +0.0816 | −0.0024 | [−0.0252, +0.0190] |
| 4.0 | +0.0742 | +0.0709 | −0.0033 | [−0.0234, +0.0165] |
| 5.0 | +0.0661 | +0.0684 | +0.0024 | [−0.0218, +0.0246] |
| 6.0 | +0.0581 | +0.0670 | +0.0089 | [−0.0148, +0.0344] |
| 8.0 | +0.0482 | +0.0750 | +0.0268 | [−0.0053, +0.0588] |

Every primary Δpenalty CI includes zero. The absolute advantage is therefore
not evidence of a smaller compression penalty.

---

## Robustness

### Five VIB seeds
The 5-seed replication at the critical rates gives nearly identical results:

| R | ΔD (3 seeds) | ΔD (5 seeds) | Δpenalty (3 seeds) | Δpenalty (5 seeds) |
|---:|---:|---:|---:|---:|
| 2.0 | +0.0734 | +0.0733 | +0.0022 | +0.0022 |
| 4.0 | +0.0679 | +0.0689 | −0.0033 | −0.0022 |

### Per-E2-seed stability
All five E2 seeds have positive ΔD at every target. Leave-one-E2-seed-out mean
ΔD ranges:

| R | Per-seed ΔD | LOO mean range |
|---:|---|---|
| 1.0 | +0.044, +0.142, +0.066, +0.071, +0.101 | [+0.071, +0.095] |
| 2.0 | +0.073, +0.125, +0.046, +0.037, +0.085 | [+0.061, +0.082] |
| 4.0 | +0.096, +0.115, +0.030, +0.021, +0.078 | [+0.056, +0.080] |
| 8.0 | +0.108, +0.158, +0.049, +0.053, +0.122 | [+0.083, +0.110] |

The result is not driven by a single E2 seed.

### Rate interaction
Mean ΔD(8 bits) − ΔD(1 bit) = **+0.0131** across seeds. The gap is roughly
constant with rate; there is no strong evidence of a rate-dependent encoder
advantage.

### Ablation summary at R=2.0 and R=4.0

| Run | R | ΔD | Δpenalty |
|---|---:|---:|---:|
| P2 main: latent 32, global scaling | 2.0 | +0.073 | +0.002 |
| P2 main: latent 32, global scaling | 4.0 | +0.068 | −0.003 |
| Latent 128 | 2.0 | +0.176 | +0.105 |
| Latent 128 | 4.0 | +0.124 | +0.053 |
| No feature scaling | 2.0 | +0.031 | +0.001 |
| No feature scaling | 4.0 | +0.044 | +0.013 |
| Per-dimension z-score | 2.0 | +0.053 | +0.030 |
| Per-dimension z-score | 4.0 | +0.061 | +0.039 |

The **sign** of ΔD is stable across all ablations, but the magnitude depends on
scaling and latent size. Latent 128 shows a larger encoder advantage and a
positive Δpenalty at low/mid rates, but the decoder expected CE at those rates is
near or above the prior, indicating a posterior-variance/optimization problem in
the 128-d decoder VIB. It is not a clean confirmation of an encoder frontier
advantage.

### Nested validation limitation
E2 checkpoints were selected on the BoolQ validation split, and P2 also uses that
split for VIB early stopping. A VIB validation split carved from the E2 training
pool was attempted and **rejected**: the backbone had already seen those
examples, so the control overfits and the test CE degrades. A clean fix requires
retraining E2 with a disjoint E2-train / E2-val / E3-val partition or
cross-fitting E2 checkpoint selection. This remains the main protocol caveat.

---

## Interpretation

The corrected E3-A result is:

> At matched variational rate on BoolQ, the 68M encoder produces lower expected
> decision CE than the 68M decoder, but its compression penalty is not
> significantly smaller. The encoder’s advantage is largely its better
> uncompressed representation, not a more efficient rate–distortion frontier.

This is **Outcome B** in the plan’s interpretation framework, with the caveat
that the primary protocol is post-pooling and task-specific.

It is not Outcome A. A claim such as “the encoder requires fewer bits for the
same decision distortion” is not supported by the primary latent-32 analysis.

---

## Limitations

- BoolQ only; binary task.
- Final last-token readout only. Attention-L18 robustness was not run because
  the E2 68M attention runs did not save checkpoints.
- Latent 32 is primary; latent 128 is a sensitivity check with a decoder
  posterior-variance problem.
- Expected stochastic CE is sensitive to posterior variance; mean-only CE is
  reported as a diagnostic, not the primary distortion.
- The KL rate is a variational upper bound on `I(X;Z)`, not exact mutual
  information.
- Nested E2/E3 validation selection remains; the VIB-validation-from-train fix
  was rejected as invalid.
- No prior-sensitivity / learned-prior check yet.

---

## Deliverables

- `runs/e3a_boolq_p2/analysis/matched_expected.md`
- `runs/e3a_boolq_p2/analysis/matched_mean.md`
- `runs/e3a_boolq_p2/analysis/robustness.md`
- `runs/e3a_boolq_p2/analysis/curves.csv`
- Ablation analyses under `runs/e3a_boolq_p2_{lat128,noscale,perdim,seeds5}/analysis/`
- `figures/e3a_absolute_rd.svg`
- `figures/e3a_delta.svg`
- `figures/e3a_penalty.svg`
- Reproduce analysis:
  ```bash
  PYTHONPATH=src $VENV/bin/python -m w2c.compare_e3_p2 --run-root runs/e3a_boolq_p2
  PYTHONPATH=src $VENV/bin/python -m w2c.e3a_summary
  PYTHONPATH=src $VENV/bin/python -m w2c.plot_e3a
  ```
