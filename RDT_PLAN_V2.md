# Experiment plan v2 — usable information in small encoders vs decoders

**Status:** proposal. Nothing in this plan has been run.
**Supersedes:** the role of E3 (bottleneck rate–distortion) as the main
instrument in `RDT.md`. E0–E2 results are kept and reused.
**Naming:** experiments here are `U0`–`U6` ("usable information") so they do not
collide with `E1`–`E6` in `RDT.md`.

## 1. Why a new plan

The 68M BoolQ results (`E1_E15_BOOLQ_68M_RESULTS.md`, `E2_BOOLQ_68M_RESULTS.md`,
`E3A_FINAL_REPORT.md`, `RDT_STORY.md`) leave three problems.

1. **The hypothesis is about size, and one size cannot test it.** The only
   second point is 150M, with unmatched readouts, wide seed variance, and an
   encoder that scores the same as the 68M encoder (about 0.740 bits).
2. **The bottleneck curves are not frontiers.** BoolQ's label entropy is 0.957
   bits and the models extract only about 0.2–0.27 bits of it. The measured
   curves spend 2–8 bits to approach a quality that sending the predicted label
   should reach in at most 1 bit (estimated from E2 accuracies, not yet run).
   The linear Gaussian bottleneck on a binary label is close to a noisy
   one-dimensional logit, so "equal compression penalty" is nearly built in.
3. **Classical mutual information cannot separate the two models.** Any
   invertible representation carries the same information about the label. What
   can differ is how much a bounded reader extracts, and how fast.

The plan therefore moves the primary measurement from rate–distortion to usable
information and description length, across a size sweep.

## 2. Hypothesis

> At small model size, an encoder has an advantage over a size-matched decoder
> on decision tasks, and the advantage shrinks as size grows.

It is split into three claims with separate measurements.

| Claim | Statement | Measured by |
|---|---|---|
| H1 — more usable information | At matched size the encoder delivers more bits about the decision | U1 |
| H2 — cheaper to extract | The encoder needs fewer labelled examples to deliver those bits | U2 |
| H3 — small-size effect | The encoder's advantage decreases with size | Slope of the U1/U2 gap against log size |

H3 is the hypothesis proper. H1 and H2 could both hold at every size, which
would be a different finding.

## 3. Quantities

All quantities are in bits per example. $K$ is the number of classes.

**Usable information.** For a fixed family of predictors $\mathcal V$
(backbone plus an allowed readout, trained under a fixed budget):

$$
I_{\mathcal V}(X \to Y) = H(Y) - \mathrm{CE}_{\mathrm{test}},
$$

with $H(Y)$ the entropy of the training label prior. This is what E1 and E2
already estimate; v2 reports it as information and fixes $\mathcal V$ in advance.

**Prequential codelength.** Order the training set once per seed and cut it at
$n_0 < n_1 < \dots < n_S$. Let $\theta_i$ be the model trained on the first $n_i$
examples. The cost of transmitting all labels is

$$
L = n_0 \log_2 K \;+\; \sum_{i=0}^{S-1} \sum_{j=n_i+1}^{n_{i+1}}
-\log_2 p_{\theta_i}(y_j \mid x_j),
$$

reported as $L / n_S$ bits per label. Each block is scored by a model that has
not seen it, so no validation split is used to select anything.

**Rate with an exact cap.** Where compression is measured (U5), the code is
discrete with $M$ symbols, so the rate is at most $\log_2 M$ bits with no
variational slack.

## 4. Shared protocol

**Models.** Paired Ettin encoder and decoder at every available size, at least
four sizes with 68M in the middle. Only 68M and 150M are in the repo today; the
full list of published sizes must be confirmed before the sweep is fixed.

**Tasks.** Chosen to span label entropy and reasoning depth.

| Task | Role | Label entropy |
|---|---|---|
| BoolQ | Continuity with E1–E3 | about 1 bit |
| A 3-way inference task | Larger test set, standard benchmark | about 1.6 bits |
| A many-way intent or tool-choice task | Enough bits for compression to be non-trivial | 6–7 bits |
| ProofWriter | Reasoning depth as a dial | per existing E1 setup |
| When2Call | Distribution shift, original target of `RDT.md` | per existing E0 setup |

Each task needs a test set of several thousand examples. BoolQ's 1,634 cannot
resolve a 0.02-bit difference and is kept for continuity only.

**Readout family, fixed in advance for both models.**

- `native`: no trained parameters. Masked answer token for the encoder, next
  token for the decoder, restricted to the answer verbalizers.
- `last`, `mean`, `attn` at the final layer, linear head.

Headline numbers use `native` and the best trained readout per model, chosen on
a split that is never used for anything else. Layer sweeps are diagnostic only.

**Splits.** Per task: train, readout-selection, test. The readout-selection
split is disjoint from any split used for early stopping. This removes the
shared-validation caveat in `E3A_FINAL_REPORT.md`.

**Budget.** Learning rate and epoch count are swept per model and per size on
the readout-selection split, so that neither arm is under-tuned. The 150M
encoder result is the motivating example.

## 5. Experiments

### U0 — Baseline check on existing E3-A features

Purpose: settle whether the E3-A curves are frontiers, using cached 68M BoolQ
features.

- Compute the predicted-label code: rate $H(\hat Y)$, distortion
  $H(Y \mid \hat Y)$, on test.
- Compute quantized-logit codes with 2, 4 and 8 bins: exact rate, test CE after
  recalibration on train.
- Predict each model's bottleneck curve from its deterministic logit
  distribution passed through a Gaussian channel.

Outcome: if the simple codes dominate the measured curves, or the Gaussian
prediction matches them, E3-B is stopped and E3-A is reported as a noise
robustness measurement.

### U1 — Usable information across size

Purpose: H1 and H3.

- Frozen and fully fine-tuned, every size, every task, the fixed readout family.
- Report $I_{\mathcal V}$ and the encoder-minus-decoder gap per size.
- Split each CE gap into discrimination and calibration by refitting a
  temperature on the readout-selection split. A gap that is mostly calibration
  is reported as such.

Prediction: positive gap at small sizes, shrinking with size.

### U2 — Prequential codelength (primary)

Purpose: H2 and H3, and a single number for the low-data crossover seen in E2.

- Block boundaries at 0.5, 1, 2, 5, 10, 25, 50 and 100% of the training set.
- Two variants: frozen backbone with a trained readout, and full fine-tuning.
- The first block is coded with the `native` readout where it beats uniform.
- Report bits per label, and the per-block curve.

Prediction: shorter code for the encoder at small sizes, gap shrinking with
size. A decoder that wins the early blocks and loses the late ones, as in E2,
shows up directly in the per-block curve.

### U3 — Position and order

Purpose: test whether the causal constraint is the mechanism. Each model is
compared with itself, so "one checkpoint is simply better trained" cannot
explain an effect.

- **Prompt order.** Passage-then-question against question-then-passage, with
  everything else fixed. Measure the within-model change in $I_{\mathcal V}$.
- **Evidence position.** On tasks where the deciding sentence can be located or
  constructed, place it early, middle or late.
- **Input length.** Bin examples by token length; add distractor sentences to
  lengthen inputs without changing the label.
- **Where the information sits.** Linear probes at every layer and at a grid of
  token positions.

Prediction: the decoder is sensitive to order and to early evidence, and holds
decision information only in late tokens; the encoder is much less sensitive;
the difference shrinks with size.

### U4 — Attention mask against pretraining objective

Purpose: separate "bidirectional attention helps" from "masked-language
pretraining helps". This is E6 in `RDT.md`, promoted.

| Arm | Checkpoint | Mask during fine-tuning |
|---|---|---|
| Encoder, native | encoder | bidirectional |
| Encoder, causal | encoder | causal |
| Decoder, native | decoder | causal |
| Decoder, bidirectional | decoder | bidirectional |

Run at the smallest size, 68M and the largest size. Measure with U1 and U2.

Prediction if attention is the cause: the mask determines the ranking whichever
checkpoint is used. If pretraining is the cause, the checkpoint determines it.

### U5 — Exact-rate compression (secondary)

Purpose: the rate–distortion question, asked only where it is non-trivial.

- Many-way task only.
- Discrete code with $M \in \{2, 4, \dots, 1024\}$ symbols between the pooled
  representation and the head.
- Plot against the predicted-label code and quantized-logit codes from U0.
- Report the smallest $M$ that reaches within 0.02 bits of the uncompressed CE.

This is kept only if the learned code beats the baselines. It is expected to
add little beyond U1.

### U6 — Practical relevance (optional)

- **Equal cost.** Repeat the U1 headline comparison at matched inference latency
  and memory.
- **Distillation.** Distil one large teacher into both small models and measure
  the bits of teacher decision each student retains.
- **Robustness.** Paraphrases and label-preserving edits; report error overlap
  between the two models.

## 6. Statistics

- At least five fine-tuning seeds per cell, paired across models by data order
  and example.
- Hierarchical bootstrap over seeds and test examples, as in E3-A.
- **Primary endpoint:** encoder-minus-decoder prequential codelength in bits per
  label, per size (U2, full fine-tuning, best trained readout).
- **Test of H3:** slope of that difference against log parameter count, pooled
  over tasks with task as a random effect.
- **Equivalence margin:** 0.01 bits per label. Differences inside it are
  reported as no difference.
- All other comparisons are secondary and reported with intervals, without
  claims of significance.

## 7. Reading the outcomes

| Outcome | Evidence |
|---|---|
| Hypothesis supported | U2 favours the encoder at small sizes, the slope is negative, U3 shows decoder order-sensitivity that fades with size |
| Encoder better, not a size effect | U1 and U2 favour the encoder with a flat slope |
| Pretraining, not architecture | U4 ranking follows the checkpoint, not the mask |
| Readout artefact | Gap with trained readouts, none with `native` |
| Calibration artefact | Gap disappears after temperature refitting |
| Not supported | Differences inside the equivalence margin, or sign varies by task |

## 8. Execution order

1. **U0** on cached 68M features. Decides the fate of E3-B.
2. **U2 on BoolQ at 68M and 150M**, reusing E2 training. Validates the
   prequential pipeline against known results.
3. **Confirm the size list and tasks**, build the three-way splits.
4. **U1 and U2 full sweep.** This is the test of the hypothesis.
5. **U3 prompt order and evidence position** at the smallest size, 68M and the
   largest size.
6. **U4** mask swap.
7. **U5 and U6** only if steps 4–6 show an effect worth characterising.

## 9. What is reused and what is retired

- **Reused:** E0 data builds, E1/E1.5 probing code, E2 fine-tuning code and
  checkpoints, the E3-A hierarchical bootstrap, the figure pipeline.
- **Retired as primary evidence:** E3-A matched-rate curves and all E3-B curves.
  E3-A is kept as a robustness measurement pending U0.
- **Paused:** the E3-B expansion in `E3B_CORRECTED_SETUP.md`, pending U0.

## 10. Open items

- Published Ettin sizes and whether any pairs differ in width at similar depth.
- Final choice of the 3-way and many-way tasks, and their licences.
- Verbalizers for the `native` readout on many-way tasks.
- Whether fine-tuning a decoder with a bidirectional mask (U4) is stable at
  68M without a warm-up phase.
- Compute estimate for step 4: sizes × tasks × 8 blocks × 5 seeds × 2 models.

## 11. Information-theory map

| Area | Used for |
|---|---|
| Usable (V-) information | U1, U3 probes, readout comparisons |
| Minimum description length, prequential coding | U2, primary endpoint |
| Data-processing inequality, sufficient statistics | U3, U4 mechanism tests |
| Rate–distortion, information bottleneck | U0, U5 |
| Proper scoring rules | Calibration split in U1 |
