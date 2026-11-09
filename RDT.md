# Information-Efficient Decision Making: A Rate–Distortion Study of Ettin Encoder and Decoder Models on When2Call

**Document type:** Research experiment plan

**Models:** Ettin Encoder 150M and Ettin Decoder 150M

**Task:** When2Call four-way tool-use decision classification

**Primary question:** At a matched representation-information budget, does an encoder-based model achieve lower decision distortion than a decoder-based model?

## 1. Executive Summary

This study compares Ettin Encoder 150M and Ettin Decoder 150M on the When2Call decision-making task through the lens of rate–distortion theory. Rather than comparing accuracy alone, the study evaluates (1) the decision information already accessible in pretrained representations, (2) the impact of fine-tuning data and adaptation, and (3) the trade-off between the amount of information transmitted through a learned representation and downstream decision error.

The central hypothesis is that the encoder may achieve lower decision distortion at a given representation-information rate, particularly in a low-rate regime. This is a testable hypothesis, not an assumed property of encoders.

The proposed study contains six experiments:

1. **Frozen representation probing:** Compare the linear and nonlinear decodability of decision labels.
2. **Data scaling and full fine-tuning:** Compare task adaptation and attainable error as training data increases.
3. **Frozen-backbone variational information bottleneck (VIB):** Estimate empirical rate–distortion frontiers without changing the pretrained representations.
4. **End-to-end VIB fine-tuning:** Test whether architecture-specific adaptation changes those frontiers.
5. **Counterfactual logic generalization:** Determine whether the retained information supports robust decisions rather than superficial correlations.
6. **Attention-mask ablation:** Explore the role of causal attention, with appropriate caveats about confounding factors.

## 2. Motivation and Research Questions

Small decision models must allocate limited representational resources among linguistic structure, tool descriptions, task instructions, and decision-relevant rules. Two models with similar parameter counts may therefore differ in how efficiently they encode the information needed for a particular decision.

Ettin provides a useful comparison because its encoder and decoder variants have broadly matched model sizes and base architectural dimensions, while differing in attention masking and pretraining objectives (masked language modeling versus causal language modeling). However, the original checkpoints do not isolate attention directionality: the pretraining objective changes as well.

The study addresses four research questions:

- **RQ1 — Representation accessibility:** Which pretrained model exposes When2Call decision information more readily to a lightweight classification head?
- **RQ2 — Adaptation and sample efficiency:** Does either model achieve lower decision distortion under matched data and optimization budgets?
- **RQ3 — Rate–distortion efficiency:** For a controlled stochastic representation bottleneck, which model achieves lower decision cross-entropy at the same measured information-rate proxy?
- **RQ4 — Logical generalization:** Do observed differences persist under controlled changes to tool availability, parameter completeness, and answerability?

## 3. Task and Models

### 3.1 When2Call decision space

Represent an example by an input $X=(Q,T)$, where $Q$ is the user request and $T$ contains the tools actually available in the scenario. The target $Y$ belongs to one of four semantic classes:

| Label | Decision | Meaning |
|-------|----------|---------|
| A | `direct` | Answer directly without a tool call. |
| B | `tool_call` | Invoke an appropriate available tool. |
| C | `request_for_info` | Ask for missing information required to proceed. |
| D | `cannot_answer` | Indicate that the request cannot be fulfilled under the available constraints. |

Use semantic class IDs as the training labels. A/B/C/D letters should not be treated as the primary prediction tokens: token probabilities can introduce tokenizer- and pretraining-specific biases.

The previously identified When2Call release contains approximately 15,000 SFT examples, 9,000 preference examples, and 3,652 MCQ test examples. Confirm the version, split sizes, schema, and licensing before implementation. The preference examples are optional for this core classification study.

### 3.2 Model specifications

| Property | Ettin Encoder 150M | Ettin Decoder 150M |
|----------|-------------------|--------------------|
| Approximate parameter count | 150M | 150M |
| Transformer blocks | 22 | 22 |
| Hidden dimension | 768 | 768 |
| Attention heads | 12 | 12 |
| FFN intermediate dimension | 1152 | 1152 |
| Attention visibility | Bidirectional | Causal |
| Original pretraining objective | MLM | CLM |

**Model references**

- Encoder: <https://huggingface.co/jhu-clsp/ettin-encoder-150m>
- Decoder: <https://huggingface.co/jhu-clsp/ettin-decoder-150m>

Verify exact checkpoint metadata and tokenizer behavior in the model cards and model configs before training.

## 4. Theoretical Framework

### 4.1 Decision distortion

Use held-out decision cross-entropy in bits per decision as the principal distortion measure:

$$
D = \mathbb{E}_{(X,Y)}\left[-\log_2 q_\psi(Y\mid Z)\right],
$$

where $Z$ is the model's decision representation and $q_\psi$ is a four-class prediction head.

Cross-entropy is preferable to accuracy as the primary distortion measure because it also captures predictive confidence. Secondary metrics should include accuracy, macro-F1, per-class F1, expected calibration error (ECE), and tool-use-specific error rates.

If the prediction head is Bayes-optimal for the induced representation, then

$$
D_{\mathrm{opt}} = H(Y\mid Z) = H(Y) - I(Y;Z).
$$

With a learned, finite-capacity head, measured cross-entropy is instead an upper bound on $H(Y\mid Z)$; the identity does not generally hold for the observed test loss.

### 4.2 Three distinct notions of capacity

| Capacity notion | Operational variable | Purpose |
|-----------------|----------------------|---------|
| Parameter/storage capacity | $L(\theta)$, model description length | Match overall model scale; investigate storage efficiency in future extensions. |
| Representation-information rate | $I(X;Z)$ | Primary rate–distortion axis. |
| Computational budget | FLOPs or latency | Optional efficiency axis; **not** interchangeable with information rate. |

Parameter count is approximately controlled in this study, but it is not equivalent to mutual information. Likewise, training-set size is a data-scaling variable, not the information rate in rate–distortion theory.

### 4.3 Information-bottleneck formulation

The idealized task-specific rate–distortion function is

$$
D^*(R) = \min_{p(z\mid x),\,q(y\mid z):\,I(X;Z)\le R} \mathbb{E}\left[-\log_2 q(Y\mid Z)\right].
$$

For the two pretrained model families and the selected optimization procedure, the actual experiments estimate attainable empirical frontiers, not universal theoretical minima:

$$
\widehat D_{\mathrm{enc}}(R) \quad\text{and}\quad \widehat D_{\mathrm{dec}}(R).
$$

Primary hypothesis (H1): In the overlapping low-rate region,

$$
\widehat D_{\mathrm{enc}}(R) < \widehat D_{\mathrm{dec}}(R).
$$

A plausible alternative is that the difference is largely eliminated by full fine-tuning or by using sufficiently expressive readout heads.

## 5. Shared Experimental Protocol

### 5.1 Canonical input template

Use identical semantic content and formatting for both backbones, for example:

```
Available Tools:
{tool_definitions}

User Request:
{question}

Decision:
```

Extract the hidden state of a designated final decision position. The decoder can attend to the full preceding prompt at that position, while the encoder can attend bidirectionally. Use a consistent explicit decision marker if needed, and verify that model-specific special tokens do not alter content visibility.

### 5.2 Dataset preparation and leakage prevention

1. Convert SFT examples into the four semantic classes using verified source metadata or auditable labeling rules. The official SFT response format may not supply an unambiguous four-class ground-truth label for every example; ambiguous examples require manual adjudication or exclusion.
2. Construct stratified training and validation partitions from training data. Preserve the official test set as a final held-out evaluation set.
3. Group semantically identical or related examples before splitting to prevent paraphrase and counterfactual leakage.
4. Never include `correct_answer`, `target_tool`, ground-truth assistant responses, or other answer-bearing metadata in model inputs.
5. Preserve the actually available tools while avoiding accidental inclusion of unavailable/oracle tool descriptions.
6. Verify that neither model truncates required tool definitions or the user request; record truncation frequency and token lengths.
7. Inspect class frequencies and report macro-averaged metrics to avoid majority-class masking.

### 5.3 Optimization and statistical controls

- Use the same data partitions, class mapping, nominal sequence content, batch size, and update budget.
- Train heads with the same architecture and loss within each experiment.
- Allow independent learning-rate selection via validation, but allocate the same hyperparameter-search budget to each model.
- Run at least three independent random seeds; preferably five for key results.
- Report mean, standard deviation, and paired bootstrap confidence intervals on the common test examples.
- Track train/validation losses and class-level confusion matrices.
- Distinguish equal training steps, equal training FLOPs, and equal wall-clock budget; these are different fairness criteria.

## 6. Experiment E1 — Frozen Representation Probing

**Objective**

Measure how readily the pretrained representations support When2Call decisions without changing the backbone weights.

**Procedure**

1. Freeze both pretrained backbones.
2. Obtain the final decision-position hidden state $h\in\mathbb{R}^{768}$.
3. Fit a four-class linear probe:

$$
q_\psi(Y\mid X) = \operatorname{softmax}(Wh+b).
$$

4. Repeat using a small matched multilayer perceptron (MLP) probe.
5. Optionally probe intermediate layers, using the same layer indices in both models.

**Measurements**

- Held-out decision cross-entropy (bits/decision)
- Accuracy and macro-F1
- Performance by layer
- Linear-to-MLP probe gap

**Interpretation**

If the encoder performs better with a frozen linear probe, decision information is more linearly accessible from its pretrained representation. This does not, on its own, prove that the encoder has higher $I(Y;Z)$: a different nonlinear head may recover information already present in the decoder representation.

## 7. Experiment E2 — Full Fine-Tuning and Data Scaling

**Objective**

Compare adaptation efficiency and task-specific distortion at approximately matched model parameter counts.

**Procedure**

1. Attach an identical four-class decision head to each backbone.
2. Fine-tune all trainable model parameters using supervised cross-entropy.
3. Repeat for nested, stratified fractions of training data:

$$
N \in \{1\%,\,5\%,\,10\%,\,25\%,\,50\%,\,100\%\}.
$$

4. Use matched validation-selection and stopping policies.
5. Fit an optional empirical data-scaling relationship:

$$
D(N) \approx D_\infty + aN^{-\alpha}.
$$

**Measurements**

- Decision CE, accuracy, macro-F1 against training-set size
- Learning curves versus optimization steps and (if feasible) FLOPs
- Class-specific errors and calibration

**Interpretation**

- **Early encoder advantage that disappears:** Supports a sample-efficiency or initialization advantage.
- **Persistent encoder advantage:** Suggests a task-specific difference in attainable distortion under the tested conditions.
- **Similar in-distribution performance but different transfer:** Points toward differences in robustness or decision-rule representation rather than simple fit quality.

**Important:** $D(N)$ is a data-scaling curve, not a formal rate–distortion curve: $N$ is not $I(X;Z)$.

## 8. Experiment E3 — Frozen-Backbone Variational Information Bottleneck

**Objective**

Construct controlled empirical distortion-versus-information-rate curves for pretrained representations.

**Method**

For each frozen backbone, pass its hidden state $h=f_\theta(X)$ through a stochastic bottleneck:

$$
q_\phi(z\mid h) = \mathcal{N}\left(\mu_\phi(h), \operatorname{diag}(\sigma^2_\phi(h))\right),
$$

$$
z = \mu_\phi(h) + \sigma_\phi(h)\odot\epsilon, \qquad \epsilon\sim\mathcal{N}(0,I).
$$

Predict the semantic decision using

$$
q_\psi(Y\mid Z) = \operatorname{softmax}(Wz+b).
$$

Optimize the VIB objective in bit units:

$$
\boxed{\mathcal{L}_{\mathrm{VIB}} = D + \beta R_{\mathrm{upper}}}
$$

with

$$
D = \mathbb{E}\left[-\log_2 q_\psi(Y\mid Z)\right],
$$

$$
R_{\mathrm{upper}} = \frac{1}{\ln 2}\mathbb{E}_X\left[D_{\mathrm{KL}}(q_\phi(Z\mid X)\parallel p(Z))\right].
$$

For a fixed prior $p(Z)$,

$$
\mathbb{E}_X D_{\mathrm{KL}}(q(Z\mid X)\parallel p(Z)) = I(X;Z) + D_{\mathrm{KL}}(q(Z)\parallel p(Z)),
$$

and thus $I(X;Z)\le R_{\mathrm{upper}}$.

**Measurement caveat:** The KL term is an upper bound, not an exact estimate, of $I(X;Z)$. Its looseness may differ between models. Compare matched observed rate ranges, and consider a learned prior or separate MI diagnostics to assess this issue.

**Initial hyperparameter grid**

| Hyperparameter | Initial setting |
|----------------|-----------------|
| Bottleneck latent dimension | 64 |
| Prior | Standard Gaussian |
| $\beta$ | 0, 0.001, 0.01, 0.1, 1 |
| Backbone | Frozen |
| Bottleneck and readout | Trainable |
| Random seeds | At least 3 |
| Model selection | Validation only |

The $\beta$ grid is only a starting point. Expand it adaptively so both models have adequate coverage of overlapping measured rates, including the low-rate regime.

**Primary output**

Plot test decision CE ($D$) against measured $R_{\mathrm{upper}}$. Each point corresponds to a trained $\beta$ setting. Show seeds/uncertainty bands and use rate-matched interpolation only within the observed overlap.

The core comparison is

$$
\Delta D(R) = D_{\mathrm{dec}}(R) - D_{\mathrm{enc}}(R).
$$

Positive $\Delta D(R)$ implies an empirical encoder advantage at that measured rate, subject to the variational-bound caveat.

## 9. Experiment E4 — End-to-End VIB Fine-Tuning

**Objective**

Test whether the relative rate–distortion curves change when the backbone is allowed to adapt to the downstream decision task.

**Procedure**

Repeat E3 with both the backbone and the stochastic bottleneck trainable:

$$
\min_{\theta,\phi,\psi}\left[D(\theta,\phi,\psi) + \beta R_{\mathrm{upper}}(\theta,\phi)\right].
$$

Use the same rate coverage, validation rules, seed count, and evaluation metrics as E3. Compare E3 versus E4 within each model in addition to comparing encoder versus decoder.

**Interpretation**

- If the encoder leads only in E3, the main advantage may be tied to pretrained feature accessibility.
- If the encoder also leads in E4, the tested model family may have better task-specific representation efficiency under adaptation.
- If the curves converge, fine-tuning may compensate for pretrained representational differences.

A full fine-tuning result still cannot, by itself, isolate the effect of the attention mask from the effect of pretraining objective.

## 10. Experiment E5 — Counterfactual Logic Generalization

**Objective**

Test whether lower distortion corresponds to robust understanding of decision rules rather than reliance on surface cues or class priors.

**Controlled factors**

| Logic factor | Controlled contrast |
|--------------|---------------------|
| Tool availability | Required tool present vs. absent |
| Required parameters | All arguments provided vs. a required argument missing |
| Tool relevance | Matching tool vs. nonmatching tool |
| Answerability | Directly answerable vs. requires external information or an unavailable action |

**Example**

Assume an environment where the only supported booking tool is

```
book_flight(origin, destination, date)
```

and there are no alternative booking channels.

| User request / tool state | Expected decision |
|---------------------------|-------------------|
| "Book a flight from Dallas to NYC on Friday." Tool available; all parameters supplied. | `tool_call` |
| "Book a flight from Dallas to NYC." Tool available; date missing. | `request_for_info` |
| "Book a flight from Dallas to NYC on Friday." No booking tool available. | `cannot_answer` |

These labels are scenario-dependent; examples should be validated against the actual task instructions and permitted fallback behavior.

**Metrics**

Report per-example decision metrics together with pairwise counterfactual consistency:

$$
\operatorname{PairConsistency} = \Pr(\hat Y_1 = Y_1 \ \land\ \hat Y_2 = Y_2).
$$

Additionally report the success rate on cases where the correct label must change after a minimal intervention, and the false-tool-call rate when tools are absent or irrelevant.

Construct counterfactual families separately from the training examples and keep members of each family in the same split.

## 11. Experiment E6 — Attention-Mask Ablation

**Objective**

Investigate whether downstream bidirectional attention contributes to the observed differences.

**Proposed comparison**

Compare the original causal Ettin Decoder 150M against a decoder variant with a bidirectional attention mask during downstream adaptation, holding the downstream data, classification head, and optimization budget as close as possible.

**Cautions**

Changing the mask at fine-tuning time creates a pretraining–fine-tuning distribution shift; it is not a clean causal test of attention directionality. Any results should be described as a downstream mask intervention. A stronger causal comparison would require matched pretraining experiments that separately vary attention masking and prediction objective, which is a substantially larger project.

## 12. Evaluation and Analysis Plan

### 12.1 Required result tables

**Table A — Frozen-probe and fine-tuning results**

| Model | Training regime | CE (bits/decision) | Accuracy | Macro-F1 | ECE |
|-------|-----------------|-------------------:|---------:|---------:|----:|
| Encoder 150M | Frozen linear probe | TBD | TBD | TBD | TBD |
| Decoder 150M | Frozen linear probe | TBD | TBD | TBD | TBD |
| Encoder 150M | Full fine-tuning | TBD | TBD | TBD | TBD |
| Decoder 150M | Full fine-tuning | TBD | TBD | TBD | TBD |

**Table B — Rate–distortion comparison**

| Training regime | Matched rate ($R_{\mathrm{upper}}$) | Encoder CE | Decoder CE | $\Delta D$ | 95% CI |
|-----------------|------------------------------------:|-----------:|-----------:|-----------:|--------|
| Frozen VIB | TBD | TBD | TBD | TBD | TBD |
| End-to-end VIB | TBD | TBD | TBD | TBD | TBD |

### 12.2 Required figures

1. Frozen linear/MLP probe performance by model and layer.
2. CE and macro-F1 versus number of fine-tuning examples.
3. Frozen-backbone $D$ versus $R_{\mathrm{upper}}$ with uncertainty bands.
4. Fine-tuned $D$ versus $R_{\mathrm{upper}}$ with uncertainty bands.
5. Rate-matched $\Delta D(R)$, restricted to overlapping empirical rate ranges.
6. Per-class confusion matrices and counterfactual consistency by logic factor.

### 12.3 Optional task-information efficiency analysis

A useful conceptual measure is

$$
\eta_{\mathrm{task}} = \frac{I(Y;Z)}{I(X;Z)}.
$$

The lower bound

$$
I(Y;Z) \ge H(Y) - D
$$

holds when $D$ is held-out cross-entropy measured under the same data distribution (up to estimation uncertainty). However, dividing this lower bound by the KL upper bound on $I(X;Z)$ gives, at best, a conservative proxy with potentially substantial estimation error. Treat such ratios as supplementary diagnostics, not definitive estimates of true mutual-information efficiency.

## 13. Threats to Validity

1. **Architecture–objective confounding:** Encoder/decoder checkpoints differ in both attention masking and pretraining objective.
2. **Variational rate looseness:** KL-to-prior may substantially overestimate $I(X;Z)$, potentially by different amounts.
3. **Label ambiguity:** SFT responses may not uniquely determine one of the four semantic decisions.
4. **Benchmark-specific artifacts:** Model performance may reflect class priors, tool templates, or superficial wording patterns.
5. **Input visibility and truncation:** Encoder and decoder must receive the same semantic context; tokenization differences can complicate direct sequence-length comparisons.
6. **Optimization fairness:** Equal epochs do not imply equal computational effort or equal hyperparameter-search opportunity.
7. **Limited generality:** Results for two 150M checkpoints and one benchmark should not be generalized to all encoder and decoder models.
8. **Finite-model frontiers:** Observed curves are empirically attained points, not proofs of global rate–distortion optima.

## 14. Recommended Execution Order

| Phase | Experiment | Deliverable | Priority |
|-------|------------|-------------|----------|
| 0 | Data audit and canonical decision-label pipeline | Reproducible, leakage-checked dataset | Essential |
| 1 | E1 — Frozen representation probe | Linear/MLP probing baseline | Essential |
| 2 | E2 — Full fine-tuning and data scaling | Supervised baselines and $D(N)$ | Essential |
| 3 | E3 — Frozen VIB | Initial empirical $D(R)$ curves | Essential |
| 4 | E4 — End-to-end VIB | Adapted empirical $D(R)$ curves | High |
| 5 | E5 — Counterfactual tests | Robustness and logic-transfer findings | High |
| 6 | E6 — Mask intervention | Exploratory mechanistic evidence | Optional |

**Minimum viable study**

The smallest convincing first study would include:

- Both official Ettin 150M checkpoints and a matched four-way semantic classifier.
- An audited When2Call SFT-to-classification training pipeline and an untouched MCQ test set.
- Three seeds for frozen probes and full fine-tuning.
- A 64-dimensional Gaussian VIB with an initial five-value $\beta$ sweep, expanded for rate overlap.
- Decision CE, macro-F1, confusion matrices, and paired uncertainty intervals.
- At least one hand-validated counterfactual set for tool availability and argument completeness.

## 15. Expected Outcomes and Interpretation

**Outcome A — Frozen-probe encoder advantage, convergence after full fine-tuning.**

*Interpretation:* Pretrained encoder representations offer more accessible task features, but adaptation largely closes the gap.

**Outcome B — Encoder advantage concentrated at low representation rates.**

*Interpretation:* Supports the hypothesis that, for this decision task and training setup, the encoder achieves better compression of decision-relevant information.

**Outcome C — Similar rate–distortion curves but different out-of-distribution behavior.**

*Interpretation:* A scalar information-rate measure may not capture how information is structured and used for generalizable decision rules.

**Outcome D — Decoder matches or outperforms the encoder.**

*Interpretation:* Rejects the central hypothesis under the tested conditions and motivates investigation of training objective, representation readout, and benchmark-specific effects.

## 16. Proposed Research Claim

The study should aim for a carefully scoped conclusion:

> **At matched model size and under a controlled stochastic representation bottleneck, encoder- and decoder-pretrained language models may exhibit different empirical decision rate–distortion frontiers. Measuring these frontiers can help disentangle pretrained representation accessibility, downstream adaptation, and the amount of task-relevant information retained at a given rate.**

The strongest possible evidence would be a statistically supported encoder advantage across an overlapping low-rate range, replicated with multiple seeds and accompanied by improved performance on counterfactual logic tests.

## References and Implementation Starting Points

- **Ettin Encoder 150M:** <https://huggingface.co/jhu-clsp/ettin-encoder-150m>
- **Ettin Decoder 150M:** <https://huggingface.co/jhu-clsp/ettin-decoder-150m>
- **When2Call:** Use the official benchmark repository, dataset card, and evaluation protocol; verify the specific release and licensing before implementation.
- **Foundational concepts:** rate–distortion theory, information bottleneck, variational information bottleneck, and empirical scaling-law analysis.
