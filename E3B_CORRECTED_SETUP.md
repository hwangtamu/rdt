# E3-B corrected experiment setup — pre-pooling token VIB

## Status and scope

This document replaces the experimental protocol implicit in
`E3B_PILOT_RESULTS.md`, `E3B_DIAG_RESULTS.md`, `E3B_BUDGET_RESULTS.md`, and
`E3B_BUDGET_CURVES.md`. Those files remain useful as a record of diagnostics,
but their quantitative encoder-versus-decoder comparisons are not valid E3-B
rate–distortion results.

The corrected E3-B asks:

> After controlling the stochastic channel, readout, total rate, backbone
> checkpoint, and optimization budget, do the encoder and decoder differ in
> how efficiently token-level states support the BoolQ decision?

E3-B is deliberately distinct from E3-A. E3-A compresses one already pooled
decision vector. E3-B places the bottleneck before token pooling and therefore
tests whether decision information is concentrated in a small number of token
states or distributed across the sequence.

## Why the previous budget curves are rejected

The previous budget implementation samples

$$
z_t \sim \mathcal N(\mu_t, \operatorname{diag}(\sigma_t^2))
$$

and passes the continuously scaled value

$$
z'_t = g_t z_t
$$

to the readout, but reports

$$
R_{\mathrm{old}} = \sum_t g_t
D_{\mathrm{KL}}\left(q(z_t\mid h_t)\parallel\mathcal N(0,I)\right).
$$

This is not the KL of the transmitted random variable. Conditional on $g_t$,

$$
z'_t \sim \mathcal N(g_t\mu_t,
                     \operatorname{diag}(g_t^2\sigma_t^2)),
$$

whose KL to a standard Gaussian is not $g_t$ times the original KL. In
addition, the deterministic, input-dependent gate $g_t(h_t)$ is an uncharged
side channel. Uniformly rescaling all gates to make the reported rate equal a
target makes budget satisfaction true by construction without establishing an
information constraint.

The old implementation also used attention pooling while citing a last-token
deterministic control. A high-rate endpoint from a different information path
does not validate the curve. Consequently, the reported 100–3200 bit budget
curves must not be interpreted as evidence that the decoder has a better
token-level rate–distortion frontier.

## Primary stochastic channel

The primary E3-B experiment uses no token gate. For each fixed token position
$t\in\{1,\ldots,T\}$,

$$
q_\phi(z_t\mid h_t)
=\mathcal N\left(\mu_\phi(h_t),
                 \operatorname{diag}(\sigma_\phi^2(h_t))\right),
$$

$$
z_t=\mu_\phi(h_t)+\sigma_\phi(h_t)\odot\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
$$

The measured rate is the exact variational upper bound implemented by this
channel:

$$
R_{\mathrm{total}}
=\frac{1}{\ln 2}\mathbb E_X
\left[\sum_{t=1}^{T}
D_{\mathrm{KL}}\left(q_\phi(z_t\mid h_t)\parallel
\mathcal N(0,I)\right)\right]
\quad\text{bits/example}.
$$

Padding positions must use $q(z_t\mid h_t)=\mathcal N(0,I)$ and therefore
contribute exactly zero KL. Total bits/example is the primary rate. Average
bits/valid-token may be reported as a length diagnostic, but models must never
be matched on the average while conclusions are stated about total rate.

Do not force the E3-A range of 1–8 bits onto E3-B. A token-level channel has
many more latent variables, so its useful overlap may be hundreds or thousands
of bits/example. The observed shared range, not a preselected numerical range,
defines the comparison.

## Preventing a length or mask side channel

At zero rate, the downstream readout must receive a distribution independent
of the input. A masked attention pool over only valid positions can leak token
length even when every $z_t$ is sampled from the prior, because the number of
pooled random variables changes with the example.

For attention-pooling arms, use a fixed support of `max_length` positions:

- valid positions use their learned posterior;
- padding positions produce independent standard-prior samples and zero KL;
- the downstream attention pool does not receive the input-dependent padding
  mask.

As an alternative sensitivity analysis, length may be declared public side
information, but the estimand must then be stated as conditional on length. It
must not be mixed with the primary unconditional-rate result.

For a last-token arm, the downstream classifier receives only the selected
latent and not its position index. The forced-prior test must verify empirically
that predictions are independent of input and match the train-label prior.

## Readout arms and matched controls

Readout is a prespecified experimental factor, not an implementation detail.
Run the following arms separately and never interpolate or average across them:

1. `last`: select the last valid token latent;
2. `attention_fixed_support`: learned single-query attention over the fixed
   support described above.

The `last` arm measures whether decision information has already been
concentrated at the final token. The attention arm measures the efficiency of a
distributed token representation. Neither is universally privileged, so both
must be reported.

Every stochastic curve needs three architecture-matched controls using the
same readout:

1. **Raw-token control:** no token projection or noise; train only the specified
   pooling/readout and classifier.
2. **Deterministic bottleneck control:** use `z_t = mu(h_t)` with the same latent
   dimension and pooling path, but no sampling or KL penalty.
3. **High-rate stochastic control:** train with beta = 0 and require expected
   stochastic CE to recover the deterministic bottleneck within 0.02 bits for
   every included E2 checkpoint and VIB seed.

The forced-prior zero-rate control sets every posterior to the standard prior.
Its CE must match the train-prior predictor within Monte Carlo error, and its
predictions must be invariant to shuffled inputs. A learned high-beta model is
not a substitute for this forced control.

## Feature preprocessing and tokenization

- Freeze the E2 backbone before extracting token states.
- Use paired E2 `last-L19`, `N100` checkpoints with seeds 0–4 for both
  backbones. Encoder seed `s` is compared with decoder seed `s`; do not compare
  one hand-selected encoder seed with a different decoder seed.
- Use identical BoolQ records, split IDs, prompt strings, and maximum character
  content.
- Tokenizers may produce different sequence lengths. Record the full length,
  truncated length, and truncation indicator for every example and backbone.
  Any nonzero truncation rate blocks the run.
- Apply train-only global scalar standardization to token states separately for
  each `(backbone, E2 seed)`, using the same rule as corrected E3-A. Save the
  scalar mean and standard deviation with the run.
- Use latent dimension 32 as the primary condition and 128 as a prespecified
  sensitivity analysis.
- Cache keys must include the checkpoint hash, tokenizer identity, ordered
  example IDs, prompt/template hash, max length, layer/readout source, and
  preprocessing mode. A checkpoint basename alone is not a valid cache key.

## Training and checkpoint selection

For beta-sweep runs, optimize in bit units:

$$
\mathcal L = D + \beta R_{\mathrm{total}},
\qquad
D=\mathbb E[-\log_2 q_\psi(Y\mid Z_{1:T})].
$$

Required settings:

- batch size 2048, unless a documented memory constraint requires gradient
  accumulation to the same effective batch size;
- at least 3 VIB seeds nested within each of 5 E2 seeds;
- Monte Carlo samples: at least 4 for training and 64 for final evaluation;
- identical optimizer, learning-rate grid, epoch cap, patience, and search
  budget for both backbones;
- checkpoint selection by validation `CE + beta * R_total`, never validation CE
  alone;
- save and restore every trained component, including pooling query, gate if a
  later valid discrete-gate arm is added, normalization metadata, and the exact
  pooling name.

Start with a logarithmic beta pilot on one paired E2 seed. Expand the grid until
each architecture has smooth coverage of a substantial shared validation-rate
range, including a forced zero-rate point and a beta = 0 high-rate endpoint.
Only then run all five E2 seeds. Do not extrapolate outside an individual
`(E2 seed, VIB seed, readout)` curve.

## Evaluation and matched-rate analysis

The primary distortion is expected stochastic test CE:

$$
D=\mathbb E_{q(Z_{1:T}\mid X)}[-\log_2 q(Y\mid Z_{1:T})].
$$

Mean-only CE is a diagnostic and must not replace the stochastic result.

For each readout and latent dimension:

1. select checkpoints and construct curves using validation data only;
2. match encoder and decoder on validation `R_total`;
3. interpolate per `(E2 seed, VIB seed)` curve only within observed coverage;
4. apply the same interpolation weights to per-example test CE;
5. report absolute distortion and compression penalty

$$
P_m(R)=D_m(R)-D_m^{\mathrm{det}},
$$

where the deterministic reference uses the same checkpoint, latent dimension,
and readout path;
6. estimate paired 95% intervals with a hierarchical bootstrap over paired E2
   seeds, VIB seeds nested within E2 seed, and test examples.

Report both

$$
\Delta D(R)=D_{\mathrm{dec}}(R)-D_{\mathrm{enc}}(R)
$$

and

$$
\Delta P(R)=P_{\mathrm{dec}}(R)-P_{\mathrm{enc}}(R).
$$

A difference in absolute $D(R)$ with no difference in $P(R)$ is a baseline
representation/readout difference, not evidence of architecture-specific
compression efficiency.

## Required artifacts

Each run must save:

- exact resolved config and git commit;
- checkpoint and tokenizer identifiers/hashes;
- split IDs, token-length and truncation audit;
- preprocessing statistics;
- per-epoch train/validation CE, rate, and objective;
- complete best-checkpoint state;
- validation and test total rates;
- per-example expected and mean-only test CE with ordered IDs;
- raw beta points and the matched-rate interpolation table.

The analysis script, aggregate CSV files, and generated Markdown report must be
tracked. Large checkpoints and prediction arrays may remain external, but the
report must include immutable paths or checksums sufficient to audit them.

## Mandatory tests and acceptance gates

E3-B is not ready for an architecture comparison until all of the following
pass:

1. **Analytic KL test:** the implementation equals the closed-form diagonal
   Gaussian KL, and padding positions contribute zero.
2. **Sum-not-mean test:** doubling two identical valid token channels doubles
   total rate.
3. **Identity-path test:** an identity deterministic bottleneck with matched
   pooling produces the same logits as its raw-token control.
4. **Checkpoint restoration test:** reloading a saved checkpoint reproduces
   logits, CE, and rate exactly and restores every pooling parameter.
5. **Zero-rate independence test:** forced-prior predictions do not change
   after permuting inputs or lengths and match the label-prior CE.
6. **High-rate recovery test:** beta = 0 is within 0.02 CE bits of the matched
   deterministic bottleneck for every included seed.
7. **Rate accounting test:** evaluation rate is example-weighted, independent
   of evaluation batch partitioning, and reported in bits/example.
8. **Coverage test:** every reported target rate is covered by both
   architectures for all included paired E2/VIB cells; otherwise show the
   reduced cell count and do not call it the primary result.
9. **No-test-selection test:** no beta, epoch, readout, or target-rate choice is
   made from test CE.

Failure of gates 1–7 blocks all E3-B quantitative conclusions. Failure of gate
8 limits the analysis to explicitly labeled exploratory points.

## Optional sparse-token experiment

Sparse token selection is a separate secondary experiment. It must not reuse
the rejected continuous-amplitude gate.

A valid stochastic gate must define a discrete random variable $G_t$, replace
an inactive token with a prior sample rather than a deterministic zero, and
include the gate-selection cost in the rate, for example

$$
R = D_{\mathrm{KL}}(q(G\mid H)\parallel p(G))
  + \mathbb E_{q(G\mid H)}
    \left[\sum_t G_t
    D_{\mathrm{KL}}(q(Z_t\mid h_t)\parallel p(Z_t))\right].
$$

A deterministic top-k selector may be useful as a mechanism ablation, but its
selected indices are an uncharged message unless their coding cost is included.
It must not be presented as a VIB rate–distortion curve.

## Interpretation rules

The corrected experiment can support only scoped statements about frozen E2
BoolQ representations under the specified token channel and readout.

- Encoder advantage in both `last` and attention arms would support a broadly
  better token representation under this variational bound.
- Decoder advantage only in `last` would support greater concentration of
  decision information at the final token.
- Encoder advantage after pooling (E3-A) but decoder advantage before pooling
  in the corrected attention arm would be consistent with distributed encoder
  information that is useful after aggregation but costly to preserve token by
  token.
- Similar compression penalties with different absolute distortions would
  again indicate a baseline representation/readout difference, not a different
  rate–distortion response.

None of these outcomes isolates causal attention from pretraining objective,
tokenization, or other checkpoint differences. The variational KL remains an
upper bound on mutual information, and its looseness may differ between model
families.
