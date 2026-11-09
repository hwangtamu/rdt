> **Superseded by [`E3B_CORRECTED_SETUP.md`](E3B_CORRECTED_SETUP.md).** This
> controller and its follow-up continuous-gate budget curves do not implement a
> valid information-rate constraint. Retain this file only as debugging history.

# E3-B budgeted token VIB — pilot result

Implementation: `src/w2c/train_e3b_budget.py`
Sparse token gates `g_t`, total rate `R = Σ_t g_t KL_t`, adaptive λ on
`R − target`. Frozen E2 decoder checkpoint seed 2, token latent 32, batch 2048.

## Target 5 bits/example (single seed)

| Quantity | Result |
|---|---:|
| Best epoch | 1 |
| Validation CE | 0.9385 |
| Validation rate | 6990.8 bits/example |
| Test CE | 0.9385 |
| Test rate | 7055.7 bits/example |
| λ final | 8973 |

The budget controller failed: λ grew very large, but the measured rate stayed at
~7000 bits/example and CE stayed near the prior. The gate did not collapse to a
budgeted solution.

## Interpretation

- The previous total-rate correction is confirmed: E3-B total rates are in the
  thousands of bits/example for the current token VIB.
- A soft gate plus a rate-target dual is not sufficient to reach 1–5 total
  bits/example.
- 1–5 bits/example is an extremely small budget for ~160 tokens (≈0.006–0.03
  bits/token); the current parameterization has no stable solution there.
- No High-Rate/low-rate matched pilot should be run with this controller.

## Next design options

1. Hard budget projection: after each forward, rescale gates so `R ≤ B` exactly,
   then optimize CE; this makes the rate constraint non-negotiable.
2. Top-k token budget: allow only k tokens to carry information, fix the rest at
   the prior; vary k/latent to realize budget.
3. Reconsider the budget definition: if the intended quantity is average
   bits/token, say so explicitly; 1–5 total bits/example is effectively a
   near-zero-information regime.

The current soft-gate + dual approach is rejected.
