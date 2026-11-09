# E3-B budgeted D(R) curves — pilot

Hard per-example budget projection on sparse token gates. Total rate is capped
at B bits/example. Frozen E2 last-L19 checkpoints:
encoder E2 seed 1, decoder E2 seed 2. 3 VIB seeds. Expected Stochastic CE.

## D(R) curves

| Budget B | Encoder CE | Decoder CE | ΔD = dec − enc [95% CI] |
|---:|---:|---:|---|
| 100 | 0.9808 | 0.9777 | −0.0031 [−0.0067, +0.0006] |
| 200 | 0.9767 | 0.9686 | −0.0081 [−0.0236, +0.0075] |
| 400 | 0.9694 | 0.9507 | −0.0187 [−0.0390, +0.0017] |
| 800 | 0.9742 | 0.9106 | −0.0636 [−0.1541, +0.0269] |
| 1600 | 0.9521 | 0.8719 | **−0.0802** [−0.1347, −0.0256] |
| 3200 | 0.9450 | 0.8421 | **−0.1029** [−0.1840, −0.0217] |

Measured rates equal the budgets exactly (hard projection).

Controls:
- high-rate deterministic last-token control: encoder 0.634–0.646,
  decoder 0.750–0.753;
- forced prior zero-rate control: CE 0.9568.

## Reading

- The decoder has the lower distortion at every budget.
- The difference grows with rate: small at 100–200 bits, clear and significant
  at 1600–3200 bits.
- The encoder curve is non-monotonic at 400→800 bits, so the raw point is kept.
- At 100–800 bits both models are close to the prior; these budgets are too small
  to expose much task information.

## Caveat

This is a one-E2-checkpoint-per-architecture pilot with 3 VIB seeds. It is not
the five-E2-seed validation. The result should be repeated across E2 seeds
before a final claim.
