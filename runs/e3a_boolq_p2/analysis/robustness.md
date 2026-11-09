# E3-A robustness summary

## Ablation summary at matched validation rate

Expected CE; ΔD = decoder − encoder (positive favors encoder); Δpenalty = decoder penalty − encoder penalty.

| Run | R | ΔD | ΔD 95% CI | Δpenalty | Δpenalty 95% CI |
|---|---:|---:|---|---:|---|
| P2 main (latent 32, global scaling) | 2.0 | +0.0734 | [+0.0371, +0.1110] | +0.0022 | [-0.0331, +0.0297] |
| P2 main (latent 32, global scaling) | 4.0 | +0.0679 | [+0.0248, +0.1056] | -0.0033 | [-0.0234, +0.0165] |
| 5 VIB seeds | 2.0 | +0.0733 | [+0.0383, +0.1112] | +0.0022 | [-0.0336, +0.0306] |
| 5 VIB seeds | 4.0 | +0.0689 | [+0.0273, +0.1075] | -0.0022 | [-0.0229, +0.0169] |
| Latent 128 | 2.0 | +0.1763 | [+0.1064, +0.2415] | +0.1051 | [+0.0100, +0.1717] |
| Latent 128 | 4.0 | +0.1238 | [+0.0579, +0.1909] | +0.0527 | [-0.0315, +0.1086] |
| No feature scaling | 2.0 | +0.0313 | [-0.0087, +0.0681] | +0.0005 | [-0.0231, +0.0302] |
| No feature scaling | 4.0 | +0.0435 | [-0.0022, +0.0881] | +0.0127 | [-0.0095, +0.0442] |
| Per-dimension z-score | 2.0 | +0.0527 | [+0.0009, +0.1016] | +0.0302 | [-0.0264, +0.0962] |
| Per-dimension z-score | 4.0 | +0.0611 | [+0.0086, +0.1157] | +0.0386 | [-0.0228, +0.1020] |

## Per-E2-seed and leave-one-out stability (P2 main)

| Target R | Per-seed ΔD | Mean | LOO mean range |
|---:|---|---:|---|
| 1.0 | +0.0440, +0.1417, +0.0656, +0.0712, +0.1014 | +0.0848 | [+0.0706, +0.0950] |
| 2.0 | +0.0733, +0.1250, +0.0463, +0.0373, +0.0850 | +0.0734 | [+0.0605, +0.0824] |
| 3.0 | +0.0920, +0.1172, +0.0330, +0.0232, +0.0782 | +0.0688 | [+0.0566, +0.0801] |
| 4.0 | +0.0963, +0.1145, +0.0304, +0.0206, +0.0775 | +0.0679 | [+0.0562, +0.0797] |
| 5.0 | +0.0968, +0.1246, +0.0363, +0.0277, +0.0820 | +0.0735 | [+0.0607, +0.0849] |
| 6.0 | +0.1002, +0.1364, +0.0375, +0.0316, +0.0946 | +0.0801 | [+0.0660, +0.0922] |
| 8.0 | +0.1076, +0.1583, +0.0488, +0.0531, +0.1217 | +0.0979 | [+0.0828, +0.1102] |

## Rate interaction

Mean ΔD(8 bits) − ΔD(1 bit) = **+0.0131** (per-seed: +0.0636, +0.0166, -0.0168, -0.0181, +0.0203).

A value near zero means the encoder/decoder gap is roughly constant across rate; a positive value would indicate a larger encoder advantage at high rate.

## Sanity checks

| Backbone | Deterministic CE | β=0 CE | β=10 max rate | β=10 max CE |
|---|---:|---:|---:|---:|
| encoder | 0.6890 | 0.6918 | 0.0153 | 0.9608 |
| decoder | 0.7601 | 0.7635 | 0.0495 | 0.9617 |
