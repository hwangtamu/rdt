# E2 BoolQ — Ettin-68M encoder vs decoder

Run: `runs/e2_boolq_68m/{encoder,decoder}`
Config: `configs/e2_boolq_68m.json`
Readouts: `attn-L12` (decoder E1.5 winner) and `attn-L18` (encoder E1.5 winner).
Seeds 0–4; fractions 5/10/25/50/100 %; fp32.

## Mean test CE / accuracy, 5 seeds

| N | Encoder `attn-L12` | Encoder `attn-L18` | Decoder `attn-L12` | Decoder `attn-L18` |
|---:|---|---|---|---|
| 5 % | **0.953 / .618** | 1.052 / .570 | 0.974 / .602 | 1.023 / .587 |
| 10 % | **0.962 / .622** | 0.986 / .617 | 0.968 / .618 | 0.995 / .611 |
| 25 % | 0.922 / .641 | 0.930 / .646 | **0.908 / .653** | 0.933 / .645 |
| 50 % | 0.869 / .682 | **0.852 / .697** | 0.885 / .675 | 0.891 / .660 |
| 100 % | 0.775 / .744 | **0.740 / .760** | 0.834 / .724 | 0.838 / .710 |

## Encoder vs decoder at matched readouts

ΔD = CE_decoder − CE_encoder; negative favors decoder.

| Readout | N | ΔD [95 % CI] | Winner |
|---|---:|---|---|
| `attn-L12` | 5 % | +0.0211 [+0.0142, +0.0280] | encoder |
| `attn-L12` | 10 % | +0.0065 [−0.0039, +0.0155] | tie |
| `attn-L12` | 25 % | −0.0140 [−0.0263, −0.0024] | **decoder** |
| `attn-L12` | 50 % | +0.0157 [+0.0013, +0.0289] | encoder |
| `attn-L12` | 100 % | +0.0583 [+0.0387, +0.0786] | encoder |
| `attn-L18` | 5 % | −0.0293 [−0.0469, −0.0123] | decoder |
| `attn-L18` | 10 % | +0.0090 [−0.0041, +0.0218] | tie |
| `attn-L18` | 25 % | +0.0030 [−0.0124, +0.0185] | tie |
| `attn-L18` | 50 % | +0.0385 [+0.0162, +0.0605] | encoder |
| `attn-L18` | 100 % | +0.0983 [+0.0706, +0.1254] | encoder |

## Reading

1. **The 68M encoder wins at full data**, especially with `attn-L18`:
   CE 0.740, accuracy 0.760.
2. **The decoder is competitive only at low/mid data** and only at `attn-L12`;
   at 100 % it is 0.058–0.098 bits worse than the encoder.
3. **Layer 18 attention is a full-data readout.** At low N it is worse than L12;
   it overtakes at 50 % and is clearly best at 100 %.
4. **The 68M encoder matches the 150M encoder’s full-data E2 result**
   (both ~0.740 CE), while the 68M decoder is worse than the 150M decoder
   (~0.834 vs ~0.728). Capacity helps the decoder arm more in this setup.
5. Readout choice still matters: the winning layer changes with N and backbone.

## Caveat

These are BoolQ results with attention pooling and fp32. The 150M E2 comparison
used `lasth-L17` / `last-L22`, so the cross-capacity comparison is not
readout-matched. Within 68M, the encoder/decoder comparison is matched on
`attn-L12` and `attn-L18`.

## Addendum: E2 `last-L19`

Added after the attention sweep: 5 seeds, 5 fractions, both backbones.

| N | Encoder `last-L19` | Decoder `last-L19` | ΔD dec − enc [95 % CI] |
|---:|---|---|---|
| 5 % | 0.975 / .619 | **0.954 / .621** | −0.0218 [−0.0334, −0.0111] |
| 10 % | 0.971 / .603 | **0.950 / .609** | −0.0214 [−0.0300, −0.0135] |
| 25 % | 0.932 / .635 | **0.904 / .657** | −0.0277 [−0.0427, −0.0129] |
| 50 % | 0.842 / .703 | 0.836 / .693 | −0.0059 [−0.0241, +0.0129] |
| 100 % | **0.741 / .771** | 0.778 / .742 | **+0.0373** [+0.0118, +0.0637] |

Key readout comparisons at 100 %:

| Backbone | `last-L19` | `attn-L18` | `attn-L12` |
|---|---:|---:|---:|
| encoder | **0.741** | **0.740** | 0.775 |
| decoder | **0.778** | 0.838 | 0.834 |

So:
- Encoder `last-L19` and `attn-L18` are effectively tied at full data.
- Decoder strongly prefers `last-L19` over attention.
- The encoder/decoder crossover remains: decoder better at 5–25 %, tie at 50 %,
  encoder better at 100 %.
