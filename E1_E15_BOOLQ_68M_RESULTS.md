# E1 / E1.5 BoolQ — Ettin-68M encoder vs decoder

Models:
- `jhu-clsp/ettin-encoder-68m` → `models/ettin-encoder-68m`
- `jhu-clsp/ettin-decoder-68m` → `models/ettin-decoder-68m`

Data: `data/boolq_e0/`
E1 run: `runs/e1_boolq_68m/` (all layers 0–19, linear/MLP, 3 seeds)
E1.5 run: `runs/e15_boolq_68m/` (last/mean/attn/concat, 3 seeds)

## E1 frozen last-token probes

Best config per backbone/probe by validation CE:

| Backbone | Probe | Layer | Val CE | Test CE | Acc |
|---|---|---:|---:|---:|---:|
| encoder | MLP | 19 | 0.9278 | **0.9056** | 0.6536 |
| encoder | linear | 19 | 0.9362 | 0.9304 | 0.6330 |
| decoder | MLP | 10 | 0.9020 | **0.9205** | 0.6675 |
| decoder | linear | 10 | 0.9062 | 0.9217 | 0.6591 |

Encoder and decoder are close under the E1 last-token readout; decoder is a
little better in accuracy, encoder in CE.

## E1.5 readout comparison

Selected by validation CE:

| Backbone | Readout | Probe | Config | Test CE | Acc | ΔD vs `last` |
|---|---|---|---|---:|---:|---:|
| encoder | attn | linear | L18 | **0.8579** | **0.6948** | **−0.0725** |
| encoder | attn | MLP | L12 | 0.8689 | 0.6824 | −0.0367 |
| encoder | mean | linear | L16 | 0.8936 | 0.6557 | −0.0369 |
| encoder | mean | MLP | L16 | 0.8860 | 0.6601 | −0.0195 |
| encoder | last | MLP | L19 | 0.9056 | 0.6536 | baseline |
| decoder | attn | linear | L12 | **0.8812** | 0.6787 | −0.0234 |
| decoder | attn | MLP | L12 | 0.8920 | 0.6693 | −0.0177 |
| decoder | mean | MLP | L12 | 0.8909 | 0.6679 | −0.0188 |
| decoder | last | linear | L12 | 0.9046 | 0.6593 | baseline |

## Reading

1. **Attention pooling is the best readout for both 68M backbones on BoolQ.**
2. **Encoder attn-L18 is the best single cell**: CE 0.858, acc 0.695.
3. Decoder attn-L12 is close: CE 0.881, acc 0.679.
4. Compared with the 150M Ettin pair, the 68M encoder is not worse on this frozen
   BoolQ probe and benefits more from attention pooling. The 150M decoder was
   slightly stronger than the 68M decoder.
5. E1 alone (`last` only) hides the encoder's attention-readout advantage; E1.5
   is essential here.
