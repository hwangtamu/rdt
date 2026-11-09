# E2-BoolQ readout pilot — results

Run: `runs/e2_boolq_readout/{encoder,decoder}`
Config: `configs/e2_boolq_readout.json`
Data: official BoolQ, split as in `data/boolq_e0`
Fine-tuning: full backbone + readout + linear head; 25 % / 100 % of train,
2 seeds; validation selection, untouched test.

Readout names:

- `last-L22` — final model output (`last_hidden_state`) last token.
- `lasth-L17`, `lasth-L22` — last token from `hidden_states[L]` (pre-final-norm
  where applicable).
- `attn-L17`, `attn-L22` — single-query attention over `hidden_states[L]`.
- `fusion-L17+L22` — per-token concat of L17/L22 hidden states, one attention pool.

`last-L17` was removed: it initially reused the final output and was a duplicate
of `last-L22`. The corrected control is `lasth-L17`.

## Hidden-state / final-norm check

The two checkpoints expose `hidden_states` differently:

- **Encoder (`ModernBertModel`):** `hidden_states` are appended *before*
  `final_norm`. `hidden_states[22]` is pre-final-norm; `last_hidden_state` is
  post-final-norm.
- **Decoder (`ModernBertDecoderModel`):** `hidden_states[-1]` already equals
  `last_hidden_state`; applying `final_norm` again is a double-normalization.

Verified on BoolQ:

| Model | `hidden_states[22]` vs `last_hidden_state` | `final_norm(hidden_states[22])` |
|---|---|---|
| encoder | different (max abs diff ≈ 3.6e4) | equals `last_hidden_state` exactly |
| decoder | identical | does **not** equal `last_hidden_state` |

Consequences:

- `lasth-L22` is the pre-final-norm last token for the encoder, but equals the
  final output for the decoder.
- `last-L22` is the post-final-norm final output for the encoder.
- `attn-L22` uses pre-final-norm states for the encoder and final states for the
  decoder.
- The encoder's large `lasth-L22` vs `last-L22` gap (0.126 bits) is therefore a
  final-norm effect, not a layer effect. This must be reported separately from
  the L17-vs-L22 comparison.

## 100 % data (2 seeds)

| Backbone | Readout | Val CE | Test CE | Acc | Macro-F1 |
|---|---|---:|---:|---:|---:|
| encoder | **lasth-L17** | **0.6853** | **0.6696** | **0.7910** | **0.7770** |
| encoder | attn-L17 | 0.7179 | 0.7127 | 0.7775 | 0.7613 |
| encoder | lasth-L22 | 0.7923 | 0.7865 | 0.7402 | 0.6968 |
| encoder | attn-L22 | 0.8067 | 0.8223 | 0.7341 | 0.7051 |
| encoder | fusion-L17+L22 | 0.8718 | 0.8496 | 0.7053 | 0.6568 |
| encoder | last-L22 | 0.9363 | 0.9122 | 0.6558 | 0.5240 |
| decoder | **last-L22** | **0.7722** | **0.7452** | **0.7711** | **0.7469** |
| decoder | lasth-L22 | 0.7722 | 0.7452 | 0.7711 | 0.7469 |
| decoder | lasth-L17 | 0.8010 | 0.7949 | 0.7509 | 0.7139 |
| decoder | fusion-L17+L22 | 0.8309 | 0.7960 | 0.7509 | 0.7039 |
| decoder | attn-L22 | 0.8554 | 0.8161 | 0.7371 | 0.6936 |
| decoder | attn-L17 | 0.8900 | 0.8632 | 0.7255 | 0.6758 |

## Key paired contrasts at 100 % (95 % bootstrap CI)

Delta = candidate − reference; negative is better.

| Backbone | Contrast | Delta CE | 95 % CI | p |
|---|---|---|---:|---:|---:|
| encoder | `lasth-L17` − `lasth-L22` | **−0.1168** | [−0.1518, −0.0858] | <0.001 |
| encoder | `attn-L17` − `lasth-L17` | +0.0431 | [+0.0154, +0.0710] | 0.003 |
| encoder | `lasth-L22` − `last-L22` | **−0.1257** | [−0.1608, −0.0880] | <0.001 |
| encoder | `attn-L22` − `lasth-L22` | +0.0359 | [−0.0111, +0.0868] | 0.122 |
| decoder | `lasth-L17` − `lasth-L22` | +0.0497 | [+0.0213, +0.0797] | 0.002 |
| decoder | `attn-L17` − `lasth-L17` | +0.0682 | [+0.0356, +0.1009] | <0.001 |
| decoder | `attn-L22` − `lasth-L22` | +0.0709 | [+0.0304, +0.1090] | 0.001 |
| decoder | `fusion-L17+L22` − `lasth-L17` | +0.0011 | [−0.0310, +0.0313] | 0.947 |

## 25 % data

The 25 % ranking reverses and is dominated by overfitting/seed variance:
encoder `last-L22` 0.8989, `attn-L17` 0.9843, `lasth-L17` 1.0505; decoder
`last-L22`/`lasth-L22` 0.9357, `lasth-L17` 0.9036. Do not select readouts at
25 %.

## Reading

1. **Strong backbone × readout interaction.** Encoder wants the last token from
   L17 hidden states; decoder wants the final-layer last token.
2. **Attention pooling is not a free win.** At matched layers it never beats the
   last-token readout for the encoder, and it is worse for the decoder.
3. **Fusion is not useful here.** It is worse than the best single-layer readout
   for both backbones.
4. **For the encoder, the final norm hurts a linear readout.** `lasth-L22`
   (hidden states) beats `last-L22` (final output) by 0.126 bits.
5. **Full fine-tuning clearly beats frozen E1.5-BoolQ** (best frozen CE ~0.87 →
   ~0.67 full FT), so E2 is the right experiment to run.
6. Two seeds are enough for the pilot signal, but not for final claims. The
   encoder `lasth-L17` seeds were 0.660/0.679; decoder `last-L22` 0.755/0.736.

## Proposed next steps

1. **Freeze the primary readouts for E2-BoolQ**: encoder `lasth-L17`, decoder
   `last-L22`; keep `attn-L17` and `lasth-L22` as controls.
2. **Add a third seed** for the winning cells at 25 %/100 % to check stability.
3. **Run the missing data fractions** (1 %, 5 %, 10 %, 50 %) for
   `lasth-L17` and `last-L22` on both backbones, then fit `D(N)`.
4. **Do a small LR/epoch sanity sweep** for the winning readouts at 100 %, since
   the 25 % runs overfit quickly and seed variance is non-trivial.
5. **Do not start E3/E4 yet**; first establish the E2 `D(N)` curve and the
   readout protocol.
