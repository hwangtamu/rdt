# Invalid experiment: VIB validation split from E2 train

This run carved a 15% VIB validation split out of the E2 training pool.

It is **not used** in the E3-A conclusion because the E2 backbone was
fine-tuned on those examples. The backbone features therefore memorize the
VIB validation examples: the deterministic control reaches very low validation
CE but degrades on test, and the VIB selects overfit checkpoints.

A clean nested-selection fix requires retraining E2 with a disjoint
E2-train/E2-val/E3-val partition, or cross-fitting E2 checkpoint selection.
That is an E2 protocol change, not an E3-A analysis fix.
