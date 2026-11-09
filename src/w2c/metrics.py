"""Decision metrics and paired bootstrap uncertainty (plan sections 4.1, 5.3, 12).

Cross-entropy is reported in bits per decision, matching the distortion D.
All functions are NumPy-only so results do not depend on sklearn versions.

Variable-K multiple choice (Lev/LevBench) is supported by passing ``n_options``:
logits at positions >= K are masked before softmax, argmax, and calibration.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 4
_MASK_VALUE = -1e30


def _masked(logits: np.ndarray, n_options: np.ndarray | None) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if n_options is None:
        return logits
    n_options = np.asarray(n_options, dtype=np.int64)
    if len(n_options) != len(logits):
        raise ValueError("n_options must align with logits")
    valid = np.arange(logits.shape[1])[None, :] < n_options[:, None]
    return np.where(valid, logits, _MASK_VALUE)


def softmax(logits: np.ndarray, n_options: np.ndarray | None = None) -> np.ndarray:
    logits = _masked(logits, n_options)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def per_example_ce_bits(logits: np.ndarray, y: np.ndarray,
                        n_options: np.ndarray | None = None) -> np.ndarray:
    probs = softmax(logits, n_options)
    true = probs[np.arange(len(y)), y]
    return -np.log2(np.clip(true, 1e-12, 1.0))


def ce_bits(logits: np.ndarray, y: np.ndarray,
            n_options: np.ndarray | None = None) -> float:
    return float(per_example_ce_bits(logits, y, n_options).mean())


def accuracy(logits: np.ndarray, y: np.ndarray,
             n_options: np.ndarray | None = None) -> float:
    return float((softmax(logits, n_options).argmax(axis=1) == y).mean())


def confusion_matrix(logits: np.ndarray, y: np.ndarray, k: int = N_CLASSES,
                     n_options: np.ndarray | None = None) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    pred = softmax(logits, n_options).argmax(axis=1)
    np.add.at(cm, (y, pred), 1)
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(logits: np.ndarray, y: np.ndarray, k: int = N_CLASSES,
             n_options: np.ndarray | None = None) -> float:
    return float(per_class_f1(confusion_matrix(logits, y, k, n_options)).mean())


def expected_calibration_error(logits: np.ndarray, y: np.ndarray,
                               n_bins: int = 15,
                               n_options: np.ndarray | None = None) -> float:
    probs = softmax(logits, n_options)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_metrics(logits: np.ndarray, y: np.ndarray,
                           n_bins: int = 15, k: int = N_CLASSES,
                           n_options: np.ndarray | None = None) -> dict:
    cm = confusion_matrix(logits, y, k, n_options)
    f1s = per_class_f1(cm)
    return {
        "ce_bits": ce_bits(logits, y, n_options),
        "accuracy": accuracy(logits, y, n_options),
        "macro_f1": macro_f1(logits, y, k, n_options),
        "ece": expected_calibration_error(logits, y, n_bins, n_options),
        "per_class_f1": {str(i): float(v) for i, v in enumerate(f1s)},
        "confusion": cm.tolist(),
        "n": int(len(y)),
    }


def subgroup_metrics(logits: np.ndarray, y: np.ndarray, groups,
                     n_bins: int = 15, k: int = N_CLASSES,
                     n_options: np.ndarray | None = None,
                     min_n: int = 25) -> dict:
    """Per-source / per-tier diagnostics on the same test logits."""
    groups = np.asarray([str(g) for g in groups])
    out = {}
    for group in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == group)
        if len(idx) < min_n:
            continue
        sub_opts = None if n_options is None else np.asarray(n_options)[idx]
        out[group] = {
            "n": int(len(idx)),
            "ce_bits": ce_bits(logits[idx], y[idx], sub_opts),
            "accuracy": accuracy(logits[idx], y[idx], sub_opts),
            "ece": expected_calibration_error(logits[idx], y[idx], n_bins, sub_opts),
        }
    return out


def bootstrap_ci(logits: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05, n_bins: int = 15,
                 k: int = N_CLASSES, n_options: np.ndarray | None = None) -> dict:
    """Percentile bootstrap over test examples for each scalar metric."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = {"ce_bits": [], "accuracy": [], "macro_f1": [], "ece": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub_opts = None if n_options is None else np.asarray(n_options)[idx]
        stats["ce_bits"].append(ce_bits(logits[idx], y[idx], sub_opts))
        stats["accuracy"].append(accuracy(logits[idx], y[idx], sub_opts))
        stats["macro_f1"].append(macro_f1(logits[idx], y[idx], k, sub_opts))
        stats["ece"].append(expected_calibration_error(logits[idx], y[idx],
                                                       n_bins, sub_opts))
    out = {}
    for name, values in stats.items():
        values = np.asarray(values)
        out[f"{name}_lo"] = float(np.quantile(values, alpha / 2))
        out[f"{name}_hi"] = float(np.quantile(values, 1 - alpha / 2))
    return out


def paired_bootstrap_delta(loss_enc: np.ndarray, loss_dec: np.ndarray,
                           n_boot: int = 2000, seed: int = 0,
                           alpha: float = 0.05) -> dict:
    """Delta D = D_dec - D_enc, with a paired bootstrap CI over the same examples."""
    loss_enc = np.asarray(loss_enc, dtype=np.float64)
    loss_dec = np.asarray(loss_dec, dtype=np.float64)
    if loss_enc.shape != loss_dec.shape:
        raise ValueError("encoder/decoder losses must be aligned on identical test examples")
    delta = loss_dec - loss_enc
    rng = np.random.default_rng(seed)
    n = len(delta)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = delta[idx].mean()
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))
    p_two = 2 * min((draws <= 1e-12).mean(), (draws >= -1e-12).mean())
    return {
        "delta_mean": float(delta.mean()),
        "delta_lo": lo,
        "delta_hi": hi,
        "p_two_sided": float(min(1.0, p_two)),
        "prob_encoder_better": float((draws < 0).mean()),
        "n": int(n),
    }
