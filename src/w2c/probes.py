"""Linear and matched-MLP probes over frozen features (plan E1).

Both probes are trained with the same recipe for both backbones; the MLP has a
single hidden layer of fixed width so the linear->MLP gap is comparable across
models.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .metrics import ce_bits


def _mask_logits(logits: torch.Tensor, n_options: torch.Tensor | None) -> torch.Tensor:
    if n_options is None:
        return logits
    valid = torch.arange(logits.size(1), device=logits.device)[None, :] < \
        n_options[:, None]
    return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)


def build_probe(kind: str, in_dim: int, hidden: int = 256, dropout: float = 0.1,
                n_classes: int = 4) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_dim, n_classes)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    raise ValueError(f"unknown probe kind '{kind}'")


def _standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def train_probe(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                kind: str, cfg: dict, seed: int, device: str = "cpu",
                n_classes: int = 4,
                n_options_train: np.ndarray | None = None,
                n_options_val: np.ndarray | None = None
                ) -> tuple[nn.Module, dict, tuple[np.ndarray, np.ndarray] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)

    mean = std = None
    if cfg.get("standardize", True):
        mean, std = _standardizer(X_train.astype(np.float64))
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

    dev = torch.device(device)
    Xtr = torch.as_tensor(X_train, dtype=torch.float32, device=dev)
    ytr = torch.as_tensor(y_train, dtype=torch.long, device=dev)
    Xva = torch.as_tensor(X_val, dtype=torch.float32, device=dev)
    yva = torch.as_tensor(y_val, dtype=torch.long, device=dev)
    ktr = (None if n_options_train is None
           else torch.as_tensor(n_options_train, dtype=torch.long, device=dev))
    kva = (None if n_options_val is None
           else torch.as_tensor(n_options_val, dtype=torch.long, device=dev))

    model = build_probe(kind, X_train.shape[1], cfg["mlp_hidden"], cfg["dropout"],
                        n_classes=n_classes).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    batch_size = int(cfg["batch_size"])
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["patience"])

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = -1
    waited = 0

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(Xtr), device=dev)
        for start in range(0, len(Xtr), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            ktr_batch = None if ktr is None else ktr[idx]
            loss = loss_fn(_mask_logits(model(Xtr[idx]), ktr_batch), ytr[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_ce = float(loss_fn(_mask_logits(model(Xva), kva), yva).item()
                          / np.log(2.0))
        if val_ce < best_val - 1e-5:
            best_val, best_epoch, waited = val_ce, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    info = {
        "kind": kind,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch + 1),
        "best_val_ce_bits": float(best_val),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "standardized": bool(mean is not None),
    }
    scaler = None
    if mean is not None:
        info["feature_mean_norm"] = float(np.linalg.norm(mean))
        info["feature_std_mean"] = float(std.mean())
        scaler = (mean, std)
    return model, info, scaler


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray,
            scaler: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Apply the *training* scaler before inference; raw features are not comparable."""
    model.eval()
    dev = next(model.parameters()).device
    if scaler is not None:
        mean, std = scaler
        X = (X - mean) / std
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    return model(Xt).float().cpu().numpy()


def evaluate_probe(model: nn.Module, X: np.ndarray, y: np.ndarray,
                   scaler: tuple[np.ndarray, np.ndarray] | None = None
                   ) -> tuple[np.ndarray, float]:
    logits = predict(model, X, scaler)
    return logits, ce_bits(logits, y)


# --------------------------------------------------------------------------- #
# learned attention pooling over ragged token states (E1.5)
# --------------------------------------------------------------------------- #

class AttnPool(nn.Module):
    """Single-query additive attention over non-pad tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.query, std=dim ** -0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        scale = x.size(-1) ** -0.5
        scores = torch.matmul(x, self.query) * scale  # [B, T]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bt,bth->bh", weights, x)
        return pooled, weights


class AttnProbe(nn.Module):
    """Attention pooling + the same probe heads used by E1."""

    def __init__(self, dim: int, kind: str, hidden: int = 256,
                 dropout: float = 0.1, n_classes: int = 4):
        super().__init__()
        self.pool = AttnPool(dim)
        self.probe = build_probe(kind, dim, hidden, dropout, n_classes=n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.pool(x, mask)
        return self.probe(pooled)


def token_stats(tokens, chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/std over all training tokens (chunked, float64 accum)."""
    n = 0
    total = np.zeros(tokens.shape[1], dtype=np.float64)
    total_sq = np.zeros(tokens.shape[1], dtype=np.float64)
    for start in range(0, len(tokens), chunk):
        block = np.asarray(tokens[start:start + chunk], dtype=np.float64)
        total += block.sum(axis=0)
        total_sq += (block * block).sum(axis=0)
        n += block.shape[0]
    mean = total / max(n, 1)
    var = np.maximum(total_sq / max(n, 1) - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def token_offsets(lengths: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return offsets


def _length_batches(lengths: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Length-sorted, non-overlapping batches (minimises padding)."""
    order = np.argsort(lengths, kind="stable")
    return [order[start:start + batch_size]
            for start in range(0, len(order), batch_size)]


def _gather_batch(tokens, offsets: np.ndarray, lengths: np.ndarray,
                  idx: np.ndarray, mean_t: torch.Tensor, std_t: torch.Tensor,
                  device: torch.device):
    idx = np.asarray(idx)
    lens = lengths[idx].astype(np.int64)
    width = int(lens.max()) if len(lens) else 1
    hidden = tokens.shape[1]
    x = np.zeros((len(idx), width, hidden), dtype=np.float16)
    mask = np.zeros((len(idx), width), dtype=bool)
    for row, (example, length) in enumerate(zip(idx, lens)):
        start = int(offsets[example])
        x[row, :int(length)] = tokens[start:start + int(length)]
        mask[row, :int(length)] = True
    xt = torch.from_numpy(x).to(device, non_blocking=True).float()
    mt = torch.from_numpy(mask).to(device, non_blocking=True)
    xt = (xt - mean_t) / std_t
    xt = xt.masked_fill(~mt.unsqueeze(-1), 0.0)
    return xt, mt


@torch.no_grad()
def predict_attn_probe(model: nn.Module, tokens, lengths: np.ndarray,
                       offsets: np.ndarray, mean: np.ndarray, std: np.ndarray,
                       batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=dev)
    n_classes = model.probe[-1].out_features if isinstance(model.probe, nn.Sequential) \
        else model.probe.out_features
    logits = np.zeros((len(lengths), n_classes), dtype=np.float32)
    for idx in _length_batches(lengths, batch_size):
        xt, mt = _gather_batch(tokens, offsets, lengths, idx, mean_t, std_t, dev)
        logits[idx] = model(xt, mt).float().cpu().numpy()
    return logits


def train_attn_probe(tokens, lengths: np.ndarray, y: np.ndarray,
                     val_tokens, val_lengths: np.ndarray, y_val: np.ndarray,
                     kind: str, cfg: dict, seed: int, device: str = "cpu",
                     n_classes: int = 4,
                     mean: np.ndarray | None = None,
                     std: np.ndarray | None = None,
                     n_options: np.ndarray | None = None,
                     n_options_val: np.ndarray | None = None
                     ) -> tuple[nn.Module, dict, np.ndarray]:
    """Train attention pooling jointly with a probe on frozen token states.

    Token states are standardised with training-token statistics before pooling.
    Returns (model, info, val_logits, attention query norm); the caller predicts
    test logits with ``predict_attn_probe``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    if mean is None or std is None:
        mean, std = token_stats(tokens)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=dev)
    offsets = token_offsets(lengths)
    val_offsets = token_offsets(val_lengths)
    ytr = torch.as_tensor(y, dtype=torch.long, device=dev)
    yva = torch.as_tensor(y_val, dtype=torch.long, device=dev)
    ktr = (None if n_options is None
           else torch.as_tensor(n_options, dtype=torch.long, device=dev))
    kva = (None if n_options_val is None
           else torch.as_tensor(n_options_val, dtype=torch.long, device=dev))

    model = AttnProbe(tokens.shape[1], kind, cfg["mlp_hidden"], cfg["dropout"],
                      n_classes=n_classes).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    batch_size = int(cfg.get("token_batch_size", 64))
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["patience"])

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = -1
    waited = 0
    rng = np.random.default_rng(seed)

    for epoch in range(max_epochs):
        model.train()
        chunks = _length_batches(lengths, batch_size)
        rng.shuffle(chunks)
        for idx in chunks:
            xt, mt = _gather_batch(tokens, offsets, lengths, idx, mean_t, std_t, dev)
            idx_t = torch.as_tensor(np.asarray(idx), device=dev)
            targets = ytr[idx_t]
            k_batch = None if ktr is None else ktr[idx_t]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(_mask_logits(model(xt, mt), k_batch), targets)
            loss.backward()
            optimizer.step()

        val_logits = predict_attn_probe(model, val_tokens, val_lengths, val_offsets,
                                        mean, std, batch_size, device)
        val_ce = ce_bits(val_logits, y_val, n_options_val)
        if val_ce < best_val - 1e-5:
            best_val, best_epoch, waited = val_ce, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    val_logits = predict_attn_probe(model, val_tokens, val_lengths, val_offsets,
                                    mean, std, batch_size, device)
    info = {
        "kind": kind,
        "pooling": "attn",
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch + 1),
        "best_val_ce_bits": float(best_val),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "token_batch_size": batch_size,
        "query_norm": float(model.pool.query.detach().norm().item()),
        "token_mean_norm": float(np.linalg.norm(mean)),
        "token_std_mean": float(std.mean()),
    }
    return model, info, val_logits
