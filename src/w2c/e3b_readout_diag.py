"""No-bottleneck token readout diagnostic for E3-B decoder."""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from .config import load_config
from .data import load_data
from .metrics import classification_metrics
from .train_e3b import extract_tokens, make_batch, set_seed


def fit_temperature(val_logits, y):
    """Grid-search a scalar temperature on validation logits."""
    best = (float("inf"), 1.0)
    z = val_logits - val_logits.max(1, keepdims=True)
    for T in np.linspace(0.25, 8.0, 256):
        logits = z / T
        logp = logits - logits.max(1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
        ce = -logp[np.arange(len(y)), y].mean() / math.log(2)
        if ce < best[0]:
            best = (float(ce), float(T))
    return best[1], best[0]


def metrics(logits, y):
    return classification_metrics(logits, y, n_bins=15, k=2)


def train_last_mean(X, ytr, Xv, yv, Xte, yte, seed, device, epochs=300):
    import torch
    set_seed(seed)
    head = torch.nn.Linear(X.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    Xt = torch.tensor(X, device=device); yt = torch.tensor(ytr, device=device)
    Xv_t = torch.tensor(Xv, device=device); yv_t = torch.tensor(yv, device=device)
    best = (float("inf"), None, -1)
    for ep in range(epochs):
        head.train(); opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(head(Xt), yt) / math.log(2)
        loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            v = torch.nn.functional.cross_entropy(head(Xv_t), yv_t).item() / math.log(2)
        if v < best[0] - 1e-9:
            best = (v, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, ep)
    head.load_state_dict(best[1]); head.eval()
    with torch.no_grad():
        vl = head(Xv_t).cpu().numpy()
        tl = head(torch.tensor(Xte, device=device)).cpu().numpy()
    return vl, tl, best[2]


def train_attention(tokens_tr, ytr, tokens_v, yv, tokens_te, yte, seed, device,
                    batch=64, epochs=30):
    import torch
    set_seed(seed)
    d = tokens_tr[0].shape[1]
    query = torch.nn.Parameter(torch.randn(d, device=device) / math.sqrt(d))
    head = torch.nn.Linear(d, 2).to(device)
    opt = torch.optim.AdamW([query, *head.parameters()], lr=1e-3, weight_decay=0.01)
    best = (float("inf"), None, -1)
    for ep in range(epochs):
        query.requires_grad_(True); head.train()
        order = np.random.permutation(len(tokens_tr)).tolist()
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            h, mask = make_batch(tokens_tr, idx, device)
            yb = torch.tensor(ytr[idx], device=device)
            scores = (h @ query) / math.sqrt(d)
            scores = scores.masked_fill(~mask.bool(), -1e9)
            w = torch.softmax(scores, -1)
            pooled = (w.unsqueeze(-1) * h).sum(1)
            loss = torch.nn.functional.cross_entropy(head(pooled), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        # validation
        vl = []
        with torch.no_grad():
            for start in range(0, len(tokens_v), batch):
                idx = list(range(start, min(start + batch, len(tokens_v))))
                h, mask = make_batch(tokens_v, idx, device)
                yb = torch.tensor(yv[idx], device=device)
                scores = (h @ query) / math.sqrt(d)
                scores = scores.masked_fill(~mask.bool(), -1e9)
                w = torch.softmax(scores, -1)
                pooled = (w.unsqueeze(-1) * h).sum(1)
                vl.append(head(pooled).cpu().numpy())
        vl = np.concatenate(vl, 0)
        v = -np.log(np.clip(np.exp(vl - vl.max(1, keepdims=True)) /
                            np.exp(vl - vl.max(1, keepdims=True)).sum(1, keepdims=True),
                            1e-12, 1))[np.arange(len(yv)), yv].mean() / math.log(2)
        if v < best[0] - 1e-9:
            best = (v, {"q": query.detach().cpu().clone(),
                        "h": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}, ep)
    query.data = best[1]["q"].to(device); head.load_state_dict(best[1]["h"]); head.eval()
    def infer(tokens):
        out = []
        with torch.no_grad():
            for start in range(0, len(tokens), batch):
                idx = list(range(start, min(start + batch, len(tokens))))
                h, mask = make_batch(tokens, idx, device)
                scores = (h @ query) / math.sqrt(d)
                scores = scores.masked_fill(~mask.bool(), -1e9)
                w = torch.softmax(scores, -1)
                out.append(head((w.unsqueeze(-1) * h).sum(1)).cpu().numpy())
        return np.concatenate(out, 0)
    return infer(tokens_v), infer(tokens_te), best[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args(argv)
    import torch
    cfg = load_config(args.config)
    device = torch.device(args.device)
    parts, _ = load_data(cfg)
    tokens, _ = extract_tokens(args.checkpoint, parts, cfg, device)
    ytr = np.array([r["label"] for r in parts["train"]])
    yv = np.array([r["label"] for r in parts["val"]])
    yte = np.array([r["label"] for r in parts["test"]])

    last = [t[-1].astype(np.float32) for t in tokens["train"]]
    # masked mean
    def mean_pool(arrays):
        return np.stack([t.astype(np.float32).mean(0) for t in arrays])
    Xtr_last = np.stack(last); Xv_last = np.stack([t[-1].astype(np.float32) for t in tokens["val"]]); Xte_last = np.stack([t[-1].astype(np.float32) for t in tokens["test"]])
    Xtr_mean = mean_pool(tokens["train"]); Xv_mean = mean_pool(tokens["val"]); Xte_mean = mean_pool(tokens["test"])

    rows = []
    for name, Xtr, Xv, Xte in [("last", Xtr_last, Xv_last, Xte_last),
                               ("mean", Xtr_mean, Xv_mean, Xte_mean)]:
        vl, tl, ep = train_last_mean(Xtr, ytr, Xv, yv, Xte, yte, 0, device)
        T, val_ce_t = fit_temperature(vl, yv)
        rows.append({"readout": name, "best_epoch": ep,
                     "val_ce": metrics(vl, yv)["ce_bits"],
                     "test": metrics(tl, yte),
                     "temperature": T,
                     "test_ce_temperature": metrics(tl / T, yte)["ce_bits"]})
    vl, tl, ep = train_attention(tokens["train"], ytr, tokens["val"], yv,
                                 tokens["test"], yte, 0, device)
    T, _ = fit_temperature(vl, yv)
    rows.append({"readout": "attention", "best_epoch": ep,
                 "val_ce": metrics(vl, yv)["ce_bits"],
                 "test": metrics(tl, yte),
                 "temperature": T,
                 "test_ce_temperature": metrics(tl / T, yte)["ce_bits"]})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "readout_diag.json"), "w") as fh:
        json.dump({"backbone": args.backbone, "checkpoint": args.checkpoint,
                   "rows": rows}, fh, indent=2)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
