"""Diagnostics for E3-A high-rate recovery.

Trains beta=0 bottlenecks in deterministic, low-variance, and stochastic modes
and evaluates both mean-only and stochastic test CE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .config import ensure_dir, load_config
from .data import load_data
from .train_e3 import _ce_bits, _cpu_clone, _extract_features, _set_seed, _kl_bits


class BN:
    def __init__(self, d_in, d, seed, mode):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.mode = mode
        self.mu = nn.Linear(d_in, d)
        self.head = nn.Linear(d, 2)
        self.logvar = nn.Linear(d_in, d) if mode == "stochastic" else None

    def to(self, device):
        self.mu.to(device); self.head.to(device)
        if self.logvar is not None:
            self.logvar.to(device)
        return self

    def parameters(self):
        p = list(self.mu.parameters()) + list(self.head.parameters())
        if self.logvar is not None:
            p += list(self.logvar.parameters())
        return p

    def __call__(self, h, mc=1, sample=True):
        return self.forward(h, mc=mc, sample=sample)

    def forward(self, h, mc=1, sample=True):
        import torch
        mu = self.mu(h)
        if self.mode == "mean":
            z = mu.unsqueeze(0)
            logvar = torch.zeros_like(mu)
        elif self.mode == "lowvar":
            logvar = torch.full_like(mu, -8.0)
            if sample:
                z = mu.unsqueeze(0) + torch.randn((mc,) + mu.shape, device=h.device) * math.exp(-4.0)
            else:
                z = mu.unsqueeze(0)
        else:
            logvar = torch.clamp(self.logvar(h), -8.0, 8.0)
            if sample:
                z = mu.unsqueeze(0) + torch.randn((mc,) + mu.shape, device=h.device) * torch.exp(0.5 * logvar).unsqueeze(0)
            else:
                z = mu.unsqueeze(0)
        logits = self.head(z.reshape(-1, z.shape[-1])).view(z.shape[0], h.shape[0], 2)
        return logits, mu, logvar


def eval_bn(bn, X, y, mc, batch, device, sample):
    import torch
    ces, probs, kl_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for start in range(0, len(X), batch):
            hb = torch.tensor(X[start:start + batch], device=device)
            yb = torch.tensor(y[start:start + batch], device=device)
            logits, mu, logvar = bn(hb, mc=mc, sample=sample)
            ce = _ce_bits(logits, yb).mean().item()
            ces.append((ce, len(yb)))
            probs.append(torch.softmax(logits, -1).mean(0).cpu().numpy())
            kl_sum += _kl_bits(mu, logvar).sum().item()
            n += len(yb)
    p = np.concatenate(probs, 0)
    return {"ce_bits": sum(c * n for c, n in ces) / sum(n for _, n in ces),
            "acc": float((p.argmax(1) == y).mean()),
            "rate_bits": kl_sum / n, "probs": p}


def train_one(Xtr, ytr, Xv, yv, Xte, yte, d, mode, lr, seed, cfg, device):
    import torch
    _set_seed(seed)
    bn = BN(Xtr.shape[1], d, seed, mode).to(device)
    opt = torch.optim.AdamW(bn.parameters(), lr=lr, weight_decay=float(cfg["weight_decay"]))
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    best = {"val": float("inf"), "state": None, "epoch": -1}
    for ep in range(int(cfg["epochs"])):
        bn.mu.train(); bn.head.train()
        if bn.logvar is not None:
            bn.logvar.train()
        opt.zero_grad(set_to_none=True)
        logits, mu, logvar = bn(Xt, mc=int(cfg["mc_train"]), sample=(mode != "mean"))
        ce = _ce_bits(logits, yt).mean()
        loss = ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bn.parameters(), float(cfg["grad_clip"]))
        opt.step()
        val = eval_bn(bn, Xv, yv, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]),
                      device, sample=(mode != "mean"))["ce_bits"]
        if val < best["val"] - 1e-9:
            best = {"val": val, "state": _cpu_clone({
                "mu": bn.mu.state_dict(), "head": bn.head.state_dict(),
                "logvar": None if bn.logvar is None else bn.logvar.state_dict()}), "epoch": ep}
        if ep - best["epoch"] >= int(cfg["patience"]):
            break
    bn.mu.load_state_dict(best["state"]["mu"]); bn.head.load_state_dict(best["state"]["head"])
    if bn.logvar is not None:
        bn.logvar.load_state_dict(best["state"]["logvar"])
    mean = eval_bn(bn, Xte, yte, 1, int(cfg["eval_batch_size"]), device, sample=False)
    stoch = eval_bn(bn, Xte, yte, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]), device,
                    sample=(mode != "mean"))
    return {"mode": mode, "latent": d, "lr": lr, "seed": seed,
            "best_epoch": best["epoch"], "val_ce": best["val"],
            "mean_test_ce": mean["ce_bits"], "mean_acc": mean["acc"],
            "stoch_test_ce": stoch["ce_bits"], "stoch_acc": stoch["acc"],
            "rate_test_bits": stoch["rate_bits"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--latents", default="32,128")
    ap.add_argument("--modes", default="mean,lowvar,stochastic")
    ap.add_argument("--lrs", default="0.001,0.0001")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args(argv)
    import torch
    cfg = load_config(args.config)
    device = torch.device(args.device)
    out = ensure_dir(args.out)
    parts, _ = load_data(cfg)
    feats = _extract_features(args.backbone, args.checkpoint, parts, cfg, device,
                              ensure_dir(cfg["feature_cache_dir"]))
    Xtr, ytr = feats["train"]["X"], feats["train"]["y"]
    Xv, yv = feats["val"]["X"], feats["val"]["y"]
    Xte, yte = feats["test"]["X"], feats["test"]["y"]
    rows = []
    for d in [int(x) for x in args.latents.split(",")]:
        for mode in args.modes.split(","):
            for lr in [float(x) for x in args.lrs.split(",")]:
                for seed in [int(x) for x in args.seeds.split(",")]:
                    t = time.time()
                    r = train_one(Xtr, ytr, Xv, yv, Xte, yte, d, mode, lr, seed, cfg, device)
                    r.update({"backbone": args.backbone, "checkpoint": args.checkpoint,
                              "elapsed_seconds": time.time() - t})
                    rows.append(r)
                    print(json.dumps(r))
    with open(os.path.join(out, "diagnostics.jsonl"), "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
