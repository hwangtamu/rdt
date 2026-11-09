"""E3-B pilot: pre-pooling token-level VIB on frozen E2 BoolQ checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone

import numpy as np

from .config import ensure_dir, load_config
from .data import load_data
from .metrics import classification_metrics
from .train_e2 import _batch_tensors, load_tokenizer, tokenize_records


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_clone(x):
    import torch
    if isinstance(x, dict):
        return {k: cpu_clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_clone(v) for v in x)
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    return x


def load_backbone(checkpoint, device):
    import torch
    from transformers import AutoModel
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = AutoModel.from_pretrained(ckpt["backbone_path"], torch_dtype=torch.float32)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        raise RuntimeError(f"missing keys: {missing[:5]}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def extract_tokens(checkpoint, parts, cfg, device):
    """Return per-split list of [T, D] float16 token tensors and lengths."""
    import torch
    model, ckpt = load_backbone(checkpoint, device)
    tokenizer = load_tokenizer({"name": ckpt["backbone_name"],
                                "path": ckpt["backbone_path"]})
    pad_id = tokenizer.pad_token_id
    max_length = int(cfg["max_length"])
    out = {}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            seqs, trunc = tokenize_records(parts[split], tokenizer, max_length)
            arrays = []
            for start in range(0, len(seqs), int(cfg["feature_batch_size"])):
                idxs = list(range(start, min(start + int(cfg["feature_batch_size"]), len(seqs))))
                input_ids, mask = _batch_tensors(seqs, idxs, pad_id, device)
                o = model(input_ids=input_ids, attention_mask=mask, return_dict=True)
                h = o.last_hidden_state
                for j, idx in enumerate(idxs):
                    length = int(mask[j].sum().item())
                    arrays.append(h[j, :length].float().cpu().numpy().astype(np.float16))
            out[split] = arrays
            print(f"[e3b] tokens {split}: {len(arrays)} examples, "
                  f"mean_len={np.mean([len(a) for a in arrays]):.1f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, ckpt


def make_batch(tokens, idxs, device):
    import torch
    lengths = [len(tokens[i]) for i in idxs]
    T = max(lengths)
    B = len(idxs)
    h = np.zeros((B, T, tokens[idxs[0]].shape[1]), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for row, idx in enumerate(idxs):
        h[row, :len(tokens[idx])] = tokens[idx]
        mask[row, :len(tokens[idx])] = 1.0
    return torch.tensor(h, device=device), torch.tensor(mask, device=device)


class TokenVIB:
    def __init__(self, d_in, d, seed, logvar_init=-4.0, pooling="attention",
                 identity_init=False):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.pooling = pooling
        self.mu = nn.Linear(d_in, d)
        self.logvar = nn.Linear(d_in, d)
        nn.init.constant_(self.logvar.bias, float(logvar_init))
        if identity_init and d == d_in:
            nn.init.eye_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
            nn.init.zeros_(self.logvar.weight)
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.gate = None
        if pooling == "last_gate":
            self.gate = nn.Linear(2 * d, 1)
            nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, 2.0)
        self.head = nn.Linear(d, 2)

    def to(self, device):
        self.mu.to(device); self.logvar.to(device); self.head.to(device)
        self.query.data = self.query.data.to(device)
        if self.gate is not None:
            self.gate.to(device)
        return self

    def parameters(self):
        p = list(self.mu.parameters()) + list(self.logvar.parameters()) + [self.query] + list(self.head.parameters())
        if self.gate is not None:
            p += list(self.gate.parameters())
        return p

    def state_dict(self):
        return {"mu": self.mu.state_dict(), "logvar": self.logvar.state_dict(),
                "query": self.query.detach().cpu(), "head": self.head.state_dict(),
                "gate": None if self.gate is None else self.gate.state_dict(),
                "pooling": self.pooling}

    def __call__(self, h, mask, mc=1, sample=True):
        return self.forward(h, mask, mc=mc, sample=sample)

    def forward(self, h, mask, mc=1, sample=True):
        import torch
        B, T, D = h.shape
        mu = self.mu(h); logvar = torch.clamp(self.logvar(h), -8.0, 8.0)
        if sample:
            eps = torch.randn((mc, B, T, mu.shape[-1]), device=h.device)
            z = mu.unsqueeze(0) + eps * torch.exp(0.5 * logvar).unsqueeze(0)
        else:
            z = mu.unsqueeze(0)
        if self.pooling == "last":
            idx = mask.sum(1).long() - 1
            pooled = z[:, torch.arange(B, device=h.device), idx]
        elif self.pooling == "mean":
            pooled = (z * mask.unsqueeze(0).unsqueeze(-1)).sum(2) / torch.clamp(mask.sum(1), min=1).unsqueeze(0).unsqueeze(-1)
        else:
            scores = (z @ self.query) / math.sqrt(z.shape[-1])
            scores = scores.masked_fill(mask.unsqueeze(0).bool() == False,
                                        torch.finfo(scores.dtype).min)
            w = torch.softmax(scores, dim=-1)
            pooled = (w.unsqueeze(-1) * z).sum(dim=2)
            if self.pooling == "last_gate":
                idx = mask.sum(1).long() - 1
                z_last = z[:, torch.arange(B, device=h.device), idx]
                g = torch.sigmoid(self.gate(torch.cat([z_last, pooled], dim=-1)))
                pooled = g * z_last + (1.0 - g) * pooled
        logits = self.head(pooled)
        kl_tok = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1) / math.log(2)
        rate = (kl_tok * mask).sum(1)
        return logits, rate


def evaluate(model, tokens, y, mc, batch, device):
    import torch
    model.mu.eval(); model.logvar.eval(); model.head.eval()
    probs = []; ces = []; rates = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch):
            idxs = list(range(start, min(start + batch, len(tokens))))
            h, mask = make_batch(tokens, idxs, device)
            yb = torch.tensor(y[idxs], device=device)
            logits, rate = model(h, mask, mc=mc, sample=True)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2), yb.unsqueeze(0).expand(logits.shape[0], -1).reshape(-1),
                reduction="none").view(logits.shape[0], -1).mean(0) / math.log(2)
            ces.append(ce.mean().item())
            probs.append(torch.softmax(logits, -1).mean(0).cpu().numpy())
            rates.append(rate.mean().item())
    p = np.concatenate(probs, 0)
    return {"ce_bits": float(np.mean(ces)), "rate_bits": float(np.mean(rates)),
            "accuracy": float((p.argmax(1) == y).mean()), "probs": p}


def train_one(tokens, ytr, tokens_v, yv, tokens_te, yte, cfg, seed, beta, device):
    import torch
    set_seed(seed)
    model = TokenVIB(tokens[0].shape[1], int(cfg["latent"]), seed,
                     logvar_init=float(cfg.get("logvar_init", -4.0)),
                     pooling=str(cfg.get("pooling", "attention"))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    best = {"val": float("inf"), "state": None, "epoch": -1}
    start_time = time.time()
    for ep in range(int(cfg["epochs"])):
        model.mu.train(); model.logvar.train(); model.head.train()
        order = list(range(len(tokens))); random.shuffle(order)
        for start in range(0, len(order), int(cfg["batch_size"])):
            idxs = order[start:start + int(cfg["batch_size"])]
            h, mask = make_batch(tokens, idxs, device)
            yb = torch.tensor(ytr[idxs], device=device)
            opt.zero_grad(set_to_none=True)
            logits, rate = model(h, mask, mc=int(cfg["mc_train"]), sample=True)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2), yb.unsqueeze(0).expand(logits.shape[0], -1).reshape(-1)) / math.log(2)
            loss = ce + beta * rate.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
        val = evaluate(model, tokens_v, yv, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]), device)
        if val["ce_bits"] < best["val"] - 1e-9:
            best = {"val": val["ce_bits"], "state": cpu_clone(model.state_dict()), "epoch": ep}
        if ep - best["epoch"] >= int(cfg["patience"]):
            break
    model.mu.load_state_dict(best["state"]["mu"]); model.logvar.load_state_dict(best["state"]["logvar"])
    model.head.load_state_dict(best["state"]["head"]); model.query.data = best["state"]["query"].to(device)
    val = evaluate(model, tokens_v, yv, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]), device)
    test = evaluate(model, tokens_te, yte, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]), device)
    return {"best_epoch": best["epoch"], "val_ce": val["ce_bits"],
            "test_ce": test["ce_bits"], "test_rate": test["rate_bits"],
            "test_acc": test["accuracy"], "elapsed": time.time() - start_time,
            "state": best["state"], "model": model}


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
    out = ensure_dir(args.out)
    parts, _ = load_data(cfg)
    tokens, ckpt = extract_tokens(args.checkpoint, parts, cfg, device)
    ytr = np.array([r["label"] for r in parts["train"]])
    yv = np.array([r["label"] for r in parts["val"]])
    yte = np.array([r["label"] for r in parts["test"]])

    # Forced prior control.
    pri = np.bincount(ytr, minlength=2).astype(float); pri /= pri.sum()
    prior_ce = float(-np.log2(pri[yte]).mean())
    print(f"[e3b] prior CE={prior_ce:.4f}")

    for beta in [float(x) for x in cfg["betas"]]:
        for seed in [int(x) for x in cfg["vib_seeds"]]:
            run_id = f"{args.backbone}__e3b__beta{beta:g}__s{seed}"
            r = train_one(tokens["train"], ytr, tokens["val"], yv, tokens["test"], yte,
                          cfg, seed, beta, device)
            rec = {"run_id": run_id, "backbone": args.backbone, "checkpoint": args.checkpoint,
                   "beta": beta, "seed": seed, "latent": int(cfg["latent"]),
                   "best_epoch": r["best_epoch"], "val_ce": r["val_ce"],
                   "test_ce": r["test_ce"], "test_rate": r["test_rate"],
                   "test_acc": r["test_acc"], "elapsed_seconds": r["elapsed"],
                   "prior_ce": prior_ce, "timestamp": now()}
            with open(os.path.join(out, "runs.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            torch.save({"run_id": run_id, "state": r["state"], "beta": beta,
                        "seed": seed, "checkpoint": args.checkpoint},
                       os.path.join(ensure_dir(os.path.join(out, "checkpoints")), run_id + ".pt"))
            print(f"[e3b] {run_id:<40} R={r['test_rate']:.2f} CE={r['test_ce']:.4f} "
                  f"acc={r['test_acc']:.4f} ep={r['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
