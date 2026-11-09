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


def resolve_backbone_path(ckpt):
    """Resolve a checkpoint's backbone path on the current machine.

    E2 checkpoints record the absolute path used when they were trained.  Allow
    a portable fallback via RDT_MODEL_ROOT or models/<basename> next to the
    working directory so checkpoints can be reused on another host.
    """
    path = ckpt["backbone_path"]
    if os.path.isdir(path):
        return path
    base = os.path.basename(os.path.normpath(path))
    candidates = []
    root = os.environ.get("RDT_MODEL_ROOT")
    if root:
        candidates.append(os.path.join(root, base))
    candidates.append(os.path.join("models", base))
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"backbone path {path!r} not found; tried {candidates}; "
        "set RDT_MODEL_ROOT or copy the model into ./models")


def load_backbone(checkpoint, device):
    import torch
    from transformers import AutoModel
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = AutoModel.from_pretrained(resolve_backbone_path(ckpt),
                                      torch_dtype=torch.float32)
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
                                "path": resolve_backbone_path(ckpt)})
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


def make_batch(tokens, idxs, device, support_length=None):
    import torch
    lengths = [len(tokens[i]) for i in idxs]
    T = int(support_length) if support_length is not None else max(lengths)
    if T < max(lengths):
        raise ValueError(f"support_length={T} < max token length {max(lengths)}")
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

    def __call__(self, h, mask, mc=1, sample=True, prior=False):
        return self.forward(h, mask, mc=mc, sample=sample, prior=prior)

    def forward(self, h, mask, mc=1, sample=True, prior=False):
        import torch
        B, T, D = h.shape
        mu = self.mu(h); logvar = torch.clamp(self.logvar(h), -8.0, 8.0)
        eps = (torch.randn((mc, B, T, mu.shape[-1]), device=h.device)
               if (sample or prior) else None)
        if prior:
            # Forced-prior zero-rate control: every position, including
            # padding, is an independent N(0, I) sample.
            z = eps
        elif sample:
            z = mu.unsqueeze(0) + eps * torch.exp(0.5 * logvar).unsqueeze(0)
            if self.pooling == "attention_fixed_support":
                # Padding must use the prior, not the posterior of a zero h.
                z = torch.where(mask.unsqueeze(0).unsqueeze(-1).bool(), z, eps)
        else:
            z = mu.unsqueeze(0)
        if self.pooling == "last":
            idx = mask.sum(1).long() - 1
            pooled = z[:, torch.arange(B, device=h.device), idx]
        elif self.pooling == "mean":
            pooled = (z * mask.unsqueeze(0).unsqueeze(-1)).sum(2) / torch.clamp(mask.sum(1), min=1).unsqueeze(0).unsqueeze(-1)
        elif self.pooling == "attention_fixed_support":
            # Fixed support of T positions.  The pool sees no padding mask, so
            # token length cannot leak through the number of pooled latents.
            scores = (z @ self.query) / math.sqrt(z.shape[-1])
            w = torch.softmax(scores, dim=-1)
            pooled = (w.unsqueeze(-1) * z).sum(dim=2)
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
        if prior:
            rate = torch.zeros(B, device=h.device)
        else:
            kl_tok = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1) / math.log(2)
            rate = (kl_tok * mask).sum(1)
        return logits, rate


def evaluate(model, tokens, y, mc, batch, device, prior=False, support_length=None):
    import torch
    model.mu.eval(); model.logvar.eval(); model.head.eval()
    probs = []; ces = []; rates = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch):
            idxs = list(range(start, min(start + batch, len(tokens))))
            h, mask = make_batch(tokens, idxs, device, support_length=support_length)
            yb = torch.tensor(y[idxs], device=device)
            logits, rate = model(h, mask, mc=mc, sample=True, prior=prior)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2), yb.unsqueeze(0).expand(logits.shape[0], -1).reshape(-1),
                reduction="none").view(logits.shape[0], -1).mean(0) / math.log(2)
            ces.append(ce.cpu().numpy())
            probs.append(torch.softmax(logits, -1).mean(0).cpu().numpy())
            rates.append(rate.cpu().numpy())
    p = np.concatenate(probs, 0)
    ce_all = np.concatenate(ces, 0)
    rate_all = np.concatenate(rates, 0)
    return {"ce_bits": float(ce_all.mean()), "rate_bits": float(rate_all.mean()),
            "accuracy": float((p.argmax(1) == y).mean()), "probs": p}


def _support_length(cfg):
    return int(cfg["max_length"]) if str(cfg.get("pooling", "")) == "attention_fixed_support" else None


def train_one(tokens, ytr, tokens_v, yv, tokens_te, yte, cfg, seed, beta, device):
    import torch
    set_seed(seed)
    model = TokenVIB(tokens[0].shape[1], int(cfg["latent"]), seed,
                     logvar_init=float(cfg.get("logvar_init", -4.0)),
                     pooling=str(cfg.get("pooling", "attention")),
                     identity_init=bool(cfg.get("identity_init", False))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    support = _support_length(cfg)
    best = {"val": float("inf"), "state": None, "epoch": -1}
    history = []
    start_time = time.time()
    for ep in range(int(cfg["epochs"])):
        model.mu.train(); model.logvar.train(); model.head.train()
        if model.gate is not None:
            model.gate.train()
        order = list(range(len(tokens))); random.shuffle(order)
        for start in range(0, len(order), int(cfg["batch_size"])):
            idxs = order[start:start + int(cfg["batch_size"])]
            h, mask = make_batch(tokens, idxs, device, support_length=support)
            yb = torch.tensor(ytr[idxs], device=device)
            opt.zero_grad(set_to_none=True)
            logits, rate = model(h, mask, mc=int(cfg["mc_train"]), sample=True)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2), yb.unsqueeze(0).expand(logits.shape[0], -1).reshape(-1)) / math.log(2)
            loss = ce + beta * rate.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
        val = evaluate(model, tokens_v, yv, int(cfg["mc_eval"]),
                       int(cfg["eval_batch_size"]), device, support_length=support)
        score = val["ce_bits"] + beta * val["rate_bits"]
        history.append({"epoch": ep, "val_ce_bits": val["ce_bits"],
                        "val_rate_bits": val["rate_bits"],
                        "val_objective": score})
        if score < best["val"] - 1e-9:
            best = {"val": score, "state": cpu_clone(model.state_dict()), "epoch": ep}
        if ep - best["epoch"] >= int(cfg["patience"]):
            break
    model.mu.load_state_dict(best["state"]["mu"]); model.logvar.load_state_dict(best["state"]["logvar"])
    model.head.load_state_dict(best["state"]["head"]); model.query.data = best["state"]["query"].to(device)
    if model.gate is not None and best["state"].get("gate") is not None:
        model.gate.load_state_dict(best["state"]["gate"])
    val = evaluate(model, tokens_v, yv, int(cfg["mc_eval"]),
                   int(cfg["eval_batch_size"]), device, support_length=support)
    test = evaluate(model, tokens_te, yte, int(cfg["mc_eval"]),
                    int(cfg["eval_batch_size"]), device, support_length=support)
    prior = evaluate(model, tokens_te, yte, int(cfg["mc_eval"]),
                     int(cfg["eval_batch_size"]), device, prior=True,
                     support_length=support)
    return {"best_epoch": best["epoch"], "val_ce": val["ce_bits"],
            "val_rate": val["rate_bits"], "val_objective": best["val"],
            "test_ce": test["ce_bits"], "test_rate": test["rate_bits"],
            "test_acc": test["accuracy"], "prior_test_ce": prior["ce_bits"],
            "prior_test_rate": prior["rate_bits"], "prior_test_acc": prior["accuracy"],
            "history": history, "elapsed": time.time() - start_time,
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

    if bool(cfg.get("standardize", False)):
        # Train-only global scalar standardization, matching corrected E3-A.
        total = 0.0
        total_sq = 0.0
        count = 0
        for arr in tokens["train"]:
            x = arr.astype(np.float64)
            total += float(x.sum())
            total_sq += float((x * x).sum())
            count += int(x.size)
        mean = total / max(count, 1)
        var = max(total_sq / max(count, 1) - mean * mean, 0.0)
        std = math.sqrt(var)
        if not math.isfinite(std) or std < 1e-12:
            raise RuntimeError(f"invalid standardization std={std}")
        for split in ("train", "val", "test"):
            tokens[split] = [((a.astype(np.float32) - mean) / std).astype(np.float16)
                             for a in tokens[split]]
        prep = {"mode": "global_scalar", "mean": mean, "std": std,
                "count": count}
        with open(os.path.join(out, "preprocessing.json"), "w", encoding="utf-8") as fh:
            json.dump(prep, fh, indent=2)
        print(f"[e3b] standardized tokens mean={mean:.6g} std={std:.6g}")

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
                   "pooling": str(cfg.get("pooling", "attention")),
                   "best_epoch": r["best_epoch"], "val_ce": r["val_ce"],
                   "val_rate": r["val_rate"], "val_objective": r["val_objective"],
                   "test_ce": r["test_ce"], "test_rate": r["test_rate"],
                   "test_acc": r["test_acc"],
                   "prior_test_ce": r["prior_test_ce"],
                   "prior_test_rate": r["prior_test_rate"],
                   "prior_test_acc": r["prior_test_acc"],
                   "history": r["history"],
                   "elapsed_seconds": r["elapsed"],
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
