"""E3-A: frozen-backbone VIB on BoolQ E2 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone

import numpy as np

from .config import ensure_dir, load_config
from .data import load_data
from .metrics import classification_metrics
from .train_e2 import (_batch_tensors, load_tokenizer, tokenize_records)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _e2_seed_from_checkpoint(checkpoint: str):
    """Extract the E2 training seed from a saved checkpoint name."""
    match = re.search(r"__s(\d+)\.pt$", os.path.basename(checkpoint))
    return int(match.group(1)) if match else None


def _cpu_clone(value):
    import torch

    if isinstance(value, dict):
        return {k: _cpu_clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_clone(v) for v in value)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_backbone(checkpoint: str, device):
    import torch
    from transformers import AutoModel

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = AutoModel.from_pretrained(ckpt["backbone_path"], torch_dtype=torch.float32)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        raise RuntimeError(f"missing weights loading checkpoint: {missing[:5]}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def _extract_features(backbone, checkpoint: str, parts, cfg: dict, device,
                      cache_dir: str) -> dict:
    import torch

    ckpt_name = os.path.basename(checkpoint).replace(".pt", "")
    cache_path = os.path.join(cache_dir, f"{ckpt_name}__features.npz")
    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as d:
            return {split: {"X": d[f"{split}_X"], "y": d[f"{split}_y"],
                            "ids": [str(x) for x in d[f"{split}_ids"]]}
                    for split in ("train", "val", "test")}

    model, ckpt = _load_backbone(checkpoint, device)
    tokenizer = load_tokenizer({"name": ckpt["backbone_name"],
                                "path": ckpt["backbone_path"]})
    pad_id = tokenizer.pad_token_id
    max_length = int(cfg["data"].get("max_length", cfg.get("max_length", 512)))
    out = {}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            seqs, trunc = tokenize_records(parts[split], tokenizer, max_length)
            rows = []
            for start in range(0, len(seqs), int(cfg["feature_batch_size"])):
                idxs = list(range(start, min(start + int(cfg["feature_batch_size"]), len(seqs))))
                input_ids, mask = _batch_tensors(seqs, idxs, pad_id, device)
                o = model(input_ids=input_ids, attention_mask=mask, return_dict=True)
                pos = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
                last = (pos * mask).max(1).values
                rr = torch.arange(mask.size(0), device=mask.device)
                rows.append(o.last_hidden_state[rr, last].float().cpu().numpy())
            X = np.concatenate(rows, axis=0).astype(np.float32)
            out[split] = {"X": X,
                          "y": np.array([r["label"] for r in parts[split]], dtype=np.int64),
                          "ids": [r["id"] for r in parts[split]]}
            print(f"[e3] features {ckpt_name} {split}: {X.shape} "
                  f"trunc={float(np.mean(trunc)):.4f}")
    np.savez(cache_path,
             **{f"{split}_{key}": (np.asarray(out[split][key])
                                   if key == "ids" else out[split][key])
                for split in out for key in ("X", "y", "ids")})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


class VIB:
    def __init__(self, d_in: int, d: int, seed: int, logvar_init: float = -4.0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.mu = nn.Linear(d_in, d)
        self.logvar = nn.Linear(d_in, d)
        nn.init.constant_(self.logvar.bias, float(logvar_init))
        self.head = nn.Linear(d, 2)

    def to(self, device):
        self.mu.to(device); self.logvar.to(device); self.head.to(device)
        return self

    def parameters(self):
        return list(self.mu.parameters()) + list(self.logvar.parameters()) + list(self.head.parameters())

    def state_dict(self):
        return {"mu": self.mu.state_dict(), "logvar": self.logvar.state_dict(),
                "head": self.head.state_dict()}

    def load_state_dict(self, state):
        self.mu.load_state_dict(state["mu"])
        self.logvar.load_state_dict(state["logvar"])
        self.head.load_state_dict(state["head"])

    def __call__(self, h, mc: int = 1, sample: bool = True):
        return self.forward(h, mc=mc, sample=sample)

    def forward(self, h, mc: int = 1, sample: bool = True):
        import torch

        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -8.0, 8.0)
        if not sample:
            z = mu.unsqueeze(0)
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn((mc,) + mu.shape, device=mu.device, dtype=mu.dtype)
            z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        logits = self.head(z.reshape(-1, z.shape[-1])).view(z.shape[0], h.shape[0], 2)
        return logits, mu, logvar


def _kl_bits(mu, logvar):
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(1) / math.log(2)


def _ce_bits(logits, y):
    import torch
    import torch.nn.functional as F
    return F.cross_entropy(logits.reshape(-1, 2).float(),
                           y.unsqueeze(0).expand(logits.shape[0], -1).reshape(-1),
                           reduction="none").view(logits.shape[0], -1).mean(0) / math.log(2)


def _eval(vib, X, y, mc: int, batch: int, device, sample: bool = True):
    import torch
    vib.mu.eval(); vib.logvar.eval(); vib.head.eval()
    ces, probs, per_example = [], [], []
    kl_sum, n_total = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(X), batch):
            hb = torch.tensor(X[start:start + batch], device=device)
            yb = torch.tensor(y[start:start + batch], device=device)
            logits, mu, logvar = vib(hb, mc=mc, sample=sample)
            ce_batch = _ce_bits(logits, yb)
            ce = ce_batch.mean().item()
            ces.append((ce, len(yb)))
            per_example.append(ce_batch.cpu().numpy())
            probs.append(torch.softmax(logits, -1).mean(0).cpu().numpy())
            kl_sum += _kl_bits(mu, logvar).sum().item()
            n_total += len(yb)
    ce = sum(c * n for c, n in ces) / sum(n for _, n in ces)
    p = np.concatenate(probs, 0)
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    return {"ce_bits": float(ce), "accuracy": acc,
            "rate_bits": float(kl_sum / n_total), "probs": p,
            "ce_per_example": np.concatenate(per_example, 0)}


def _deterministic(X, y, Xv, yv, cfg, seed, device, patience=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _set_seed(seed)
    head = nn.Linear(X.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    Xt = torch.tensor(X, device=device); yt = torch.tensor(y, device=device)
    Xv_t = torch.tensor(Xv, device=device); yv_t = torch.tensor(yv, device=device)
    patience = int(cfg.get("det_patience", cfg.get("epochs", 250)) if patience is None else patience)
    best = (float("inf"), None, -1)
    for ep in range(int(cfg["epochs"])):
        head.train(); opt.zero_grad()
        loss = F.cross_entropy(head(Xt), yt) / math.log(2)
        loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            v = F.cross_entropy(head(Xv_t), yv_t).item() / math.log(2)
        if v < best[0] - 1e-9:
            best = (v, _cpu_clone(head.state_dict()), ep)
        elif ep - best[2] >= patience:
            break
    head.load_state_dict(best[1]); head.eval()
    with torch.no_grad():
        logits = head(torch.tensor(Xv, device=device))
    return head, best[2], best[0]


def _deterministic_bottleneck(X, y, Xv, yv, Xte, cfg, seed, device, latent):
    """Capacity-matched deterministic control: linear projection + linear head."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _set_seed(seed)
    proj = nn.Linear(X.shape[1], int(latent)).to(device)
    head = nn.Linear(int(latent), 2).to(device)
    opt = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()),
                            lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    Xt = torch.tensor(X, device=device); yt = torch.tensor(y, device=device)
    Xv_t = torch.tensor(Xv, device=device); yv_t = torch.tensor(yv, device=device)
    patience = int(cfg.get("det_patience", cfg.get("epochs", 250)))
    best = (float("inf"), None, None, -1)
    for ep in range(int(cfg["epochs"])):
        proj.train(); head.train(); opt.zero_grad()
        loss = F.cross_entropy(head(proj(Xt)), yt) / math.log(2)
        loss.backward(); opt.step()
        proj.eval(); head.eval()
        with torch.no_grad():
            v = F.cross_entropy(head(proj(Xv_t)), yv_t).item() / math.log(2)
        if v < best[0] - 1e-9:
            best = (v, _cpu_clone(proj.state_dict()), _cpu_clone(head.state_dict()), ep)
        elif ep - best[3] >= patience:
            break
    proj.load_state_dict(best[1]); head.load_state_dict(best[2])
    proj.eval(); head.eval()
    with torch.no_grad():
        logits = head(proj(torch.tensor(Xte, device=device))).cpu().numpy()
    return logits, best[3], best[0]


def run(cfg: dict, backbone_key: str, checkpoint: str, out: str, device: str) -> None:
    import torch

    out = ensure_dir(out)
    device = torch.device(device)
    parts, _ = load_data(cfg)
    cache_dir = ensure_dir(cfg["feature_cache_dir"])
    feats = _extract_features(backbone_key, checkpoint, parts, cfg, device, cache_dir)
    Xtr, ytr = feats["train"]["X"], feats["train"]["y"]
    Xv, yv = feats["val"]["X"], feats["val"]["y"]
    Xte, yte = feats["test"]["X"], feats["test"]["y"]

    # Train-only feature scaling.  "global" removes only the overall scale
    # (safe for a linear readout); "per_dim" z-scores each feature and is kept
    # as an ablation because it can amplify low-variance noise dimensions.
    scale_mode = cfg.get("standardize", False)
    if scale_mode is True:
        scale_mode = "per_dim"
    if scale_mode == "global":
        mu = Xtr.mean(); sd = Xtr.std() + 1e-6
        Xtr = (Xtr - mu) / sd
        Xv = (Xv - mu) / sd
        Xte = (Xte - mu) / sd
    elif scale_mode == "per_dim":
        mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd; Xv = (Xv - mu) / sd; Xte = (Xte - mu) / sd

    runs_path = os.path.join(out, "runs.jsonl")
    ensure_dir(os.path.join(out, "runs"))
    ensure_dir(os.path.join(out, "preds"))

    e2_seed = _e2_seed_from_checkpoint(checkpoint)
    selection = str(cfg.get("selection", "ce"))

    # Deterministic control, no bottleneck.
    det_head, det_ep, det_val = _deterministic(Xtr, ytr, Xv, yv, cfg, int(cfg["vib_seeds"][0]), device)
    with torch.no_grad():
        det_test_logits = det_head(torch.tensor(Xte, device=device)).cpu().numpy()
    det_metrics = classification_metrics(det_test_logits, yte, n_bins=10, k=2)
    det = {"run_id": f"{backbone_key}__deterministic__s0",
           "backbone_key": backbone_key, "checkpoint": checkpoint, "e2_seed": e2_seed,
           "kind": "deterministic", "beta": None, "latent": None,
           "seed": int(cfg["vib_seeds"][0]), "best_epoch": det_ep,
           "val_ce_bits": det_val, "test": det_metrics,
           "rate_train_bits": 0.0, "rate_val_bits": 0.0, "rate_test_bits": 0.0,
           "timestamp": _now()}
    with open(runs_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(det) + "\n")
    det_ce = -np.log2(np.clip(
        np.exp(det_test_logits - det_test_logits.max(1, keepdims=True)) /
        np.exp(det_test_logits - det_test_logits.max(1, keepdims=True)).sum(1, keepdims=True),
        1e-12, 1.0))[np.arange(len(yte)), yte]
    np.savez(os.path.join(out, "preds", det["run_id"] + ".npz"),
             logits=det_test_logits, labels=yte, ids=np.asarray(feats["test"]["ids"]),
             expected_ce=det_ce, mean_ce=det_ce)
    print(f"[e3] deterministic val={det_val:.4f} testCE={det_metrics['ce_bits']:.4f}")

    # Capacity-matched deterministic bottleneck control (z = mu, no KL).
    for det_latent in [int(x) for x in cfg.get("det_bottleneck_latents", [])]:
        detb_logits, detb_ep, detb_val = _deterministic_bottleneck(
            Xtr, ytr, Xv, yv, Xte, cfg, int(cfg["vib_seeds"][0]), device, det_latent)
        detb_metrics = classification_metrics(detb_logits, yte, n_bins=10, k=2)
        detb = {"run_id": f"{backbone_key}__det_bottleneck_d{det_latent}__s0",
                "backbone_key": backbone_key, "checkpoint": checkpoint, "e2_seed": e2_seed,
                "kind": "det_bottleneck", "beta": None, "latent": int(det_latent),
                "seed": int(cfg["vib_seeds"][0]), "best_epoch": detb_ep,
                "val_ce_bits": detb_val, "test": detb_metrics,
                "rate_train_bits": 0.0, "rate_val_bits": 0.0, "rate_test_bits": 0.0,
                "timestamp": _now()}
        with open(runs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(detb) + "\n")
        detb_ce = -np.log2(np.clip(
            np.exp(detb_logits - detb_logits.max(1, keepdims=True)) /
            np.exp(detb_logits - detb_logits.max(1, keepdims=True)).sum(1, keepdims=True),
            1e-12, 1.0))[np.arange(len(yte)), yte]
        np.savez(os.path.join(out, "preds", detb["run_id"] + ".npz"),
                 logits=detb_logits, labels=yte, ids=np.asarray(feats["test"]["ids"]),
                 expected_ce=detb_ce, mean_ce=detb_ce)
        print(f"[e3] det-bottleneck d{det_latent} val={detb_val:.4f} "
              f"testCE={detb_metrics['ce_bits']:.4f}")

    for latent in [int(x) for x in cfg.get("latents", [cfg.get("latent", 32)])]:
      for beta in [float(x) for x in cfg["betas"]]:
        for seed in [int(x) for x in cfg["vib_seeds"]]:
            run_id = f"{backbone_key}__vib__d{latent}__beta{beta:g}__s{seed}"
            if os.path.exists(os.path.join(out, "runs", run_id + ".json")):
                continue
            _set_seed(seed)
            init_map = cfg.get("logvar_init_by_beta", {})
            beta_init = float(init_map.get(str(beta), cfg.get("logvar_init", 0.0)))
            warm_epochs = 0
            if beta <= float(cfg.get("warmstart_max_beta", -1.0)):
                warm_epochs = int(cfg.get("warmstart_epochs", 0))
            vib = VIB(Xtr.shape[1], int(latent), seed,
                      logvar_init=beta_init).to(device)
            lr = float(cfg.get("lr_by_beta", {}).get(str(beta), cfg["lr"]))
            opt = torch.optim.AdamW(vib.parameters(), lr=lr,
                                    weight_decay=float(cfg["weight_decay"]))
            Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
            Xv_t = torch.tensor(Xv, device=device); yv_t = torch.tensor(yv, device=device)
            start_time = time.time()
            # Deterministic mean-only warm start.
            if warm_epochs > 0:
                for _ in range(warm_epochs):
                    vib.mu.train(); vib.logvar.train(); vib.head.train()
                    opt.zero_grad(set_to_none=True)
                    logits, mu, logvar = vib(Xt, mc=1, sample=False)
                    loss = _ce_bits(logits, yt).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(vib.parameters(), float(cfg["grad_clip"]))
                    opt.step()
                torch.nn.init.constant_(vib.logvar.bias,
                                        float(cfg.get("warmstart_logvar", -4.0)))
            patience = int(cfg.get("patience_by_beta", {}).get(
                str(beta), cfg.get("patience", 20)))
            best = {"score": float("inf"), "val_ce": float("inf"),
                    "val_rate": float("inf"), "state": None, "epoch": -1}
            history = []
            for ep in range(int(cfg["epochs"])):
                vib.mu.train(); vib.logvar.train(); vib.head.train()
                perm = torch.randperm(len(Xt), device=device)
                kl_warmup = int(cfg.get("kl_warmup_epochs", 0))
                beta_eff = beta
                if kl_warmup > 0:
                    beta_eff = beta * min(1.0, float(ep + 1) / kl_warmup)
                ce_sum = kl_sum = loss_sum = 0.0
                n_batches = 0
                for start in range(0, len(Xt), int(cfg["batch_size"])):
                    idx = perm[start:start + int(cfg["batch_size"])]
                    hb, yb = Xt[idx], yt[idx]
                    opt.zero_grad(set_to_none=True)
                    logits, mu, logvar = vib(hb, mc=int(cfg["mc_train"]), sample=True)
                    ce = _ce_bits(logits, yb).mean()
                    kl = _kl_bits(mu, logvar).mean()
                    loss = ce + beta_eff * kl
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(vib.parameters(), float(cfg["grad_clip"]))
                    opt.step()
                    ce_sum += float(ce.detach())
                    kl_sum += float(kl.detach())
                    loss_sum += float(loss.detach())
                    n_batches += 1
                val = _eval(vib, Xv, yv, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]), device)
                score = val["ce_bits"]
                if selection == "objective":
                    score += beta * val["rate_bits"]
                history.append({
                    "epoch": ep,
                    "train_ce_bits": ce_sum / max(1, n_batches),
                    "train_kl_bits": kl_sum / max(1, n_batches),
                    "train_loss": loss_sum / max(1, n_batches),
                    "val_ce_bits": val["ce_bits"],
                    "val_rate_bits": val["rate_bits"],
                    "val_objective": score,
                })
                if score < best["score"] - 1e-9:
                    best = {"score": score, "val_ce": val["ce_bits"],
                            "val_rate": val["rate_bits"],
                            "state": _cpu_clone(vib.state_dict()), "epoch": ep}
                if ep - best["epoch"] >= patience:
                    break
            vib.load_state_dict(best["state"])
            val = _eval(vib, Xv, yv, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]), device)
            test = _eval(vib, Xte, yte, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]), device)
            test_mean = _eval(vib, Xte, yte, 1, int(cfg["eval_batch_size"]), device, sample=False)
            train = _eval(vib, Xtr, ytr, int(cfg["mc_eval"]), int(cfg["eval_batch_size"]), device)
            metrics = classification_metrics(np.log(np.clip(test["probs"], 1e-12, 1)),
                                             yte, n_bins=10, k=2)
            metrics["expected_ce_bits"] = test["ce_bits"]
            metrics["mc_ce_bits"] = float(-np.log(np.clip(
                test["probs"][np.arange(len(yte)), yte], 1e-12, 1)).mean() / math.log(2))
            rec = {
                "run_id": run_id, "timestamp": _now(), "backbone_key": backbone_key,
                "checkpoint": checkpoint, "e2_seed": e2_seed,
                "kind": "vib", "beta": beta,
                "latent": int(latent), "seed": seed, "best_epoch": best["epoch"],
                "logvar_init": beta_init,
                "warmstart_epochs": warm_epochs,
                "lr": lr,
                "selection": selection,
                "patience": patience,
                "val_ce_bits": best["val_ce"],
                "val_rate_bits": best["val_rate"],
                "val_objective": best["score"],
                "test": metrics,
                "rate_train_bits": train["rate_bits"],
                "rate_val_bits": val["rate_bits"],
                "rate_test_bits": test["rate_bits"],
                "expected_test_ce_bits": test["ce_bits"],
                "mean_test_ce_bits": test_mean["ce_bits"],
                "mean_test_accuracy": test_mean["accuracy"],
                "history": history,
                "elapsed_seconds": time.time() - start_time,
            }
            with open(os.path.join(out, "runs", run_id + ".json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2)
            with open(runs_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            np.savez(os.path.join(out, "preds", run_id + ".npz"),
                     probs=test["probs"], labels=yte, ids=np.asarray(feats["test"]["ids"]),
                     expected_ce=test["ce_per_example"],
                     mean_ce=test_mean["ce_per_example"])
            ckpt_dir = ensure_dir(os.path.join(out, "checkpoints"))
            torch.save({
                "run_id": run_id, "backbone_key": backbone_key, "checkpoint": checkpoint,
                "beta": beta, "latent": int(latent), "seed": seed,
                "state": best["state"], "best_epoch": best["epoch"],
                "val_ce_bits": best["val_ce"],
                "logvar_init": float(cfg.get("logvar_init", 0.0)),
                "warmstart_epochs": int(cfg.get("warmstart_epochs", 0)),
                "warmstart_logvar": float(cfg.get("warmstart_logvar", -6.0)),
            }, os.path.join(ckpt_dir, run_id + ".pt"))
            print(f"[e3] {run_id:<45} R={test['rate_bits']:.4f} "
                  f"CE={test['ce_bits']:.4f} acc={test['accuracy']:.4f} "
                  f"ep={best['epoch']}")
            del vib, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E3-A frozen VIB")
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True, choices=["encoder", "decoder"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.backbone, args.checkpoint, args.out, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
