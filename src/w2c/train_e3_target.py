"""E3-A target-rate training with an adaptive KL coefficient."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .config import ensure_dir, load_config
from .data import load_data
from .train_e3 import (VIB, _ce_bits, _cpu_clone, _eval, _extract_features,
                       _kl_bits, _set_seed)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--targets", default="5,4,3,2,1,0.5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
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
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    ensure_dir(os.path.join(out, "checkpoints"))

    for target in [float(x) for x in args.targets.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            _set_seed(seed)
            vib = VIB(Xtr.shape[1], int(cfg.get("latents", [32])[0]), seed,
                      logvar_init=0.0).to(device)
            opt = torch.optim.AdamW(vib.parameters(), lr=float(cfg["lr"]),
                                    weight_decay=float(cfg["weight_decay"]))
            beta = float(cfg.get("beta_init", 0.1))
            best = {"val": float("inf"), "state": None, "epoch": -1}
            history = []
            for ep in range(int(cfg["epochs"])):
                vib.mu.train(); vib.logvar.train(); vib.head.train()
                opt.zero_grad(set_to_none=True)
                logits, mu, logvar = vib(Xt, mc=int(cfg["mc_train"]), sample=True)
                ce = _ce_bits(logits, yt).mean()
                kl = _kl_bits(mu, logvar).mean()
                loss = ce + beta * kl
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    vib.parameters(), float(cfg["grad_clip"])))
                opt.step()
                val = _eval(vib, Xv, yv, int(cfg["mc_eval"]),
                            int(cfg["eval_batch_size"]), device)
                # Dual update on the measured training KL.
                beta = float(np.clip(beta * math.exp(float(cfg.get("beta_lr", 0.05)) *
                                                     (float(kl) - target)),
                                     1e-6, 1e4))
                history.append({"epoch": ep, "beta": beta,
                                "train_ce_bits": float(ce),
                                "train_kl_bits": float(kl),
                                "objective_bits": float(ce + beta * kl),
                                "val_ce_bits": float(val["ce_bits"]),
                                "val_kl_bits": float(val["rate_bits"]),
                                "grad_norm": grad_norm})
                if val["ce_bits"] < best["val"] - 1e-9:
                    best = {"val": float(val["ce_bits"]),
                            "state": _cpu_clone(vib.state_dict()), "epoch": ep}
                if ep - best["epoch"] >= int(cfg["patience"]):
                    break
            vib.load_state_dict(best["state"])
            val = _eval(vib, Xv, yv, int(cfg["mc_eval"]),
                        int(cfg["eval_batch_size"]), device)
            test = _eval(vib, Xte, yte, int(cfg["mc_eval"]),
                         int(cfg["eval_batch_size"]), device)
            mean = _eval(vib, Xte, yte, 1, int(cfg["eval_batch_size"]), device,
                         sample=False)
            run_id = f"{args.backbone}__target{target:g}__s{seed}"
            rec = {"run_id": run_id, "backbone": args.backbone,
                   "checkpoint": args.checkpoint, "target_rate": target,
                   "seed": seed, "best_epoch": best["epoch"],
                   "final_beta": beta, "val_ce": val["ce_bits"],
                   "val_rate": val["rate_bits"], "test_ce": test["ce_bits"],
                   "test_rate": test["rate_bits"], "test_acc": test["accuracy"],
                   "mean_test_ce": mean["ce_bits"], "mean_test_acc": mean["accuracy"],
                   "history": history, "timestamp": time.time()}
            with open(os.path.join(out, "runs.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            torch.save({"run_id": run_id, "state": best["state"],
                        "target_rate": target, "seed": seed,
                        "best_epoch": best["epoch"], "beta": beta},
                       os.path.join(out, "checkpoints", run_id + ".pt"))
            print(f"[target] {run_id:<35} target={target:g} R={test['rate_bits']:.3f} "
                  f"CE={test['ce_bits']:.4f} mean={mean['ce_bits']:.4f} "
                  f"beta={beta:.4g} ep={best['epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
