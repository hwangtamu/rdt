"""Evaluate saved E3-B checkpoints with mean-only inference."""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from .config import load_config
from .data import load_data
from .train_e3b import TokenVIB, extract_tokens, make_batch


def evaluate_mean(model, tokens, y, batch, device):
    import torch
    model.mu.eval(); model.logvar.eval(); model.head.eval()
    probs = []; ces = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch):
            idxs = list(range(start, min(start + batch, len(tokens))))
            h, mask = make_batch(tokens, idxs, device)
            yb = torch.tensor(y[idxs], device=device)
            logits, rate = model(h, mask, mc=1, sample=False)
            ce = torch.nn.functional.cross_entropy(logits[0], yb) / math.log(2)
            ces.append(float(ce))
            probs.append(torch.softmax(logits[0], -1).cpu().numpy())
    p = np.concatenate(probs, 0)
    return {"ce_bits": float(np.mean(ces)), "accuracy": float((p.argmax(1) == y).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args(argv)
    import torch
    cfg = load_config(args.config)
    device = torch.device(args.device)
    parts, _ = load_data(cfg)
    tokens, _ = extract_tokens(args.checkpoint, parts, cfg, device)
    yte = np.array([r["label"] for r in parts["test"]])
    out = []
    for path in sorted(os.listdir(os.path.join(args.run_dir, "checkpoints"))):
        if not path.endswith(".pt"):
            continue
        ck = torch.load(os.path.join(args.run_dir, "checkpoints", path),
                        map_location="cpu", weights_only=False)
        m = TokenVIB(tokens["test"][0].shape[1], int(cfg["latent"]),
                     int(ck["seed"])).to(device)
        m.mu.load_state_dict(ck["state"]["mu"])
        m.logvar.load_state_dict(ck["state"]["logvar"])
        m.query.data = ck["state"]["query"].to(device)
        m.head.load_state_dict(ck["state"]["head"])
        r = evaluate_mean(m, tokens["test"], yte, int(cfg["eval_batch_size"]), device)
        r.update({"run_id": ck["run_id"], "beta": ck["beta"], "seed": ck["seed"]})
        out.append(r)
        print(r)
    with open(os.path.join(args.run_dir, "mean_eval.jsonl"), "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
