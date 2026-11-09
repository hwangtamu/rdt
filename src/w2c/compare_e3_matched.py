"""DEPRECATED for P1/P2: use compare_e3_p2.py.

This older helper groups by VIB seed only and matches on test rate, so it can
mix E2 checkpoints and is not the protocol used for the fixed E3-A analysis.
Keep it only for the older single-checkpoint v6 runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np


def load(path: str, metric: str):
    by_seed = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") != "vib":
            continue
        key = "expected_test_ce_bits" if metric == "expected" else "mean_test_ce_bits"
        if key not in r:
            continue
        by_seed[int(r["seed"])].append((float(r["rate_test_bits"]), float(r[key]),
                                        float(r["beta"])))
    return by_seed


def interp(points, target):
    pts = sorted((r, ce) for r, ce, _ in points if math.isfinite(r) and math.isfinite(ce))
    if not pts:
        return None
    rs = np.array([p[0] for p in pts])
    ces = np.array([p[1] for p in pts])
    if target < rs.min() or target > rs.max():
        return None
    # Average CE for exact duplicate rates before interpolation.
    uniq = {}
    for r, ce in zip(rs, ces):
        uniq.setdefault(float(r), []).append(float(ce))
    ur = np.array(sorted(uniq))
    uc = np.array([np.mean(uniq[r]) for r in ur])
    return float(np.interp(target, ur, uc))


def ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    sem = values.std(ddof=1) / math.sqrt(n)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 1.96)
    return float(values.mean() - t * sem), float(values.mean() + t * sem)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-runs", required=True)
    ap.add_argument("--decoder-runs", required=True)
    ap.add_argument("--targets", default="1.6,2,2.5,3,3.5,4,4.5")
    ap.add_argument("--metric", choices=["expected", "mean"], default="expected")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    enc = load(args.encoder_runs, args.metric)
    dec = load(args.decoder_runs, args.metric)
    rows = []
    for target in [float(x) for x in args.targets.split(",")]:
        e = {s: interp(v, target) for s, v in enc.items()}
        d = {s: interp(v, target) for s, v in dec.items()}
        common = sorted(s for s in e if e[s] is not None and d.get(s) is not None)
        if not common:
            continue
        ev = [e[s] for s in common]
        dv = [d[s] for s in common]
        delta = [d[s] - e[s] for s in common]
        rows.append({
            "target_rate": target, "n_seeds": len(common),
            "enc_ce_mean": float(np.mean(ev)), "enc_ce_sd": float(np.std(ev, ddof=1)),
            "dec_ce_mean": float(np.mean(dv)), "dec_ce_sd": float(np.std(dv, ddof=1)),
            "delta_dec_minus_enc": float(np.mean(delta)),
            "delta_sd": float(np.std(delta, ddof=1)),
            "delta_lo95": ci95(delta)[0], "delta_hi95": ci95(delta)[1],
        })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["target_rate"])
        w.writeheader(); w.writerows(rows)
    md = os.path.splitext(args.out)[0] + ".md"
    with open(md, "w", encoding="utf-8") as fh:
        fh.write(f"# Matched-rate comparison ({args.metric} CE)\n\n")
        fh.write("| Target R | Seeds | Encoder CE | Decoder CE | ΔD = dec − enc [95% CI] |\n")
        fh.write("|---:|---:|---:|---:|---|\n")
        for r in rows:
            fh.write(f"| {r['target_rate']:.2f} | {r['n_seeds']} | "
                     f"{r['enc_ce_mean']:.4f} | {r['dec_ce_mean']:.4f} | "
                     f"{r['delta_dec_minus_enc']:+.4f} "
                     f"[{r['delta_lo95']:+.4f}, {r['delta_hi95']:+.4f}] |\n")
    print(f"wrote {args.out} and {md}")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
