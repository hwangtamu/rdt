"""Summarize E3-A P2 robustness and ablations."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np


def records(root):
    out = []
    for path in glob.glob(os.path.join(root, "**", "runs.jsonl"), recursive=True):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def interp(points, target):
    pts = sorted((r, c) for r, c in points
                 if math.isfinite(r) and math.isfinite(c))
    if not pts:
        return None
    rs = np.asarray([p[0] for p in pts])
    cs = np.asarray([p[1] for p in pts])
    if target < rs.min() or target > rs.max():
        return None
    uniq = {}
    for r, c in zip(rs, cs):
        uniq.setdefault(float(r), []).append(float(c))
    ur = np.asarray(sorted(uniq))
    uc = np.asarray([np.mean(uniq[r]) for r in ur])
    return float(np.interp(target, ur, uc))


def matched_delta(root, target):
    rows = records(root)
    vals = {}
    for backbone in ("encoder", "decoder"):
        by = defaultdict(list)
        for rec in rows:
            if rec.get("kind") == "vib" and rec.get("backbone_key") == backbone:
                by[(int(rec["e2_seed"]), int(rec["seed"]))].append(
                    (float(rec["rate_val_bits"]), float(rec["expected_test_ce_bits"])))
        vals[backbone] = {k: interp(v, target) for k, v in by.items()}
    out = {}
    common = set(vals["encoder"]) & set(vals["decoder"])
    for e2 in sorted({k[0] for k in common}):
        ek = [vals["encoder"][k] for k in common if k[0] == e2]
        dk = [vals["decoder"][k] for k in common if k[0] == e2]
        ek = [x for x in ek if x is not None]
        dk = [x for x in dk if x is not None]
        if ek and dk:
            out[e2] = float(np.mean(dk) - np.mean(ek))
    return out


def sanity(root):
    rows = records(root)
    res = {}
    for backbone in ("encoder", "decoder"):
        rr = [r for r in rows if r.get("backbone_key") == backbone]
        det = [r["test"]["ce_bits"] for r in rr if r.get("kind") == "deterministic"]
        beta0 = [r["expected_test_ce_bits"] for r in rr
                 if r.get("kind") == "vib" and float(r["beta"]) == 0.0]
        beta10 = [r for r in rr if r.get("kind") == "vib" and float(r["beta"]) == 10.0]
        res[backbone] = {
            "det": float(np.mean(det)) if det else float("nan"),
            "beta0": float(np.mean(beta0)) if beta0 else float("nan"),
            "zero_max_rate": max((r["rate_test_bits"] for r in beta10), default=float("nan")),
            "zero_max_ce": max((r["expected_test_ce_bits"] for r in beta10), default=float("nan")),
            "n_zero": len(beta10),
        }
    return res


def csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/e3a_boolq_p2/analysis/robustness.md")
    args = ap.parse_args(argv)

    roots = [
        ("P2 main (latent 32, global scaling)", "runs/e3a_boolq_p2"),
        ("5 VIB seeds", "runs/e3a_boolq_p2_seeds5"),
        ("Latent 128", "runs/e3a_boolq_p2_lat128"),
        ("No feature scaling", "runs/e3a_boolq_p2_noscale"),
        ("Per-dimension z-score", "runs/e3a_boolq_p2_perdim"),
    ]
    lines = ["# E3-A robustness summary\n"]
    lines.append("## Ablation summary at matched validation rate\n")
    lines.append("Expected CE; ΔD = decoder − encoder (positive favors encoder); "
                 "Δpenalty = decoder penalty − encoder penalty.\n")
    lines.append("| Run | R | ΔD | ΔD 95% CI | Δpenalty | Δpenalty 95% CI |")
    lines.append("|---|---:|---:|---|---:|---|")
    for label, root in roots:
        path = os.path.join(root, "analysis", "matched_expected.csv")
        for row in csv_rows(path):
            t = float(row["target"])
            if abs(t - 2.0) > 1e-9 and abs(t - 4.0) > 1e-9:
                continue
            lines.append(
                f"| {label} | {t:.1f} | {float(row['delta']):+.4f} | "
                f"[{float(row['delta_lo']):+.4f}, {float(row['delta_hi']):+.4f}] | "
                f"{float(row['dpen']):+.4f} | "
                f"[{float(row['dpen_lo']):+.4f}, {float(row['dpen_hi']):+.4f}] |")

    lines.append("\n## Per-E2-seed and leave-one-out stability (P2 main)\n")
    lines.append("| Target R | Per-seed ΔD | Mean | LOO mean range |")
    lines.append("|---:|---|---:|---|")
    for target in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        d = matched_delta("runs/e3a_boolq_p2", target)
        vals = [d[k] for k in sorted(d)]
        if not vals:
            continue
        loo = [float(np.mean([v for j, v in enumerate(vals) if j != i]))
               for i in range(len(vals))]
        lines.append(f"| {target:.1f} | {', '.join(f'{v:+.4f}' for v in vals)} | "
                     f"{np.mean(vals):+.4f} | [{min(loo):+.4f}, {max(loo):+.4f}] |")

    d1 = matched_delta("runs/e3a_boolq_p2", 1.0)
    d8 = matched_delta("runs/e3a_boolq_p2", 8.0)
    common = sorted(set(d1) & set(d8))
    if common:
        diffs = [d8[e] - d1[e] for e in common]
        lines.append("\n## Rate interaction\n")
        lines.append(f"Mean ΔD(8 bits) − ΔD(1 bit) = **{np.mean(diffs):+.4f}** "
                     f"(per-seed: {', '.join(f'{x:+.4f}' for x in diffs)}).\n")
        lines.append("A value near zero means the encoder/decoder gap is roughly "
                     "constant across rate; a positive value would indicate a "
                     "larger encoder advantage at high rate.")

    lines.append("\n## Sanity checks\n")
    lines.append("| Backbone | Deterministic CE | β=0 CE | β=10 max rate | β=10 max CE |")
    lines.append("|---|---:|---:|---:|---:|")
    for backbone, s in sanity("runs/e3a_boolq_p2").items():
        lines.append(f"| {backbone} | {s['det']:.4f} | {s['beta0']:.4f} | "
                     f"{s['zero_max_rate']:.4f} | {s['zero_max_ce']:.4f} |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
