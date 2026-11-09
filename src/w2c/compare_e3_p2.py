"""Matched-rate E3-A analysis with per-seed curves and hierarchical uncertainty.

Matches on validation rate, interpolates each (E2 seed, VIB seed) curve at the
target rate, and bootstraps over E2 seeds, VIB seeds, and test examples.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np


def _e2_seed(record: dict) -> int | None:
    if record.get("e2_seed") is not None:
        return int(record["e2_seed"])
    match = re.search(r"__s(\d+)\.pt$", os.path.basename(str(record.get("checkpoint", ""))))
    return int(match.group(1)) if match else None


def load_records(root: str) -> list[dict]:
    records = []
    for path in glob.glob(os.path.join(root, "**", "runs.jsonl"), recursive=True):
        out_dir = os.path.dirname(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_out_dir"] = out_dir
                rec["e2_seed"] = _e2_seed(rec)
                records.append(rec)
    return records


def _softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_predictions(record: dict, ref_ids, metric: str):
    path = os.path.join(record["_out_dir"], "preds", str(record["run_id"]) + ".npz")
    with np.load(path, allow_pickle=False) as d:
        labels = d["labels"].astype(np.int64)
        ids = [str(x) for x in d["ids"]] if "ids" in d.files else None
        if metric == "expected" and "expected_ce" in d.files:
            ce = d["expected_ce"].astype(np.float64)
        elif metric == "mean" and "mean_ce" in d.files:
            ce = d["mean_ce"].astype(np.float64)
        else:
            if "probs" in d.files:
                probs = d["probs"].astype(np.float64)
            elif "logits" in d.files:
                probs = _softmax(d["logits"].astype(np.float64))
            else:
                raise KeyError(f"{path}: no probs/logits/ce")
            ce = -np.log2(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))
    if ref_ids is not None and ids is not None:
        lookup = {rid: i for i, rid in enumerate(ids)}
        order = np.asarray([lookup[rid] for rid in ref_ids], dtype=np.int64)
        ce = ce[order]
        labels = labels[order]
    elif ref_ids is None and ids is not None:
        ref_ids = ids
    return ce, labels, ref_ids


def collapse_duplicate_rates(points):
    by_rate = defaultdict(list)
    for rate, ce in points:
        if math.isfinite(rate) and np.all(np.isfinite(ce)):
            by_rate[float(rate)].append(ce)
    rates = sorted(by_rate)
    ces = [np.mean(np.stack(by_rate[r], axis=0), axis=0) for r in rates]
    return np.asarray(rates, dtype=np.float64), ces


def interp_at_rate(points, target):
    rates, ces = collapse_duplicate_rates(points)
    if len(rates) == 0 or target < rates[0] or target > rates[-1]:
        return None
    if target == rates[0]:
        return ces[0]
    if target == rates[-1]:
        return ces[-1]
    hi = int(np.searchsorted(rates, target, side="right"))
    lo = hi - 1
    if hi >= len(rates):
        return ces[-1]
    if rates[hi] == rates[lo]:
        return ces[lo]
    w = (target - rates[lo]) / (rates[hi] - rates[lo])
    return ces[lo] * (1.0 - w) + ces[hi] * w


def build_target_cells(records, ref_ids, metric, targets):
    """Return target -> backbone -> (e2,vib) -> per-example CE array."""
    cells = {t: {"encoder": {}, "decoder": {}} for t in targets}
    det = {}
    for backbone in ("encoder", "decoder"):
        vib = defaultdict(list)
        det_recs = {}
        for rec in records:
            if rec.get("backbone_key") != backbone:
                continue
            if rec.get("kind") == "deterministic":
                det_recs[int(rec["e2_seed"])] = rec
            elif rec.get("kind") == "vib":
                vib[(int(rec["e2_seed"]), int(rec["seed"]))].append(rec)

        det_arrays = {}
        for e2, rec in det_recs.items():
            ce, _, _ = load_predictions(rec, ref_ids, "expected")
            det_arrays[e2] = ce

        curves = {}
        for key, recs in vib.items():
            recs = sorted(recs, key=lambda r: float(r["rate_val_bits"]))
            points = []
            for rec in recs:
                ce, _, _ = load_predictions(rec, ref_ids, metric)
                points.append((float(rec["rate_val_bits"]), ce))
            curves[key] = points

        for target in targets:
            for key, points in curves.items():
                interp = interp_at_rate(points, target)
                if interp is not None:
                    cells[target][backbone][key] = interp
        det[backbone] = det_arrays
    return cells, det


def _mean_arrays(arrays):
    return np.mean(np.stack(arrays, axis=0), axis=0)


def bootstrap(cells, det, targets, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for target in targets:
        enc = cells[target]["encoder"]
        dec = cells[target]["decoder"]
        common = sorted(set(enc) & set(dec))
        if not common:
            out[target] = None
            continue
        e2_seeds = sorted({k[0] for k in common})
        vib_by_e2 = {e2: sorted(k[1] for k in common if k[0] == e2) for e2 in e2_seeds}
        n_test = len(next(iter(enc.values())))

        def point(store):
            vals = []
            for e2 in e2_seeds:
                parts = [store[(e2, v)] for v in vib_by_e2[e2] if (e2, v) in store]
                if parts:
                    vals.append(_mean_arrays(parts))
            return _mean_arrays(vals)

        def point_pen(store):
            vals = []
            for e2 in e2_seeds:
                parts = [store[(e2, v)] - det["encoder" if store is enc else "decoder"][e2]
                         for v in vib_by_e2[e2] if (e2, v) in store]
                if parts:
                    vals.append(_mean_arrays(parts))
            return _mean_arrays(vals)

        enc_ce = point(enc)
        dec_ce = point(dec)
        enc_pen = point_pen(enc)
        dec_pen = point_pen(dec)

        draws = {k: [] for k in ("enc", "dec", "delta", "enc_pen", "dec_pen", "dpen")}
        for _ in range(int(n_boot)):
            chosen_e2 = rng.choice(e2_seeds, size=len(e2_seeds), replace=True)
            test_idx = rng.integers(0, n_test, n_test)
            enc_sel, dec_sel, encp_sel, decp_sel = [], [], [], []
            for e2 in chosen_e2:
                e2 = int(e2)
                vibs = vib_by_e2[e2]
                chosen_vib = rng.choice(vibs, size=len(vibs), replace=True)
                e_parts = [enc[(e2, int(v))] for v in chosen_vib if (e2, int(v)) in enc]
                d_parts = [dec[(e2, int(v))] for v in chosen_vib if (e2, int(v)) in dec]
                ep_parts = [enc[(e2, int(v))] - det["encoder"][e2]
                            for v in chosen_vib if (e2, int(v)) in enc]
                dp_parts = [dec[(e2, int(v))] - det["decoder"][e2]
                            for v in chosen_vib if (e2, int(v)) in dec]
                if e_parts:
                    enc_sel.append(_mean_arrays(e_parts))
                if d_parts:
                    dec_sel.append(_mean_arrays(d_parts))
                if ep_parts:
                    encp_sel.append(_mean_arrays(ep_parts))
                if dp_parts:
                    decp_sel.append(_mean_arrays(dp_parts))
            if not enc_sel or not dec_sel:
                continue
            e = _mean_arrays(enc_sel)[test_idx].mean()
            d = _mean_arrays(dec_sel)[test_idx].mean()
            ep = _mean_arrays(encp_sel)[test_idx].mean()
            dp = _mean_arrays(decp_sel)[test_idx].mean()
            draws["enc"].append(e)
            draws["dec"].append(d)
            draws["delta"].append(d - e)
            draws["enc_pen"].append(ep)
            draws["dec_pen"].append(dp)
            draws["dpen"].append(dp - ep)

        def summary(key, point_val):
            v = np.asarray(draws[key], dtype=np.float64)
            return {
                "point": float(point_val),
                "lo": float(np.quantile(v, 0.025)),
                "hi": float(np.quantile(v, 0.975)),
                "boot_mean": float(v.mean()),
                "p_le0": float((v <= 0).mean()),
            }

        out[target] = {
            "n_e2": len(e2_seeds),
            "n_cells": len(common),
            "enc": summary("enc", enc_ce.mean()),
            "dec": summary("dec", dec_ce.mean()),
            "delta": summary("delta", dec_ce.mean() - enc_ce.mean()),
            "enc_pen": summary("enc_pen", enc_pen.mean()),
            "dec_pen": summary("dec_pen", dec_pen.mean()),
            "dpen": summary("dpen", dec_pen.mean() - enc_pen.mean()),
        }
    return out


def fmt_ci(s):
    return f"[{s['lo']:+.4f}, {s['hi']:+.4f}]"


def write_markdown(path, rows, metric):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# E3-A matched-rate analysis ({metric} CE)\n\n")
        fh.write("Matched on validation rate. Per (E2 seed, VIB seed) interpolation; "
                 "95% CI from hierarchical bootstrap over E2 seeds, VIB seeds, and test examples.\n\n")
        fh.write("| Target R | E2 seeds | Cells | Encoder CE | Decoder CE | "
                 "Delta D | Delta 95% CI |\n")
        fh.write("|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            fh.write(f"| {r['target']:.2f} | {r['n_e2']} | {r['n_cells']} | "
                     f"{r['enc']['point']:.4f} | {r['dec']['point']:.4f} | "
                     f"{r['delta']['point']:+.4f} | {fmt_ci(r['delta'])} |\n")
        fh.write("\n### Compression penalty relative to deterministic control\n\n")
        fh.write("Delta penalty = decoder penalty - encoder penalty; negative means "
                 "the encoder has the larger penalty.\n\n")
        fh.write("| Target R | Encoder penalty | Decoder penalty | Delta penalty | "
                 "Delta penalty 95% CI |\n")
        fh.write("|---:|---:|---:|---:|---|\n")
        for r in rows:
            fh.write(f"| {r['target']:.2f} | {r['enc_pen']['point']:+.4f} | "
                     f"{r['dec_pen']['point']:+.4f} | {r['dpen']['point']:+.4f} | "
                     f"{fmt_ci(r['dpen'])} |\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="E3-A P2 matched-rate analysis")
    ap.add_argument("--run-root", default="runs/e3a_boolq_p2")
    ap.add_argument("--targets", default="1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0,8.0")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="runs/e3a_boolq_p2/analysis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)
    records = load_records(args.run_root)
    ref_ids = None
    for rec in records:
        if rec.get("kind") == "deterministic":
            _, _, ref_ids = load_predictions(rec, None, "expected")
            break
    if ref_ids is None:
        raise RuntimeError("no deterministic predictions found")

    for metric, label in (("expected", "expected"), ("mean", "mean-only")):
        cells, det = build_target_cells(records, ref_ids, metric, targets)
        result = bootstrap(cells, det, targets, args.n_boot, args.seed)
        rows = [{"target": t, **result[t]} for t in targets if result.get(t) is not None]
        write_markdown(os.path.join(args.out, f"matched_{metric}.md"), rows, label)
        with open(os.path.join(args.out, f"matched_{metric}.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["target", "n_e2", "n_cells",
                             "enc", "enc_lo", "enc_hi",
                             "dec", "dec_lo", "dec_hi",
                             "delta", "delta_lo", "delta_hi",
                             "enc_pen", "enc_pen_lo", "enc_pen_hi",
                             "dec_pen", "dec_pen_lo", "dec_pen_hi",
                             "dpen", "dpen_lo", "dpen_hi"])
            for r in rows:
                writer.writerow([
                    r["target"], r["n_e2"], r["n_cells"],
                    r["enc"]["point"], r["enc"]["lo"], r["enc"]["hi"],
                    r["dec"]["point"], r["dec"]["lo"], r["dec"]["hi"],
                    r["delta"]["point"], r["delta"]["lo"], r["delta"]["hi"],
                    r["enc_pen"]["point"], r["enc_pen"]["lo"], r["enc_pen"]["hi"],
                    r["dec_pen"]["point"], r["dec_pen"]["lo"], r["dec_pen"]["hi"],
                    r["dpen"]["point"], r["dpen"]["lo"], r["dpen"]["hi"],
                ])
        print(f"wrote {os.path.join(args.out, 'matched_' + metric + '.md')}")

    with open(os.path.join(args.out, "curves.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["backbone", "e2_seed", "vib_seed", "beta", "rate_val_bits",
                         "rate_test_bits", "expected_ce_bits", "mean_ce_bits",
                         "det_ce_bits"])
        for backbone in ("encoder", "decoder"):
            det = {int(r["e2_seed"]): r["test"]["ce_bits"] for r in records
                   if r.get("backbone_key") == backbone and r.get("kind") == "deterministic"}
            for rec in records:
                if rec.get("backbone_key") != backbone or rec.get("kind") != "vib":
                    continue
                writer.writerow([backbone, rec["e2_seed"], rec["seed"], rec["beta"],
                                 rec["rate_val_bits"], rec["rate_test_bits"],
                                 rec["expected_test_ce_bits"], rec["mean_test_ce_bits"],
                                 det.get(int(rec["e2_seed"]), "")])
    print(f"wrote {os.path.join(args.out, 'curves.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
