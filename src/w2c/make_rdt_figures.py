"""Generate the E1/E1.5/E2/E3-A figure set and the cross-experiment summary."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from .figure_utils import Plot

ENC = "#1f5fbf"
DEC = "#c23b22"
GRAY = "#666666"
GREEN = "#2b7a4b"
PURPLE = "#4a3fbf"
PRIOR_BOOLQ = 0.9567754419733319


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dir(path):
    rows = []
    for f in glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True):
        rows.extend(read_jsonl(f))
    return rows


def mean_sd(values):
    v = np.asarray([float(x) for x in values], dtype=float)
    if len(v) == 0:
        return float("nan"), 0.0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def t_ci(values):
    v = np.asarray([float(x) for x in values], dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float(v.mean()), float(v.mean())
    sem = v.std(ddof=1) / math.sqrt(n)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 1.96)
    return float(v.mean()), float(v.mean() - t * sem), float(v.mean() + t * sem)


def best_e1(path):
    rows = load_dir(path)
    out = {}
    for backbone in ("encoder", "decoder"):
        rr = [r for r in rows if r["backbone_key"] == backbone]
        if rr:
            out[backbone] = min(rr, key=lambda r: float(r["val_ce_bits"]))
    return out


def dataset_prior(config_path):
    import math
    from collections import Counter
    cfg = json.load(open(config_path, encoding="utf-8"))
    d = cfg["data"]
    classes = list(d["classes"])
    field = d["label_field"]
    counts = Counter()
    for line in open(d["test"], encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        value = raw.get(field)
        if isinstance(value, int):
            lid = value
        elif isinstance(value, str) and value.isdigit():
            lid = int(value)
        else:
            try:
                lid = classes.index(value)
            except ValueError:
                continue
        counts[lid] += 1
    n = sum(counts.values())
    probs = [counts[i] / n for i in range(len(classes)) if counts[i]]
    return -sum(p * math.log2(p) for p in probs)


def e1_layer_plot(path, title, out):
    rows = load_dir(path)
    layers = sorted({int(r["layer"]) for r in rows})
    series = {}
    for backbone in ("encoder", "decoder"):
        for probe in ("linear", "mlp"):
            vals = []
            for layer in layers:
                xs = [r["test"]["ce_bits"] for r in rows
                      if r["backbone_key"] == backbone and r["probe"] == probe
                      and int(r["layer"]) == layer]
                if xs:
                    vals.append((layer, float(np.mean(xs))))
            series[(backbone, probe)] = vals
    all_y = [y for pts in series.values() for _, y in pts] + [PRIOR_BOOLQ]
    p = Plot(title, "layer", "test CE (bits)",
             xlim=(min(layers) - 0.5, max(layers) + 0.5),
             ylim=(min(all_y) - 0.02, max(all_y) + 0.02))
    step = 4 if max(layers) > 10 else 1
    p.xticks = sorted(set(list(range(0, max(layers) + 1, step)) + [max(layers)]))
    p.line(min(layers), PRIOR_BOOLQ, max(layers), PRIOR_BOOLQ,
           color=GRAY, width=1, dash="4,4")
    for (backbone, probe), pts in series.items():
        color = ENC if backbone == "encoder" else DEC
        dash = "7,5" if probe == "mlp" else None
        p.polyline(pts, color=color, dash=dash)
        for x, y in pts:
            p.marker(x, y, color=color, kind="circle" if probe == "linear" else "square")
    p.legend([
        (ENC, "encoder linear", None, "circle"),
        (ENC, "encoder MLP", "7,5", "square"),
        (DEC, "decoder linear", None, "circle"),
        (DEC, "decoder MLP", "7,5", "square"),
    ], x=140, y=85)
    p.text(max(layers) - 0.5, PRIOR_BOOLQ + 0.004, "label prior", size=11, anchor="end", color=GRAY)
    p.save(out)


def e15_readout_plot(path, title, out):
    rows = list(csv.DictReader(open(os.path.join(path, "comparison.csv"), encoding="utf-8")))
    readouts = ["last", "mean", "attn", "concat"]
    selected = {}
    for backbone in ("encoder", "decoder"):
        for readout in readouts:
            rr = [r for r in rows if r["backbone"] == backbone and r["readout"] == readout]
            if rr:
                selected[(backbone, readout)] = min(rr, key=lambda r: float(r["val_ce_bits"]))
    vals = [float(r["ce_bits"]) for r in selected.values()]
    vals += [float(r["ce_lo"]) for r in selected.values()]
    vals += [float(r["ce_hi"]) for r in selected.values()]
    p = Plot(title, "readout", "test CE (bits)",
             xlim=(-0.5, len(readouts) - 0.5),
             ylim=(min(vals + [PRIOR_BOOLQ]) - 0.01, max(vals + [PRIOR_BOOLQ]) + 0.01))
    p.xticks = []
    p.line(-0.5, PRIOR_BOOLQ, len(readouts) - 0.5, PRIOR_BOOLQ,
           color=GRAY, width=1, dash="4,4")
    width = 0.34
    for i, readout in enumerate(readouts):
        for j, backbone in enumerate(("encoder", "decoder")):
            rec = selected.get((backbone, readout))
            if not rec:
                continue
            x = i + (-width / 2 if j == 0 else width / 2)
            y = float(rec["ce_bits"])
            color = ENC if backbone == "encoder" else DEC
            p.bar(x, y, width, color=color, baseline=min(p.ylim))
            p.errorbar(x, y, float(rec["ce_lo"]), float(rec["ce_hi"]), color="#222", cap=4, width=1.3)
        p.text(i, min(p.ylim) + 0.006, readout, size=12, anchor="middle")
    p.legend([(ENC, "encoder", None, "square"), (DEC, "decoder", None, "square")],
             x=150, y=88)
    p.text(len(readouts) - 0.6, PRIOR_BOOLQ + 0.004, "label prior",
           size=11, anchor="end", color=GRAY)
    p.save(out)


def e1_cross_task_plot(out):
    tasks = [
        ("BoolQ\n68M", "configs/e1_boolq_68m.json", "runs/e1_boolq_68m"),
        ("BoolQ\n150M", "configs/e0_boolq.json", "runs/e0_boolq"),
        ("LevBench\n150M", "configs/e1_levbench_12k.json", "runs/e1_levbench_12k"),
        ("W2C\n150M", "configs/e1_w2c_pref.json", "runs/e1_w2c_pref"),
        ("ProofWriter\n150M", "configs/e1_proofwriter.json", "runs/e1_proofwriter"),
    ]
    data = []
    for label, config_path, run_path in tasks:
        prior = dataset_prior(config_path)
        best = best_e1(run_path)
        for backbone in ("encoder", "decoder"):
            if backbone in best:
                data.append((label, backbone, prior - float(best[backbone]["test"]["ce_bits"])))
    labels = [t[0] for t in tasks]
    vals = [v for _, _, v in data]
    p = Plot("E1: best frozen probe improvement over label prior", "task",
             "prior CE $-$ test CE (bits)", xlim=(-0.5, len(labels) - 0.5),
             ylim=(min(vals + [0]) - 0.03, max(vals + [0]) + 0.03))
    p.xticks = []
    p.line(-0.5, 0, len(labels) - 0.5, 0, color=GRAY, width=1, dash="4,4")
    width = 0.34
    for i, label in enumerate(labels):
        for j, backbone in enumerate(("encoder", "decoder")):
            xs = [v for lab, b, v in data if lab == label and b == backbone]
            if not xs:
                continue
            x = i + (-width / 2 if j == 0 else width / 2)
            color = ENC if backbone == "encoder" else DEC
            p.bar(x, xs[0], width, color=color, baseline=0)
        p.text(i, min(p.ylim) + 0.008, label, size=10, anchor="middle")
    p.legend([(ENC, "encoder", None, "square"), (DEC, "decoder", None, "square")],
             x=150, y=88)
    p.save(out)


def e2_records(path):
    rows = []
    for f in glob.glob(os.path.join(path, "**", "runs.jsonl"), recursive=True):
        rows.extend(read_jsonl(f))
    return rows


def e2_curve(rows, backbone, readout, fraction):
    rr = [r for r in rows if r["backbone_key"] == backbone and r["readout"] == readout
          and abs(float(r["fraction"]) - fraction) < 1e-9]
    return mean_sd([r["test"]["ce_bits"] for r in rr]) if rr else (float("nan"), 0.0)


def e2_delta(rows, readout, fraction):
    by_seed = defaultdict(dict)
    for r in rows:
        if r["readout"] == readout and abs(float(r["fraction"]) - fraction) < 1e-9:
            by_seed[int(r["seed"])][r["backbone_key"]] = float(r["test"]["ce_bits"])
    ds = [v["decoder"] - v["encoder"] for v in by_seed.values()
          if "encoder" in v and "decoder" in v]
    return t_ci(ds) if ds else (float("nan"), float("nan"), float("nan"))


def e2_scaling_plot(rows, readout, title, out):
    fractions = sorted({float(r["fraction"]) for r in rows})
    series = {}
    for backbone in ("encoder", "decoder"):
        pts = []
        for f in fractions:
            y, sd = e2_curve(rows, backbone, readout, f)
            pts.append((f * 100.0, y, sd))
        series[backbone] = pts
    all_y = [y for pts in series.values() for _, y, _ in pts] + [PRIOR_BOOLQ]
    all_y += [y - sd for pts in series.values() for _, y, sd in pts]
    all_y += [y + sd for pts in series.values() for _, y, sd in pts]
    p = Plot(title, "training fraction (%)", "test CE (bits)",
             xlim=(0, 105), ylim=(min(all_y) - 0.01, max(all_y) + 0.01))
    p.xticks = [5, 10, 25, 50, 100]
    p.line(0, PRIOR_BOOLQ, 105, PRIOR_BOOLQ, color=GRAY, width=1, dash="4,4")
    for backbone, pts in series.items():
        color = ENC if backbone == "encoder" else DEC
        p.polyline([(x, y) for x, y, _ in pts], color=color)
        for x, y, sd in pts:
            p.errorbar(x, y, y - sd, y + sd, color=color, cap=4, width=1.5)
            p.marker(x, y, color=color, kind="circle")
    p.legend([(ENC, "encoder", None, "circle"), (DEC, "decoder", None, "circle")],
             x=150, y=88)
    p.text(103, PRIOR_BOOLQ + 0.004, "label prior", size=11, anchor="end", color=GRAY)
    p.save(out)


def e2_delta_plot(rows68, rows150, out):
    fractions = sorted({float(r["fraction"]) for r in rows68})
    series = [
        ("68M last-L19", rows68, "last-L19", ENC, None, "circle"),
        ("68M attn-L12", rows68, "attn-L12", PURPLE, "7,5", "square"),
        ("68M attn-L18", rows68, "attn-L18", GREEN, "3,4", "diamond"),
        ("150M last-L22", rows150, "last-L22", DEC, "5,4", "triangle"),
    ]
    data = {}
    all_lo, all_hi = [], []
    for label, rows, readout, color, dash, kind in series:
        pts = []
        for f in fractions:
            m, lo, hi = e2_delta(rows, readout, f)
            pts.append((f * 100, m, lo, hi))
            all_lo.append(lo)
            all_hi.append(hi)
        data[label] = (pts, color, dash, kind)
    p = Plot("E2: encoder–decoder gap vs data", "training fraction (%)",
             "decoder - encoder CE (bits)", xlim=(0, 105),
             ylim=(min(all_lo + [0]) - 0.02, max(all_hi + [0]) + 0.02))
    p.xticks = [5, 10, 25, 50, 100]
    p.line(0, 0, 105, 0, color=GRAY, width=1, dash="4,4")
    for label, (pts, color, dash, kind) in data.items():
        p.polyline([(x, y) for x, y, _, _ in pts], color=color, dash=dash)
        for x, y, lo, hi in pts:
            p.errorbar(x, y, lo, hi, color=color, cap=3, width=1.2)
            p.marker(x, y, color=color, kind=kind)
    p.legend([(c, lab, d, k) for lab, (_, c, d, k) in data.items()],
             x=130, y=85)
    p.save(out)


def e2_learning_plot(rows, out):
    # 100% data, matched final readout. Show train and validation dynamics.
    series = {}
    for backbone in ("encoder", "decoder"):
        rr = [r for r in rows if r["backbone_key"] == backbone
              and r["readout"] == "last-L19" and abs(float(r["fraction"]) - 1.0) < 1e-9]
        train_by_epoch = defaultdict(list)
        val_by_epoch = defaultdict(list)
        for r in rr:
            for h in r.get("history", []):
                train_by_epoch[int(h["epoch"])].append(float(h["train_loss"]) / math.log(2))
                val_by_epoch[int(h["epoch"])].append(float(h["val_ce_bits"]))
        series[(backbone, "train")] = [(e, float(np.mean(v))) for e, v in sorted(train_by_epoch.items())]
        series[(backbone, "val")] = [(e, float(np.mean(v))) for e, v in sorted(val_by_epoch.items())]
    max_epoch = max(e for pts in series.values() for e, _ in pts)
    all_y = [y for pts in series.values() for _, y in pts]
    p = Plot("E2: fine-tuning dynamics, 100% BoolQ", "epoch",
             "CE (bits)", xlim=(0, max_epoch), ylim=(min(all_y) - 0.02, max(all_y) + 0.02))
    p.xticks = list(range(0, max_epoch + 1))
    for (backbone, split), pts in series.items():
        color = ENC if backbone == "encoder" else DEC
        dash = "6,4" if split == "train" else None
        p.polyline(pts, color=color, dash=dash)
        for x, y in pts:
            p.marker(x, y, color=color, kind="circle" if split == "val" else "square")
    p.line(1, min(p.ylim), 1, max(p.ylim), color=GRAY, width=1, dash="3,3")
    p.text(1.08, max(p.ylim) - 0.08, "best validation region", size=11, color=GRAY)
    p.legend([(ENC, "encoder val", None, "circle"),
              (DEC, "decoder val", None, "circle"),
              (ENC, "encoder train", "6,4", "square"),
              (DEC, "decoder train", "6,4", "square")], x=120, y=80)
    p.save(out)


def e2_readout_plot(rows, out):
    readouts = ["last-L19", "attn-L12", "attn-L18"]
    p = None
    data = {}
    for backbone in ("encoder", "decoder"):
        for r in readouts:
            rr = [x for x in rows if x["backbone_key"] == backbone and x["readout"] == r
                  and abs(float(x["fraction"]) - 1.0) < 1e-9]
            if rr:
                data[(backbone, r)] = mean_sd([x["test"]["ce_bits"] for x in rr])
    vals = [v[0] for v in data.values()] + [PRIOR_BOOLQ]
    vals += [v[0] - v[1] for v in data.values()]
    vals += [v[0] + v[1] for v in data.values()]
    p = Plot("E2: readout at 100% BoolQ, 68M", "readout", "test CE (bits)",
             xlim=(-0.5, len(readouts) - 0.5),
             ylim=(min(vals) - 0.01, max(vals) + 0.01))
    p.xticks = []
    p.line(-0.5, PRIOR_BOOLQ, len(readouts) - 0.5, PRIOR_BOOLQ,
           color=GRAY, width=1, dash="4,4")
    width = 0.34
    for i, readout in enumerate(readouts):
        for j, backbone in enumerate(("encoder", "decoder")):
            if (backbone, readout) not in data:
                continue
            m, sd = data[(backbone, readout)]
            x = i + (-width / 2 if j == 0 else width / 2)
            color = ENC if backbone == "encoder" else DEC
            p.bar(x, m, width, color=color, baseline=min(p.ylim))
            p.errorbar(x, m, m - sd, m + sd, color="#222", cap=4, width=1.3)
        p.text(i, min(p.ylim) + 0.006, readout, size=11, anchor="middle")
    p.legend([(ENC, "encoder", None, "square"), (DEC, "decoder", None, "square")],
             x=150, y=88)
    p.text(len(readouts) - 0.6, PRIOR_BOOLQ + 0.004, "label prior",
           size=11, anchor="end", color=GRAY)
    p.save(out)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def e3_plot(kind, out):
    rows = read_csv("runs/e3a_boolq_p2/analysis/matched_expected.csv")
    if kind == "absolute":
        vals = [float(r["enc"]) for r in rows] + [float(r["dec"]) for r in rows]
        vals += [float(r["enc_lo"]) for r in rows] + [float(r["enc_hi"]) for r in rows]
        vals += [float(r["dec_lo"]) for r in rows] + [float(r["dec_hi"]) for r in rows]
        p = Plot("E3-A: absolute rate-distortion", "validation KL rate (bits/example)",
                 "test CE (bits)", xlim=(0, 8.8), ylim=(min(vals) - 0.02, max(vals) + 0.02))
        for r in rows:
            x = float(r["target"])
            p.errorbar(x, float(r["enc"]), float(r["enc_lo"]), float(r["enc_hi"]), color=ENC, cap=4)
            p.errorbar(x, float(r["dec"]), float(r["dec_lo"]), float(r["dec_hi"]), color=DEC, cap=4)
        p.polyline([(float(r["target"]), float(r["enc"])) for r in rows], color=ENC)
        p.polyline([(float(r["target"]), float(r["dec"])) for r in rows], color=DEC)
        for r in rows:
            p.marker(float(r["target"]), float(r["enc"]), color=ENC, kind="circle")
            p.marker(float(r["target"]), float(r["dec"]), color=DEC, kind="triangle")
        p.legend([(ENC, "encoder", None, "circle"), (DEC, "decoder", None, "triangle")],
                 x=630, y=85)
    elif kind == "delta":
        lo = min(float(r["delta_lo"]) for r in rows)
        hi = max(float(r["delta_hi"]) for r in rows)
        p = Plot("E3-A: matched-rate delta D", "validation KL rate (bits/example)",
                 "decoder - encoder CE (bits)", xlim=(0, 8.8),
                 ylim=(min(0, lo) - 0.02, max(0, hi) + 0.02))
        p.line(0, 0, 8.8, 0, color=GRAY, width=1, dash="4,4")
        p.polyline([(float(r["target"]), float(r["delta"])) for r in rows], color=PURPLE)
        for r in rows:
            x = float(r["target"])
            p.errorbar(x, float(r["delta"]), float(r["delta_lo"]), float(r["delta_hi"]), color=PURPLE)
            p.marker(x, float(r["delta"]), color=PURPLE, kind="circle")
    else:
        lo = min(float(r["dpen_lo"]) for r in rows)
        hi = max(float(r["dpen_hi"]) for r in rows)
        p = Plot("E3-A: compression-penalty delta", "validation KL rate (bits/example)",
                 "decoder penalty - encoder penalty", xlim=(0, 8.8),
                 ylim=(min(0, lo) - 0.02, max(0, hi) + 0.02))
        p.line(0, 0, 8.8, 0, color=GRAY, width=1, dash="4,4")
        p.polyline([(float(r["target"]), float(r["dpen"])) for r in rows], color=GREEN)
        for r in rows:
            x = float(r["target"])
            p.errorbar(x, float(r["dpen"]), float(r["dpen_lo"]), float(r["dpen_hi"]), color=GREEN)
            p.marker(x, float(r["dpen"]), color=GREEN, kind="circle")
    p.xticks = [1, 2, 3, 4, 5, 6, 8]
    p.save(out)


def e3_ablation_plot(out):
    roots = [
        ("main", "runs/e3a_boolq_p2", ENC, None, "circle"),
        ("latent 128", "runs/e3a_boolq_p2_lat128", DEC, "3,4", "diamond"),
        ("no scaling", "runs/e3a_boolq_p2_noscale", GRAY, "6,3", "square"),
        ("per-dim", "runs/e3a_boolq_p2_perdim", GREEN, None, "circle"),
    ]
    data = {}
    all_y = []
    for label, root, color, dash, kind in roots:
        rows = read_csv(os.path.join(root, "analysis", "matched_expected.csv"))
        if label == "5 VIB seeds":
            rows = [r for r in rows if float(r["target"]) <= 5.0]
        pts = [(float(r["target"]), float(r["delta"])) for r in rows]
        data[label] = (pts, color, dash, kind)
        all_y.extend(y for _, y in pts)
    p = Plot("E3-A: matched-rate delta across ablations",
             "validation KL rate (bits/example)", "decoder - encoder CE (bits)",
             xlim=(0, 8.8), ylim=(min(all_y + [0]) - 0.02, max(all_y + [0]) + 0.02))
    p.xticks = [1, 2, 3, 4, 5, 6, 8]
    p.line(0, 0, 8.8, 0, color=GRAY, width=1, dash="4,4")
    for label, (pts, color, dash, kind) in data.items():
        p.polyline(pts, color=color, dash=dash)
        for x, y in pts:
            p.marker(x, y, color=color, kind=kind)
    p.legend([(c, lab, d, k) for lab, (_, c, d, k) in data.items()], x=610, y=80)
    p.save(out)


def e3_variance_plot(out):
    exp = {float(r["target"]): r for r in read_csv("runs/e3a_boolq_p2/analysis/matched_expected.csv")}
    mean = {float(r["target"]): r for r in read_csv("runs/e3a_boolq_p2/analysis/matched_mean.csv")}
    ts = sorted(set(exp) & set(mean))
    enc = [(t, float(exp[t]["enc"]) - float(mean[t]["enc"])) for t in ts]
    dec = [(t, float(exp[t]["dec"]) - float(mean[t]["dec"])) for t in ts]
    all_y = [y for _, y in enc + dec] + [0]
    p = Plot("E3-A: expected minus mean-only CE", "validation KL rate (bits/example)",
             "CE gap (bits)", xlim=(0, 8.8), ylim=(min(all_y) - 0.01, max(all_y) + 0.01))
    p.xticks = [1, 2, 3, 4, 5, 6, 8]
    p.line(0, 0, 8.8, 0, color=GRAY, width=1, dash="4,4")
    p.polyline(enc, color=ENC)
    p.polyline(dec, color=DEC)
    for x, y in enc:
        p.marker(x, y, color=ENC, kind="circle")
    for x, y in dec:
        p.marker(x, y, color=DEC, kind="triangle")
    p.legend([(ENC, "encoder", None, "circle"), (DEC, "decoder", None, "triangle")],
             x=630, y=85)
    p.save(out)


def story_summary(out):
    # E1.5 best 68M frozen
    e15 = list(csv.DictReader(open("runs/e15_boolq_68m/comparison.csv", encoding="utf-8")))
    best = {}
    for b in ("encoder", "decoder"):
        best[b] = min([r for r in e15 if r["backbone"] == b], key=lambda r: float(r["val_ce_bits"]))
    e2 = {}
    for f in glob.glob("runs/e2_boolq_68m/**/runs.jsonl", recursive=True):
        for r in read_jsonl(f):
            if r["readout"] == "last-L19" and abs(float(r["fraction"]) - 1.0) < 1e-9:
                e2.setdefault(r["backbone_key"], []).append(float(r["test"]["ce_bits"]))
    e3 = {float(r["target"]): r for r in read_csv("runs/e3a_boolq_p2/analysis/matched_expected.csv")}
    stages = [
        ("Frozen\nE1.5 best", {"encoder": float(best["encoder"]["ce_bits"]),
                               "decoder": float(best["decoder"]["ce_bits"])}),
        ("Fine-tuned\nE2 100%", {"encoder": float(np.mean(e2["encoder"])),
                                  "decoder": float(np.mean(e2["decoder"]))}),
        ("Compressed\nE3-A R=2", {"encoder": float(e3[2.0]["enc"]),
                                   "decoder": float(e3[2.0]["dec"])}),
    ]
    vals = [v for _, d in stages for v in d.values()]
    p = Plot("68M BoolQ: encoder vs decoder across E1/E2/E3-A", "stage",
             "test CE (bits)", xlim=(-0.5, len(stages) - 0.5),
             ylim=(min(vals) - 0.03, max(vals) + 0.03))
    p.xticks = []
    width = 0.34
    for i, (label, d) in enumerate(stages):
        for j, backbone in enumerate(("encoder", "decoder")):
            x = i + (-width / 2 if j == 0 else width / 2)
            color = ENC if backbone == "encoder" else DEC
            p.bar(x, d[backbone], width, color=color, baseline=min(p.ylim))
        p.text(i, min(p.ylim) + 0.006, label, size=10, anchor="middle")
    p.legend([(ENC, "encoder", None, "square"), (DEC, "decoder", None, "square")],
             x=150, y=88)
    p.save(out)


def main():
    os.makedirs("figures", exist_ok=True)
    e1_layer_plot("runs/e1_boolq_68m", "E1: frozen probe by layer, BoolQ 68M",
                  "figures/fig01_e1_layer_68m.svg")
    e1_layer_plot("runs/e0_boolq", "E1: frozen probe by layer, BoolQ 150M",
                  "figures/fig02_e1_layer_150m.svg")
    e15_readout_plot("runs/e15_boolq_68m", "E1.5: readout comparison, BoolQ 68M",
                     "figures/fig03_e15_readout_68m.svg")
    e15_readout_plot("runs/e15_boolq", "E1.5: readout comparison, BoolQ 150M",
                     "figures/fig04_e15_readout_150m.svg")
    e1_cross_task_plot("figures/fig05_e1_cross_task.svg")

    rows68 = e2_records("runs/e2_boolq_68m")
    rows150 = e2_records("runs/e2_boolq_full")
    e2_scaling_plot(rows68, "last-L19", "E2: data scaling, BoolQ 68M",
                    "figures/fig06_e2_scaling_68m.svg")
    e2_scaling_plot(rows150, "last-L22", "E2: data scaling, BoolQ 150M",
                    "figures/fig07_e2_scaling_150m.svg")
    e2_delta_plot(rows68, rows150, "figures/fig08_e2_delta_n.svg")
    e2_learning_plot(rows68, "figures/fig09_e2_learning_curves.svg")
    e2_readout_plot(rows68, "figures/fig10_e2_readout_100.svg")

    e3_plot("absolute", "figures/fig11_e3a_absolute_rd.svg")
    e3_plot("delta", "figures/fig12_e3a_delta.svg")
    e3_plot("penalty", "figures/fig13_e3a_penalty.svg")
    e3_ablation_plot("figures/fig14_e3a_ablation_delta.svg")
    e3_variance_plot("figures/fig15_e3a_variance.svg")
    story_summary("figures/fig16_story_summary.svg")
    print("wrote figures 01-16")


if __name__ == "__main__":
    main()
