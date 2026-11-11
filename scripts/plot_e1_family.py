"""Dependency-free SVG plots for the E1 full-family BoolQ comparison.

    PYTHONPATH=src python scripts/plot_e1_family.py --run-dir runs/e1_boolq_family

Inputs (written by ``w2c.compare_e1_family``):
  family_summary.csv, family_layers.csv

Outputs:
  family_ce_vs_size.svg   validation-selected test CE vs model size
  family_depth_1b.svg     test CE vs relative depth for the 1b models
"""

from __future__ import annotations

import argparse
import csv
import math
import os

from w2c.figure_utils import Plot

PARAMS = {"17m": 17, "32m": 32, "68m": 67, "150m": 150, "400m": 395, "1b": 1000}
COLORS = {"enc": "#1f5fbf", "dec": "#c0392b",
          "encfd": "#2ca02c", "decfe": "#e08a1e"}
LABEL = {"enc": "encoder (scratch)", "dec": "decoder (scratch)",
         "encfd": "encoder<-decoder", "decfe": "decoder<-encoder"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ce_vs_size(summary, out_dir):
    rows = [r for r in summary if r["size"] in PARAMS]
    ce = [float(r["test_ce_bits"]) for r in rows]
    lo, hi = min(ce), max(ce)
    pad = (hi - lo) * 0.15 or 0.01
    x = [math.log10(PARAMS[r["size"]]) for r in rows]
    plot = Plot(
        title="BoolQ: frozen linear/MLP probe CE vs Ettin model size",
        xlabel="parameters (millions, log scale)", ylabel="test CE (bits), "
        "validation-selected layer",
        xlim=(min(x) - 0.1, max(x) + 0.1), ylim=(lo - pad, hi + pad))
    ticks = sorted({round(math.log10(v), 6) for v in PARAMS.values()})
    plot.xticks = ticks
    plot.xtick_labels = [f"{round(10 ** t)}" for t in ticks]
    plot.yticks = [lo - pad + i * (hi - lo + 2 * pad) / 5 for i in range(6)]

    for probe, dash in (("linear", None), ("mlp", "6,4")):
        for family in ("enc", "dec", "encfd", "decfe"):
            pts = sorted(
                ((math.log10(PARAMS[r["size"]]), float(r["test_ce_bits"]))
                 for r in rows if r["probe"] == probe and r["family"] == family))
            if not pts:
                continue
            plot.polyline(pts, color=COLORS[family], width=2.5, dash=dash)
            for px, py in pts:
                plot.marker(px, py, color=COLORS[family],
                            kind="circle" if probe == "linear" else "square")
    entries = [(COLORS[f], LABEL[f], None, "circle") for f in
               ("enc", "dec", "encfd", "decfe")]
    entries += [("#555", "solid = linear probe", None, "circle"),
                ("#555", "dashed = MLP probe", "6,4", "square")]
    plot.legend(entries)
    path = os.path.join(out_dir, "family_ce_vs_size.svg")
    plot.save(path)
    return path


def depth_profile(layers, out_dir, size="1b", probe="linear"):
    rows = [r for r in layers if r["size"] == size and r["probe"] == probe]
    if not rows:
        return None
    max_layer = max(int(r["layer"]) for r in rows)
    agg: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        agg.setdefault((r["family"], int(r["layer"])), []).append(
            float(r["test_ce_bits"]))
    ce = [v for vals in agg.values() for v in vals]
    lo, hi = min(ce), max(ce)
    pad = (hi - lo) * 0.1 or 0.01
    plot = Plot(
        title=f"BoolQ {probe} probe: test CE vs depth ({size} models)",
        xlabel="relative depth (0 = embeddings, 1 = final block)",
        ylabel="test CE (bits)",
        xlim=(-0.02, 1.02), ylim=(lo - pad, hi + pad))
    for family in ("enc", "dec", "encfd", "decfe"):
        pts = sorted((layer / max_layer, sum(v) / len(v))
                     for (fam, layer), v in agg.items() if fam == family)
        if not pts:
            continue
        plot.polyline(pts, color=COLORS[family], width=2.5)
        for px, py in pts:
            plot.marker(px, py, color=COLORS[family], size=2.5)
    plot.legend([(COLORS[f], LABEL[f], None, "circle")
                 for f in ("enc", "dec", "encfd", "decfe")])
    path = os.path.join(out_dir, f"family_depth_{size}.svg")
    plot.save(path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/e1_boolq_family")
    args = parser.parse_args(argv)
    summary = read_csv(os.path.join(args.run_dir, "family_summary.csv"))
    layers = read_csv(os.path.join(args.run_dir, "family_layers.csv"))
    print(ce_vs_size(summary, args.run_dir))
    for size in ("1b", "150m"):
        for probe in ("linear", "mlp"):
            path = depth_profile(layers, args.run_dir, size=size, probe=probe)
            if path:
                print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
