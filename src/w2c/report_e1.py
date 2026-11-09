"""Render a human-readable E1 report from a completed run directory.

    PYTHONPATH=src python -m w2c.report_e1 --run-dir runs/e1_w2c_pref
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, pstdev


def _runs(run_dir: str) -> list[dict]:
    by_id = {}
    with open(os.path.join(run_dir, "runs.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                by_id[rec["run_id"]] = rec
    return list(by_id.values())


def _comparison(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "comparison.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_report(run_dir: str) -> str:
    runs = _runs(run_dir)
    comp = _comparison(run_dir)
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))

    classes = manifest.get("classes", [])
    keys = sorted({r["backbone_key"] for r in runs})
    layers = sorted({r["layer"] for r in runs})
    probes = sorted({r["probe"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    n_test = manifest["counts"]["test"]
    trunc = {r["backbone_key"]: r.get("truncation_rate", 0.0) for r in runs}

    lines = [
        "# E1 report — frozen representation probing",
        "",
        f"Run directory: `{run_dir}`",
        "",
        f"- Classes ({len(classes)}): {', '.join(classes)}",
        f"- Splits: " + ", ".join(f"{k}={v}" for k, v in manifest["counts"].items()),
        f"- Layers probed: {layers}",
        f"- Probes: {probes}; seeds: {seeds}",
        f"- Target: last non-pad token of `\"Decision:\"`",
        f"- Test truncation rate: " + ", ".join(
            f"{k}={v:.3%}" for k, v in trunc.items()),
        "",
        "## Label distribution",
        "",
        "| Split | " + " | ".join(classes) + " |",
        "|---|" + "---:|" * len(classes),
    ]
    for split in ("train", "val", "test"):
        counts = manifest["label_counts"].get(split, {})
        lines.append(f"| {split} | " + " | ".join(str(counts.get(c, 0)) for c in classes) + " |")
    if manifest.get("dropped"):
        lines += ["", f"Dropped during loading: `{manifest['dropped']}`"]

    lines += ["", "## Mean test metrics by layer (averaged over seeds)", ""]
    by_cell = defaultdict(list)
    for rec in runs:
        by_cell[(rec["backbone_key"], rec["layer"], rec["probe"])].append(rec)
    for probe in probes:
        lines += [f"### {probe}", "",
                  "| Backbone | Layer | CE (bits) | Accuracy | Macro-F1 | ECE |",
                  "|---|---:|---:|---:|---:|---:|"]
        for bb in keys:
            for layer in layers:
                cell = by_cell.get((bb, layer, probe))
                if not cell:
                    continue
                lines.append(
                    f"| {bb} | {layer} | "
                    f"{mean(r['test']['ce_bits'] for r in cell):.4f} | "
                    f"{mean(r['test']['accuracy'] for r in cell):.4f} | "
                    f"{mean(r['test']['macro_f1'] for r in cell):.4f} | "
                    f"{mean(r['test']['ece'] for r in cell):.4f} |"
                )
        lines.append("")

    if comp:
        lines += ["## Paired encoder-vs-decoder comparison", "",
                  "Delta D = D_decoder − D_encoder (negative favors the encoder).", "",
                  "| Layer | Probe | Delta D | 95% CI | p |",
                  "|---:|---|---:|---:|---:|"]
        for row in comp:
            lines.append(
                f"| {row['layer']} | {row['probe']} | {float(row['delta_mean']):+.4f} | "
                f"[{float(row['delta_lo']):+.4f}, {float(row['delta_hi']):+.4f}] | "
                f"{float(row['p_two_sided']):.3f} |"
            )
        lines.append("")

    # Best linear configuration per backbone, by mean CE.
    lines += ["## Best linear probe per backbone (by mean CE)", "",
              "| Backbone | Layer | CE | Accuracy | Macro-F1 | ECE |",
              "|---|---:|---:|---:|---:|---:|"]
    for bb in keys:
        cells = [(layer, by_cell[(bb, layer, "linear")]) for layer in layers
                 if by_cell.get((bb, layer, "linear"))]
        layer, cell = min(cells, key=lambda x: mean(r["test"]["ce_bits"] for r in x[1]))
        lines.append(
            f"| {bb} | {layer} | {mean(r['test']['ce_bits'] for r in cell):.4f} | "
            f"{mean(r['test']['accuracy'] for r in cell):.4f} | "
            f"{mean(r['test']['macro_f1'] for r in cell):.4f} | "
            f"{mean(r['test']['ece'] for r in cell):.4f} |"
        )
    lines.append("")

    # Confusion matrix of the best linear config, first seed.
    lines += ["## Confusion matrix, best linear config (seed 0)", "",
              f"Rows = true label, columns = predicted; order = {classes}.", ""]
    for bb in keys:
        cells = [(layer, by_cell[(bb, layer, "linear")]) for layer in layers
                 if by_cell.get((bb, layer, "linear"))]
        layer, _ = min(cells, key=lambda x: mean(r["test"]["ce_bits"] for r in x[1]))
        rec = next(r for r in by_cell[(bb, layer, "linear")] if r["seed"] == 0)
        lines += [f"**{bb}, layer {layer}**", "", "```"]
        for row in rec["test"]["confusion"]:
            lines.append(" ".join(f"{v:5d}" for v in row))
        lines += ["```", ""]

    lines += [
        "## Interpretation guardrails",
        "",
        "* Frozen probing measures **linear/MLP accessibility** of decision information, "
        "not differences in `I(Y;Z)` (plan section 6).",
        "* The official MCQ test set is a different distribution from the preference/SFT "
        "training pool; prior shift and domain shift both contribute to absolute scores.",
        "* Layer 0 is the embedding of the shared final token (`:`) and is therefore "
        "input-invariant; its probes can only learn the class prior.",
        "* Differences that flip sign across layers/probes are not evidence for a "
        "uniform architecture advantage. Report the layer and probe with the comparison.",
        "* This release has no usable `direct` examples (5/24,000 responses; 0 in test), "
        "so E1 is run over the three supported classes unless stated otherwise.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render E1 report")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    text = build_report(args.run_dir)
    out = args.out or os.path.join(args.run_dir, "REPORT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"[report] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
