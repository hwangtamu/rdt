"""Aggregate E1-style frozen-probing runs across the whole Ettin family.

Reads one or more E1 run directories (the family is run as two GPU shards) and
emits a single cross-model comparison for BoolQ:

    runs/e1_boolq_family/family_layers.csv   every (model, probe, layer, seed)
    runs/e1_boolq_family/family_summary.csv  best layer per model/probe
    runs/e1_boolq_family/family_deltas.csv   matched-depth paired deltas
    runs/e1_boolq_family/family_report.md    human-readable tables

Layer selection is by **validation** CE (mean over seeds), so the test column is
not selected on test. An oracle column (best test CE) is reported separately and
labelled as such.

    PYTHONPATH=src python -m w2c.compare_e1_family \
        --run-dirs runs/e1_boolq_family_g1 runs/e1_boolq_family_g2 \
        --out runs/e1_boolq_family
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .metrics import bootstrap_ci, classification_metrics, paired_bootstrap_delta

SIZES = ["17m", "32m", "68m", "150m", "400m", "1b"]
SIZE_ORDER = {s: i for i, s in enumerate(SIZES)}
FAMILY_LABEL = {
    "enc": "encoder",
    "dec": "decoder",
    "encfd": "encoder\u2190decoder",
    "decfe": "decoder\u2190encoder",
}
PAIRS = {
    # (reference, candidate): Delta D = D_candidate - D_reference
    "enc_vs_dec": ("enc", "dec"),
    "encfd_vs_decfe": ("encfd", "decfe"),
    "enc_vs_decfe": ("enc", "decfe"),
    "encfd_vs_dec": ("encfd", "dec"),
    "enc_vs_encfd_init": ("enc", "encfd"),
    "dec_vs_decfe_init": ("dec", "decfe"),
}


def split_key(key: str) -> tuple[str, str]:
    """'encfd-150m' -> ('encfd', '150m'); bare legacy keys pass through."""
    if "-" not in key:
        return key, "unknown"
    return tuple(key.rsplit("-", 1))


def index_runs(run_dirs: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    records: dict[str, dict] = {}
    pred_dirs: dict[str, str] = {}
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "runs.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records[rec["run_id"]] = rec
                pred_dirs[rec["run_id"]] = os.path.join(run_dir, "preds")
    return records, pred_dirs


def load_ce(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data["ids"]]
        labels = data["labels"].astype(np.int64)
        ce = data["ce_bits"].astype(np.float64)
    return ids, labels, ce


def load_logits(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data["ids"]]
        labels = data["labels"].astype(np.int64)
        logits = data["logits"].astype(np.float64)
    return ids, labels, logits


class RunIndex:
    def __init__(self, run_dirs: list[str]):
        self.records, self.pred_dirs = index_runs(run_dirs)
        # (backbone_key, probe, layer) -> {seed: record}
        self.cells: dict[tuple[str, str, int], dict[int, dict]] = defaultdict(dict)
        for rec in self.records.values():
            self.cells[(rec["backbone_key"], rec["probe"], int(rec["layer"]))][
                int(rec["seed"])] = rec
        self._ref_ids: list[str] | None = None
        self._ref_labels: np.ndarray | None = None

    # -- reference test order: first run encountered, all runs share the split --
    @property
    def ref_ids(self) -> list[str]:
        if self._ref_ids is None:
            self._load_reference()
        return self._ref_ids

    @property
    def ref_labels(self) -> np.ndarray:
        if self._ref_labels is None:
            self._load_reference()
        return self._ref_labels

    def _load_reference(self) -> None:
        for rec in self.records.values():
            path = os.path.join(self.pred_dirs[rec["run_id"]], rec["run_id"] + ".npz")
            ids, labels, _ = load_ce(path)
            self._ref_ids, self._ref_labels = ids, labels
            return
        raise RuntimeError("no runs found")

    def per_example_ce(self, run_id: str) -> np.ndarray:
        path = os.path.join(self.pred_dirs[run_id], run_id + ".npz")
        ids, labels, ce = load_ce(path)
        if ids != self.ref_ids:
            index = {x: i for i, x in enumerate(ids)}
            ce = np.asarray([ce[index[x]] for x in self.ref_ids])
        return ce

    def seed_averaged_ce(self, key: str, probe: str, layer: int) -> np.ndarray:
        seeds = self.cells[(key, probe, layer)]
        if not seeds:
            raise KeyError(f"no runs for {key} {probe} L{layer}")
        return np.mean([self.per_example_ce(rec["run_id"])
                        for rec in seeds.values()], axis=0)

    def seed_averaged_metrics(self, key: str, probe: str, layer: int) -> dict:
        """Mean over seeds of the per-seed logits metrics."""
        rows = []
        for seed, rec in sorted(self.cells[(key, probe, layer)].items()):
            path = os.path.join(self.pred_dirs[rec["run_id"]], rec["run_id"] + ".npz")
            ids, labels, logits = load_logits(path)
            if ids != self.ref_ids:
                index = {x: i for i, x in enumerate(ids)}
                order = [index[x] for x in self.ref_ids]
                logits, labels = logits[order], labels[order]
            m = classification_metrics(logits, labels, n_bins=10, k=2)
            rows.append(m)
        out = {}
        for name in ("ce_bits", "accuracy", "macro_f1", "ece"):
            values = np.asarray([r[name] for r in rows], dtype=float)
            out[name] = float(values.mean())
            out[name + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out["n_seeds"] = len(rows)
        return out

    def val_ce(self, key: str, probe: str, layer: int) -> float:
        values = [float(rec["val_ce_bits"])
                  for rec in self.cells[(key, probe, layer)].values()]
        return float(np.mean(values))

    def layers(self, key: str, probe: str) -> list[int]:
        return sorted({layer for (k, p, layer) in self.cells if k == key and p == probe})

    def has(self, key: str, probe: str) -> bool:
        return any(k == key and p == probe for (k, p, _l) in self.cells)


def best_layer(index: RunIndex, key: str, probe: str, criterion: str) -> tuple[int, float]:
    layers = index.layers(key, probe)
    scores = [(index.val_ce(key, probe, layer), layer) for layer in layers]
    best_val, layer = min(scores)
    if criterion == "val":
        return layer, best_val
    # oracle: pick by test CE (reported only as an upper bound)
    tests = [(index.seed_averaged_metrics(key, probe, lay)["ce_bits"], lay)
             for lay in layers]
    return min(tests)[1], best_val


def layer_rows(index: RunIndex, backbone_keys: list[str]) -> list[dict]:
    rows = []
    for key in backbone_keys:
        family, size = split_key(key)
        for probe in ("linear", "mlp"):
            if not index.has(key, probe):
                continue
            for layer in index.layers(key, probe):
                for seed, rec in sorted(index.cells[(key, probe, layer)].items()):
                    rows.append({
                        "model": key,
                        "family": family,
                        "size": size,
                        "probe": probe,
                        "layer": layer,
                        "seed": seed,
                        "val_ce_bits": float(rec["val_ce_bits"]),
                        "test_ce_bits": float(rec["test"]["ce_bits"]),
                        "test_accuracy": float(rec["test"]["accuracy"]),
                        "test_macro_f1": float(rec["test"]["macro_f1"]),
                        "test_ece": float(rec["test"]["ece"]),
                        "probe_params": int(rec["probe_info"]["n_params"]),
                    })
    return rows


def summary_rows(index: RunIndex, backbone_keys: list[str]) -> list[dict]:
    rows = []
    for key in backbone_keys:
        family, size = split_key(key)
        for probe in ("linear", "mlp"):
            if not index.has(key, probe):
                continue
            layer, best_val = best_layer(index, key, probe, "val")
            metrics = index.seed_averaged_metrics(key, probe, layer)
            oracle_layer, _ = best_layer(index, key, probe, "test")
            oracle_metrics = index.seed_averaged_metrics(key, probe, oracle_layer)
            rows.append({
                "model": key,
                "family": family,
                "size": size,
                "probe": probe,
                "n_layers": len(index.layers(key, probe)),
                "best_layer_val": layer,
                "val_ce_bits": best_val,
                "test_ce_bits": metrics["ce_bits"],
                "test_ce_sd": metrics["ce_bits_sd"],
                "test_accuracy": metrics["accuracy"],
                "test_accuracy_sd": metrics["accuracy_sd"],
                "test_macro_f1": metrics["macro_f1"],
                "test_ece": metrics["ece"],
                "oracle_layer_test": oracle_layer,
                "oracle_test_ce_bits": oracle_metrics["ce_bits"],
                "oracle_test_accuracy": oracle_metrics["accuracy"],
            })
    rows.sort(key=lambda r: (SIZE_ORDER.get(r["size"], 99), r["family"], r["probe"]))
    return rows


def delta_rows(index: RunIndex, backbone_keys: list[str],
               n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> list[dict]:
    have = set(backbone_keys)
    rows = []
    for label, (ref_family, cand_family) in PAIRS.items():
        for size in SIZES:
            ref_key, cand_key = f"{ref_family}-{size}", f"{cand_family}-{size}"
            if ref_key not in have or cand_key not in have:
                continue
            for probe in ("linear", "mlp"):
                if not (index.has(ref_key, probe) and index.has(cand_key, probe)):
                    continue
                # matched depth = the reference model's best validation layer
                layer, _ = best_layer(index, ref_key, probe, "val")
                ref_ce = index.seed_averaged_ce(ref_key, probe, layer)
                cand_ce = index.seed_averaged_ce(cand_key, probe, layer)
                stats = paired_bootstrap_delta(ref_ce, cand_ce, n_boot=n_boot,
                                               seed=seed, alpha=alpha)
                rows.append({
                    "comparison": label,
                    "size": size,
                    "probe": probe,
                    "layer": layer,
                    "ce_reference": float(ref_ce.mean()),
                    "ce_candidate": float(cand_ce.mean()),
                    "delta_mean": stats["delta_mean"],
                    "delta_lo": stats["delta_lo"],
                    "delta_hi": stats["delta_hi"],
                    "p_two_sided": stats["p_two_sided"],
                    # paired_bootstrap_delta reports P(candidate better) as
                    # prob_encoder_better (delta < 0), so invert for the reference.
                    "prob_reference_better": 1.0 - float(stats["prob_encoder_better"]),
                })
    rows.sort(key=lambda r: (r["comparison"], SIZE_ORDER.get(r["size"], 99),
                             r["probe"]))
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown(summary: list[dict], deltas: list[dict]) -> str:
    lines = ["# E1 on BoolQ — full Ettin family", ""]
    lines += [
        "Frozen backbones, last-token pooling at the end of the prompt, linear and",
        "MLP probes (768->256, GELU, dropout 0.1), 3 seeds, standardized features,",
        "layer chosen by **validation** CE. Test = 1,634 held-out BoolQ examples.",
        "Chance CE = 1.0 bit; majority-class accuracy = 0.622.",
        "",
    ]
    for probe in ("linear", "mlp"):
        lines += [f"## {probe} probe (best validation layer)", "",
                  "| Size | Model | Layers | L | Test CE (bits) | Acc | Macro-F1 | ECE |",
                  "|------|-------|---:|---:|---:|---:|---:|---:|"]
        for row in summary:
            if row["probe"] != probe:
                continue
            lines.append(
                f"| {row['size']} | {FAMILY_LABEL.get(row['family'], row['family'])} | {row['n_layers']} | "
                f"{row['best_layer_val']} | {_fmt(row['test_ce_bits'])} ± "
                f"{_fmt(row['test_ce_sd'])} | {_fmt(row['test_accuracy'])} | "
                f"{_fmt(row['test_macro_f1'])} | {_fmt(row['test_ece'])} |")
        lines.append("")

    lines += ["## Matched-depth paired deltas", "",
              "Delta D = D_candidate - D_reference on seed-averaged per-example CE;",
              "paired bootstrap over test examples. Negative favours the candidate.",
              ""]
    for label, (ref_family, cand_family) in PAIRS.items():
        subset = [r for r in deltas if r["comparison"] == label]
        if not subset:
            continue
        lines += [f"### {FAMILY_LABEL.get(ref_family, ref_family)} (ref) vs "
                  f"{FAMILY_LABEL.get(cand_family, cand_family)} (cand)",
                  "",
                  "| Size | Probe | Layer | CE ref | CE cand | Delta D [95% CI] |",
                  "|------|-------|------:|-------:|--------:|------------------|"]
        for row in subset:
            sig = "" if row["delta_lo"] <= 0 <= row["delta_hi"] else " *"
            lines.append(
                f"| {row['size']} | {row['probe']} | {row['layer']} | "
                f"{_fmt(row['ce_reference'], 4)} | {_fmt(row['ce_candidate'], 4)} | "
                f"{row['delta_mean']:+.4f} [{row['delta_lo']:+.4f}, "
                f"{row['delta_hi']:+.4f}]{sig} |")
        lines.append("")
    lines += ["`*` = 95% CI excludes zero.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare E1 runs across model families")
    parser.add_argument("--run-dirs", nargs="+",
                        default=["runs/e1_boolq_family_g1", "runs/e1_boolq_family_g2"])
    parser.add_argument("--out", default="runs/e1_boolq_family")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default=None,
                        help="comma-separated backbone keys (default: all found)")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    index = RunIndex(args.run_dirs)
    keys = sorted({rec["backbone_key"] for rec in index.records.values()})
    if args.models:
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        keys = [k for k in keys if k in wanted]
    print(f"[family] {len(keys)} backbones, {len(index.records)} runs")

    layers = layer_rows(index, keys)
    summary = summary_rows(index, keys)
    deltas = delta_rows(index, keys, n_boot=args.n_boot, alpha=args.alpha,
                        seed=args.seed)
    write_csv(os.path.join(args.out, "family_layers.csv"), layers)
    write_csv(os.path.join(args.out, "family_summary.csv"), summary)
    write_csv(os.path.join(args.out, "family_deltas.csv"), deltas)
    with open(os.path.join(args.out, "family_report.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown(summary, deltas))
    print(f"[family] wrote {args.out}/family_{{layers,summary,deltas}}.csv + "
          f"family_report.md")

    for row in summary:
        print(f"[family] {row['model']:<12s} {row['probe']:<6s} "
              f"L{row['best_layer_val']:<3d} CE={row['test_ce_bits']:.4f} "
              f"acc={row['test_accuracy']:.4f} F1={row['test_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
