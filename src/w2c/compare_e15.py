"""Compare E1.5 readouts (last / mean / attention / layer-concat).

Selects one configuration per (backbone, readout, probe) by mean **validation**
CE over seeds, then reports test metrics and a paired bootstrap of the selected
readout against that backbone's ``last`` baseline on identical test examples.

Because the selection sees validation only, the test columns are not selection-
contaminated. Parameter counts are reported for every readout: attention pooling
and layer concatenation are not parameter-matched to the linear last-token
baseline.

    PYTHONPATH=src python -m w2c.compare_e15 --run-dir runs/e15_w2c_pref
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .compare_e1 import load_pred, read_runs
from .metrics import paired_bootstrap_delta


def _config_key(record: dict):
    if record.get("layer_set"):
        return ("concat", tuple(int(x) for x in record["layer_set"]))
    if record.get("layer") is not None:
        return ("layer", int(record["layer"]))
    raise ValueError(f"run {record.get('run_id')} has neither layer nor layer_set")


def _config_label(config) -> str:
    kind, value = config
    return f"L{value}" if kind == "layer" else "concat[" + ",".join(
        str(x) for x in value) + "]"


def _mean(values) -> float:
    return float(np.mean(values))


def _bootstrap_mean(values: np.ndarray, n_boot: int, seed: int,
                    alpha: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    n = len(values)
    for b in range(n_boot):
        draws[b] = values[rng.integers(0, n, n)].mean()
    return (float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))


def _pred_path(run_dir: str, record: dict) -> str:
    return os.path.join(run_dir, "preds", record["run_id"] + ".npz")


def compare(run_dir: str, baseline: str = "last", n_boot: int = 2000,
            alpha: float = 0.05, seed: int = 0) -> dict:
    runs = read_runs(os.path.join(run_dir, "runs.jsonl"))
    if not runs:
        raise ValueError(f"no runs in {run_dir}")

    # (backbone, readout, probe) -> config -> seed -> record
    grouped = defaultdict(lambda: defaultdict(dict))
    for record in runs:
        grouped[(record["backbone_key"], record["readout"], record["probe"])][
            _config_key(record)][int(record["seed"])] = record

    # Reference test ordering / labels.
    ids_ref: list[str] | None = None
    labels_ref: np.ndarray | None = None
    for record in runs:
        with np.load(_pred_path(run_dir, record), allow_pickle=False) as data:
            ids_ref = [str(x) for x in data["ids"]]
            labels_ref = data["labels"].astype(np.int64)
        break

    # Aggregate every configuration over seeds: validation selects, test reports.
    all_rows = []
    selected: dict[tuple, tuple] = {}
    for key, configs in sorted(grouped.items()):
        for config, by_seed in sorted(configs.items()):
            seeds = sorted(by_seed)
            val_ce = _mean([by_seed[s]["val_ce_bits"] for s in seeds])
            test = [by_seed[s]["test"] for s in seeds]
            row = {
                "backbone": key[0],
                "readout": key[1],
                "probe": key[2],
                "config": _config_label(config),
                "n_seeds": len(seeds),
                "n_params": int(by_seed[seeds[0]]["probe_info"]["n_params"]),
                "val_ce_bits": val_ce,
                "ce_bits": _mean([t["ce_bits"] for t in test]),
                "ce_seed_sd": float(np.std([t["ce_bits"] for t in test], ddof=1))
                if len(test) > 1 else 0.0,
                "accuracy": _mean([t["accuracy"] for t in test]),
                "macro_f1": _mean([t["macro_f1"] for t in test]),
                "ece": _mean([t["ece"] for t in test]),
            }
            all_rows.append(row)
            current = selected.get(key)
            if current is None or val_ce < current[0]:
                selected[key] = (val_ce, config, by_seed)

    all_path = os.path.join(run_dir, "all_configs.csv")
    with open(all_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    def mean_ce(by_seed: dict, seeds: list[int]) -> np.ndarray:
        stack = [load_pred(_pred_path(run_dir, by_seed[s]), ids_ref, labels_ref)
                 for s in seeds]
        return np.stack(stack).mean(axis=0)

    # Selected table + paired delta against the selected `last` baseline.
    selected_rows = []
    for key, (val_ce, config, by_seed) in sorted(selected.items()):
        backbone, readout, probe = key
        seeds = sorted(by_seed)
        ce = mean_ce(by_seed, seeds)
        lo, hi = _bootstrap_mean(ce, n_boot, seed, alpha)
        record = by_seed[seeds[0]]
        row = {
            "backbone": backbone,
            "readout": readout,
            "probe": probe,
            "config": _config_label(config),
            "n_seeds": len(seeds),
            "n_params": int(record["probe_info"]["n_params"]),
            "val_ce_bits": val_ce,
            "ce_bits": float(ce.mean()),
            "ce_lo": lo,
            "ce_hi": hi,
            "accuracy": _mean([by_seed[s]["test"]["accuracy"] for s in seeds]),
            "macro_f1": _mean([by_seed[s]["test"]["macro_f1"] for s in seeds]),
            "ece": _mean([by_seed[s]["test"]["ece"] for s in seeds]),
            "delta_vs_last": 0.0,
            "delta_lo": 0.0,
            "delta_hi": 0.0,
            "p_two_sided": 1.0,
        }
        base = selected.get((backbone, baseline, probe))
        if readout != baseline and base is not None:
            base_seeds = sorted(base[2])
            common = [s for s in seeds if s in base_seeds]
            if common:
                cand_ce = mean_ce(by_seed, common)
                ref_ce = mean_ce(base[2], common)
                stats = paired_bootstrap_delta(ref_ce, cand_ce, n_boot=n_boot,
                                               seed=seed, alpha=alpha)
                row.update({
                    "delta_vs_last": stats["delta_mean"],
                    "delta_lo": stats["delta_lo"],
                    "delta_hi": stats["delta_hi"],
                    "p_two_sided": stats["p_two_sided"],
                })
        selected_rows.append(row)

    csv_path = os.path.join(run_dir, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    # Encoder-vs-decoder at matched (readout, probe, config).
    pair_rows = []
    configs_by_arm = defaultdict(dict)  # (readout, probe, config) -> backbone -> by_seed
    for (backbone, readout, probe), configs in grouped.items():
        for config, by_seed in configs.items():
            configs_by_arm[(readout, probe, config)][backbone] = by_seed
    for (readout, probe, config), arms in sorted(configs_by_arm.items(),
                                                 key=lambda x: (x[0][0], x[0][1],
                                                                str(x[0][2]))):
        if "encoder" not in arms or "decoder" not in arms:
            continue
        common = sorted(set(arms["encoder"]) & set(arms["decoder"]))
        if not common:
            continue
        enc_ce = mean_ce(arms["encoder"], common)
        dec_ce = mean_ce(arms["decoder"], common)
        stats = paired_bootstrap_delta(enc_ce, dec_ce, n_boot=n_boot,
                                       seed=seed, alpha=alpha)
        pair_rows.append({
            "readout": readout,
            "probe": probe,
            "config": _config_label(config),
            "n_seeds": len(common),
            "ce_encoder": float(enc_ce.mean()),
            "ce_decoder": float(dec_ce.mean()),
            "delta_decoder_minus_encoder": stats["delta_mean"],
            "delta_lo": stats["delta_lo"],
            "delta_hi": stats["delta_hi"],
            "p_two_sided": stats["p_two_sided"],
            "prob_encoder_better": stats["prob_encoder_better"],
        })

    pair_path = os.path.join(run_dir, "encoder_vs_decoder.csv")
    with open(pair_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys())
                                if pair_rows else ["readout"])
        writer.writeheader()
        writer.writerows(pair_rows)

    md = [
        "# E1.5 — frozen backbone, readout comparison",
        "",
        f"Run directory: `{run_dir}`",
        f"Selection: minimum mean **validation** CE over seeds; test is held out.",
        f"Baseline readout for ΔD: `{baseline}`.",
        "ΔD = D_readout − D_baseline (negative favors the alternative readout);",
        "paired bootstrap over test examples on seed-averaged per-example CE.",
        "",
        "## Selected configuration per backbone / readout / probe",
        "",
        "| Backbone | Readout | Probe | Config | Params | Val CE | CE (bits) [95% CI] | Acc | Macro-F1 | ECE | ΔD vs last [95% CI] | p |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in selected_rows:
        delta = "—"
        if row["readout"] != baseline and row["delta_vs_last"] != 0.0:
            delta = (f"{row['delta_vs_last']:+.4f} "
                     f"[{row['delta_lo']:+.4f}, {row['delta_hi']:+.4f}]")
        md.append(
            f"| {row['backbone']} | {row['readout']} | {row['probe']} | "
            f"{row['config']} | {row['n_params']:,} | {row['val_ce_bits']:.4f} | "
            f"{row['ce_bits']:.4f} [{row['ce_lo']:.4f}, {row['ce_hi']:.4f}] | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['ece']:.4f} | "
            f"{delta} | {row['p_two_sided']:.3f} |"
        )
    md += [
        "",
        "## Encoder vs decoder at matched readout / probe / config",
        "",
        "ΔD = CE_decoder − CE_encoder (negative favors the encoder).",
        "",
        "| Readout | Probe | Config | Enc CE | Dec CE | ΔD [95% CI] | p |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in pair_rows:
        md.append(
            f"| {row['readout']} | {row['probe']} | {row['config']} | "
            f"{row['ce_encoder']:.4f} | {row['ce_decoder']:.4f} | "
            f"{row['delta_decoder_minus_encoder']:+.4f} "
            f"[{row['delta_lo']:+.4f}, {row['delta_hi']:+.4f}] | "
            f"{row['p_two_sided']:.3f} |"
        )
    md += [
        "",
        "## Guardrails",
        "",
        "* Frozen readouts measure linear/MLP accessibility, not `I(Y;Z)`.",
        "* Attention pooling and layer concatenation add parameters; they are not",
        "  parameter-matched to the last-token linear baseline.",
        "* Layer/config selection uses validation CE only; the test set is untouched.",
        "* `all_configs.csv` has the full layer curves, so the selection step can be audited.",
        "",
    ]
    md_path = os.path.join(run_dir, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    return {"selected": selected_rows, "pairs": pair_rows,
            "comparison": csv_path, "all_configs": all_path,
            "encoder_vs_decoder": pair_path, "markdown": md_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare E1.5 readouts")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline", default="last")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    result = compare(args.run_dir, baseline=args.baseline,
                     n_boot=args.bootstrap, alpha=args.alpha)
    for row in result["selected"]:
        print(f"[e15] {row['backbone']:<8} {row['readout']:<7} {row['probe']:<6} "
              f"{row['config']:<20} CE={row['ce_bits']:.4f} "
              f"delta={row['delta_vs_last']:+.4f} p={row['p_two_sided']:.3f}")
    print(f"[e15] wrote {result['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
