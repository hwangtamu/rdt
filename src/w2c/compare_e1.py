"""Compare E1 encoder vs decoder runs on identical test examples.

Delta D(R) here is delta D = D_decoder - D_encoder at a matched (layer, probe).
Reported CIs are paired bootstraps over test examples, using per-example CE
averaged across seeds first (so seed noise does not masquerade as example noise).

Example
-------
    PYTHONPATH=src python -m w2c.compare_e1 --run-dir runs/e1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .metrics import paired_bootstrap_delta


def read_runs(path: str) -> list[dict]:
    """Read run records, de-duplicating repeated invocations by run_id (last wins)."""
    by_id: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                record = json.loads(line)
                by_id[record["run_id"]] = record
    return list(by_id.values())


def load_pred(path: str, ids_order: list[str], labels_order: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data["ids"]]
        labels = data["labels"].astype(np.int64)
        ce = data["ce_bits"].astype(np.float64)
    if ids != ids_order:
        index = {example_id: i for i, example_id in enumerate(ids)}
        missing = [x for x in ids_order if x not in index]
        if missing:
            raise AssertionError(f"{path}: missing {len(missing)} test examples")
        order = [index[x] for x in ids_order]
        ids = [ids[i] for i in order]
        labels = labels[order]
        ce = ce[order]
    if not np.array_equal(labels, labels_order):
        raise AssertionError(f"{path}: labels differ from reference test set")
    return ce


def compare(run_dir: str, reference: str = "encoder", candidate: str = "decoder",
            n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    runs = [r for r in read_runs(os.path.join(run_dir, "runs.jsonl"))
            if r["backbone_key"] in (reference, candidate)]
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for rec in runs:
        by_key[(rec["layer"], rec["probe"], rec["seed"])][rec["backbone_key"]] = rec

    paired = defaultdict(dict)  # (layer, probe) -> seed -> (ce_ref, ce_cand)
    test_ids: list[str] | None = None
    labels: np.ndarray | None = None
    for (layer, probe, s), backs in sorted(by_key.items()):
        if reference not in backs or candidate not in backs:
            continue
        ref_id, cand_id = backs[reference]["run_id"], backs[candidate]["run_id"]
        if test_ids is None:
            with np.load(os.path.join(run_dir, "preds", ref_id + ".npz")) as data:
                test_ids = [str(x) for x in data["ids"]]
                labels = data["labels"].astype(np.int64)
        ref_ce = load_pred(os.path.join(run_dir, "preds", ref_id + ".npz"), test_ids, labels)
        cand_ce = load_pred(os.path.join(run_dir, "preds", cand_id + ".npz"), test_ids, labels)
        paired[(layer, probe)][s] = (ref_ce, cand_ce, backs[reference], backs[candidate])

    rows = []
    for (layer, probe), seeds in sorted(paired.items()):
        ref_stack = np.stack([v[0] for v in seeds.values()])
        cand_stack = np.stack([v[1] for v in seeds.values()])
        ref_ce = ref_stack.mean(axis=0)
        cand_ce = cand_stack.mean(axis=0)
        stats = paired_bootstrap_delta(ref_ce, cand_ce, n_boot=n_boot, seed=seed, alpha=alpha)
        per_seed_delta = [float((v[1] - v[0]).mean()) for v in seeds.values()]
        rows.append({
            "layer": layer,
            "probe": probe,
            "n_seeds": len(seeds),
            "seeds": ",".join(str(x) for x in sorted(seeds)),
            "ce_reference": float(ref_ce.mean()),
            "ce_candidate": float(cand_ce.mean()),
            "delta_mean": stats["delta_mean"],
            "delta_lo": stats["delta_lo"],
            "delta_hi": stats["delta_hi"],
            "p_two_sided": stats["p_two_sided"],
            "prob_encoder_better": stats["prob_encoder_better"],
            "delta_seed_sd": float(np.std(per_seed_delta, ddof=1)) if len(per_seed_delta) > 1 else 0.0,
        })

    csv_path = os.path.join(run_dir, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["layer"])
        writer.writeheader()
        writer.writerows(rows)

    md = [
        f"# E1 comparison: {reference} (R) vs {candidate} (C)",
        "",
        f"Delta D = D_{candidate} - D_{reference}; negative favors the encoder.",
        "Paired bootstrap over test examples on seed-averaged per-example CE.",
        "",
        "| Layer | Probe | Seeds | CE ref | CE cand | Delta D | 95% CI | p | P(ref better) |",
        "|------:|-------|------:|-------:|--------:|--------:|-------:|--:|--------------:|",
    ]
    for row in rows:
        md.append(
            f"| {row['layer']} | {row['probe']} | {row['n_seeds']} | "
            f"{row['ce_reference']:.4f} | {row['ce_candidate']:.4f} | "
            f"{row['delta_mean']:+.4f} | [{row['delta_lo']:+.4f}, {row['delta_hi']:+.4f}] | "
            f"{row['p_two_sided']:.3f} | {row['prob_encoder_better']:.3f} |"
        )
    md.append("")
    md.append("Reminder: frozen-probe differences describe linear/MLP *accessibility*, "
              "not differences in I(Y;Z). See plan section 6.")
    md_path = os.path.join(run_dir, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    return {"rows": rows, "csv": csv_path, "markdown": md_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare E1 encoder vs decoder probes")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference", default="encoder")
    parser.add_argument("--candidate", default="decoder")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    result = compare(args.run_dir, args.reference, args.candidate,
                     n_boot=args.bootstrap, alpha=args.alpha)
    if not result["rows"]:
        print("[compare] no matched encoder/decoder runs found")
        return 1
    for row in result["rows"]:
        print(f"[compare] L{row['layer']:<2} {row['probe']:<6} "
              f"delta={row['delta_mean']:+.4f} "
              f"[{row['delta_lo']:+.4f}, {row['delta_hi']:+.4f}] "
              f"p={row['p_two_sided']:.3f}")
    print(f"[compare] wrote {result['csv']} and {result['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
