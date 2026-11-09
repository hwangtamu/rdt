"""Build ProofWriter E1/E2 data with theory-disjoint depth splits.

Primary depth configs: depth-0, depth-1, depth-2, depth-3, depth-5.
`allProofs` is never rendered: it leaks the answer.

Outputs under data/proofwriter_e1/:
  train.jsonl / val.jsonl / test.jsonl   mixed-depth sampled pool
  depth-<d>/{train,val,test}.jsonl       within-depth subsets
  extrap/{train,val,test}.jsonl          train 0/1/2, val 2, test 3/5
  stats.json

Groups are theory hashes and are split globally, so a theory never crosses
splits even when it appears under multiple depth configs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

DEPTHS = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-5"]
LABEL = {"False": 0, "True": 1, "Unknown": 2}
CLASSES = ["false", "true", "unknown"]


def group_key(theory: str) -> str:
    return hashlib.sha1(theory.strip().encode("utf-8")).hexdigest()[:16]


def row_key(row: dict) -> str:
    q = hashlib.sha1(str(row["question"]).strip().encode("utf-8")).hexdigest()[:10]
    return f"pw:{row['config']}:{row['id']}:{q}"


def render_prompt(theory: str, question: str) -> str:
    return (
        f"Theory:\n{theory.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer with one of: True, False, Unknown.\n\n"
        "Answer:"
    )


def to_record(row: dict) -> dict:
    answer = str(row["answer"])
    if answer not in LABEL:
        raise ValueError(f"unexpected answer: {answer}")
    group = group_key(str(row["theory"]))
    return {
        "id": row_key(row),
        "group_id": f"pw:{group}",
        "question": str(row["question"]).strip(),
        "prompt": render_prompt(str(row["theory"]), str(row["question"])),
        "label": LABEL[answer],
        "n_options": len(CLASSES),
        "source": str(row["config"]),
        "tier": str(row["config"]),
        "depth_config": str(row["config"]),
        "QDep": int(row["QDep"]),
        "answer": answer,
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_config_split(rows: list[dict], groups: list[str],
                        target: int, rng: random.Random) -> list[dict]:
    """Add whole theory groups until the target row count is reached."""
    groups = list(groups)
    rng.shuffle(groups)
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    chosen = []
    for group in groups:
        if len(chosen) >= target:
            break
        chosen.extend(by_group[group])
    return chosen


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "by_config": dict(Counter(r["depth_config"] for r in rows)),
        "by_label": dict(Counter(r["answer"] for r in rows)),
        "by_qdep": dict(sorted(Counter(r["QDep"] for r in rows).items())),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build ProofWriter depth-controlled E1 data")
    ap.add_argument("--raw-dir", default="data/proofwriter_raw/data")
    ap.add_argument("--out-dir", default="data/proofwriter_e1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-per-depth", type=int, default=4000)
    ap.add_argument("--val-per-depth", type=int, default=800)
    ap.add_argument("--test-per-depth", type=int, default=800)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    raw = Path(args.raw_dir)
    records = []
    for path in sorted(raw.glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            if str(row["config"]) in DEPTHS:
                records.append(to_record(row))

    print(f"[pw] usable rows: {len(records):,}; groups: "
          f"{len({r['group_id'] for r in records}):,}")

    # Global theory-level split. Same theory always goes to the same split,
    # even if it occurs under several depth configs.
    groups = sorted({r["group_id"] for r in records})
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    group_split = {}
    for i, group in enumerate(groups):
        if i < n_train:
            group_split[group] = "train"
        elif i < n_train + n_val:
            group_split[group] = "val"
        else:
            group_split[group] = "test"

    by_split = defaultdict(list)
    for row in records:
        by_split[group_split[row["group_id"]]].append(row)

    # Depth-stratified, group-safe sampling within each split.
    sampled = {"train": [], "val": [], "test": []}
    targets = {"train": args.train_per_depth,
               "val": args.val_per_depth,
               "test": args.test_per_depth}
    for split in ("train", "val", "test"):
        split_rng = random.Random(args.seed * 1009 + {"train": 1, "val": 2, "test": 3}[split])
        for depth in DEPTHS:
            sub = [r for r in by_split[split] if r["depth_config"] == depth]
            groups_here = sorted({r["group_id"] for r in sub})
            chosen = sample_config_split(sub, groups_here, targets[split], split_rng)
            sampled[split].extend(chosen)
            print(f"[pw] {split:5s} {depth:8s} rows={len(chosen):6d} "
                  f"groups={len({r['group_id'] for r in chosen}):5d}")
        sampled[split].sort(key=lambda r: r["id"])

    for split in ("train", "val", "test"):
        write_jsonl(sampled[split], out / f"{split}.jsonl")

    # Within-depth subsets.
    for depth in DEPTHS:
        for split in ("train", "val", "test"):
            rows = [r for r in sampled[split] if r["depth_config"] == depth]
            write_jsonl(rows, out / depth / f"{split}.jsonl")

    # Depth extrapolation: train on 0/1/2, validate on held-out depth-2
    # theories, test on depth-3 and depth-5.
    extrap = {
        "train": [r for r in sampled["train"]
                  if r["depth_config"] in {"depth-0", "depth-1", "depth-2"}],
        "val": [r for r in sampled["val"] if r["depth_config"] == "depth-2"],
        "test": [r for r in sampled["test"]
                 if r["depth_config"] in {"depth-3", "depth-5"}],
    }
    for split, rows in extrap.items():
        write_jsonl(rows, out / "extrap" / f"{split}.jsonl")

    stats = {
        "raw_dir": str(raw),
        "depths": DEPTHS,
        "classes": CLASSES,
        "label_map": LABEL,
        "sampling": {
            "train_per_depth": args.train_per_depth,
            "val_per_depth": args.val_per_depth,
            "test_per_depth": args.test_per_depth,
            "seed": args.seed,
        },
        "mixed": {split: summarize(rows) for split, rows in sampled.items()},
        "extrap": {split: summarize(rows) for split, rows in extrap.items()},
        "within_depth": {
            depth: {split: summarize([r for r in sampled[split]
                                      if r["depth_config"] == depth])
                    for split in ("train", "val", "test")}
            for depth in DEPTHS
        },
    }
    with (out / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"[pw] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
