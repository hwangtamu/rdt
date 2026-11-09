"""Build E0 BoolQ data: frozen-backbone binary decision sanity check.

E0 asks the simplest possible transfer question: can either frozen backbone
separate two classes (No / Yes) on a clean, well-known binary task?  If a
linear probe cannot beat the 50 % prior on BoolQ, it is not evidence that the
harder LevBench task is purely a capacity problem.

Inputs are the official BoolQ parquet files (google/boolq).  The official
validation split is divided once, stratified by label, into E0 val and test so
that probe selection and the reported number never share examples.

    PYTHONPATH=src python -m w2c.build_boolq_e0 \
        --train-parquet data/boolq_raw/data/train-00000-of-00001.parquet \
        --val-parquet data/boolq_raw/data/validation-00000-of-00001.parquet \
        --out-dir data/boolq_e0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

LABELS = {False: 0, True: 1}
OPTIONS = ["No", "Yes"]


def render_prompt(passage: str, question: str) -> str:
    lines = [
        "Passage:",
        passage.strip(),
        "",
        f"Question: {question.strip()}",
        "",
        "Answer the question using only the passage.",
        "",
    ]
    lines.extend(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(OPTIONS))
    return "\n".join(lines).strip() + "\n\nAnswer:"


def to_record(row: dict, split: str, index: int) -> dict:
    label = LABELS[bool(row["answer"])]
    rid = f"boolq:{split}/{index}"
    return {
        "id": rid,
        "group_id": rid,
        "question": row["question"].strip(),
        "prompt": render_prompt(row["passage"], row["question"]),
        "label": label,
        "n_options": len(OPTIONS),
        "source": "boolq",
        "tier": split,
    }


def split_validation(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_label[LABELS[bool(row["answer"])]].append(i)
    rng = random.Random(seed)
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(by_label):
        indices = by_label[label][:]
        rng.shuffle(indices)
        # Alternate within each class to keep both halves class-balanced.
        val_idx.extend(indices[0::2])
        test_idx.extend(indices[1::2])
    val_idx.sort()
    test_idx.sort()
    return (
        [to_record(rows[i], "validation", i) for i in val_idx],
        [to_record(rows[i], "validation", i) for i in test_idx],
    )


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build E0 BoolQ data")
    ap.add_argument(
        "--train-parquet",
        default="data/boolq_raw/data/train-00000-of-00001.parquet",
    )
    ap.add_argument(
        "--val-parquet",
        default="data/boolq_raw/data/validation-00000-of-00001.parquet",
    )
    ap.add_argument("--out-dir", default="data/boolq_e0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = pq.read_table(args.train_parquet).to_pylist()
    val_rows = pq.read_table(args.val_parquet).to_pylist()

    train = [to_record(row, "train", i) for i, row in enumerate(train_rows)]
    val, test = split_validation(val_rows, args.seed)

    for name, records in (("train", train), ("val", val), ("test", test)):
        write_jsonl(records, out / f"{name}.jsonl")

    stats = {
        "sources": {
            "train": args.train_parquet,
            "val": args.val_parquet,
            "test": args.val_parquet,
        },
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "label_counts": {
            name: dict(sorted(Counter(r["label"] for r in records).items()))
            for name, records in (("train", train), ("val", val), ("test", test))
        },
        "n_options": len(OPTIONS),
        "option_text": OPTIONS,
        "validation_split": {
            "method": "stratified alternating halves of google/boolq validation",
            "seed": args.seed,
            "val_fraction": 0.5,
        },
    }
    with (out / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"[e0] wrote {out}: " + ", ".join(
        f"{k}={v}" for k, v in stats["counts"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
