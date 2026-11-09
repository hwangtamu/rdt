"""Build E1-v2 data: Lev-train-v2 (train) -> LevBench (val/test).

The When2Call E1 scaffold assumes a fixed semantic label space.  Lev/LevBench
are variable-K multiple-choice tasks, so this converter flattens every row to a
common ``A..F`` option head (K <= 6), keeping the gold option index and K per
example.  ``data.label_mode = "index"`` and the metric/probe masking added in
this revision consume those fields.

Split policy (mirrors the E1 spirit: train and benchmark never share a state):

* train: every usable row from Lev-train-v2 ``s120`` splits;
* val:   LevBench ``development`` (used for early stopping/selection only);
* test:  LevBench ``test`` (held out until comparison).

Rows that are explicitly unanswerable, carry a uniform soft target, have more
than six options, or repeat option text are dropped and counted.  Lev-train-v2
was designed to be state-disjoint from LevBench; this script verifies that and
removes any accidental overlap.

    PYTHONPATH=src python -m w2c.build_lev_e1 \
        --levtrain /mnt/local-disk/bench/levtrain \
        --levbench /mnt/local-disk/bench/levbench \
        --out-dir data/levbench_e1
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]


def text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)


def state_key(state) -> str:
    return hashlib.sha256(text(state).strip().encode("utf-8")).hexdigest()


def opt_text(key, val) -> str:
    """Lev sometimes stores the option text in the key and leaves the value null."""
    if val is None or (isinstance(val, str) and not val.strip()):
        return str(key)
    return text(val)


def _uniform_target(target) -> bool:
    return isinstance(target, dict) and bool(target) and len(set(target.values())) == 1


def convert(raw: dict, max_k: int) -> tuple[dict | None, str | None]:
    meta = raw.get("_meta") or {}
    questions = raw.get("questions") or {}
    if not questions:
        return None, "no_questions"
    q = next(iter(questions.values()))
    t = q.get("type")
    crit = q.get("criteria")

    if t == "choice":
        if not isinstance(crit, dict):
            return None, "choice_without_criteria"
        keys = list(crit)
        if q.get("label") not in crit:
            return None, "choice_label_missing"
        options = [opt_text(k, crit[k]) for k in keys]
        gold = keys.index(q["label"])
    elif t == "noul":
        if isinstance(crit, dict) and crit:
            keys = list(crit)
            options = [opt_text(k, crit[k]) for k in keys]
        else:
            keys = ["false", "true"]
            options = ["No", "Yes"]
        try:
            gold = int(bool(q["label"]))
        except (TypeError, ValueError):
            return None, "noul_bad_label"
        if gold >= len(options):
            return None, "noul_label_out_of_range"
    elif t == "score":
        if not isinstance(crit, list) or not crit:
            return None, "score_without_levels"
        options = [text(c) for c in crit]
        try:
            gold = int(q["label"])
        except (TypeError, ValueError):
            return None, "score_bad_label"
        if not 0 <= gold < len(options):
            return None, "score_label_out_of_range"
    else:
        return None, f"unsupported_type:{t}"

    # Drop repeated option strings.  If the gold text is duplicated the item is
    # ambiguous; otherwise drop the repeated distractor(s) and remap the gold.
    seen: dict[str, int] = {}
    keep: list[int] = []
    for i, option in enumerate(options):
        key = text(option).strip()
        if key in seen:
            continue
        seen[key] = i
        keep.append(i)
    gold_text = text(options[gold]).strip()
    if sum(1 for o in options if text(o).strip() == gold_text) > 1:
        return None, "duplicate_gold_text"
    if len(keep) != len(options):
        if gold not in keep:
            return None, "gold_dropped_with_duplicate"
        options = [options[i] for i in keep]
        gold = keep.index(gold)

    if len(options) < 2:
        return None, "fewer_than_two_options"
    if len(options) > max_k:
        return None, f"too_many_options:{len(options)}"

    if meta.get("unanswerable") or _uniform_target(q.get("target")):
        return None, "unanswerable_or_uniform_target"

    source = str(meta.get("source") or q.get("src") or "lev")
    group = str(meta.get("group_id") or meta.get("id") or state_key(raw.get("state")))
    record_id = str(meta.get("id") or meta.get("row_sha256") or state_key(raw.get("state")))
    return {
        "state": raw.get("state"),
        "instructions": text(q.get("instructions", "")).strip(),
        "options": options,
        "gold": int(gold),
        "source": source,
        "family": str(meta.get("family") or source),
        "tier": str(meta.get("tier") or "train"),
        "group_id": group,
        "id": record_id,
        "state_key": state_key(raw.get("state")),
    }, None


def render_prompt(rec: dict) -> str:
    lines = [text(rec["state"]).strip(), "",
             rec["instructions"], ""]
    lines.extend(f"{LETTERS[i]}. {o}" for i, o in enumerate(rec["options"]))
    return "\n".join(lines).strip() + "\n\nAnswer:"


def to_e1(rec: dict, prefix: str) -> dict:
    return {
        "id": f"{prefix}:{rec['id']}",
        "question": text(rec["state"]).strip(),
        "prompt": render_prompt(rec),
        "label": int(rec["gold"]),
        "n_options": len(rec["options"]),
        "group_id": f"{prefix}:{rec['group_id']}",
        "source": rec["source"],
        "family": rec["family"],
        "tier": rec["tier"],
    }


def load_split(bench_dir: str, split: str, max_k: int) -> tuple[list[dict], Counter]:
    rows, stats = [], Counter()
    for p in sorted(glob.glob(os.path.join(bench_dir, "data", "*", split + ".jsonl"))):
        for line_no, line in enumerate(open(p, encoding="utf-8"), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            rec, reason = convert(raw, max_k)
            if rec is None:
                stats[reason] += 1
                continue
            rows.append(rec)
    return rows, stats


def load_train(levtrain_dir: str, splits: list[str], max_k: int) -> tuple[list[dict], Counter]:
    rows, stats = [], Counter()
    for split in splits:
        for p in sorted(glob.glob(os.path.join(levtrain_dir, "data", "*", split + ".jsonl"))):
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                raw = json.loads(line)
                rec, reason = convert(raw, max_k)
                if rec is None:
                    stats[reason] += 1
                    continue
                rows.append(rec)
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lev-train-v2 + LevBench -> E1-v2 data")
    ap.add_argument("--levtrain", default="/mnt/local-disk/bench/levtrain")
    ap.add_argument("--levbench", default="/mnt/local-disk/bench/levbench")
    ap.add_argument("--out-dir", default="data/levbench_e1")
    ap.add_argument("--splits", nargs="+", default=["s120"])
    ap.add_argument("--max-k", type=int, default=6)
    args = ap.parse_args(argv)

    dev, dev_stats = load_split(args.levbench, "development", args.max_k)
    test, test_stats = load_split(args.levbench, "test", args.max_k)
    train, train_stats = load_train(args.levtrain, args.splits, args.max_k)

    # Leakage guard: remove train rows whose state also appears in LevBench.
    bench_keys = {r["state_key"] for r in dev + test}
    before = len(train)
    train = [r for r in train if r["state_key"] not in bench_keys]
    removed = before - len(train)
    train_stats["dropped_benchmark_state_overlap"] += removed

    # Defensive de-duplication by (state, option list, instructions).
    def dedupe(rows: list[dict], stats: Counter) -> list[dict]:
        seen, out = set(), []
        for r in rows:
            key = (r["state_key"], tuple(r["options"]), r["instructions"])
            if key in seen:
                stats["dropped_duplicate_item"] += 1
                continue
            seen.add(key)
            out.append(r)
        return out

    train = dedupe(train, train_stats)
    dev = dedupe(dev, dev_stats)
    test = dedupe(test, test_stats)

    parts = {
        "train": [to_e1(r, "levtrain") for r in train],
        "val": [to_e1(r, "levbench-dev") for r in dev],
        "test": [to_e1(r, "levbench-test") for r in test],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def summarize(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "n_groups": len({r["group_id"] for r in rows}),
            "n_options": dict(sorted(Counter(r["n_options"] for r in rows).items())),
            "gold_position": dict(sorted(Counter(r["label"] for r in rows).items())),
            "per_source": dict(Counter(r["source"] for r in rows).most_common()),
            "per_tier": dict(Counter(r["tier"] for r in rows).most_common()),
        }

    stats = {
        "sources": {"levtrain": args.levtrain, "levbench": args.levbench},
        "splits": args.splits,
        "max_k": args.max_k,
        "counts": {name: len(rows) for name, rows in parts.items()},
        "summaries": {name: summarize(rows) for name, rows in parts.items()},
        "filter": {
            "train": dict(train_stats),
            "val": dict(dev_stats),
            "test": dict(test_stats),
        },
        "train_benchmark_overlap_removed": removed,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}")
    for name in ("train", "val", "test"):
        s = stats["summaries"][name]
        print(f"  {name}: K={s['n_options']} sources={len(s['per_source'])}")
    print(f"  removed train/benchmark state overlap: {removed}")
    print(f"wrote {out}/{{train,val,test}}.jsonl + stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
