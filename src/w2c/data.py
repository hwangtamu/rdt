"""Dataset loading, canonical-label validation, and leakage-safe splitting.

Expected JSONL record (field names configurable):

    {"id": "...", "question": "...", "tools": [...], "decision": "tool_call",
     "group_id": "optional-family-id"}

Rules enforced here (plan sections 5.2):
  * labels are mapped to the four canonical classes; ambiguous -> dropped;
  * examples that share ``group_id`` (or, if absent, a normalized-question hash)
    always land in the same split;
  * splits are stratified by class at the group level.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from .labels import CANONICAL, normalize_label
from .template import build_prompt

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _norm_text(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", text.lower())).strip()


def _stable_hash(*parts: str) -> str:
    h = hashlib.sha1("\x1f".join(parts).encode("utf-8"))
    return h.hexdigest()[:16]


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc


def normalize_record(raw: dict, cfg: dict, stats: Counter) -> dict | None:
    dcfg, tcfg = cfg["data"], cfg["template"]

    classes = list(dcfg.get("classes") or CANONICAL)
    label_mode = dcfg.get("label_mode", "name")
    question = raw.get(dcfg["question_field"])
    if not isinstance(question, str) or not question.strip():
        stats["dropped_missing_question"] += 1
        return None

    if label_mode == "index":
        # Positional multiple-choice labels: 0..K-1 into ``classes``.  This is
        # used by Lev/LevBench, where the label is an option index and K varies
        # by example (the readout is masked to K logits at loss/metric time).
        label_value = raw.get(dcfg["label_field"])
        try:
            lid = int(label_value)
        except (TypeError, ValueError):
            stats["dropped_ambiguous"] += 1
            stats[f"dropped_value:{str(label_value)[:40]}"] += 1
            return None
        if lid < 0 or lid >= len(classes):
            stats["dropped_excluded_class"] += 1
            return None
        label_name = classes[lid]
        n_options = raw.get(dcfg.get("n_options_field", "n_options"))
        if n_options is None:
            n_options = len(classes)
        try:
            n_options = int(n_options)
        except (TypeError, ValueError):
            stats["dropped_bad_n_options"] += 1
            return None
        if not 2 <= n_options <= len(classes) or lid >= n_options:
            stats["dropped_bad_n_options"] += 1
            return None
    else:
        label_value = raw.get(dcfg["label_field"])
        label_name = normalize_label(label_value)
        if label_name is None:
            stats["dropped_ambiguous"] += 1
            stats[f"dropped_value:{str(label_value)[:40]}"] += 1
            return None
        if label_name not in classes:
            stats[f"dropped_excluded_class:{label_name}"] += 1
            return None
        lid = classes.index(label_name)
        n_options = len(classes)

    tools = raw.get(dcfg["tools_field"])
    record_id = raw.get(dcfg["id_field"])
    record_id = str(record_id) if record_id not in (None, "") else _stable_hash(
        label_name, question, json.dumps(tools, sort_keys=True, default=str)
    )

    group = raw.get(dcfg["group_field"])
    group = str(group) if group not in (None, "") else "q:" + _stable_hash(_norm_text(question))

    prompt_field = dcfg.get("prompt_field")
    rendered = raw.get(prompt_field) if prompt_field else None
    if isinstance(rendered, str) and rendered.strip():
        prompt = rendered
    else:
        prompt = build_prompt(
            question=question,
            tools=tools,
            template=tcfg["text"],
            tool_format=tcfg["tool_format"],
            empty_tools=tcfg["empty_tools"],
        )

    stats["kept"] += 1
    rec = {
        "id": record_id,
        "question": question,
        "tools": tools,
        "label": lid,
        "label_name": label_name,
        "n_options": n_options,
        "group": group,
        "prompt": prompt,
    }
    for meta_key in ("source", "tier", "family"):
        if raw.get(meta_key) not in (None, ""):
            rec[meta_key] = raw[meta_key]
    return rec


def load_records(path: str, cfg: dict) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    records = []
    seen_ids: set[str] = set()
    for raw in iter_jsonl(path):
        rec = normalize_record(raw, cfg, stats)
        if rec is None:
            continue
        if rec["id"] in seen_ids:
            stats["dropped_duplicate_id"] += 1
            continue
        seen_ids.add(rec["id"])
        records.append(rec)

    limit = cfg["data"]["max_examples"]
    if limit and len(records) > limit:
        rng = random.Random(cfg["seed"])
        by_class: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            by_class[rec["label"]].append(rec)
        keep: list[dict] = []
        per_class = max(1, int(limit) // max(1, len(by_class)))
        for lid in sorted(by_class):
            bucket = by_class[lid]
            rng.shuffle(bucket)
            keep.extend(bucket[:per_class])
        records = sorted(keep, key=lambda r: r["id"])
        stats["subsampled_to"] = len(records)
    return records, stats


def stratified_group_split(records: list[dict], val_fraction: float,
                           test_fraction: float, seed: int) -> dict[str, list[dict]]:
    """Group-level, class-stratified split. Groups never cross splits."""
    if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1):
        raise ValueError("val_fraction/test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["group"]].append(rec)

    by_class: dict[int, list[str]] = defaultdict(list)
    for group, recs in groups.items():
        majority = Counter(r["label"] for r in recs).most_common(1)[0][0]
        by_class[majority].append(group)

    parts: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for _lid, group_ids in sorted(by_class.items()):
        rng.shuffle(group_ids)
        total = sum(len(groups[g]) for g in group_ids)
        targets = {
            "test": round(total * test_fraction),
            "val": round(total * val_fraction),
        }
        for part in ("test", "val"):
            taken = 0
            while group_ids and taken < targets[part]:
                group = group_ids.pop()
                parts[part].extend(groups[group])
                taken += len(groups[group])
        for group in group_ids:
            parts["train"].extend(groups[group])

    # Guarantee non-empty splits when enough data exists, without splitting groups.
    if not parts["train"] and (parts["val"] or parts["test"]):
        donor = "val" if parts["val"] else "test"
        group = parts[donor][0]["group"]
        moved = [r for r in parts[donor] if r["group"] == group]
        parts[donor] = [r for r in parts[donor] if r["group"] != group]
        parts["train"].extend(moved)
    if not parts["val"] and len({r["group"] for r in parts["train"]}) > 1:
        donor = "train"
        group = parts[donor][0]["group"]
        moved = [r for r in parts[donor] if r["group"] == group]
        parts[donor] = [r for r in parts[donor] if r["group"] != group]
        parts["val"].extend(moved)

    for name, recs in parts.items():
        recs.sort(key=lambda r: r["id"])
    _assert_disjoint(parts)
    return parts


def _assert_disjoint(parts: dict[str, list[dict]]) -> None:
    seen_groups: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    for name, recs in parts.items():
        for rec in recs:
            if rec["group"] in seen_groups and seen_groups[rec["group"]] != name:
                raise AssertionError(
                    f"group {rec['group']} leaks across {seen_groups[rec['group']]} and {name}"
                )
            if rec["id"] in seen_ids and seen_ids[rec["id"]] != name:
                raise AssertionError(f"example {rec['id']} leaks across splits")
            seen_groups[rec["group"]] = name
            seen_ids[rec["id"]] = name


def load_data(cfg: dict) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """Load explicit split files, or split ``data.raw`` with grouped stratification."""
    dcfg = cfg["data"]
    stats: Counter = Counter()

    if dcfg.get("train") and dcfg.get("val") and dcfg.get("test"):
        parts = {name: load_records(dcfg[name], cfg)[0] for name in ("train", "val", "test")}
        sources = {name: dcfg[name] for name in ("train", "val", "test")}
        for name in parts:
            if not parts[name]:
                raise ValueError(f"split '{name}' is empty: {dcfg[name]}")
        _assert_disjoint(parts)
    elif dcfg.get("raw"):
        records, stats = load_records(dcfg["raw"], cfg)
        if not records:
            raise ValueError(f"no usable records in raw file {dcfg['raw']}")
        if dcfg.get("test"):
            # Official held-out test set: split the raw pool into train/val only.
            parts = stratified_group_split(records, float(dcfg["val_fraction"]), 0.0,
                                           int(cfg["seed"]))
            test_cfg = copy.deepcopy(cfg)
            test_cfg["data"]["max_examples"] = None
            parts["test"] = load_records(dcfg["test"], test_cfg)[0]
            if not parts["test"]:
                raise ValueError(f"test split is empty: {dcfg['test']}")
            _assert_disjoint(parts)
            sources = {"raw": dcfg["raw"], "test": dcfg["test"]}
        else:
            parts = stratified_group_split(
                records, float(dcfg["val_fraction"]), float(dcfg["test_fraction"]),
                int(cfg["seed"]),
            )
            sources = {"raw": dcfg["raw"]}
    else:
        raise ValueError("config must set data.train/val/test or data.raw")

    classes = list(cfg["data"].get("classes") or CANONICAL)
    manifest = {
        "sources": sources,
        "classes": classes,
        "counts": {name: len(recs) for name, recs in parts.items()},
        "label_counts": {
            name: {cls: int(sum(1 for r in recs if r["label_name"] == cls))
                   for cls in classes}
            for name, recs in parts.items()
        },
        "n_groups": {name: len({r["group"] for r in recs}) for name, recs in parts.items()},
        "n_options_counts": {
            name: {str(k): int(v) for k, v in
                   sorted(Counter(r.get("n_options", len(classes)) for r in recs).items())}
            for name, recs in parts.items()
        },
        "source_counts": {
            name: dict(Counter(r.get("source", "unknown") for r in recs).most_common())
            for name, recs in parts.items()
        },
        "dropped": dict(stats),
        "grouped_split": {"val_fraction": dcfg["val_fraction"],
                          "test_fraction": dcfg["test_fraction"], "seed": cfg["seed"]},
    }
    return parts, manifest


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({
                "id": rec["id"],
                "group_id": rec["group"],
                "prompt": rec["prompt"],
                "decision": rec["label_name"],
                "label": rec["label"],
                "label_name": rec["label_name"],
            }, ensure_ascii=False) + "\n")
