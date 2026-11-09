"""Phase 0 converter: When2Call SFT/preference data -> audited 4-way E1 records.

Why preference rows: the SFT split is the "decline or clarify" half only
(`tool_call` = 0; cannot 7,561 / request 7,337 / direct 102). The preference
split's *chosen* responses are balanced (tool_call 3,000 / cannot 2,991 /
request 2,963 / direct 46), so it is the only train source with all four
classes. SFT is optionally merged to enlarge the rare `direct` class.

Labels come from explicit, logged surface rules on the assistant response --
not from answer-bearing metadata. Every rule hit is counted and sampled to
`audit_samples.jsonl` for manual review. Rows whose normalized question matches
the official MCQ test set are dropped, and test examples are written separately.

    PYTHONPATH=src python -m w2c.build_w2c_e1 \
        --when2call-dir /mnt/local-disk/bench/when2call \
        --out-dir data/when2call

Outputs: raw.jsonl (train pool), test.jsonl (held-out MCQ), stats.json,
audit_samples.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

from .config import ensure_dir
from .labels import ID_TO_LABEL, label_id

CLASSES = [ID_TO_LABEL[i] for i in range(4)]

_TOOLCALL = re.compile(r"<TOOLCALL>")
_REFUSAL = re.compile(
    r"(\bunable to\b|\bcannot\b|\bcan't\b|\bcan not\b|\bnot able to\b|"
    r"\bno access to\b|\bdon't have access\b|\bdo not have access\b|"
    r"\bdon't have real-time\b|\bdo not have real-time\b|"
    r"\bbeyond my (?:capabilities|ability)\b|"
    r"\bnot within my (?:capabilities|ability)\b|"
    r"\b(?:don't|do not) have the (?:ability|capability|capabilities)\b|"
    r"\bdesigned to (?:provide|assist)[^.]{0,80}\bnot\b|"
    r"\bai language model\b[^.]{0,60}\b(?:don't|do not|cannot|can't)\b)",
    re.IGNORECASE,
)
_REQUEST = re.compile(
    r"(could you|can you (tell|provide|clarify|specify|share|confirm)|"
    r"please (provide|specify|clarify|share|confirm)|"
    r"i(?:'ll| will| do)? need (?:to know|the|a|an|you|more|specific)|"
    r"which .* (?:do you|would you)|what .* (?:are you|do you))",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def norm_question(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", str(text).lower())).strip()


def qhash(text: str) -> str:
    return hashlib.sha1(norm_question(text).encode("utf-8")).hexdigest()[:16]


def text_of(value) -> str:
    if isinstance(value, dict):
        return str(value.get("content") or "")
    return str(value or "")


def user_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    return " ".join(str(m.get("content") or "") for m in messages
                    if m.get("role") == "user").strip()


def assistant_text(messages) -> str:
    if isinstance(messages, str):
        return ""
    replies = [str(m.get("content") or "") for m in messages
               if m.get("role") == "assistant"]
    return replies[-1].strip() if replies else ""


def parse_tools(tools) -> list[dict]:
    """Source tools are JSON strings; normalize to dicts for rendering."""
    parsed = []
    for tool in tools or []:
        if isinstance(tool, str):
            try:
                tool = json.loads(tool)
            except json.JSONDecodeError:
                parsed.append({"name": "unparsed_tool", "description": tool})
                continue
        if isinstance(tool, dict):
            parsed.append(tool)
        else:
            parsed.append({"name": "unparsed_tool", "description": str(tool)})
    return parsed


def classify(response: str) -> tuple[str | None, str]:
    """Return (label, rule). Rules are ordered; multi-rule hits are reported."""
    text = response.strip()
    if not text:
        return None, "empty"
    if _TOOLCALL.search(text) or (text.startswith("{") and '"name"' in text[:300]):
        return "tool_call", "toolcall_structural"
    asks = text.endswith("?") or bool(_REQUEST.search(text))
    refuses = bool(_REFUSAL.search(text))
    if refuses and asks:
        return "cannot_answer", "ambiguous_refusal_request"
    if refuses:
        return "cannot_answer", "refusal_phrase"
    if asks:
        return "request_for_info", "clarification_question"
    return "direct", "fallback_direct"


def _maybe_drop(record: dict, test_hashes: set[str]) -> str | None:
    if qhash(record["question"]) in test_hashes:
        return "test_overlap"
    if not record["question"].strip():
        return "empty_question"
    if record["label_name"] == "tool_call" and not record["tools"]:
        return "tool_call_without_tools"
    return None


def _make(record_id: str, question: str, tools, label: str, source: str) -> dict:
    return {
        "id": record_id,
        "question": question.strip(),
        "tools": parse_tools(tools),
        "decision": label,
        "label_name": label,
        "group_id": qhash(question),
        "source": source,
    }


def sample_for_audit(labelled: list[tuple[dict, str, str]], per_class: int,
                     seed: int) -> list[dict]:
    """Deterministic random audit sample, stratified by class.

    Sampling the first N rows of each class (as the scaffold originally did)
    over-represents whichever APIs happen to appear first in the preference
    file. Draw evenly from each class instead, then sort by id so the file is
    stable and diffable.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[tuple[dict, str, str]]] = defaultdict(list)
    for item in labelled:
        by_class[item[0]["label_name"]].append(item)
    chosen = []
    for label in CLASSES:
        pool = sorted(by_class.get(label, []), key=lambda x: x[0]["id"])
        take = min(per_class, len(pool))
        chosen.extend(rng.sample(pool, take))
    chosen.sort(key=lambda x: x[0]["id"])
    return [{
        "id": rec["id"], "source": rec["source"], "rule": rule,
        "decision": rec["label_name"], "question": rec["question"],
        "response": response[:400], "n_tools": len(rec["tools"]),
    } for rec, response, rule in chosen]


def build(when2call_dir: str, out_dir: str, include_sft: bool = True,
          sample_per_class: int = 25, audit_seed: int = 0,
          drop_classes: set[str] | None = None) -> dict:
    drop_classes = set(drop_classes or ())
    test_path = os.path.join(when2call_dir, "test", "when2call_test_mcq.jsonl")
    pref_path = os.path.join(when2call_dir, "train", "when2call_train_pref.jsonl")
    sft_path = os.path.join(when2call_dir, "train", "when2call_train_sft.jsonl")

    test_rows = [json.loads(line) for line in open(test_path, encoding="utf-8") if line.strip()]
    test_hashes = {qhash(r["question"]) for r in test_rows}

    stats: Counter = Counter()
    rules: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    labelled: list[tuple[dict, str, str]] = []
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(record_id, question, tools, response, source):
        label, rule = classify(response)
        rules[rule] += 1
        if label is None:
            stats["dropped_empty_response"] += 1
            return
        if label in drop_classes:
            stats[f"dropped_excluded_class:{label}"] += 1
            return
        rec = _make(record_id, question, tools, label, source)
        reason = _maybe_drop(rec, test_hashes)
        if reason:
            stats[f"dropped_{reason}"] += 1
            return
        key = (rec["group_id"], rec["label_name"])
        if key in seen:
            stats["dropped_duplicate_question_label"] += 1
            return
        seen.add(key)
        records.append(rec)
        by_source[source][rec["label_name"]] += 1
        labelled.append((rec, response, rule))

    for line in open(pref_path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        question = user_text(row.get("messages"))
        response = text_of(row.get("chosen_response")) or assistant_text(row.get("messages"))
        add(f"pref-{len(records):06d}", question, row.get("tools"), response, "pref_chosen")

    if include_sft:
        for line in open(sft_path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            question = user_text(row.get("messages"))
            response = assistant_text(row.get("messages"))
            add(f"sft-{len(records):06d}", question, row.get("tools"), response, "sft_assistant")

    test_records = []
    for row in test_rows:
        label = row.get("correct_answer")
        lid = label_id(label)
        if lid is None:
            stats["test_dropped_ambiguous"] += 1
            continue
        if ID_TO_LABEL[lid] in drop_classes:
            stats[f"test_dropped_excluded_class:{ID_TO_LABEL[lid]}"] += 1
            continue
        rid = row.get("uuid") or f"{len(test_records):06d}"
        rec = _make(f"test-{rid}", row.get("question", ""), row.get("tools"),
                    ID_TO_LABEL[lid], "mcq_test")
        test_records.append(rec)

    ensure_dir(out_dir)
    for name, rows in (("raw.jsonl", records), ("test.jsonl", test_records)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "audit_samples.jsonl"), "w", encoding="utf-8") as fh:
        for sample in sample_for_audit(labelled, sample_per_class, audit_seed):
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    label_counts = Counter(r["label_name"] for r in records)
    classes = [c for c in CLASSES if c not in drop_classes and label_counts.get(c, 0)]
    summary = {
        "when2call_dir": when2call_dir,
        "include_sft": include_sft,
        "audit_seed": audit_seed,
        "train_pool": len(records),
        "test": len(test_records),
        "classes": classes,
        "label_counts": {c: int(label_counts.get(c, 0)) for c in CLASSES},
        "test_label_counts": {c: int(sum(1 for r in test_records if r["label_name"] == c))
                              for c in CLASSES},
        "by_source": {src: dict(counts) for src, counts in by_source.items()},
        "rule_hits": dict(rules),
        "drops": {k: v for k, v in stats.items() if k.startswith(("dropped", "test_dropped"))},
        "n_groups": len({r["group_id"] for r in records}),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--when2call-dir", default="/mnt/local-disk/bench/when2call")
    parser.add_argument("--out-dir", default="data/when2call")
    parser.add_argument("--no-sft", action="store_true", help="use preference rows only")
    parser.add_argument("--samples-per-class", type=int, default=25)
    parser.add_argument("--audit-seed", type=int, default=0)
    parser.add_argument("--drop-class", action="append", default=[],
                        help="canonical class to exclude, repeatable (e.g. direct)")
    args = parser.parse_args(argv)

    summary = build(args.when2call_dir, args.out_dir,
                    include_sft=not args.no_sft,
                    sample_per_class=args.samples_per_class,
                    audit_seed=args.audit_seed,
                    drop_classes=set(args.drop_class or ()))
    print(json.dumps(summary, indent=2))
    print(f"\nReview {args.out_dir}/audit_samples.jsonl before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
