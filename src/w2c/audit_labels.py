"""Reproducible label audit for the When2Call E1 dataset.

The Phase-0 converter labels assistant responses with ordered surface rules.
This module audits those labels without a human in the loop where the check is
mechanical, and prepares/marks up a deterministic manual sample where it is not.

Mechanical checks
  * every tool-call row parses as JSON inside ``<TOOLCALL>...</TOOLCALL>``,
    names only tools offered in that example, and carries no trailing prose;
  * the training pool is rebuilt from the source file and compared by id,
    question, label, and group with ``raw.jsonl`` (drops included);
  * questions that appear with more than one label are split into pairs whose
    available tool sets differ (expected: the right decision depends on tools)
    and pairs whose tool sets are identical (irreducible prompt ambiguity);
  * the official MCQ test labels are checked against ``target_tool`` and
    against duplicate questions.

Manual audit
  ``audit_samples.jsonl`` is drawn deterministically by ``build_w2c_e1``
  (``--audit-seed``). Reviewers record verdicts in ``audit_review.jsonl``::

      {"id": "pref-000171", "decision": "request_for_info",
       "verdict": "correct", "adjudicated": null, "note": ""}

  ``verdict`` is one of ``correct``, ``ambiguous``, or ``incorrect``.
  ``adjudicated`` is the label a reviewer would give instead, if any.

Usage
    PYTHONPATH=src python -m w2c.audit_labels \
        --when2call-dir /mnt/local-bench/when2call \
        --out-dir data/when2call_pref

Outputs: audit_report.json, AUDIT.md under ``out_dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

from .build_w2c_e1 import (
    _maybe_drop,
    _make,
    assistant_text,
    classify,
    parse_tools,
    qhash,
    text_of,
    user_text,
)
from .config import ensure_dir

CLASSES = ["tool_call", "request_for_info", "cannot_answer"]
_TAG = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)
_TRAILING = re.compile(r"</TOOLCALL>\s*$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def _tool_names(tools) -> tuple[str, ...]:
    return tuple(sorted(str(t.get("name")) for t in parse_tools(tools)))


def replay_train_pool(when2call_dir: str, drop_classes: set[str]) -> list[tuple[dict, str, str]]:
    """Rebuild the exact kept training records and retain their responses."""
    pref_path = os.path.join(when2call_dir, "train", "when2call_train_pref.jsonl")
    test_path = os.path.join(when2call_dir, "test", "when2call_test_mcq.jsonl")
    test_hashes = {
        qhash(json.loads(line)["question"])
        for line in open(test_path, encoding="utf-8") if line.strip()
    }
    kept: list[tuple[dict, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in open(pref_path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        question = user_text(row.get("messages"))
        response = text_of(row.get("chosen_response")) or assistant_text(row.get("messages"))
        label, rule = classify(response)
        if label is None or label in drop_classes:
            continue
        rec = _make(f"pref-{len(kept):06d}", question, row.get("tools"), label, "pref_chosen")
        if _maybe_drop(rec, test_hashes):
            continue
        key = (rec["group_id"], rec["label_name"])
        if key in seen:
            continue
        seen.add(key)
        kept.append((rec, response, rule))
    return kept


def audit_tool_calls(kept: list[tuple[dict, str, str]]) -> dict:
    total = 0
    errors = Counter()
    examples = []
    for rec, response, _ in kept:
        if rec["label_name"] != "tool_call":
            continue
        total += 1
        match = _TAG.search(response)
        if not match:
            errors["missing_or_unclosed_tag"] += 1
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            errors["invalid_json"] += 1
            continue
        entries = payload if isinstance(payload, list) else [payload]
        names = {str(e.get("name")) for e in entries if isinstance(e, dict)}
        available = {str(t.get("name")) for t in parse_tools(rec["tools"])}
        if not names:
            errors["no_tool_name"] += 1
        if not names <= available:
            errors["name_not_in_available_tools"] += 1
        if response.count("<TOOLCALL>") != 1:
            errors["multiple_blocks"] += 1
        if not _TRAILING.search(response):
            errors["trailing_prose"] += 1
        if errors and len(examples) < 5:
            examples.append({"id": rec["id"], "response": response[:300]})
    return {"rows": total, "errors": dict(errors), "examples": examples}


def audit_label_conflicts(kept: list[tuple[dict, str, str]]) -> dict:
    by_q: dict[str, list[tuple[dict, str, str]]] = defaultdict(list)
    for item in kept:
        by_q[item[0]["group_id"]].append(item)
    same_tools, diff_tools = [], []
    for group in by_q.values():
        if len({item[0]["label_name"] for item in group}) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a[0]["label_name"] == b[0]["label_name"]:
                    continue
                pair = {
                    "question": a[0]["question"],
                    "a": {"id": a[0]["id"], "label": a[0]["label_name"], "response": a[1][:300]},
                    "b": {"id": b[0]["id"], "label": b[0]["label_name"], "response": b[1][:300]},
                    "tools": list(_tool_names(a[0]["tools"])),
                }
                if _tool_names(a[0]["tools"]) == _tool_names(b[0]["tools"]):
                    same_tools.append(pair)
                else:
                    diff_tools.append(pair)
    return {
        "groups_with_conflicting_labels": sum(
            1 for g in by_q.values() if len({i[0]["label_name"] for i in g}) > 1),
        "pairs_different_tools": len(diff_tools),
        "pairs_same_tools": len(same_tools),
        "same_tool_pairs": same_tools,
        "same_tool_examples": same_tools[:15],
        "different_tool_example": diff_tools[:3],
    }


def audit_edge_cases(kept_ids: set[str], when2call_dir: str) -> dict:
    """Every fallback-direct and refusal+clarification row in the source."""
    pref_path = os.path.join(when2call_dir, "train", "when2call_train_pref.jsonl")
    edges = []
    counts = Counter()
    for i, line in enumerate(open(pref_path, encoding="utf-8")):
        if not line.strip():
            continue
        row = json.loads(line)
        response = text_of(row.get("chosen_response")) or assistant_text(row.get("messages"))
        label, rule = classify(response)
        counts[rule] += 1
        if rule in ("fallback_direct", "ambiguous_refusal_request"):
            edges.append({
                "source_row": i,
                "question": user_text(row.get("messages"))[:200],
                "response": response[:300],
                "rule": rule,
                "label": label,
            })
    return {"rule_counts": dict(counts), "edges": edges,
            "kept_ids_available": len(kept_ids)}


def audit_test(when2call_dir: str) -> dict:
    test_path = os.path.join(when2call_dir, "test", "when2call_test_mcq.jsonl")
    rows = [json.loads(line) for line in open(test_path, encoding="utf-8") if line.strip()]
    labels = Counter(r.get("correct_answer") for r in rows)
    target = Counter()
    for r in rows:
        has_target = str(r.get("target_tool")) not in ("None", "", "null", "nan")
        target[(r.get("correct_answer"), has_target)] += 1
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[_norm(r.get("question", ""))].append(r)
    same_tools = 0
    diff_tools = 0
    for group in by_q.values():
        if len({r.get("correct_answer") for r in group}) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.get("correct_answer") == b.get("correct_answer"):
                    continue
                if _tool_names(a.get("tools")) == _tool_names(b.get("tools")):
                    same_tools += 1
                else:
                    diff_tools += 1
    return {
        "rows": len(rows),
        "label_counts": dict(labels),
        "labels_with_target_tool": {
            "with_target": {k[0]: v for k, v in target.items() if k[1]},
            "without_target": {k[0]: v for k, v in target.items() if not k[1]},
        },
        "empty_tools_by_label": dict(Counter(
            r.get("correct_answer") for r in rows if not r.get("tools"))),
        "conflicting_label_pairs_same_tools": same_tools,
        "conflicting_label_pairs_different_tools": diff_tools,
        "duplicate_uuid": len({r.get("uuid") for r in rows}) != len(rows),
    }


def load_reviews(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, "audit_review.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def summarize_reviews(reviews: list[dict], sample_ids: set[str]) -> dict:
    verdicts = Counter(r.get("verdict") for r in reviews)
    reviewed = {r.get("id") for r in reviews}
    by_class = Counter((r.get("decision"), r.get("verdict")) for r in reviews
                       if r.get("stratum") == "random_sample")
    strata = Counter(r.get("stratum", "unspecified") for r in reviews)
    return {
        "reviews": len(reviews),
        "verdicts": dict(verdicts),
        "strata": dict(strata),
        "by_class_and_verdict": {f"{c}/{v}": n for (c, v), n in sorted(by_class.items())},
        "sample_rows": len(sample_ids),
        "unreviewed_sample_rows": len(sample_ids - reviewed),
        "adjudications": [
            {"id": r["id"], "from": r.get("decision"), "to": r.get("adjudicated"),
             "verdict": r.get("verdict"), "note": r.get("note", "")}
            for r in reviews if r.get("adjudicated")
        ],
        "incorrect": [r for r in reviews if r.get("verdict") == "incorrect"],
        "ambiguous": [r for r in reviews if r.get("verdict") == "ambiguous"],
    }


def render_markdown(report: dict) -> str:
    t = report["tool_calls"]
    c = report["label_conflicts"]
    te = report["test"]
    rev = report["manual_review"]
    e = report["edge_cases"]
    lines = [
        "# Label audit — When2Call E1 (preference chosen)",
        "",
        f"Training pool: **{report['train_pool']['rows']}** rows "
        f"(groups: {report['train_pool']['groups']}), "
        f"test: **{te['rows']}** rows. Scope: preference-only E1 train pool; "
        "the merged-SFT sensitivity pool uses the same rules but is not re-audited.",
        "",
        "## Mechanical checks",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Tool-call rows | {t['rows']} |",
        f"| Tool-call structural errors | {sum(t['errors'].values())} {t['errors'] or ''} |",
        f"| Rebuild matches `raw.jsonl` | {report['train_pool']['rebuild_matches_raw']} |",
        f"| Conflicting questions (all tool sets) | {c['groups_with_conflicting_labels']} groups |",
        f"| … pairs with different tools (expected) | {c['pairs_different_tools']} |",
        f"| … pairs with identical tools (ambiguity) | {c['pairs_same_tools']} |",
        f"| Test labels with `target_tool` when `tool_call` | {te['labels_with_target_tool']['with_target'].get('tool_call', 0)}/{te['label_counts'].get('tool_call', 0)} |",
        f"| Test rows with empty tools (all `cannot_answer`) | {te['empty_tools_by_label']} |",
        f"| Test same-tool conflicting gold pairs | {te['conflicting_label_pairs_same_tools']} |",
        "",
        "## Rule edge cases",
        "",
        f"Rule hits over the source preference file: `{e['rule_counts']}`.",
        "",
        "All `fallback_direct` and `ambiguous_refusal_request` rows are listed in "
        "`audit_report.json` (`edge_cases.edges`). With `direct` excluded from E1, "
        "the only ambiguous row retained in training is a refusal that also asks "
        "for a missing parameter.",
        "",
        "## Manual review",
        "",
        f"Sample: {rev['sample_rows']} rows; reviewed rows (sample + targeted "
        f"strata): {rev['reviews']}; verdicts: {rev['verdicts'] or 'pending'}; "
        f"sample rows not yet reviewed: {rev['unreviewed_sample_rows']}.",
        "",
        f"Review strata: `{rev['strata']}`. Random sample verdicts: "
        f"`{rev['by_class_and_verdict']}`.",
        "",
    ]
    if rev["adjudications"]:
        lines += ["| id | original | adjudicated | verdict | note |", "|---|---|---|---|---|"]
        for a in rev["adjudications"]:
            lines.append(f"| {a['id']} | {a['from']} | {a['to']} | {a['verdict']} | {a['note']} |")
        lines.append("")
    if c["same_tool_examples"]:
        lines += [
            "## Irreducible same-tool ambiguity",
            "",
            "These prompts are paired with the same tool set but two different "
            "chosen decisions. Both responses are defensible; the corpus itself "
            "does not define a unique gold action. Kept in the primary run and "
            "reported as a noise floor rather than silently relabelled.",
            "",
            "| Question | Decision A | Decision B |",
            "|---|---|---|",
        ]
        for x in c["same_tool_examples"]:
            q = x["question"].replace("|", "\\|")[:110]
            lines.append(f"| {q} | {x['a']['label']} | {x['b']['label']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def audit(when2call_dir: str, out_dir: str, drop_classes: set[str] | None = None) -> dict:
    drop_classes = set(drop_classes or {"direct"})
    kept = replay_train_pool(when2call_dir, drop_classes)

    raw_path = os.path.join(out_dir, "raw.jsonl")
    raw = [json.loads(line) for line in open(raw_path, encoding="utf-8") if line.strip()]
    replay = [rec for rec, _, _ in kept]
    matches = [a["id"] == b["id"] and a["question"] == b["question"]
               and a["label_name"] == b["label_name"] and a["group_id"] == b["group_id"]
               for a, b in zip(replay, raw)]
    rebuild_matches = len(replay) == len(raw) and all(matches)

    sample_path = os.path.join(out_dir, "audit_samples.jsonl")
    sample_ids = {
        json.loads(line)["id"]
        for line in open(sample_path, encoding="utf-8") if line.strip()
    } if os.path.exists(sample_path) else set()

    report = {
        "when2call_dir": when2call_dir,
        "classes": [c for c in CLASSES if c not in drop_classes],
        "train_pool": {
            "rows": len(kept),
            "groups": len({rec["group_id"] for rec, _, _ in kept}),
            "rebuild_matches_raw": rebuild_matches,
            "raw_rows": len(raw),
        },
        "tool_calls": audit_tool_calls(kept),
        "label_conflicts": audit_label_conflicts(kept),
        "edge_cases": audit_edge_cases({rec["id"] for rec, _, _ in kept}, when2call_dir),
        "test": audit_test(when2call_dir),
        "manual_review": summarize_reviews(load_reviews(out_dir), sample_ids),
    }
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "audit_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "AUDIT.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--when2call-dir", default="/mnt/local-disk/bench/when2call")
    parser.add_argument("--out-dir", default="data/when2call_pref")
    parser.add_argument("--drop-class", action="append", default=["direct"],
                        help="class excluded from the training pool (default: direct)")
    args = parser.parse_args(argv)
    report = audit(args.when2call_dir, args.out_dir, set(args.drop_class or ()))
    print(json.dumps({
        "train_pool": report["train_pool"],
        "tool_calls": report["tool_calls"],
        "same_tool_conflicts": report["label_conflicts"]["pairs_same_tools"],
        "test_same_tool_conflicts":
            report["test"]["conflicting_label_pairs_same_tools"],
        "manual_review": report["manual_review"],
    }, indent=2))
    print(f"\nWrote {args.out_dir}/audit_report.json and AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
