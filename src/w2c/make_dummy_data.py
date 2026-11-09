"""Generate synthetic When2Call-shaped JSONL for smoke tests only.

    PYTHONPATH=src python -m w2c.make_dummy_data --out data/dummy/raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

CLASSES = ("direct", "tool_call", "request_for_info", "cannot_answer")

QUESTIONS = {
    "direct": ["What is the capital of France?", "Explain what a p-value is.",
               "Summarize the difference between mean and median."],
    "tool_call": ["Book a flight from Dallas to NYC on Friday.",
                  "Email alice@example.com the meeting notes.",
                  "Check the weather in Lisbon tomorrow."],
    "request_for_info": ["Book a flight from Dallas to NYC.",
                         "Email the meeting notes to the team.",
                         "Check the weather in Lisbon."],
    "cannot_answer": ["Book a flight from Dallas to NYC on Friday.",
                      "Email alice@example.com the meeting notes.",
                      "Check the weather in Lisbon tomorrow."],
}

TOOL_SETS = {
    "direct": [],
    "tool_call": [{"name": "book_flight", "description": "Book a flight.",
                   "parameters": {"origin": "string (required)", "destination": "string (required)",
                                  "date": "string (required)"}}],
    "request_for_info": [{"name": "book_flight", "description": "Book a flight.",
                          "parameters": {"origin": "string (required)", "destination": "string (required)",
                                         "date": "string (required)"}}],
    "cannot_answer": [],
}

PARAPHRASES = ("", " Please proceed.", " Thanks.", " If possible.")


def generate(n_per_class: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for label in CLASSES:
        for i in range(n_per_class):
            group = f"{label}-{i // 4}"  # 4 variants share one group
            question = rng.choice(QUESTIONS[label]) + rng.choice(PARAPHRASES)
            records.append({
                "id": f"{label}-{i}",
                "question": question,
                "tools": TOOL_SETS[label],
                "decision": label,
                "group_id": group,
            })
    rng.shuffle(records)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dummy/raw.jsonl")
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    records = generate(args.n_per_class, args.seed)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[dummy] wrote {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
