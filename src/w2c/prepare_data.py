"""Materialize grouped, stratified splits from a raw JSONL file.

    PYTHONPATH=src python -m w2c.prepare_data --config configs/e1.json \
        --raw data/raw/when2call_sft.jsonl --out-dir data/splits
"""

from __future__ import annotations

import argparse
import json
import os

from .config import ensure_dir, load_config
from .data import load_data, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build E1 train/val/test splits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw", required=True, help="raw JSONL with labels")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg["data"]["train"] = cfg["data"]["val"] = cfg["data"]["test"] = None
    cfg["data"]["raw"] = args.raw
    if args.seed is not None:
        cfg["seed"] = args.seed

    parts, manifest = load_data(cfg)
    ensure_dir(args.out_dir)
    for name, records in parts.items():
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        write_jsonl(records, path)
        print(f"[prepare] {name}: {len(records)} examples -> {path}")
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[prepare] groups: {manifest['n_groups']}")
    if manifest["dropped"]:
        print(f"[prepare] dropped: {manifest['dropped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
