"""Build the E1-style BoolQ probing configs for the full Ettin family (24 models).

Emits:
  configs/e1_boolq_family.json      canonical config, all 24 backbones, all layers
  configs/e1_boolq_family_g1.json   shard: ~half the parameter mass, device cuda:1
  configs/e1_boolq_family_g2.json   shard: the rest,             device cuda:2

Shards are balanced on a cost proxy (parameters x layer count) and are written
independently so two GPUs can run at once. Feature caches are keyed by backbone
name + split ids, so shards can share one cache directory.

    python3 scripts/make_e1_family_configs.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = "/mnt/local-disk/decision-models/models"

SIZES = ["17m", "32m", "68m", "150m", "400m", "1b"]
PARAMS_M = {"17m": 17, "32m": 32, "68m": 67, "150m": 150, "400m": 395, "1b": 1000}
LAYERS = {"17m": 7, "32m": 10, "68m": 19, "150m": 22, "400m": 28, "1b": 28}

# key prefix -> (repo prefix, human label)
FAMILIES = {
    "enc": ("ettin-encoder-", "encoder (from scratch)"),
    "dec": ("ettin-decoder-", "decoder (from scratch)"),
    "encfd": ("ettin-enc-from-dec-", "encoder (init from decoder)"),
    "decfe": ("ettin-dec-from-enc-", "decoder (init from encoder)"),
}

DATA = {
    "train": "data/boolq_e0/train.jsonl",
    "val": "data/boolq_e0/val.jsonl",
    "test": "data/boolq_e0/test.jsonl",
    "classes": ["a", "b"],
    "label_mode": "index",
    "label_field": "label",
    "n_options_field": "n_options",
    "prompt_field": "prompt",
    "id_field": "id",
    "question_field": "question",
    "tools_field": "tools",
    "group_field": "group_id",
    "val_fraction": 0.0,
    "test_fraction": 0.0,
    "max_examples": None,
}

TEMPLATE = {
    "text": "{question}",
    "tool_format": "- {name}\n  description: {description}\n  parameters: {parameters}",
    "empty_tools": "(none)",
}

PROBES = {
    "kinds": ["linear", "mlp"],
    "mlp_hidden": 256,
    "dropout": 0.1,
    "lr": 0.001,
    "weight_decay": 0.01,
    "batch_size": 4096,
    "max_epochs": 200,
    "patience": 20,
    "standardize": True,
}

EVAL = {"bootstrap": 2000, "n_bins": 10, "alpha": 0.05}


def backbones() -> list[dict]:
    out = []
    for size in SIZES:
        for key_prefix, (repo_prefix, _label) in FAMILIES.items():
            repo = repo_prefix + size
            out.append({
                "key": f"{key_prefix}-{size}",
                "name": f"jhu-clsp/{repo}",
                "path": os.path.join(MODELS, repo),
                "dtype": "float16",
                "family": key_prefix,
                "size": size,
            })
    return out


def base_config(device: str, out_dir: str, keys: list[str]) -> dict:
    bbs = [b for b in backbones() if b["key"] in keys]
    return {
        "seed": 0,
        "seeds": [0, 1, 2],
        "data": DATA,
        "template": TEMPLATE,
        "backbones": bbs,
        "features": {
            "batch_size": 16,
            "max_length": 512,
            "pooling": "last",
            "layers": None,
            "device": device,
            "cache_dir": "runs/features_boolq_family",
            "token_cache_dir": None,
            "trust_remote_code": False,
        },
        "probes": PROBES,
        "eval": EVAL,
        "out": out_dir,
    }


def main() -> None:
    all_bbs = backbones()
    assert len(all_bbs) == 24, len(all_bbs)
    weight = {b["key"]: PARAMS_M[b["size"]] * LAYERS[b["size"]] for b in all_bbs}
    ordered = sorted(all_bbs, key=lambda b: -weight[b["key"]])

    # Longest-processing-time greedy split into two balanced shards.
    shards: list[list[str]] = [[], []]
    loads = [0, 0]
    for b in ordered:
        i = 0 if loads[0] <= loads[1] else 1
        shards[i].append(b["key"])
        loads[i] += weight[b["key"]]
    print(f"shard loads (params*layers): g1={loads[0]:,} g2={loads[1]:,}")

    canonical = base_config("cuda:1", "runs/e1_boolq_family",
                            [b["key"] for b in all_bbs])
    canonical["backbones"] = all_bbs  # keep original ordering
    writes = {
        "e1_boolq_family.json": canonical,
        "e1_boolq_family_g1.json": base_config("cuda:1", "runs/e1_boolq_family_g1",
                                               shards[0]),
        "e1_boolq_family_g2.json": base_config("cuda:2", "runs/e1_boolq_family_g2",
                                               shards[1]),
    }
    for name, cfg in writes.items():
        path = os.path.join(ROOT, "configs", name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"wrote {path}  ({len(cfg['backbones'])} backbones)")


if __name__ == "__main__":
    main()
