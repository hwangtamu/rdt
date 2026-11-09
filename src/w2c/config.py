"""Config loading and small helpers."""

from __future__ import annotations

import copy
import json
import os
from typing import Any


DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "seeds": [0, 1, 2],
    "data": {
        "raw": None, "train": None, "val": None, "test": None,
        "classes": ["direct", "tool_call", "request_for_info", "cannot_answer"],
        "label_field": "decision", "id_field": "id",
        "question_field": "question", "tools_field": "tools",
        "group_field": "group_id",
        "val_fraction": 0.15, "test_fraction": 0.15, "max_examples": None,
    },
    "template": {
        "text": "Available Tools:\n{tool_definitions}\n\nUser Request:\n{question}\n\nDecision:",
        "tool_format": "- {name}\n  description: {description}\n  parameters: {parameters}",
        "empty_tools": "(none)",
    },
    "features": {
        "batch_size": 8, "max_length": 1024, "pooling": "last",
        "layers": None, "device": "cuda", "cache_dir": "runs/features",
        "token_cache_dir": None, "trust_remote_code": False,
    },
    "probes": {
        "kinds": ["linear", "mlp"], "mlp_hidden": 256, "dropout": 0.1,
        "lr": 1e-3, "weight_decay": 0.01, "batch_size": 4096,
        "max_epochs": 200, "patience": 20, "standardize": True,
    },
    "eval": {"bootstrap": 2000, "n_bins": 15, "alpha": 0.05},
    "e15": {
        "readouts": ["last", "mean", "attn", "concat"],
        "concat": {"pooling": "last", "layer_sets": []},
        "attn": {"layers": None, "device": None, "batch_size": 64,
                 "lr": None, "weight_decay": None,
                 "max_epochs": None, "patience": None},
    },
    "out": "runs/e1",
}


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str) -> dict:
    """Load JSON config (YAML avoided to keep dependencies minimal)."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return _deep_update(copy.deepcopy(DEFAULTS), cfg)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
