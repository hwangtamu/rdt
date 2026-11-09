"""E1: frozen representation probing for one or both backbones.

Example
-------
    PYTHONPATH=src python -m w2c.train_e1 --config configs/e1.json
    PYTHONPATH=src python -m w2c.train_e1 --config configs/e1_dummy.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from .config import ensure_dir, load_config
from .data import load_data
from .features import get_features
from .metrics import (bootstrap_ci, classification_metrics,
                      per_example_ce_bits, subgroup_metrics)
from .probes import predict, train_probe


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_list(value: str | None, cast=str):
    if value is None:
        return None
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def _slice_option(X: np.ndarray, layers: list[int], layer: int) -> np.ndarray:
    return X[:, layers.index(layer), :]


def run(cfg: dict) -> dict:
    parts, manifest = load_data(cfg)
    out = ensure_dir(cfg["out"])
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[e1] splits: " + ", ".join(f"{k}={v}" for k, v in manifest["counts"].items()))

    probe_cfg = cfg["probes"]
    eval_cfg = cfg["eval"]
    classes = list(cfg["data"].get("classes") or manifest["classes"])
    n_classes = len(classes)
    seeds = [int(s) for s in cfg["seeds"]]
    ensure_dir(os.path.join(out, "preds"))
    ensure_dir(os.path.join(out, "runs"))
    runs_path = os.path.join(out, "runs.jsonl")

    summaries = []
    for backbone in cfg["backbones"]:
        print(f"[e1] backbone {backbone['key']} ({backbone['name']})")
        feats = get_features(backbone, parts, cfg)
        layers = feats["train"]["layers"]
        trunc_rate = float(np.mean(feats["test"]["trunc"])) if len(feats["test"]["trunc"]) else 0.0
        print(f"[e1]   hidden dim={feats['train']['X'].shape[-1]} layers={layers} "
              f"test_truncation={trunc_rate:.3f}")

        y_train = np.array([r["label"] for r in parts["train"]])
        y_val = np.array([r["label"] for r in parts["val"]])
        y_test = np.array([r["label"] for r in parts["test"]])
        k_train = np.array([r.get("n_options", n_classes) for r in parts["train"]])
        k_val = np.array([r.get("n_options", n_classes) for r in parts["val"]])
        k_test = np.array([r.get("n_options", n_classes) for r in parts["test"]])
        test_ids = [r["id"] for r in parts["test"]]

        for layer in layers:
            X_train = _slice_option(feats["train"]["X"], layers, layer).astype(np.float32)
            X_val = _slice_option(feats["val"]["X"], layers, layer).astype(np.float32)
            X_test = _slice_option(feats["test"]["X"], layers, layer).astype(np.float32)

            for kind in probe_cfg["kinds"]:
                for seed in seeds:
                    model, info, scaler = train_probe(
                        X_train, y_train, X_val, y_val, kind, probe_cfg,
                        seed=seed, device=cfg["features"]["device"],
                        n_classes=n_classes,
                        n_options_train=k_train, n_options_val=k_val,
                    )
                    logits = predict(model, X_test, scaler)
                    metrics = classification_metrics(
                        logits, y_test, n_bins=eval_cfg["n_bins"], k=n_classes,
                        n_options=k_test)
                    metrics.update(bootstrap_ci(
                        logits, y_test, n_boot=eval_cfg["bootstrap"],
                        seed=seed, alpha=eval_cfg["alpha"], n_bins=eval_cfg["n_bins"],
                        k=n_classes, n_options=k_test,
                    ))
                    metrics["per_source"] = subgroup_metrics(
                        logits, y_test,
                        [r.get("source", "unknown") for r in parts["test"]],
                        n_bins=eval_cfg["n_bins"], k=n_classes, n_options=k_test)
                    metrics["per_tier"] = subgroup_metrics(
                        logits, y_test,
                        [r.get("tier", "unknown") for r in parts["test"]],
                        n_bins=eval_cfg["n_bins"], k=n_classes, n_options=k_test)
                    val_logits = predict(model, X_val, scaler)
                    val_ce = classification_metrics(
                        val_logits, y_val, n_bins=eval_cfg["n_bins"], k=n_classes,
                        n_options=k_val)["ce_bits"]

                    run_id = f"{backbone['key']}__L{layer}__{kind}__s{seed}"
                    np.savez(
                        os.path.join(out, "preds", run_id + ".npz"),
                        logits=logits.astype(np.float32),
                        labels=y_test.astype(np.int64),
                        ids=np.asarray(test_ids),
                        ce_bits=per_example_ce_bits(logits, y_test, k_test).astype(np.float32),
                        n_options=k_test.astype(np.int64),
                    )
                    record = {
                        "run_id": run_id,
                        "timestamp": _now(),
                        "backbone_key": backbone["key"],
                        "backbone_name": backbone["name"],
                        "layer": int(layer),
                        "probe": kind,
                        "seed": int(seed),
                        "val_ce_bits": float(val_ce),
                        "test": metrics,
                        "probe_info": info,
                        "classes": classes,
                        "truncation_rate": trunc_rate,
                    }
                    with open(os.path.join(out, "runs", run_id + ".json"), "w",
                              encoding="utf-8") as fh:
                        json.dump(record, fh, indent=2)
                    with open(runs_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    summaries.append(record)
                    print(f"[e1]   {run_id:<34} CE={metrics['ce_bits']:.4f} "
                          f"acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f} "
                          f"ECE={metrics['ece']:.4f}")
    return {"out": out, "n_runs": len(summaries)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E1 frozen representation probing")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="override output directory")
    parser.add_argument("--seeds", default=None, help="comma-separated seed override")
    parser.add_argument("--models", default=None, help="comma-separated backbone keys")
    parser.add_argument("--probes", default=None, help="comma-separated probe kinds")
    parser.add_argument("--layers", default=None, help="comma-separated layer indices")
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.out:
        cfg["out"] = args.out
    seeds = _parse_list(args.seeds, int)
    if seeds:
        cfg["seeds"] = seeds
    models = _parse_list(args.models)
    if models:
        cfg["backbones"] = [b for b in cfg["backbones"] if b["key"] in models]
    probes = _parse_list(args.probes)
    if probes:
        cfg["probes"]["kinds"] = probes
    layers = _parse_list(args.layers, int)
    if layers:
        cfg["features"]["layers"] = layers
    if args.bootstrap:
        cfg["eval"]["bootstrap"] = args.bootstrap

    result = run(cfg)
    print(f"[e1] done: {result['n_runs']} runs -> {result['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
