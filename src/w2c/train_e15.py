"""E1.5: frozen backbone, better readout (RDT plan sections 6-7 bridge).

E1 fixes the readout to the last non-pad token of ``"Decision:"`` and asks which
layer is most linearly decodable. E1.5 holds the backbone frozen and asks the
orthogonal question: *how much does the choice of readout matter?*

Readouts compared, on the same grouped splits and untouched MCQ test as E1:

* ``last``   -- last non-pad token (E1 baseline)
* ``mean``   -- masked mean over all prompt tokens
* ``attn``   -- learned single-query attention over all prompt tokens, trained
                jointly with the probe (extra ``hidden`` parameters)
* ``concat`` -- decision-position states concatenated over a configured layer set

For ``last``/``mean``/``attn`` every probed layer is swept, so each readout also
gets a layer curve. The comparison selects a configuration by *validation* CE
only; the test set is never used for selection.

    PYTHONPATH=src python -m w2c.train_e15 --config configs/e15_w2c_pref.json

Like E1, this measures accessibility with a lightweight head, not I(Y;Z). A
learned attention readout adds parameters, so it is not a pure representation
comparison; report parameter counts alongside it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone

import numpy as np

from .config import ensure_dir, load_config
from .data import load_data
from .features import get_features, get_token_features
from .metrics import (bootstrap_ci, classification_metrics,
                      per_example_ce_bits, subgroup_metrics)
from .probes import (predict, predict_attn_probe, token_offsets, token_stats,
                     train_attn_probe, train_probe)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_list(value: str | None, cast=str):
    if value is None:
        return None
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def _metrics(logits: np.ndarray, y: np.ndarray, eval_cfg: dict,
             n_classes: int, seed: int,
             n_options: np.ndarray | None = None,
             records: list[dict] | None = None) -> dict:
    out = classification_metrics(logits, y, n_bins=int(eval_cfg["n_bins"]),
                                 k=n_classes, n_options=n_options)
    out.update(bootstrap_ci(logits, y, n_boot=int(eval_cfg["bootstrap"]),
                            seed=seed, alpha=float(eval_cfg["alpha"]),
                            n_bins=int(eval_cfg["n_bins"]), k=n_classes,
                            n_options=n_options))
    if records is not None:
        out["per_source"] = subgroup_metrics(
            logits, y, [r.get("source", "unknown") for r in records],
            n_bins=int(eval_cfg["n_bins"]), k=n_classes, n_options=n_options)
        out["per_tier"] = subgroup_metrics(
            logits, y, [r.get("tier", "unknown") for r in records],
            n_bins=int(eval_cfg["n_bins"]), k=n_classes, n_options=n_options)
    return out


def _save_run(out: str, record: dict, logits: np.ndarray, y: np.ndarray,
              ids: list[str], n_options: np.ndarray | None = None) -> None:
    run_id = record["run_id"]
    payload = {
        "logits": logits.astype(np.float32),
        "labels": y.astype(np.int64),
        "ids": np.asarray(ids),
        "ce_bits": per_example_ce_bits(logits, y, n_options).astype(np.float32),
    }
    if n_options is not None:
        payload["n_options"] = np.asarray(n_options, dtype=np.int64)
    np.savez(os.path.join(out, "preds", run_id + ".npz"), **payload)
    with open(os.path.join(out, "runs", run_id + ".json"), "w",
              encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    with open(os.path.join(out, "runs.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _already(out: str, run_id: str) -> bool:
    return os.path.exists(os.path.join(out, "runs", run_id + ".json"))


def _concat_features(feats: dict, available: list[int],
                     layer_set: list[int]) -> np.ndarray:
    missing = [layer for layer in layer_set if layer not in available]
    if missing:
        raise ValueError(f"concat layers {missing} not in cached layers {available}")
    return np.concatenate(
        [feats["X"][:, available.index(layer), :] for layer in layer_set], axis=1)


def _base_record(backbone: dict, readout: str, probe: str, seed: int,
                 cfg: dict, classes: list[str], truncation_rate: float,
                 layer: int | None = None,
                 layer_set: list[int] | None = None) -> dict:
    return {
        "run_id": (f"{backbone['key']}__{readout}"
                   + (f"__L{layer}" if layer is not None else "")
                   + (f"__concat-{'-'.join(str(x) for x in layer_set)}"
                      if layer_set is not None else "")
                   + f"__{probe}__s{seed}"),
        "timestamp": _now(),
        "backbone_key": backbone["key"],
        "backbone_name": backbone["name"],
        "readout": readout,
        "layer": layer,
        "layer_set": layer_set,
        "probe": probe,
        "seed": int(seed),
        "classes": classes,
        "truncation_rate": truncation_rate,
        "template": cfg["template"]["text"],
    }


# --------------------------------------------------------------------------- #
# readouts over cached pooled features (last / mean / concat)
# --------------------------------------------------------------------------- #

def run_pooled_readout(backbone: dict, feats: dict, readout: str,
                       parts: dict, cfg: dict, out: str,
                       layer: int | None = None, layer_set: list[int] | None = None,
                       pooling: str = "last") -> None:
    probe_cfg, eval_cfg = cfg["probes"], cfg["eval"]
    classes = list(cfg["data"].get("classes"))
    n_classes = len(classes)
    y = {split: np.array([r["label"] for r in recs]) for split, recs in parts.items()}
    k = {split: np.array([r.get("n_options", n_classes) for r in recs])
         for split, recs in parts.items()}
    test_ids = [r["id"] for r in parts["test"]]
    truncation = float(np.mean(feats["test"]["trunc"])) if len(parts["test"]) else 0.0
    available = feats["train"]["layers"]

    def features(split: str) -> np.ndarray:
        if layer_set is not None:
            return _concat_features(feats[split], available, layer_set).astype(np.float32)
        index = available.index(int(layer))
        return feats[split]["X"][:, index, :].astype(np.float32)

    X_train, X_val, X_test = (features(s) for s in ("train", "val", "test"))
    for kind in probe_cfg["kinds"]:
        for seed in cfg["seeds"]:
            record = _base_record(backbone, readout, kind, int(seed), cfg, classes,
                                  truncation, layer=layer, layer_set=layer_set)
            if _already(out, record["run_id"]):
                print(f"[e15]   {record['run_id']:<46} exists; skipping")
                continue
            model, info, scaler = train_probe(
                X_train, y["train"], X_val, y["val"], kind, probe_cfg,
                seed=int(seed), device=cfg["features"]["device"],
                n_classes=n_classes,
                n_options_train=k["train"], n_options_val=k["val"],
            )
            val_logits = predict(model, X_val, scaler)
            test_logits = predict(model, X_test, scaler)
            record["pooling"] = pooling
            record["val_ce_bits"] = float(classification_metrics(
                val_logits, y["val"], n_bins=int(eval_cfg["n_bins"]),
                k=n_classes, n_options=k["val"])["ce_bits"])
            record["test"] = _metrics(test_logits, y["test"], eval_cfg,
                                      n_classes, int(seed), n_options=k["test"],
                                      records=parts["test"])
            record["probe_info"] = info
            _save_run(out, record, test_logits, y["test"], test_ids,
                      n_options=k["test"])
            t = record["test"]
            label = f"L{layer}" if layer is not None else f"concat{layer_set}"
            print(f"[e15]   {backbone['key']:<8} {readout:<7} {label:<18} "
                  f"{kind:<6} s{seed} CE={t['ce_bits']:.4f} "
                  f"acc={t['accuracy']:.4f} F1={t['macro_f1']:.4f}")


# --------------------------------------------------------------------------- #
# attention readout over cached token states
# --------------------------------------------------------------------------- #

def run_attn_readout(backbone: dict, tokens: dict, parts: dict, cfg: dict,
                     out: str) -> None:
    probe_cfg, eval_cfg = cfg["probes"], cfg["eval"]
    attn_cfg = cfg["e15"]["attn"]
    classes = list(cfg["data"].get("classes"))
    n_classes = len(classes)
    y = {split: np.array([r["label"] for r in recs]) for split, recs in parts.items()}
    k = {split: np.array([r.get("n_options", n_classes) for r in recs])
         for split, recs in parts.items()}
    test_ids = [r["id"] for r in parts["test"]]
    hidden = int(tokens["train"]["hidden"])
    truncation = float(np.mean(tokens["test"]["trunc"])) if len(parts["test"]) else 0.0
    layers = [int(x) for x in (attn_cfg.get("layers") or tokens["train"]["layers"])]
    device = attn_cfg.get("device") or cfg["features"]["device"]

    for layer in layers:
        if layer not in tokens["train"]["layers"]:
            raise ValueError(f"attn layer {layer} was not cached; "
                             f"cached={tokens['train']['layers']}")
        arrays, lengths, offsets = {}, {}, {}
        for split in ("train", "val", "test"):
            arrays[split] = np.load(tokens[split]["paths"][layer], mmap_mode="r")
            lengths[split] = tokens[split]["lengths"]
            offsets[split] = token_offsets(lengths[split])
        # Training-token statistics are shared by every seed/probe at this layer.
        token_mean, token_std = token_stats(arrays["train"])

        for kind in probe_cfg["kinds"]:
            for seed in cfg["seeds"]:
                record = _base_record(backbone, "attn", kind, int(seed), cfg,
                                      classes, truncation, layer=layer)
                if _already(out, record["run_id"]):
                    print(f"[e15]   {record['run_id']:<46} exists; skipping")
                    continue
                tcfg = copy.deepcopy(probe_cfg)
                tcfg.update({
                    "token_batch_size": int(attn_cfg.get("batch_size", 64)),
                    "lr": float(attn_cfg.get("lr", probe_cfg["lr"])),
                    "weight_decay": float(attn_cfg.get("weight_decay",
                                                       probe_cfg["weight_decay"])),
                    "max_epochs": int(attn_cfg.get("max_epochs",
                                                   probe_cfg["max_epochs"])),
                    "patience": int(attn_cfg.get("patience",
                                                 probe_cfg["patience"])),
                })
                model, info, val_logits = train_attn_probe(
                    arrays["train"], lengths["train"], y["train"],
                    arrays["val"], lengths["val"], y["val"],
                    kind, tcfg, seed=int(seed), device=device,
                    n_classes=n_classes, mean=token_mean, std=token_std,
                    n_options=k["train"], n_options_val=k["val"],
                )
                test_logits = predict_attn_probe(
                    model, arrays["test"], lengths["test"], offsets["test"],
                    token_mean, token_std,
                    batch_size=tcfg["token_batch_size"], device=device)
                val_logits = predict_attn_probe(
                    model, arrays["val"], lengths["val"], offsets["val"],
                    token_mean, token_std,
                    batch_size=tcfg["token_batch_size"], device=device)
                record["val_ce_bits"] = float(classification_metrics(
                    val_logits, y["val"], n_bins=int(eval_cfg["n_bins"]),
                    k=n_classes, n_options=k["val"])["ce_bits"])
                record["test"] = _metrics(test_logits, y["test"], eval_cfg,
                                          n_classes, int(seed),
                                          n_options=k["test"],
                                          records=parts["test"])
                record["probe_info"] = info
                record["hidden"] = hidden
                record["attn"] = {"batch_size": tcfg["token_batch_size"],
                                  "lr": tcfg["lr"], "max_epochs": tcfg["max_epochs"],
                                  "patience": tcfg["patience"]}
                _save_run(out, record, test_logits, y["test"], test_ids,
                          n_options=k["test"])
                t = record["test"]
                print(f"[e15]   {backbone['key']:<8} attn    L{layer:<2} "
                      f"{kind:<6} s{seed} CE={t['ce_bits']:.4f} "
                      f"acc={t['accuracy']:.4f} F1={t['macro_f1']:.4f} "
                      f"ep={info['epochs_run']}")
        del arrays



# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def run(cfg: dict) -> dict:
    parts, manifest = load_data(cfg)
    out = ensure_dir(cfg["out"])
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    ensure_dir(os.path.join(out, "preds"))
    ensure_dir(os.path.join(out, "runs"))

    e15 = cfg["e15"]
    readouts = [x for x in e15["readouts"]]
    print(f"[e15] splits: " + ", ".join(f"{k}={v}" for k, v in manifest["counts"].items()))
    print(f"[e15] readouts={readouts} seeds={cfg['seeds']}")

    for backbone in cfg["backbones"]:
        print(f"[e15] backbone {backbone['key']} ({backbone['name']})")

        for pooling in ("last", "mean"):
            if pooling not in readouts:
                continue
            fcfg = copy.deepcopy(cfg)
            fcfg["features"]["pooling"] = pooling
            feats = get_features(backbone, parts, fcfg)
            print(f"[e15]   {pooling} features: layers={feats['train']['layers']} "
                  f"dim={feats['train']['X'].shape[-1]}")
            for layer in feats["train"]["layers"]:
                run_pooled_readout(backbone, feats, pooling, parts, cfg, out,
                                   layer=int(layer), pooling=pooling)

        if "concat" in readouts:
            concat_cfg = e15.get("concat", {})
            pooling = concat_cfg.get("pooling", "last")
            fcfg = copy.deepcopy(cfg)
            fcfg["features"]["pooling"] = pooling
            feats = get_features(backbone, parts, fcfg)
            for layer_set in concat_cfg.get("layer_sets", [[]]):
                layer_set = [int(x) for x in layer_set]
                if not layer_set:
                    raise ValueError("concat.layer_sets contains an empty set")
                print(f"[e15]   concat({pooling}) layers={layer_set}")
                run_pooled_readout(backbone, feats, "concat", parts, cfg, out,
                                   layer_set=layer_set, pooling=pooling)

        if "attn" in readouts:
            tokens = get_token_features(backbone, parts, cfg)
            print(f"[e15]   token cache: hidden={tokens['train']['hidden']} "
                  f"layers={tokens['train']['layers']} "
                  f"train_tokens={len(tokens['train']['lengths'])} examples / "
                  f"{int(tokens['train']['lengths'].sum())} tokens")
            run_attn_readout(backbone, tokens, parts, cfg, out)

    return {"out": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E1.5 frozen backbone readout comparison")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--probes", default=None)
    parser.add_argument("--readouts", default=None,
                        help="comma-separated subset of last,mean,attn,concat")
    parser.add_argument("--layers", default=None)
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
    readouts = _parse_list(args.readouts)
    if readouts:
        cfg["e15"]["readouts"] = readouts
    layers = _parse_list(args.layers, int)
    if layers:
        cfg["features"]["layers"] = layers
        cfg["e15"].setdefault("attn", {})["layers"] = layers
    if args.bootstrap:
        cfg["eval"]["bootstrap"] = args.bootstrap

    result = run(cfg)
    print(f"[e15] done -> {result['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
