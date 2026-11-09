"""Frozen hidden-state extraction with on-disk caching.

Both backbones are read-only here (plan E1). The decision position is the last
non-pad token of the canonical prompt ("Decision:"), found independently of the
padding side. All layers are cached so E1 can probe layer-by-layer with matched
layer indices across models.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings

import numpy as np

from .config import ensure_dir


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #

def model_ref(backbone: dict) -> str:
    """Local checkpoint directory if provided, else the hub id."""
    return backbone.get("path") or backbone["name"]


def is_dummy(backbone: dict) -> bool:
    return model_ref(backbone) == "dummy" or backbone["name"] == "dummy"


def _cache_key(backbone: dict, fcfg: dict, tcfg: dict, parts: dict) -> str:
    payload = {
        "model": backbone["name"],
        "ref": model_ref(backbone),
        "template": tcfg["text"],
        "max_length": fcfg["max_length"],
        "pooling": fcfg["pooling"],
        "split_ids": {name: [r["id"] for r in recs] for name, recs in parts.items()},
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _cache_path(cache_dir: str, backbone_key: str, split: str, key: str) -> str:
    return os.path.join(cache_dir, f"{backbone_key}__{split}__{key}.npz")


def _token_cache_key(backbone: dict, fcfg: dict, tcfg: dict, parts: dict,
                     layers: list[int]) -> str:
    payload = {
        "model": backbone["name"],
        "ref": model_ref(backbone),
        "template": tcfg["text"],
        "max_length": fcfg["max_length"],
        "pooling": "tokens",
        "layers": [int(x) for x in layers],
        "split_ids": {name: [r["id"] for r in recs] for name, recs in parts.items()},
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _token_paths(cache_dir: str, backbone_key: str, split: str, key: str,
                 layers: list[int]) -> dict[int, str]:
    return {
        int(layer): os.path.join(
            cache_dir, f"{backbone_key}__{split}__tokens__L{int(layer)}__{key}.npy")
        for layer in layers
    }


def _token_meta_path(cache_dir: str, backbone_key: str, split: str, key: str) -> str:
    return os.path.join(cache_dir, f"{backbone_key}__{split}__tokens__meta__{key}.npz")


def _load_cached(path: str, expected_ids: list[str], layers: list[int] | None):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data["ids"]]
        if ids != expected_ids:
            return None
        available = [int(x) for x in data["layers"]]
        if layers is not None and any(layer not in available for layer in layers):
            return None
        X = data["X"]
        trunc = data["trunc"].astype(bool)
    if layers is not None:
        index = [available.index(layer) for layer in layers]
        X = X[:, index, :]
        available = list(layers)
    return {"X": X, "ids": ids, "trunc": trunc, "layers": available}


# --------------------------------------------------------------------------- #
# real backbones
# --------------------------------------------------------------------------- #

def _resolve_device(requested: str):
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"device '{requested}' unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_dtype(name: str, device):
    import torch

    if name in (None, "float32", "fp32"):
        return torch.float32
    dtype = {"float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[name]
    if device.type == "cpu" and dtype == torch.float16:
        warnings.warn("float16 inference on CPU is unsupported; using float32")
        return torch.float32
    return dtype


def _count_token_lengths(prompts: list[str], tokenizer, batch_size: int) -> np.ndarray:
    """Full (untruncated) token lengths, used only for the truncation audit."""
    lengths = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False,
                        return_length=True)
        lengths.extend(int(x) for x in enc["length"])
    return np.asarray(lengths, dtype=np.int32)


def _stream_real_tokens(records: list[dict], backbone: dict, fcfg: dict,
                        layers: list[int], paths: dict[int, str]) -> dict:
    """Write token-level hidden states for ``layers`` to per-layer .npy files.

    Tokens are stored ragged: one ``[sum(lengths), H]`` array per layer, in
    example order, without padding. This is what learned attention pooling needs
    and it is ~3x smaller than padding to the split maximum.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _resolve_device(fcfg["device"])
    dtype = _resolve_dtype(backbone.get("dtype"), device)
    ref = model_ref(backbone)
    trust = bool(fcfg.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(ref, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (tokenizer.eos_token or tokenizer.sep_token
                               or tokenizer.unk_token)
    model = AutoModel.from_pretrained(ref, torch_dtype=dtype, trust_remote_code=trust)
    model.eval().to(device)

    prompts = [r["prompt"] for r in records]
    batch_size = int(fcfg["batch_size"])
    max_length = int(fcfg["max_length"])
    full_lengths = _count_token_lengths(prompts, tokenizer, batch_size)
    trunc = full_lengths >= max_length
    lengths = np.minimum(full_lengths, max_length).astype(np.int32)
    total = int(lengths.sum())
    hidden = int(model.config.hidden_size)

    tmp_paths = {layer: paths[layer] + ".tmp" for layer in layers}
    maps = {layer: np.lib.format.open_memmap(
        tmp_paths[layer], mode="w+", dtype=np.float16, shape=(total, hidden))
        for layer in layers}
    cursor = 0
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start:start + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True, return_dict=True)
                valid = enc["attention_mask"].bool()
                batch_tokens = {
                    layer: out.hidden_states[layer][valid].float().cpu().numpy()
                    for layer in layers
                }
                n = batch_tokens[layers[0]].shape[0]
                for layer in layers:
                    maps[layer][cursor:cursor + n] = batch_tokens[layer].astype(np.float16)
                cursor += n
        if cursor != total:
            raise RuntimeError(f"token cache wrote {cursor} rows, expected {total}")
    finally:
        for layer in layers:
            maps[layer].flush()
            del maps[layer]
    for layer in layers:
        os.replace(tmp_paths[layer], paths[layer])
    return {"ids": [r["id"] for r in records], "lengths": lengths, "trunc": trunc,
            "layers": [int(x) for x in layers], "hidden": hidden}


def _stream_dummy_tokens(records: list[dict], fcfg: dict, layers: list[int],
                         paths: dict[int, str]) -> dict:
    rng = np.random.default_rng(1234)
    hidden = 768
    class_dirs = rng.normal(size=(4, hidden)) * 6.0
    lengths = np.array([
        4 + int(hashlib.sha1(r["id"].encode()).hexdigest()[:4], 16) % 9
        for r in records
    ], dtype=np.int32)
    total = int(lengths.sum())
    maps = {layer: np.lib.format.open_memmap(
        paths[layer] + ".tmp", mode="w+", dtype=np.float16, shape=(total, hidden))
        for layer in layers}
    cursor = 0
    max_layer = max(layers) if layers else 1
    try:
        for rec, length in zip(records, lengths):
            seed = int(hashlib.sha1(rec["id"].encode()).hexdigest()[:8], 16)
            local = np.random.default_rng(seed)
            base = local.normal(size=(int(length), hidden))
            for layer in layers:
                strength = layer / max_layer
                toks = base + strength * class_dirs[rec["label"]]
                maps[layer][cursor:cursor + int(length)] = toks.astype(np.float16)
            cursor += int(length)
    finally:
        for layer in layers:
            maps[layer].flush()
            del maps[layer]
    for layer in layers:
        os.replace(paths[layer] + ".tmp", paths[layer])
    return {"ids": [r["id"] for r in records], "lengths": lengths,
            "trunc": np.zeros(len(records), dtype=bool),
            "layers": [int(x) for x in layers], "hidden": hidden}


def _extract_real(records: list[dict], backbone: dict, fcfg: dict) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _resolve_device(fcfg["device"])
    dtype = _resolve_dtype(backbone.get("dtype"), device)
    ref = model_ref(backbone)
    trust = bool(fcfg.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(ref, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (tokenizer.eos_token or tokenizer.sep_token
                               or tokenizer.unk_token)
    model = AutoModel.from_pretrained(ref, torch_dtype=dtype, trust_remote_code=trust)
    model.eval().to(device)

    prompts = [r["prompt"] for r in records]
    batch_size = int(fcfg["batch_size"])
    max_length = int(fcfg["max_length"])
    pooling = fcfg["pooling"]
    if pooling not in ("last", "mean"):
        raise ValueError(f"unknown pooling '{pooling}'")

    full_lengths = _count_token_lengths(prompts, tokenizer, batch_size)
    trunc = full_lengths >= max_length

    per_batch = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states  # tuple of [B, T, H]
            mask = enc["attention_mask"].to(torch.float32)  # [B, T]
            if pooling == "last":
                # Works for left or right padding.
                positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
                last_idx = (positions * mask.long()).max(dim=1).values  # [B]
                rows = torch.arange(mask.size(0), device=mask.device)
                features = torch.stack([h[rows, last_idx] for h in hidden], dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                features = torch.stack(
                    [(h * mask.unsqueeze(-1)).sum(dim=1) / denom for h in hidden], dim=1
                )
            per_batch.append(features.float().cpu().numpy())

    X = np.concatenate(per_batch, axis=0)  # [N, n_layers, H]
    return {"X": X.astype(np.float32), "ids": [r["id"] for r in records],
            "trunc": trunc, "layers": list(range(X.shape[1]))}


# --------------------------------------------------------------------------- #
# dummy backbone (smoke tests only -- never report these numbers)
# --------------------------------------------------------------------------- #

def _extract_dummy(records: list[dict], backbone: dict, fcfg: dict) -> dict:
    rng = np.random.default_rng(1234)
    hidden_dim = 768
    n_layers = 23
    class_dirs = rng.normal(size=(4, hidden_dim)) * 6.0

    xs = []
    for rec in records:
        seed = int(hashlib.sha1(rec["id"].encode()).hexdigest()[:8], 16)
        local = np.random.default_rng(seed)
        X = local.normal(scale=1.0, size=(n_layers, hidden_dim))
        for layer in range(n_layers):
            strength = layer / (n_layers - 1)
            X[layer] += strength * class_dirs[rec["label"]]
        xs.append(X)
    X = np.stack(xs).astype(np.float32)
    writes = np.zeros(len(records), dtype=bool)
    return {"X": X, "ids": [r["id"] for r in records], "trunc": writes,
            "layers": list(range(n_layers))}


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def _load_token_meta(meta_path: str, expected_ids: list[str],
                     layers: list[int]) -> dict | None:
    if not os.path.exists(meta_path):
        return None
    with np.load(meta_path, allow_pickle=False) as data:
        ids = [str(x) for x in data["ids"]]
        if ids != expected_ids:
            return None
        available = [int(x) for x in data["layers"]]
        if any(layer not in available for layer in layers):
            return None
        return {
            "ids": ids,
            "lengths": data["lengths"].astype(np.int32),
            "trunc": data["trunc"].astype(bool),
            "layers": available,
            "hidden": int(data["hidden"]),
        }


def get_token_features(backbone: dict, parts: dict[str, list[dict]],
                       cfg: dict) -> dict:
    """Return ragged per-layer token states for learned pooling readouts.

    Result: ``{split: {paths: {layer: .npy path}, lengths, ids, trunc, layers,
    hidden}}``. Tokens are float16, shape ``[sum(lengths), hidden]`` in example
    order; slices are recovered with ``lengths``.
    """
    fcfg, tcfg = cfg["features"], cfg["template"]
    layers = [int(x) for x in (fcfg.get("layers") or [])]
    if not layers:
        raise ValueError("token readouts need an explicit features.layers list")
    key = _token_cache_key(backbone, fcfg, tcfg, parts, layers)
    cache_dir = ensure_dir(fcfg.get("token_cache_dir")
                           or str(fcfg["cache_dir"]) + "_tokens")

    result = {}
    for split, records in parts.items():
        ids = [r["id"] for r in records]
        meta_path = _token_meta_path(cache_dir, backbone["key"], split, key)
        paths = _token_paths(cache_dir, backbone["key"], split, key, layers)
        meta = _load_token_meta(meta_path, ids, layers)
        if meta is None or any(not os.path.exists(p) for p in paths.values()):
            if is_dummy(backbone):
                meta = _stream_dummy_tokens(records, fcfg, layers, paths)
            else:
                meta = _stream_real_tokens(records, backbone, fcfg, layers, paths)
            np.savez(meta_path,
                     ids=np.asarray(meta["ids"]),
                     lengths=meta["lengths"],
                     trunc=meta["trunc"],
                     layers=np.asarray(meta["layers"], dtype=np.int32),
                     hidden=np.asarray(meta["hidden"], dtype=np.int32))
        meta["paths"] = paths
        result[split] = meta
    return result


def get_features(backbone: dict, parts: dict[str, list[dict]], cfg: dict) -> dict:
    """Return {split: {"X": [N, L, H], "ids", "trunc", "layers"}} for one backbone."""
    fcfg, tcfg = cfg["features"], cfg["template"]
    key = _cache_key(backbone, fcfg, tcfg, parts)
    cache_dir = ensure_dir(fcfg["cache_dir"])
    requested_layers = fcfg.get("layers")
    requested_layers = None if requested_layers is None else [int(x) for x in requested_layers]

    result = {}
    for split, records in parts.items():
        path = _cache_path(cache_dir, backbone["key"], split, key)
        cached = _load_cached(path, [r["id"] for r in records], requested_layers)
        if cached is None:
            if is_dummy(backbone):
                feats = _extract_dummy(records, backbone, fcfg)
            else:
                feats = _extract_real(records, backbone, fcfg)
            available = feats["layers"]
            if requested_layers is not None:
                missing = [x for x in requested_layers if x not in available]
                if missing:
                    raise ValueError(
                        f"requested layers {missing} not in model output {available}"
                    )
                index = [available.index(x) for x in requested_layers]
                feats["X"] = feats["X"][:, index, :]
                feats["layers"] = list(requested_layers)
            np.savez(path, X=feats["X"].astype(np.float16),
                     ids=np.asarray(feats["ids"]), trunc=feats["trunc"],
                     layers=np.asarray(feats["layers"], dtype=np.int32))
            cached = feats
        result[split] = cached
    return result
