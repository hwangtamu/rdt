"""E2: full fine-tuning and data scaling (RDT plan section 7).

Both Ettin backbones are fully fine-tuned with an identical decision head on
nested, class-stratified fractions of the E1 training pool. The protocol reuses
E1's grouped train/val split and untouched official MCQ test set, so E2 differs
from E1 only in whether the backbone is adapted.

    PYTHONPATH=src python -m w2c.train_e2 --config configs/e2_w2c_pref.json

Fractions are N in {1, 5, 10, 25, 50, 100}% of the training pool. For a fixed
seed, each smaller fraction is a prefix of the next (group-level prefixes within
class), so the D(N) curve is nested. The same prefixes are used across model
seeds; seed-to-seed variance is optimization variance, not data variance.

Important: D(N) is a data-scaling curve, not a rate-distortion curve. N is not
I(X;Z); rate-distortion belongs to E3/E4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from .config import ensure_dir, load_config
from .data import load_data
from .metrics import (bootstrap_ci, ce_bits, classification_metrics,
                      per_example_ce_bits)
from .features import model_ref

CLASSES_ORDER = ("tool_call", "request_for_info", "cannot_answer")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cpu_clone(value):
    import torch

    if isinstance(value, dict):
        return {k: _cpu_clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_clone(v) for v in value)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str):
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {name}, but CUDA is unavailable")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# nested, stratified data fractions
# --------------------------------------------------------------------------- #

def nested_fraction_indices(records: list[dict], fractions: list[float],
                            order_seed: int = 0) -> dict[float, list[int]]:
    """Return nested index sets, stratified by class at the group level.

    Groups inside each class are shuffled once with ``order_seed``; a fraction
    takes a prefix of that ordering. Thus fraction f1 < f2 implies
    indices[f1] ⊆ indices[f2], and the same subsets are reused across model
    seeds.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        groups[rec["group"]].append(idx)

    by_class: dict[int, list[str]] = defaultdict(list)
    for group, idxs in groups.items():
        label = Counter(records[i]["label"] for i in idxs).most_common(1)[0][0]
        by_class[label].append(group)

    rng = random.Random(order_seed)
    ordered: dict[int, list[str]] = {}
    for label, group_ids in sorted(by_class.items()):
        group_ids = list(group_ids)
        rng.shuffle(group_ids)
        ordered[label] = group_ids

    class_totals = {
        label: sum(len(groups[g]) for g in group_ids)
        for label, group_ids in ordered.items()
    }

    result = {}
    for fraction in fractions:
        chosen: list[str] = []
        if fraction >= 1.0:
            for group_ids in ordered.values():
                chosen.extend(group_ids)
        else:
            for label, group_ids in ordered.items():
                target = int(round(fraction * class_totals[label]))
                taken = 0
                for group in group_ids:
                    if taken >= target and taken > 0:
                        break
                    chosen.append(group)
                    taken += len(groups[group])
        indices = sorted(i for g in chosen for i in groups[g])
        result[float(fraction)] = indices
    return result


# --------------------------------------------------------------------------- #
# tokenization / batching
# --------------------------------------------------------------------------- #

def tokenize_records(records: list[dict], tokenizer, max_length: int,
                     batch_size: int = 256) -> tuple[list[list[int]], np.ndarray]:
    """Tokenize prompts once; also audit untruncated lengths."""
    prompts = [r["prompt"] for r in records]
    seqs: list[list[int]] = []
    full_lengths: list[int] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(batch, add_special_tokens=True, truncation=True,
                        max_length=max_length)
        seqs.extend([list(ids) for ids in enc["input_ids"]])
        lengths = tokenizer(batch, add_special_tokens=True, truncation=False,
                            return_length=True)["length"]
        full_lengths.extend(int(x) for x in lengths)
    trunc = np.asarray(full_lengths) >= max_length
    return seqs, trunc


def _batch_tensors(seqs: list[list[int]], indices: list[int], pad_id: int, device):
    import torch

    lengths = [len(seqs[i]) for i in indices]
    width = max(lengths)
    input_ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, (idx, length) in enumerate(zip(indices, lengths)):
        input_ids[row, :length] = torch.tensor(seqs[idx], dtype=torch.long)
        mask[row, :length] = 1
    return input_ids.to(device), mask.to(device)


# --------------------------------------------------------------------------- #
# model / forward
# --------------------------------------------------------------------------- #

def load_tokenizer(backbone: dict):
    from transformers import AutoTokenizer

    ref = model_ref(backbone)
    tokenizer = AutoTokenizer.from_pretrained(ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (tokenizer.eos_token or tokenizer.sep_token
                               or tokenizer.unk_token)
    return tokenizer


def load_model(backbone: dict):
    import torch
    from transformers import AutoModel

    ref = model_ref(backbone)
    tokenizer = load_tokenizer(backbone)
    model = AutoModel.from_pretrained(ref, torch_dtype=torch.float32)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return tokenizer, model


class DecisionHead:
    """Standardization + linear or one-hidden-layer MLP decision head."""

    def __init__(self, hidden: int, n_classes: int, dropout: float, seed: int,
                 head_kind: str = "linear", mlp_hidden: int = 256,
                 standardize: bool = False):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.head_kind = head_kind
        self.standardize = (
            nn.BatchNorm1d(hidden, affine=False, momentum=0.05)
            if standardize else None)
        self.dropout = nn.Dropout(dropout)
        self.linear = None
        self.mlp = None
        if head_kind == "linear":
            self.linear = nn.Linear(hidden, n_classes)
        elif head_kind == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, n_classes),
            )
        else:
            raise ValueError(f"unknown head kind: {head_kind}")

    def to(self, device):
        if self.standardize is not None:
            self.standardize.to(device)
        self.dropout.to(device)
        if self.linear is not None:
            self.linear.to(device)
        if self.mlp is not None:
            self.mlp.to(device)
        return self

    def train(self):
        if self.standardize is not None:
            self.standardize.train()
        self.dropout.train()
        if self.mlp is not None:
            self.mlp.train()
        return self

    def eval(self):
        if self.standardize is not None:
            self.standardize.eval()
        self.dropout.eval()
        if self.mlp is not None:
            self.mlp.eval()
        return self

    def parameters(self):
        params = list(self.dropout.parameters())
        if self.standardize is not None:
            params += list(self.standardize.parameters())
        if self.linear is not None:
            params += list(self.linear.parameters())
        if self.mlp is not None:
            params += list(self.mlp.parameters())
        return params

    def state_dict(self):
        return {
            "head_kind": self.head_kind,
            "standardize": None if self.standardize is None else self.standardize.state_dict(),
            "dropout": self.dropout.state_dict(),
            "linear": None if self.linear is None else self.linear.state_dict(),
            "mlp": None if self.mlp is None else self.mlp.state_dict(),
        }

    def __call__(self, hidden):
        if self.standardize is not None:
            hidden = self.standardize(hidden)
        hidden = self.dropout(hidden)
        if self.linear is not None:
            return self.linear(hidden)
        return self.mlp(hidden)


def parse_readout(spec: str) -> tuple[str, list[int]]:
    """Parse ``last-L22``, ``lasth-L17``, ``attn-L17`` or ``fusion-L17+L22``."""
    def _layer(token: str) -> int:
        return int(token[1:] if token.startswith("L") else token)

    if spec.startswith("last-L"):
        return "last", [_layer(spec.split("-", 1)[1])]
    if spec.startswith("lasth-L"):
        # Last token from a hidden_states entry (pre-final-norm where applicable).
        return "lasth", [_layer(spec.split("-", 1)[1])]
    if spec.startswith("attn-L"):
        return "attn", [_layer(spec.split("-", 1)[1])]
    if spec.startswith("fusion-L"):
        layers = [_layer(x) for x in spec.split("-", 1)[1].split("+")]
        if len(layers) != 2:
            raise ValueError(f"fusion readout needs exactly two layers: {spec}")
        return "fusion", layers
    raise ValueError(f"unknown readout spec: {spec}")


class Readout:
    """Last-token, single-layer attention, or two-layer attention fusion.

    ``fusion-L17+L22`` concatenates the per-token states from both layers and
    applies one attention pool over the 2*hidden token vectors.  This is a
    deliberately small head; parameter counts are recorded so the extra
    capacity is visible.
    """

    def __init__(self, spec: str, hidden: int, dropout: float, seed: int):
        import torch
        import torch.nn as nn

        self.spec = spec
        self.kind, self.layers = parse_readout(spec)
        self.needs_hidden = self.kind in ("lasth", "attn", "fusion")
        self.out_dim = hidden * (len(self.layers) if self.kind == "fusion" else 1)
        torch.manual_seed(seed)
        self.query = None
        self.dropout = None
        if self.kind in ("attn", "fusion"):
            self.query = nn.Parameter(torch.randn(self.out_dim) / math.sqrt(self.out_dim))
            self.dropout = nn.Dropout(dropout)

    def to(self, device):
        if self.query is not None:
            self.query = type(self.query)(self.query.data.to(device))
        if self.dropout is not None:
            self.dropout.to(device)
        return self

    def parameters(self):
        return [] if self.query is None else [self.query]

    def train(self):
        if self.dropout is not None:
            self.dropout.train()
        return self

    def eval(self):
        if self.dropout is not None:
            self.dropout.eval()
        return self

    def state_dict(self):
        return {"kind": self.kind, "layers": list(self.layers),
                "query": None if self.query is None else self.query.detach().cpu()}

    def __call__(self, out, mask):
        import torch

        if self.kind == "last":
            hidden = out.last_hidden_state
            positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
            last_idx = (positions * mask).max(dim=1).values
            rows = torch.arange(mask.size(0), device=mask.device)
            return hidden[rows, last_idx]

        if self.kind == "lasth":
            hidden = out.hidden_states[self.layers[0]]
            positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
            last_idx = (positions * mask).max(dim=1).values
            rows = torch.arange(mask.size(0), device=mask.device)
            return hidden[rows, last_idx]

        states = out.hidden_states
        max_layer = len(states) - 1
        missing = [layer for layer in self.layers if layer > max_layer]
        if missing:
            raise ValueError(
                f"readout {self.spec} requests layers {missing}, but model has "
                f"hidden_states 0..{max_layer}")
        if self.kind == "attn":
            hidden = states[self.layers[0]]
        else:
            hidden = torch.cat([states[layer] for layer in self.layers], dim=-1)

        scores = torch.matmul(hidden, self.query.to(hidden.dtype))
        scores = scores / math.sqrt(self.out_dim)
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        return self.dropout(pooled)


def _logits(model, head, readout, input_ids, mask):
    out = model(input_ids=input_ids, attention_mask=mask, return_dict=True,
                output_hidden_states=readout.needs_hidden)
    pooled = readout(out, mask)
    return head(pooled)


def evaluate(model, head, readout, seqs, indices, labels, pad_id, device, batch_size,
             autocast=True):
    import torch

    model.eval()
    head.eval()
    readout.eval()
    logits_all = []
    order = list(indices)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            input_ids, mask = _batch_tensors(seqs, idxs, pad_id, device)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and autocast):
                logits = _logits(model, head, readout, input_ids, mask)
            logits_all.append(logits.float().cpu().numpy())
    if not logits_all:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(logits_all, axis=0)


# --------------------------------------------------------------------------- #
# one fine-tuning run
# --------------------------------------------------------------------------- #

def finetune_run(backbone: dict, cfg: dict, parts: dict, seqs: dict, trunc: dict,
                 fraction: float, seed: int, readout_spec: str) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers.optimization import get_linear_schedule_with_warmup

    ft = cfg["finetune"]
    eval_cfg = cfg["eval"]
    autocast = bool(ft.get("autocast", True))
    device = _resolve_device(ft["device"])
    classes = list(cfg["data"]["classes"])
    n_classes = len(classes)
    labels = {split: np.array([r["label"] for r in recs])
              for split, recs in parts.items()}

    fractions = [float(x) for x in cfg["fractions"]]
    frac_indices = nested_fraction_indices(parts["train"], fractions,
                                           order_seed=int(cfg["seed"]))
    train_idx = frac_indices[float(fraction)]
    n_train = len(train_idx)
    if n_train == 0:
        raise ValueError(f"empty training fraction {fraction}")

    max_epochs = int(ft["max_epochs_by_fraction"][str(fraction)])
    batch_size = int(ft["batch_size"])
    grad_accum = int(ft["grad_accum"])
    steps_per_epoch = math.ceil(n_train / (batch_size * grad_accum))
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(round(total_steps * float(ft.get("warmup_ratio", 0.06))))

    set_seed(seed)
    tokenizer, model = load_model(backbone)
    model.to(device)
    hidden = int(model.config.hidden_size)
    readout = Readout(readout_spec, hidden, float(ft["dropout"]), seed=seed).to(device)
    head = DecisionHead(
        readout.out_dim, n_classes, float(ft["dropout"]), seed=seed,
        head_kind=str(ft.get("head", "linear")),
        mlp_hidden=int(ft.get("mlp_hidden", 256)),
        standardize=bool(ft.get("standardize", False)),
    ).to(device)
    trainable = list(model.parameters()) + readout.parameters() + head.parameters()

    optimizer = torch.optim.AdamW(
        trainable, lr=float(ft["lr"]), weight_decay=float(ft["weight_decay"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    rng = random.Random(seed * 100003 + int(round(fraction * 1000)))
    pad_id = tokenizer.pad_token_id
    history = []
    best = {"val_ce": float("inf"), "epoch": -1, "test_logits": None,
            "val_logits": None, "train_loss": None,
            "model_state": None, "readout_state": None, "head_state": None}
    tokens_seen = 0
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        head.train()
        readout.train()
        order = list(train_idx)
        rng.shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_no, start in enumerate(range(0, len(order), batch_size)):
            idxs = order[start:start + batch_size]
            input_ids, mask = _batch_tensors(seqs["train"], idxs, pad_id, device)
            y = torch.tensor(labels["train"][idxs], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and autocast):
                logits = _logits(model, head, readout, input_ids, mask)
                loss = F.cross_entropy(logits, y)
            (loss / grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
            tokens_seen += int(mask.sum().item())
            if (batch_no + 1) % grad_accum == 0 or start + batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(ft.get("grad_clip", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_logits = evaluate(model, head, readout, seqs["val"],
                              list(range(len(parts["val"]))),
                              labels["val"], pad_id, device,
                              int(ft["eval_batch_size"]), autocast=autocast)
        val_ce = ce_bits(val_logits, labels["val"])
        history.append({"epoch": epoch, "step": global_step,
                        "train_loss": train_loss, "val_ce_bits": val_ce})
        print(f"[e2]   {backbone['key']} f={fraction:g} s={seed} "
              f"ep={epoch} step={global_step} train={train_loss:.4f} val={val_ce:.4f}")
        if val_ce < best["val_ce"] - 1e-9:
            test_logits = evaluate(
                model, head, readout, seqs["test"], list(range(len(parts["test"]))),
                labels["test"], pad_id, device,
                int(ft["eval_batch_size"]), autocast=autocast)
            best = {"val_ce": float(val_ce), "epoch": epoch,
                    "test_logits": test_logits,
                    "val_logits": val_logits, "train_loss": train_loss,
                    "model_state": _cpu_clone(model.state_dict()),
                    "readout_state": _cpu_clone(readout.state_dict()),
                    "head_state": _cpu_clone(head.state_dict())}
        elif epoch - best["epoch"] >= int(ft["patience"]):
            break

    if best["test_logits"] is None:
        raise RuntimeError("no validation improvement recorded")
    test_logits = best["test_logits"]
    y_test = labels["test"]
    metrics = classification_metrics(test_logits, y_test,
                                     n_bins=int(eval_cfg["n_bins"]),
                                     k=n_classes)
    metrics.update(bootstrap_ci(
        test_logits, y_test, n_boot=int(eval_cfg["bootstrap"]), seed=seed,
        alpha=float(eval_cfg["alpha"]), n_bins=int(eval_cfg["n_bins"]),
        k=n_classes))
    val_metrics = classification_metrics(best["val_logits"], labels["val"],
                                         n_bins=int(eval_cfg["n_bins"]),
                                         k=n_classes)

    run_id = (f"{backbone['key']}__{readout_spec}__N"
              f"{int(round(fraction * 100)):03d}__s{seed}")
    n_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in readout.parameters()) + sum(
        p.numel() for p in head.parameters())
    record = {
        "run_id": run_id,
        "timestamp": _now(),
        "backbone_key": backbone["key"],
        "backbone_name": backbone["name"],
        "fraction": float(fraction),
        "n_train": n_train,
        "n_train_groups": len({parts["train"][i]["group"] for i in train_idx}),
        "seed": int(seed),
        "classes": classes,
        "readout": readout_spec,
        "readout_layers": list(readout.layers),
        "readout_params": int(sum(p.numel() for p in readout.parameters())),
        "head_kind": str(ft.get("head", "linear")),
        "standardize": bool(ft.get("standardize", False)),
        "head_params": int(sum(p.numel() for p in head.parameters())),
        "n_params": int(n_params),
        "epochs_run": len(history),
        "best_epoch": int(best["epoch"]),
        "steps": int(global_step),
        "train_tokens": int(tokens_seen),
        "approx_train_flops": float(6 * n_params * tokens_seen),
        "train_truncation_rate": float(np.mean(trunc["train"][train_idx])),
        "test_truncation_rate": float(np.mean(trunc["test"])),
        "finetune": {k: ft[k] for k in
                     ("lr", "weight_decay", "batch_size", "grad_accum",
                      "max_length", "dropout", "warmup_ratio", "patience",
                      "autocast") if k in ft},
        "val_ce_bits": float(best["val_ce"]),
        "history": history,
        "test": metrics,
        "test_val_metrics": val_metrics,
        "best_train_loss": float(best["train_loss"]),
        "checkpoint": None,
    }

    if bool(ft.get("save_checkpoints", False)):
        ckpt_dir = ensure_dir(os.path.join(cfg["out"], "checkpoints"))
        ckpt_path = os.path.join(ckpt_dir, run_id + ".pt")
        torch.save({
            "run_id": run_id,
            "backbone_key": backbone["key"],
            "backbone_name": backbone["name"],
            "backbone_path": backbone["path"],
            "readout": readout_spec,
            "readout_layers": list(readout.layers),
            "head_kind": str(ft.get("head", "linear")),
            "standardize": bool(ft.get("standardize", False)),
            "classes": classes,
            "seed": int(seed),
            "fraction": float(fraction),
            "best_epoch": int(best["epoch"]),
            "val_ce_bits": float(best["val_ce"]),
            "model_state": best["model_state"],
            "readout_state": best["readout_state"],
            "head_state": best["head_state"],
        }, ckpt_path)
        record["checkpoint"] = ckpt_path

    # Free GPU memory before the next run.
    del model, head, readout, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, test_logits, y_test


def run(cfg: dict) -> dict:
    parts, manifest = load_data(cfg)
    out = ensure_dir(cfg["out"])
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    ensure_dir(os.path.join(out, "preds"))
    ensure_dir(os.path.join(out, "runs"))

    fractions = [float(x) for x in cfg["fractions"]]
    seeds = [int(s) for s in cfg["seeds"]]
    ft = cfg["finetune"]
    readouts = list(ft.get("readouts") or [ft.get("readout", "last-L22")])
    print(f"[e2] splits: " + ", ".join(f"{k}={v}" for k, v in manifest["counts"].items()))
    print(f"[e2] fractions={fractions} seeds={seeds} readouts={readouts}")

    started = time.time()
    n_runs = 0
    for backbone in cfg["backbones"]:
        print(f"[e2] backbone {backbone['key']} ({backbone['name']})")
        tokenizer = load_tokenizer(backbone)
        max_length = int(cfg["finetune"]["max_length"])
        seqs, trunc = {}, {}
        for split in ("train", "val", "test"):
            seqs[split], trunc[split] = tokenize_records(
                parts[split], tokenizer, max_length)
            print(f"[e2]   tokenized {split}: n={len(seqs[split])} "
                  f"trunc={float(np.mean(trunc[split])):.4f}")
        del tokenizer

        run_tag = str(ft.get("run_tag", "") or "")
        for fraction in fractions:
            for seed in seeds:
                for readout_spec in readouts:
                    tag = f"{run_tag}__" if run_tag else ""
                    run_id = (f"{backbone['key']}__{tag}{readout_spec}__N"
                              f"{int(round(fraction * 100)):03d}__s{seed}")
                    record_path = os.path.join(out, "runs", run_id + ".json")
                    if os.path.exists(record_path):
                        print(f"[e2]   {run_id:<34} exists; skipping")
                        n_runs += 1
                        continue
                    record, test_logits, y_test = finetune_run(
                        backbone, cfg, parts, seqs, trunc, fraction, seed,
                        readout_spec)
                    test_ids = [r["id"] for r in parts["test"]]
                    np.savez(
                        os.path.join(out, "preds", run_id + ".npz"),
                        logits=test_logits.astype(np.float32),
                        labels=y_test.astype(np.int64),
                        ids=np.asarray(test_ids),
                        ce_bits=per_example_ce_bits(test_logits, y_test).astype(np.float32),
                    )
                    with open(os.path.join(out, "runs", run_id + ".json"), "w",
                              encoding="utf-8") as fh:
                        json.dump(record, fh, indent=2)
                    with open(os.path.join(out, "runs.jsonl"), "a",
                              encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    n_runs += 1
                    t = record["test"]
                    print(f"[e2]   {run_id:<34} N={record['n_train']:<5} "
                          f"CE={t['ce_bits']:.4f} acc={t['accuracy']:.4f} "
                          f"F1={t['macro_f1']:.4f}")
    elapsed = time.time() - started
    print(f"[e2] done: {n_runs} runs in {elapsed/60:.1f} min -> {out}")
    return {"out": out, "n_runs": n_runs, "elapsed_seconds": elapsed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2 full fine-tuning and data scaling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--fractions", default=None)
    parser.add_argument("--readouts", default=None,
                        help="comma-separated readout specs, e.g. last-L22,attn-L17")
    parser.add_argument("--device", default=None)
    parser.add_argument("--head", default=None, choices=["linear", "mlp"])
    parser.add_argument("--mlp-hidden", type=int, default=None)
    parser.add_argument("--standardize", action="store_true", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--save-checkpoints", action="store_true", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.out:
        cfg["out"] = args.out
    if args.seeds:
        cfg["seeds"] = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.models:
        keys = {x.strip() for x in args.models.split(",") if x.strip()}
        cfg["backbones"] = [b for b in cfg["backbones"] if b["key"] in keys]
    if args.fractions:
        cfg["fractions"] = [float(x) for x in args.fractions.split(",") if x.strip()]
    if args.readouts:
        cfg["finetune"]["readouts"] = [
            x.strip() for x in args.readouts.split(",") if x.strip()]
    if args.device:
        cfg["finetune"]["device"] = args.device
    if args.head:
        cfg["finetune"]["head"] = args.head
    if args.mlp_hidden:
        cfg["finetune"]["mlp_hidden"] = args.mlp_hidden
    if args.standardize:
        cfg["finetune"]["standardize"] = True
    if args.run_tag:
        cfg["finetune"]["run_tag"] = args.run_tag
    if args.save_checkpoints:
        cfg["finetune"]["save_checkpoints"] = True
    if args.bootstrap:
        cfg["eval"]["bootstrap"] = args.bootstrap

    result = run(cfg)
    print(f"[e2] result: {json.dumps(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
