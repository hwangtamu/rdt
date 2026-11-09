import json
import os
from collections import Counter

import numpy as np

from w2c.compare_e1 import compare
from w2c.config import load_config
from w2c.data import normalize_record, stratified_group_split
from w2c.labels import label_id
from w2c.make_dummy_data import generate
from w2c.train_e1 import run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_label_mapping():
    assert label_id("B") == 1
    assert label_id("tool_call") == 1
    assert label_id("ambiguous") is None
    assert label_id("not-a-label") is None


def test_grouped_split_no_leakage():
    cfg = load_config(os.path.join(ROOT, "configs", "e1_dummy.json"))
    stats: Counter = Counter()
    records = [normalize_record(r, cfg, stats) for r in generate(8, seed=1)]
    records = [r for r in records if r is not None]
    parts = stratified_group_split(records, val_fraction=0.2, test_fraction=0.2, seed=0)
    groups = {name: {r["group"] for r in recs} for name, recs in parts.items()}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert all(parts[name] for name in parts)


def test_dummy_end_to_end(tmp_path):
    raw = tmp_path / "raw.jsonl"
    with open(raw, "w", encoding="utf-8") as fh:
        for rec in generate(20, seed=2):
            fh.write(json.dumps(rec) + "\n")

    cfg = load_config(os.path.join(ROOT, "configs", "e1_dummy.json"))
    cfg["data"].update({
        "raw": str(raw), "train": None, "val": None, "test": None,
        "max_examples": None,
    })
    cfg["features"].update({
        "cache_dir": str(tmp_path / "features"), "batch_size": 64,
        "layers": [0, 4, 8], "device": "cpu",
    })
    cfg["probes"].update({"max_epochs": 10, "patience": 3, "mlp_hidden": 16})
    cfg["eval"].update({"bootstrap": 30})
    cfg["seeds"] = [0]
    cfg["out"] = str(tmp_path / "e1")

    result = run(cfg)
    assert result["n_runs"] == 2 * 3 * 2  # 2 backbones x 3 layers x 2 probes

    with open(os.path.join(cfg["out"], "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert sum(manifest["counts"].values()) == 80
    assert (tmp_path / "e1" / "preds").exists()

    comparison = compare(cfg["out"], n_boot=30)
    assert len(comparison["rows"]) == 3 * 2
    for row in comparison["rows"]:
        assert np.isfinite(row["delta_mean"])
        assert row["delta_lo"] <= row["delta_hi"]


def test_audit_sample_is_seeded_and_stratified():
    from w2c.build_w2c_e1 import sample_for_audit

    labelled = []
    for label in ("tool_call", "request_for_info", "cannot_answer"):
        for i in range(20):
            rec = {"id": f"pref-{label}-{i:03d}", "label_name": label,
                   "source": "pref_chosen", "question": f"q{i}", "tools": []}
            labelled.append((rec, "response", "rule"))
    a = sample_for_audit(labelled, per_class=5, seed=0)
    b = sample_for_audit(labelled, per_class=5, seed=0)
    c = sample_for_audit(labelled, per_class=5, seed=1)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert [x["id"] for x in a] != [x["id"] for x in c]
    counts = Counter(x["decision"] for x in a)
    assert counts == {"tool_call": 5, "request_for_info": 5, "cannot_answer": 5}


def test_e15_dummy_end_to_end(tmp_path):
    from w2c.compare_e15 import compare as compare_e15
    from w2c.train_e15 import run as run_e15

    raw = tmp_path / "raw.jsonl"
    with open(raw, "w", encoding="utf-8") as fh:
        for rec in generate(24, seed=3):
            fh.write(json.dumps(rec) + "\n")

    cfg = load_config(os.path.join(ROOT, "configs", "e15_dummy.json"))
    cfg["data"].update({
        "raw": str(raw), "train": None, "val": None, "test": None,
        "max_examples": None,
    })
    cfg["features"].update({
        "cache_dir": str(tmp_path / "features"),
        "token_cache_dir": str(tmp_path / "features_tokens"),
        "batch_size": 32, "layers": [0, 4, 8], "device": "cpu",
    })
    cfg["probes"].update({"max_epochs": 8, "patience": 3, "mlp_hidden": 16})
    cfg["eval"].update({"bootstrap": 20})
    cfg["e15"]["attn"].update({"layers": [0, 4, 8], "batch_size": 16,
                               "max_epochs": 6, "patience": 2})
    cfg["e15"]["concat"].update({"layer_sets": [[0, 4, 8]]})
    cfg["seeds"] = [0]
    cfg["out"] = str(tmp_path / "e15")

    run_e15(cfg)

    records = [json.loads(line) for line in
               open(os.path.join(cfg["out"], "runs.jsonl"), encoding="utf-8")]
    # 2 backbones x (3 last + 3 mean + 3 attn + 1 concat) x 2 probes x 1 seed
    assert len(records) == 2 * (3 + 3 + 3 + 1) * 2
    assert {r["readout"] for r in records} == {"last", "mean", "attn", "concat"}
    assert all(np.isfinite(r["test"]["ce_bits"]) for r in records)
    assert all(os.path.exists(os.path.join(
        cfg["out"], "preds", r["run_id"] + ".npz")) for r in records)

    result = compare_e15(cfg["out"], n_boot=20)
    assert len(result["selected"]) == 2 * 4 * 2  # backbones x readouts x probes
    assert result["pairs"]
    for row in result["selected"]:
        assert np.isfinite(row["ce_bits"])
        assert row["ce_lo"] <= row["ce_hi"]
