"""train.py: CPU smoke test on a tiny model, the C7 record, config handling and safety checks.

C1/C2 and the evidence model are mocked in damian_mocks.py until X8/X9/G1 land.
"""
import json
import time

import pytest
import yaml

import qa.train as train_mod
from damian_mocks import ToyEvidenceModel, smoke_config
from qa.train import load_config, official_scores, train

C7_KEYS = {"run_id", "system", "variant", "seed", "split", "exact", "f1", "recall_at_1",
           "recall_at_3", "mrr", "n", "params_M", "train_gpu_hours", "gpu", "infer_qps",
           "git_sha", "config"}


def test_smoke_train_on_cpu_under_two_minutes(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "ROOT", tmp_path)  # --record writes under tmp, not the repo
    cfg = smoke_config(tmp_path)
    started = time.time()
    record = train(cfg, record=True, log=lambda _: None)
    assert time.time() - started < 120

    assert set(record) == C7_KEYS
    assert record["run_id"] == "smoke_s42" and record["split"] == "val" and record["n"] == 45
    assert 0 <= record["exact"] <= record["f1"] <= 100
    assert record["recall_at_1"] is None and record["train_gpu_hours"] is None
    run_dir = tmp_path / "outputs/smoke_s42"
    assert json.loads((run_dir / "run.json").read_text()) == record
    assert json.loads((tmp_path / "results/runs/smoke_s42.json").read_text()) == record
    for name in ("model.pt", "config.yaml", "encoder/config.json", "tokenizer.json"):
        assert (run_dir / "best" / name).exists()


def test_same_seed_same_result(tmp_path):
    a = train(smoke_config(tmp_path / "a", epochs=1), log=lambda _: None)
    b = train(smoke_config(tmp_path / "b", epochs=1), log=lambda _: None)
    assert (a["exact"], a["f1"]) == (b["exact"], b["f1"])


def test_new_model_class_plugs_in_through_config_only(tmp_path):
    ToyEvidenceModel.seen_neg_modes.clear()
    cfg = smoke_config(tmp_path, model_class="damian_mocks:ToyEvidenceModel", epochs=1, neg_mode="model")
    record = train(cfg, log=lambda _: None)
    assert ToyEvidenceModel.seen_neg_modes == ["model"]
    assert record["config"]["model"]["neg_mode"] == "model"


def test_config_layers_and_overrides(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"run_id": "x", "train": {"epochs": 3}}))
    cfg = load_config(exp, overrides=["train.learning_rate=2e-5", "model.neg_mode=random"])
    assert cfg["run_id"] == "x" and cfg["train"]["epochs"] == 3
    assert cfg["train"]["learning_rate"] == 2e-5 and cfg["model"]["neg_mode"] == "random"
    assert cfg["train"]["weight_decay"] == 0.01  # untouched base default


def test_refuses_to_tune_on_the_test_set(tmp_path):
    cfg = smoke_config(tmp_path)
    cfg["data"]["val_file"] = cfg["data"]["test_file"]
    with pytest.raises(ValueError, match="test set"):
        train(cfg, log=lambda _: None)


def test_missing_data_hooks_name_the_blocker(tmp_path):
    cfg = smoke_config(tmp_path)
    cfg["data"]["features_fn"] = None
    with pytest.raises(ValueError, match="X8/X9"):
        train(cfg, log=lambda _: None)


def test_official_scores_require_every_question(tmp_path):
    cfg = smoke_config(tmp_path)
    val = json.loads(open(cfg["data"]["val_file"]).read())
    qas = [qa for p in val["data"][0]["paragraphs"] for qa in p["qas"]]
    gold = {qa["id"]: qa["answers"][0]["text"] for qa in qas}
    assert official_scores(cfg["data"]["val_file"], gold) == {"exact": 100.0, "f1": 100.0, "n": 45}
    gold.popitem()
    with pytest.raises(ValueError, match="no prediction"):
        official_scores(cfg["data"]["val_file"], gold)
