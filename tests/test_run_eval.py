"""Official evaluator parity and C6 -> C7 integration, using temporary data only."""
import copy
import json
import subprocess
import sys

import pytest

from qa.eval.run_eval import OFFICIAL_EVAL, official_scores, run_eval, validate_record


def metadata():
    return {"system": "synthetic_span", "variant": "span_only", "seed": 42,
            "git_sha": "synthetic-test-only", "config": {"seed": 42, "model": {"encoder": "tiny"}}}


def fixture_files(tmp_path):
    answers = [["The Blue whale!", "whale"], ["cat cat dog"], [], ["the"], ["café 東京"]]
    qas = [{"id": str(i), "question": "?", "answers": [
        {"text": a, "answer_start": 0} for a in refs]} for i, refs in enumerate(answers)]
    data = tmp_path / "synthetic.json"
    data.write_text(json.dumps({"data": [{"title": "synthetic", "paragraphs": [
        {"context": "synthetic", "qas": qas}]}]}), encoding="utf-8")
    predictions = {"0": " WHALE ", "1": "cat dog dog", "2": "", "3": "", "4": "CAFÉ 東京"}
    pred = tmp_path / "predictions.json"
    pred.write_text(json.dumps(predictions), encoding="utf-8")
    return data, pred, predictions


def test_official_cli_and_training_parity(tmp_path):
    from qa.train import official_scores as training_scores

    data, pred, predictions = fixture_files(tmp_path)
    output = tmp_path / "official.json"
    subprocess.run([sys.executable, str(OFFICIAL_EVAL), str(data), str(pred),
                    "--out-file", str(output)], check=True, capture_output=True)
    expected = json.loads(output.read_text())
    actual = official_scores(data, predictions)
    assert actual == {"exact": expected["exact"], "f1": expected["f1"], "n": expected["total"]}
    assert actual == training_scores(data, predictions)
    assert actual["exact"] == 80
    assert actual["f1"] == pytest.approx(100 * (4 + 2 / 3) / 5)
    # Extra IDs have the same ignored behavior as the official evaluator.
    assert official_scores(data, {**predictions, "extra": "ignored"}) == actual


def test_cli_writes_frozen_c7_and_preserves_source(tmp_path, capsys):
    from qa.eval.run_eval import main

    data, pred, _ = fixture_files(tmp_path)
    source = metadata()
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps(source))
    out = tmp_path / "runs/synthetic_val.json"
    main(["--data-file", str(data), "--predictions", str(pred), "--metadata", str(meta),
          "--split", "val", "--run-id", "synthetic_val", "--out-file", str(out)])
    record = json.loads(out.read_text())
    validate_record(record)
    assert record == json.loads(capsys.readouterr().out)
    assert record["config"] == source["config"] and record["git_sha"] == source["git_sha"]
    assert record["recall_at_1"] is None and record["train_gpu_hours"] is None
    assert record["n"] == 5 and record["split"] == "val"
    original = copy.deepcopy(record)
    new = run_eval(data, pred, record, "addsent", "synthetic_addsent", tmp_path / "new.json")
    assert record == original and new["split"] == "addsent"
    with pytest.raises(FileExistsError):
        run_eval(data, pred, source, "val", "other", out)
    assert json.loads(out.read_text()) == record


@pytest.mark.parametrize("bad", [{}, {"0": 3}, [], {0: "x"}])
def test_bad_predictions_fail_without_output(tmp_path, bad):
    data, _, _ = fixture_files(tmp_path)
    with pytest.raises(ValueError):
        official_scores(data, bad)


def test_empty_and_duplicate_ids_rejected(tmp_path):
    data, _, predictions = fixture_files(tmp_path)
    obj = json.loads(data.read_text())
    qas = obj["data"][0]["paragraphs"][0]["qas"]
    qas.append(qas[0])
    data.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="unique"):
        official_scores(data, predictions)
    data.write_text('{"data": []}')
    with pytest.raises(ValueError, match="nonempty"):
        official_scores(data, predictions)


def test_test_guards_before_reading(tmp_path):
    with pytest.raises(ValueError, match="test set"):
        official_scores(tmp_path / "dev-v1.1.json", {})
    with pytest.raises(ValueError, match="test set"):
        run_eval("absent", "absent", metadata(), "test", "x", tmp_path / "x.json")
    # Only synthetic data exercises the explicit final-test gate.
    data, pred, _ = fixture_files(tmp_path)
    record = run_eval(data, pred, metadata(), "test", "synthetic_test",
                      tmp_path / "allowed.json", allow_test=True)
    assert record["n"] == 5


def test_training_record_metadata_preserved_and_invalid_metadata_rejected(tmp_path):
    from qa.train import run_record
    import torch

    data, pred, predictions = fixture_files(tmp_path)
    cfg = {**metadata()["config"], "system": "synthetic", "variant": "span_only"}
    source = run_record(cfg, "train_s42", official_scores(data, predictions), 1.25, 0.5,
                        torch.device("cpu"))
    result = run_eval(data, pred, source, "val", "eval_s42", tmp_path / "eval.json")
    for key in ("config", "git_sha", "params_M", "train_gpu_hours", "gpu", "seed"):
        assert result[key] == source[key]
    for broken in ({}, {**metadata(), "f1": float("nan")}, {**metadata(), "unknown": 1}):
        # f1 is recomputed, so use an invalid optional measurement for validation.
        if "f1" in broken:
            broken["params_M"] = float("nan")
        with pytest.raises(ValueError):
            run_eval(data, pred, broken, "val", "bad", tmp_path / "bad.json")
        assert not (tmp_path / "bad.json").exists()
