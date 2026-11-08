"""Synthetic C7 aggregation; no experiment records are created in the repository."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from qa.eval.run_eval import C7_KEYS

spec = importlib.util.spec_from_file_location(
    "make_tables", Path(__file__).resolve().parents[1] / "scripts/make_tables.py")
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def record(seed=13, score=70):
    r = dict.fromkeys(C7_KEYS)
    r.update(run_id=f"synthetic_s{seed}", system="bert_&_test", variant="span_only",
             seed=seed, split="val", n=10, exact=score, f1=score + 5,
             git_sha="synthetic", gpu="cpu", params_M=1,
             config={"seed": seed, "run_id": "synthetic", "model": {"encoder": "tiny"},
                     "decode": {"lambda": 0}, "data": {"val_file": "synthetic.json"}})
    return r


def test_sample_std_csv_latex_and_cli(tmp_path, capsys):
    runs, out = tmp_path / "runs", tmp_path / "tables"
    runs.mkdir()
    records = [record(s, v) for s, v in zip((13, 42, 87), (70, 72, 74))]
    for r in records:
        (runs / f"{r['run_id']}.json").write_text(json.dumps(r))
    before = copy.deepcopy(records)
    row, = tables.aggregate_records(records)
    assert records == before
    assert row["exact_mean"] == 72 and row["exact_std"] == 2
    assert row["seed_count"] == 3 and row["recall_at_1_mean"] is None
    tables.main(["--runs-dir", str(runs), "--out-dir", str(out)])
    assert "1 compatible groups" in capsys.readouterr().out
    with (out / "run_summary.csv").open(encoding="utf-8", newline="") as f:
        csv_row, = list(csv.DictReader(f))
    assert float(csv_row["exact_std"]) == 2 and csv_row["mrr_mean"] == ""
    tex = (out / "run_summary.tex").read_text(encoding="utf-8")
    assert r"72.00 $\pm$ 2.00" in tex and r"bert\_\&\_test" in tex
    assert "% Group 1:" in tex


@pytest.mark.parametrize("field", ["system", "variant", "split", "n", "gpu", "git_sha",
                                   "params_M", "encoder", "decode", "data", "learning_rate"])
def test_incompatible_groups_stay_separate(field):
    a, b = record(), record(42)
    if field == "encoder":
        b["config"]["model"]["encoder"] = "other"
    elif field == "decode":
        b["config"]["decode"]["lambda"] = 1
    elif field == "data":
        b["config"]["data"]["val_file"] = "other.json"
    elif field == "learning_rate":
        b["config"]["train"] = {"learning_rate": 1e-5}
    else:
        b[field] = 20 if field in ("n", "params_M") else "other"
    assert len(tables.aggregate_records([a, b])) == 2


def test_duplicate_runs_seeds_missing_values_and_invalid_records():
    a, b = record(), record(42)
    with pytest.raises(ValueError, match="duplicate run_id"):
        tables.aggregate_records([a, a])
    b["seed"] = a["seed"]
    with pytest.raises(ValueError, match="duplicate seed"):
        tables.aggregate_records([a, b])
    b = record(42)
    b["mrr"] = 0.5
    with pytest.raises(ValueError, match="mixed null"):
        tables.aggregate_records([a, b])
    for key, value in (("exact", float("nan")), ("f1", 101), ("n", 0), ("seed", True)):
        bad = {**a, key: value}
        with pytest.raises(ValueError):
            tables.aggregate_records([bad])
    del a["config"]
    with pytest.raises(ValueError, match="C7"):
        tables.aggregate_records([a])


def test_singleton_and_no_records(tmp_path):
    row, = tables.aggregate_records([record()])
    assert row["exact_std"] is None and row["exact_mean"] == 70
    with pytest.raises(ValueError, match="no C7"):
        tables.make_tables(tmp_path / "absent", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_literature_cli_preserves_gaps_and_never_reads_runs(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "out"
    runs.mkdir()
    # Invalid C7 input must not be read by the literature-only mode.
    (runs / "invalid.json").write_text("not JSON")
    tables.main(["--literature", "--runs-dir", str(runs), "--out-dir", str(out)])
    with (out / "literature_squad11.csv").open(encoding="utf-8", newline="") as stream:
        rows = {r["model"]: r for r in csv.DictReader(stream)}
    assert rows["Logistic regression"]["em"] == ""
    assert rows["SpanBERT-base"]["em"] == ""
    assert rows["SpanBERT-base"]["f1"] != ""
    assert rows["DeBERTaV3-base"]["f1"] == ""
    assert all(r["url"].startswith("https://") and r["locator"] and r["caveat"]
               for r in rows.values())
    assert not (out / "run_summary.csv").exists()
    tex = (out / "literature_squad11.tex").read_text(encoding="utf-8")
    assert "Internal validation results are excluded" in tex
    assert " & None" not in tex and " & nan" not in tex
