"""Official SQuAD scoring and explicit C7 export, without model dependencies.

CLI: python -m qa.eval.run_eval --data-file data/processed/val.json
  --predictions outputs/run/val.predictions.json --metadata outputs/run/run.json
  --split val --run-id run_val --out-file results/runs/run_val.json

Metadata is a training C7 record (or its identity/config/provenance fields).
Training metadata is preserved; scores, split and the explicit run ID are replaced.
Existing records are never overwritten. Final test use is reserved for run_all.sh
with --allow-test and frozen checkpoints, as for qa.predict.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path

OFFICIAL_EVAL = Path(__file__).with_name("evaluate-v2.0.py")
C7_KEYS = ("run_id", "system", "variant", "seed", "split", "exact", "f1",
           "recall_at_1", "recall_at_3", "mrr", "n", "params_M",
           "train_gpu_hours", "gpu", "infer_qps", "git_sha", "config")
METRICS = ("exact", "f1", "recall_at_1", "recall_at_3", "mrr",
           "params_M", "train_gpu_hours", "infer_qps")


def _official_module():
    spec = importlib.util.spec_from_file_location("squad_official_eval", OFFICIAL_EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_data_path(squad_file, allow_test):
    if Path(squad_file).resolve().name.lower() == "dev-v1.1.json" and not allow_test:
        raise ValueError("test set is reserved for final checkpoints via run_all.sh")


def official_scores(squad_file, predictions, *, allow_test=False):
    """Return exact/F1 (0–100) and n, compatible with qa.train.official_scores.

    Calls the official raw scoring and aggregation functions directly. As in
    the training engine, incomplete predictions fail instead of shrinking n.
    Extra prediction IDs are ignored by the official evaluator. No v2 no-answer
    probability file or threshold tuning is applied to this v1.1/C6 interface.
    """
    _check_data_path(squad_file, allow_test)
    if not isinstance(predictions, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in predictions.items()
    ):
        raise ValueError("C6 predictions must be {question_id: answer_text} strings")
    dataset = json.loads(Path(squad_file).read_text(encoding="utf-8"))["data"]
    ids = [qa["id"] for a in dataset for p in a["paragraphs"] for qa in p["qas"]]
    if not ids or any(not isinstance(qid, str) for qid in ids) or len(set(ids)) != len(ids):
        raise ValueError("dataset must contain nonempty, unique string question IDs")
    missing = set(ids) - predictions.keys()
    if missing:
        raise ValueError(f"{len(missing)} questions have no prediction, e.g. {sorted(missing)[:3]}")
    official = _official_module()
    exact, f1 = official.get_raw_scores(dataset, predictions)
    result = official.make_eval_dict(exact, f1)
    return {"exact": result["exact"], "f1": result["f1"], "n": result["total"]}


def validate_record(record):
    """Validate the frozen C7 shape; unknown measurements remain null, never zero."""
    if not isinstance(record, dict) or set(record) != set(C7_KEYS):
        raise ValueError("record must contain exactly the C7 fields")
    for key in ("run_id", "system", "variant", "split"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if record["seed"] is not None and type(record["seed"]) is not int:
        raise ValueError("seed must be an integer or null (unseeded ranking baseline)")
    if type(record["n"]) is not int or record["n"] <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(record["config"], dict):
        raise ValueError("config must be the resolved configuration object")
    for key in ("git_sha", "gpu"):
        if record[key] is not None and not isinstance(record[key], str):
            raise ValueError(f"{key} must be a string or null")
    for key in METRICS:
        value = record[key]
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite, nonnegative or null")
        if key in ("exact", "f1") and value > 100:
            raise ValueError(f"{key} must be on the official 0–100 scale")
    # Ranking scales are intentionally preserved as supplied by J6; C7 does not
    # freeze whether these are fractions or percentages. Never rescale them here.
    json.dumps(record, allow_nan=False)


def run_eval(squad_file, predictions_file, metadata, split, run_id, out_file,
             *, allow_test=False):
    """Evaluate C6, preserve caller provenance, and write a new C7 record."""
    if split == "test" and not allow_test:
        raise ValueError("test set is reserved for final checkpoints via run_all.sh")
    if Path(out_file).exists():
        raise FileExistsError(out_file)
    for key in ("system", "variant", "seed", "git_sha", "config"):
        if key not in metadata:
            raise ValueError(f"metadata requires {key}")
    unknown = set(metadata) - set(C7_KEYS)
    if unknown:
        raise ValueError(f"unknown metadata fields: {sorted(unknown)}")
    predictions = json.loads(Path(predictions_file).read_text(encoding="utf-8"))
    scores = official_scores(squad_file, predictions, allow_test=allow_test)
    record = {key: copy.deepcopy(metadata.get(key)) for key in C7_KEYS}
    record.update(scores, run_id=run_id, split=split)
    # Ranking metrics describe a particular prediction/ranking run, not training.
    # Supply them only in metadata for this evaluation (never infer them).
    validate_record(record)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for arg in ("data-file", "predictions", "metadata", "split", "run-id", "out-file"):
        parser.add_argument(f"--{arg}", required=True)
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    result = run_eval(args.data_file, args.predictions, metadata, args.split,
                      args.run_id, args.out_file, allow_test=args.allow_test)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
