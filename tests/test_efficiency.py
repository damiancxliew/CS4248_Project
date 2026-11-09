"""D5 efficiency: timing stages on the tiny offline models, and the table from timings + C7 records."""
import json

import pytest

from damian_mocks import evidence_records, smoke_config, squad_examples
from qa.eval import efficiency
from qa.train import ROOT, load_config


def write_c3(squad_file, path):
    records = evidence_records(squad_examples(squad_file))
    for record in records.values():
        cands = [i for i in range(len(record["sentences"])) if i != record["gold_sent"]]
        record.update(neg_candidates=cands, lex_neg=cands[0] if cands else None,
                      rand_neg=cands[-1] if cands else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records.values()))


@pytest.fixture(scope="module")
def timings(tmp_path_factory):
    span_cfg = smoke_config(tmp_path_factory.mktemp("span"))
    root = tmp_path_factory.mktemp("evidence")
    evidence_cfg = smoke_config(root, model_class="qa.models.evidence_model:EvidenceModel",
                                neg_mode="model", alpha=0.3, beta=0.5, margin=1.0,
                                evidence_dir=str(root / "c3"))
    evidence_cfg.update(system="tiny_evidence", variant="model_mined")
    evidence_cfg["decode"]["lam"] = 0.5
    write_c3(evidence_cfg["data"]["val_file"], root / "c3" / "evidence_val.jsonl")
    systems = {"tiny_span": (span_cfg, ("span",)), "tiny_evidence": (evidence_cfg, efficiency.DECODERS)}
    return efficiency.measure(systems, repeats=2, device="cpu", log=lambda _: None)


def test_one_row_per_decoder_with_every_stage_timed(timings):
    rows = {(r["name"], r["decoder"]): r for r in timings["rows"]}
    assert set(rows) == {("tiny_span", "span"), ("tiny_evidence", "span"),
                         ("tiny_evidence", "joint"), ("tiny_evidence", "sequential")}
    for r in rows.values():
        assert r["n_questions"] == 45 and r["n_windows"] > 45  # contexts are split into windows
        assert all(len(v) == 2 for v in r["runs_s"].values())
        assert r["forward_s"] > 0 and r["decode_s"] > 0 and r["e2e_qps"] < r["model_qps"]
    assert rows["tiny_span", "span"]["segment_s"] == 0  # only the evidence model splits sentences
    assert rows["tiny_evidence", "joint"]["segment_s"] > 0
    assert (rows["tiny_evidence", "joint"]["lam"], rows["tiny_evidence", "sequential"]["k"]) == (0.5, 1)
    # The ranker head is Linear(hidden -> 1): 32 + 1 parameters on the tiny encoder.
    extra = rows["tiny_evidence", "span"]["params_M"] - rows["tiny_span", "span"]["params_M"]
    assert extra == pytest.approx(33e-6, abs=1e-3)
    assert timings["meta"]["gpu"] == "cpu" and timings["meta"]["split"] == "val"


def test_refuses_the_test_split():
    with pytest.raises(SystemExit, match="test"):
        efficiency.time_system("x", {}, ("span",), split="test")


def test_span_model_cannot_use_evidence_decoders(tmp_path):
    with pytest.raises(ValueError, match="evidence model"):
        efficiency.time_system("x", smoke_config(tmp_path), ("joint",), device="cpu")


def test_table_adds_training_hours_and_overheads(tmp_path):
    rows = [
        {"system": "deberta_span_only", "variant": "span_only", "decoder": "span", "params_M": 183.8,
         "segment_s": 0.0, "features_s": 10.0, "forward_s": 40.0, "decode_s": 9.34,
         "model_qps": 189.3, "e2e_qps": 157.3, "peak_mem_gb": 3.1, "n_questions": 9340},
        {"system": "deberta_evidence", "variant": "model_mined", "decoder": "joint", "params_M": 183.8,
         "segment_s": 5.0, "features_s": 12.0, "forward_s": 42.0, "decode_s": 18.68,
         "model_qps": 153.9, "e2e_qps": 117.0, "peak_mem_gb": 3.3, "n_questions": 9340},
    ]
    meta = {"split": "val", "gpu": "NVIDIA H100 NVL MIG 3g.47gb", "batch_size": 64, "repeats": 3}
    runs = tmp_path / "runs"
    runs.mkdir()
    for seed, hours in zip((13, 42, 87), (0.30, 0.32, 0.34)):
        (runs / f"r3_s{seed}.json").write_text(json.dumps(
            {"system": "deberta_span_only", "variant": "span_only", "split": "val",
             "train_gpu_hours": hours}))
    (runs / "other.json").write_text(json.dumps(
        {"system": "deberta_span_only", "variant": "span_only", "split": "test", "train_gpu_hours": 9}))

    span, joint = efficiency.table_rows({"meta": meta, "rows": rows}, runs)
    assert (span["train_gpu_hours_mean"], span["train_gpu_hours_std"], span["train_seeds"]) == (0.32, 0.02, 3)
    assert joint["train_gpu_hours_mean"] is None and joint["train_seeds"] == 0
    assert (span["forward_vs_base_pct"], joint["forward_vs_base_pct"]) == (0.0, 5.0)
    assert (span["decode_ms_per_q"], joint["decode_ms_per_q"]) == (1.0, 2.0)
    assert joint["label"] == "Evidence model, joint (R4)"

    csv_path, tex_path = efficiency.write_table([span, joint], meta, tmp_path / "tables")
    assert csv_path.read_text().splitlines()[0].startswith("label,system,")
    tex = tex_path.read_text()
    assert r"0.32 $\pm$ 0.02" in tex and "-- &" in tex and "MIG 3g.47gb" in tex


def test_cli_table_reads_the_timings_file(tmp_path):
    timings = {"meta": {"split": "val", "gpu": "cpu", "batch_size": 64, "repeats": 1},
               "rows": [{"system": "bert_span_only", "variant": "span_only", "decoder": "span",
                         "params_M": 109.5, "segment_s": 0.0, "features_s": 1.0, "forward_s": 2.0,
                         "decode_s": 1.0, "model_qps": 10.0, "e2e_qps": 7.5, "peak_mem_gb": None,
                         "n_questions": 30}]}
    path = tmp_path / "timings.json"
    path.write_text(json.dumps(timings))
    efficiency.main(["table", "--timings", str(path), "--runs-dir", str(tmp_path),
                     "--out-dir", str(tmp_path / "out")])
    assert (tmp_path / "out" / "efficiency.tex").exists()


@pytest.mark.parametrize("name", sorted(efficiency.SYSTEMS))
def test_registered_systems_match_their_configs(name):
    config, decoders = efficiency.SYSTEMS[name]
    if not (ROOT / config).exists():
        pytest.skip(f"{config} is not on this branch")
    cfg = load_config(ROOT / config)
    is_evidence = cfg["model"]["class"].endswith("EvidenceModel")
    assert bool(set(decoders) & set(efficiency.EVIDENCE_DECODERS)) == is_evidence
    for decoder in decoders:
        assert (cfg["system"], decoder) in efficiency.LABELS
