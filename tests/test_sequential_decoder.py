"""Tests for sequential_decoder, run against mocks of the C3/C6/C8 contracts.

The mocks live here only. The `k = all` equality test runs against Gokul's real
evidence_decoder.py (needs torch, so it is skipped where torch is missing).
"""
import json

import numpy as np
import pytest

from qa.decoding import sequential_decoder as sd

# q1: gold sentence 0, ranker puts it first, but a higher-logit distractor sits in sentence 1.
# q2: gold sentence 2, ranker puts it last -> ranker error for k < 3.
# q3: gold sentence 1, ranked second, span correct once kept.
NBEST = {
    "q1": [
        {"text": "wrong one", "start_logit": 5.0, "end_logit": 5.0, "sent_idx": 1},
        {"text": "Paris", "start_logit": 4.0, "end_logit": 4.0, "sent_idx": 0},
    ],
    "q2": [
        {"text": "1999", "start_logit": 6.0, "end_logit": 6.0, "sent_idx": 0},
        {"text": "in 2004", "start_logit": 3.0, "end_logit": 3.0, "sent_idx": 2},
    ],
    "q3": [
        {"text": "blue", "start_logit": 2.0, "end_logit": 2.0, "sent_idx": 1},
        {"text": "red", "start_logit": 1.0, "end_logit": 1.0, "sent_idx": 0},
    ],
}
SENT_RANK = {
    "q1": [[0, -0.1], [1, -1.0], [2, -3.0]],
    "q2": [[0, -0.1], [1, -1.0], [2, -3.0]],
    "q3": [[0, -0.2], [1, -0.5], [2, -2.0]],
}
EVIDENCE = {"q1": {"gold_sent": 0}, "q2": {"gold_sent": 2}, "q3": {"gold_sent": 1}}
GOLD = {"q1": ["Paris"], "q2": ["2004"], "q3": ["blue"]}


def test_top_k_sentences():
    assert sd.top_k_sentences(SENT_RANK["q1"], 2) == {0, 1}
    assert sd.top_k_sentences(SENT_RANK["q1"], sd.K_ALL) == {0, 1, 2}
    assert sd.top_k_sentences(SENT_RANK["q1"], 10) == {0, 1, 2}


def test_k1_restricts_to_top_sentence():
    assert sd.decode(NBEST, SENT_RANK, 1) == {"q1": "Paris", "q2": "1999", "q3": "red"}


def test_k_all_is_plain_best_span():
    assert sd.decode(NBEST, SENT_RANK, sd.K_ALL) == {"q1": "wrong one", "q2": "1999", "q3": "blue"}


def test_empty_when_no_candidate_in_kept_sentences():
    nbest = {"q": [{"text": "x", "start_logit": 1, "end_logit": 1, "sent_idx": 2}]}
    assert sd.decode(nbest, {"q": [[0, 0.0], [2, -1.0]]}, 1) == {"q": ""}


def test_recall_at_k_ceiling_is_monotone():
    r = [sd.recall_at_k(SENT_RANK, EVIDENCE, k) for k in (1, 2, 3, sd.K_ALL)]
    assert r == pytest.approx([1 / 3, 1.0 * 2 / 3, 1.0, 1.0])
    assert r == sorted(r)


def test_error_split_partitions_questions():
    preds = sd.decode(NBEST, SENT_RANK, 1)
    s = sd.error_split(preds, SENT_RANK, EVIDENCE, GOLD, 1)
    # q1 correct, q2 ranker error (gold sentence 2 not in top-1), q3 ranker error (gold sentence 1 not in top-1)
    assert (s["correct"], s["ranker_error"], s["reader_error"]) == (1, 2, 0)
    assert s["correct"] + s["ranker_error"] + s["reader_error"] == s["n"]


def test_reader_error_when_gold_sentence_kept_but_span_wrong():
    preds = sd.decode(NBEST, SENT_RANK, sd.K_ALL)
    s = sd.error_split(preds, SENT_RANK, EVIDENCE, GOLD, sd.K_ALL)
    # q1 picks the distractor although gold sentence 0 is kept; q2 likewise
    assert (s["correct"], s["ranker_error"], s["reader_error"]) == (1, 0, 2)


def test_sweep_covers_every_k_and_keeps_split_consistent():
    res = sd.sweep(NBEST, SENT_RANK, EVIDENCE, GOLD)
    assert set(res) == {1, 2, 3, sd.K_ALL}
    for r in res.values():
        assert r["correct"] + r["ranker_error"] + r["reader_error"] == r["n"]


def test_exact_can_exceed_recall_when_answer_text_repeats_in_another_sentence():
    # Gold evidence is sentence 2, but the same answer text "2004" also occurs in sentence 0,
    # which the ranker puts first. At k=1 the span is right by text (EM = 1) although the gold
    # sentence is not kept (recall@1 = 0), so exact > recall_at_k. EM is text-based on purpose.
    nbest = {"q": [{"text": "2004", "start_logit": 1.0, "end_logit": 1.0, "sent_idx": 0}]}
    rank = {"q": [[0, -0.1], [1, -1.0], [2, -2.0]]}
    res = sd.sweep(nbest, rank, {"q": {"gold_sent": 2}}, {"q": ["2004"]}, ks=(1,))
    assert res[1]["recall_at_k"] == 0.0
    assert res[1]["exact"] == 1.0
    assert res[1]["ranker_error"] == 0


def test_plot_recall_ceiling_writes_png(tmp_path):
    res = sd.sweep(NBEST, SENT_RANK, EVIDENCE, GOLD)
    path = tmp_path / "ceiling.png"
    sd.plot_recall_ceiling({"ranker A": res, "ranker B": res}, path)
    assert path.read_bytes().startswith(b"\x89PNG")


def test_candidate_outside_every_sentence_is_kept_only_by_k_all():
    nbest = {"q": [
        {"text": "stray", "start_logit": 9.0, "end_logit": 9.0, "sent_idx": None},
        {"text": "in 0", "start_logit": 1.0, "end_logit": 1.0, "sent_idx": 0},
    ]}
    rank = {"q": [[0, -0.1], [1, -1.0]]}
    assert sd.decode(nbest, rank, 1) == {"q": "in 0"}
    assert sd.decode(nbest, rank, sd.K_ALL) == {"q": "stray"}


def test_k_all_equals_the_joint_decoder_at_lambda_zero():
    pytest.importorskip("torch")  # evidence_decoder imports qa.predict
    from test_evidence_decoder import sample
    from qa.decoding.evidence_decoder import decode_joint

    contexts, evidence, example_ids, offsets = sample()
    rng = np.random.default_rng(0)
    start, end = rng.normal(size=(2, *offsets.shape[:2]))
    ranking = {qid: [[s, -0.5 * s] for s in range(len(rec["sentences"]))] for qid, rec in evidence.items()}
    preds, nbest = decode_joint(contexts, example_ids, offsets, start, end, ranking, evidence, lam=0.0)
    assert sd.decode(nbest, ranking, sd.K_ALL) == preds
    # k = 1 keeps only the top-ranked sentence's candidates, and reads the best of those
    top1 = sd.decode(nbest, ranking, 1)
    for qid, text in top1.items():
        kept = [c for c in nbest[qid] if c["sent_idx"] == ranking[qid][0][0]]
        assert text == (max(kept, key=lambda c: c["start_logit"] + c["end_logit"])["text"] if kept else "")


def _write_cli_inputs(tmp_path):
    squad = {"data": [{"paragraphs": [{"qas": [
        {"id": q, "question": "what?", "answers": [{"text": g[0]}]} for q, g in GOLD.items()]}]}]}
    paths = {
        "nbest": tmp_path / "val.joint_lam0.25.nbest.json",
        "sent_rank": tmp_path / "val.sent_rank.json",
        "evidence": tmp_path / "evidence_val.jsonl",
        "data_file": tmp_path / "val.json",
    }
    paths["nbest"].write_text(json.dumps(NBEST))
    paths["sent_rank"].write_text(json.dumps(SENT_RANK))
    paths["evidence"].write_text("\n".join(json.dumps({"id": q, **ev}) for q, ev in EVIDENCE.items()))
    paths["data_file"].write_text(json.dumps(squad))
    return [arg for key, path in paths.items() for arg in (f"--{key.replace('_', '-')}", str(path))]


def test_cli_sweep_and_single_k(tmp_path):
    args = _write_cli_inputs(tmp_path)
    sd.main(args)
    sweep = json.loads((tmp_path / "val.sequential_sweep.json").read_text())
    assert set(sweep) == {"1", "2", "3", "all"}
    assert sweep["all"]["recall_at_k"] == 1.0
    sd.main(args + ["--k", "1"])
    preds = json.loads((tmp_path / "val.sequential_k1.predictions.json").read_text())
    assert preds == sd.decode(NBEST, SENT_RANK, 1)


def test_cli_refuses_test_split(tmp_path):
    args = _write_cli_inputs(tmp_path)
    args[args.index("--data-file") + 1] = str(tmp_path / "dev-v1.1.json")
    with pytest.raises(SystemExit):
        sd.main(args)
