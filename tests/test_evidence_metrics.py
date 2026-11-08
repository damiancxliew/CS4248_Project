"""evidence_metrics: Recall@1, Recall@3 and MRR on hand-built C3 records and C8 / C6 inputs."""
import csv
import json

import pytest

from qa.eval import evidence_metrics as em

# Gold ranked 1st (q1), 2nd (q2), 4th (q3), 1st in a one-sentence passage (q4).
EVIDENCE = {
    "q1": {"id": "q1", "gold_sent": 1, "sentences": [[0, 5]] * 3},
    "q2": {"id": "q2", "gold_sent": 0, "sentences": [[0, 5]] * 3},
    "q3": {"id": "q3", "gold_sent": 2, "sentences": [[0, 5]] * 4},
    "q4": {"id": "q4", "gold_sent": 0, "sentences": [[0, 5]]},
}
RANKING = {
    "q1": [[1, 3.0], [0, 2.0], [2, 1.0]],
    "q2": [[2, 3.0], [0, 2.0], [1, 1.0]],
    "q3": [[0, 4.0], [1, 3.0], [3, 2.0], [2, 1.0]],
    "q4": [[0, 0.0]],
}


def test_gold_ranks():
    assert em.gold_ranks(RANKING, EVIDENCE) == {"q1": 1, "q2": 2, "q3": 4, "q4": 1}


def test_summarize_matches_hand_computed_metrics():
    summary = em.summarize(em.gold_ranks(RANKING, EVIDENCE))
    assert summary["n"] == 4
    assert summary["recall_at_1"] == 2 / 4                       # q1, q4
    assert summary["recall_at_3"] == 3 / 4                       # + q2
    assert summary["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4 + 1) / 4)


def test_per_question_recall_at_1_is_binary_per_qid():
    scores = em.per_question(em.gold_ranks(RANKING, EVIDENCE))
    assert scores["recall_at_1"] == {"q1": 1, "q2": 0, "q3": 0, "q4": 1}


def test_recall_matches_sequential_decoder_definition():
    from qa.decoding.sequential_decoder import recall_at_k
    summary = em.summarize(em.gold_ranks(RANKING, EVIDENCE))
    assert summary["recall_at_1"] == recall_at_k(RANKING, EVIDENCE, 1)
    assert summary["recall_at_3"] == recall_at_k(RANKING, EVIDENCE, 3)


def test_refuses_ranking_for_another_split():
    with pytest.raises(ValueError, match="1 question"):
        em.gold_ranks({k: v for k, v in RANKING.items() if k != "q4"}, EVIDENCE)
    with pytest.raises(ValueError, match="1 extra"):
        em.gold_ranks({**RANKING, "q5": [[0, 0.0]]}, EVIDENCE)


def test_refuses_ranking_that_drops_or_repeats_a_sentence():
    with pytest.raises(ValueError, match="exactly once"):
        em.gold_ranks({**RANKING, "q1": [[1, 3.0], [0, 2.0]]}, EVIDENCE)
    with pytest.raises(ValueError, match="exactly once"):
        em.gold_ranks({**RANKING, "q1": [[1, 3.0], [1, 2.0], [2, 1.0]]}, EVIDENCE)


def _span(sent_idx, start, end):
    return {"text": "x", "start_logit": start, "end_logit": end, "sent_idx": sent_idx,
            "sent_logprob": None, "score": start + end}


def test_span_implied_ranking_orders_sentences_by_their_best_span():
    evidence = {"q1": {"id": "q1", "gold_sent": 2, "sentences": [[0, 5]] * 4}}
    nbest = {"q1": [_span(2, 3.0, 3.0), _span(0, 2.0, 2.0), _span(2, 1.0, 1.0), _span(None, 9.0, 9.0)]}
    ranking = em.span_implied_ranking(nbest, evidence)
    # Sentence 2 holds the top span (6.0); 0 next (4.0); 1 and 3 have no span, passage order.
    # The null-sentence span (18.0) is skipped.
    assert [s for s, _ in ranking["q1"]] == [2, 0, 1, 3]
    assert em.gold_ranks(ranking, evidence) == {"q1": 1}


def test_span_implied_ranking_needs_sent_idx():
    evidence = {"q1": {"id": "q1", "gold_sent": 0, "sentences": [[0, 5]] * 2}}
    with pytest.raises(ValueError, match="--evidence"):
        em.span_implied_ranking({"q1": [_span(None, 1.0, 1.0)]}, evidence)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write_evidence(dir_, split, records):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"evidence_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records.values()))


def test_score_cli_prints_metrics_and_writes_significance_input(tmp_path, capsys):
    _write_evidence(tmp_path, "val", EVIDENCE)
    _write(tmp_path / "rank.json", RANKING)
    em.main(["score", "--split", "val", "--evidence-dir", str(tmp_path),
             "--ranking", str(tmp_path / "rank.json"), "--scores-out", str(tmp_path / "r1.json")])
    report = json.loads(capsys.readouterr().out)
    assert (report["n"], report["recall_at_1"], report["recall_at_3"]) == (4, 0.5, 0.75)

    from qa.eval.significance import compare_scores
    scores = json.loads((tmp_path / "r1.json").read_text())
    assert scores == {"q1": 1, "q2": 0, "q3": 0, "q4": 1}
    compare_scores(scores, scores, metric="recall_at_1", units="fraction", n_resamples=10,
                   n_randomisations=10)  # accepted as significance.py input


def test_table_averages_seeds_and_leaves_out_missing_systems(tmp_path, capsys):
    out = tmp_path / "outputs"
    _write(out / "r1_bm25" / "val.sent_rank.json", RANKING)                  # Recall@1 50%
    perfect = {qid: sorted(r, key=lambda x: x[0] != EVIDENCE[qid]["gold_sent"]) for qid, r in RANKING.items()}
    _write(out / "r4_deberta_evidence_s13" / "val.sent_rank.json", perfect)  # 100%
    _write(out / "r4_deberta_evidence_s42" / "val.sent_rank.json", RANKING)  # 50%

    rows = em.table_rows("val", EVIDENCE, out)
    assert [r["run_id"] for r in rows] == ["r1_bm25", "r4_deberta_evidence"]
    bm25, r4 = rows
    assert (bm25["seeds"], bm25["recall_at_1_mean"], bm25["recall_at_1_std"]) == ("", 50.0, None)
    assert r4["seeds"] == "13 42"
    assert r4["recall_at_1_mean"] == 75.0
    assert r4["recall_at_1_std"] == pytest.approx(35.3553, abs=1e-3)   # sample std of 100, 50
    assert "Span-implied (R3)" in capsys.readouterr().err

    csv_path, tex_path = em.write_table(rows, "val", tmp_path / "tables")
    with csv_path.open() as f:
        assert [r["system"] for r in csv.DictReader(f)] == ["BM25 (R1)", "Full model (R4)"]
    assert r"Full model (R4) & 75.00 $\pm$ 35.36" in tex_path.read_text()


def test_cli_refuses_test_split_without_flag(tmp_path):
    _write_evidence(tmp_path, "test", EVIDENCE)
    with pytest.raises(SystemExit, match="allow-test"):
        em.main(["table", "--split", "test", "--evidence-dir", str(tmp_path),
                 "--output-dir", str(tmp_path), "--table-dir", str(tmp_path / "tables")])
    assert not (tmp_path / "tables").exists()
