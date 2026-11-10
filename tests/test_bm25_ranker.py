"""bm25_ranker: C3 bm25 scores -> C8 ranking, on hand-built C3 records."""
import json

import pytest

from qa.baselines.bm25_ranker import main, rank_sentences, rank_split
from qa.decoding.sequential_decoder import recall_at_k

# q1: gold sentence ranked first. q2: gold ranked second. q3: gold ranked last. q4: one sentence.
EVIDENCE = {
    qid: {"id": qid, "gold_sent": gold, "bm25": bm25, "sentences": [[0, 5]] * len(bm25)}
    for qid, gold, bm25 in [
        ("q1", 1, [3.6, 4.8, 1.3, 0.0, 3.2]),
        ("q2", 0, [2.0, 5.0, 1.0]),
        ("q3", 2, [1.0, 2.0, 0.5, 3.0]),
        ("q4", 0, [0.0]),
    ]
}


def test_rank_sentences_sorts_best_first():
    assert rank_sentences([3.6, 4.8, 1.3, 0.0, 3.2]) == [
        [1, 4.8], [0, 3.6], [4, 3.2], [2, 1.3], [3, 0.0],
    ]


def test_ties_go_to_the_earlier_sentence():
    # Questions sharing no word with the passage score 0 everywhere: passage order is kept.
    assert [i for i, _ in rank_sentences([0.0, 0.0, 0.0])] == [0, 1, 2]
    assert [i for i, _ in rank_sentences([1.0, 2.0, 1.0, 2.0])] == [1, 3, 0, 2]


def test_ranking_covers_every_sentence_once_and_keeps_raw_scores():
    for qid, ranking in rank_split(EVIDENCE).items():
        bm25 = EVIDENCE[qid]["bm25"]
        assert sorted(i for i, _ in ranking) == list(range(len(bm25)))
        assert all(score == bm25[i] for i, score in ranking)
        assert [s for _, s in ranking] == sorted(bm25, reverse=True)


def test_recall_at_k_on_hand_built_rankings():
    ranking = rank_split(EVIDENCE)
    assert recall_at_k(ranking, EVIDENCE, 1) == 2 / 4   # q1, q4
    assert recall_at_k(ranking, EVIDENCE, 3) == 3 / 4   # + q2; q3's gold is 4th


def _write_evidence(path, records):
    path.write_text("".join(json.dumps(rec) + "\n" for rec in records.values()))


def test_cli_writes_c8_file_and_prints_recall(tmp_path, capsys):
    _write_evidence(tmp_path / "evidence_val.jsonl", EVIDENCE)
    main(["--split", "val", "--evidence-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])

    written = json.loads((tmp_path / "out" / "r1_bm25" / "val.sent_rank.json").read_text())
    assert written == rank_split(EVIDENCE)
    report = json.loads(capsys.readouterr().out)
    assert report["n"] == 4
    assert report["recall_at_1"] == 0.5
    assert report["recall_at_3"] == 0.75


def test_cli_refuses_test_split_without_flag(tmp_path):
    _write_evidence(tmp_path / "evidence_test.jsonl", EVIDENCE)
    with pytest.raises(SystemExit, match="allow-test"):
        main(["--split", "test", "--evidence-dir", str(tmp_path), "--out-dir", str(tmp_path)])
    assert not (tmp_path / "r1_bm25").exists()
