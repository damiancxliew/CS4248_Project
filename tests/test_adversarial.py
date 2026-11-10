import json

import pytest

from qa.eval import adversarial as adv

ORIG = "Paris is in France. It is big."
DISTRACT = " Berlin is in Spain."
ORIG_B = "Rome is old"  # no final period: the splitter merges the distractor into it
DISTRACT_B = " Oslo is old."


def _dataset():
    def qa(qid, answer, start):
        return {"id": qid, "question": "q?", "answers": [{"text": answer, "answer_start": start}]}

    return {"version": "1.1", "data": [{"title": "t", "paragraphs": [
        {"context": ORIG, "qas": [qa("a", "France", 13)]},
        {"context": ORIG + DISTRACT, "qas": [qa("a-high-conf-turk0", "France", 13)]},
        {"context": ORIG + DISTRACT, "qas": [qa("a-high-conf-turk1", "France", 13)]},
        {"context": ORIG_B, "qas": [qa("b", "Rome", 0)]},
        {"context": ORIG_B + DISTRACT_B, "qas": [qa("b-high-conf-turk0", "Rome", 0)]},
        {"context": "Cats purr.", "qas": [qa("c", "Cats", 0)]},  # an original with no variant
    ]}]}


def _evidence():
    n = len(ORIG)
    two = [[0, 19], [20, n]]
    three = two + [[n + 1, n + len(DISTRACT)]]
    merged = [[0, len(ORIG_B) + len(DISTRACT_B)]]
    return [
        {"id": "a", "sentences": two, "gold_sent": 0},
        {"id": "a-high-conf-turk0", "sentences": three, "gold_sent": 0},
        {"id": "a-high-conf-turk1", "sentences": three, "gold_sent": 0},
        {"id": "b", "sentences": [[0, len(ORIG_B)]], "gold_sent": 0},
        {"id": "b-high-conf-turk0", "sentences": merged, "gold_sent": 0},
        {"id": "c", "sentences": [[0, 10]], "gold_sent": 0},
    ]


@pytest.fixture
def split(tmp_path):
    (tmp_path / "addsent.json").write_text(json.dumps(_dataset()))
    (tmp_path / "evidence_addsent.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _evidence()))
    return adv.Split("addsent", tmp_path)


def _top(text, sent):
    return [{"text": text, "start_logit": 0.0, "end_logit": 0.0, "sent_idx": sent, "sent_logprob": None, "score": 0.0}]


def test_base_id():
    assert adv.base_id("56be4db0acb8001400a502ee-high-conf-turk3") == "56be4db0acb8001400a502ee"
    assert adv.base_id("56be4db0acb8001400a502ee") == "56be4db0acb8001400a502ee"


def test_distractor_located_by_offset_including_merged_sentence(split):
    start = len(ORIG) + 1
    assert split.spans["a-high-conf-turk0"] == (start, len(ORIG + DISTRACT))
    assert split.adv_sents["a-high-conf-turk0"] == (2, False)
    assert split.adv_sents["b-high-conf-turk0"] == (0, True)  # merged with "Rome is old"
    assert "a" not in split.spans and "c" not in split.spans


def test_context_not_prefixed_by_original_is_rejected():
    with pytest.raises(ValueError, match="appended"):
        adv.adversarial_spans({"a": "One.", "a-high-conf-turk0": "Two. One."})


def test_worst_case_takes_group_minimum_and_keeps_lone_originals():
    scores = {"a": 1.0, "a-x": 0.0, "a-y": 1.0, "c": 1.0}
    worst, clean, n = adv.worst_case(scores, list(scores))
    assert (worst, clean, n) == (0.5, 1.0, 2)


def test_group_worst_scores_for_significance():
    scores = {"a": 1.0, "a-x": 0.25, "c": 0.5}
    assert adv.group_worst(scores, list(scores)) == {"a": 0.25, "c": 0.5}


def test_score_run_end_to_end(split):
    predictions = {"a": "France", "a-high-conf-turk0": "Spain", "a-high-conf-turk1": "France",
                   "b": "Rome", "b-high-conf-turk0": "Oslo", "c": "Cats"}
    nbest = {"a": _top("France", 0), "a-high-conf-turk0": _top("Spain", 2),
             "a-high-conf-turk1": _top("France", 0), "b": _top("Rome", 0),
             "b-high-conf-turk0": _top("Oslo", 0), "c": _top("Cats", 0)}
    ranking = {"a-high-conf-turk0": [[2, 0.0], [0, -1.0], [1, -2.0]],
               "a-high-conf-turk1": [[0, 0.0], [2, -1.0], [1, -2.0]],
               "b-high-conf-turk0": [[0, 0.0]]}
    out = adv.score_run(split, predictions, nbest, ranking)
    # groups a (worst 0), b (worst 0), c (1): worst case 1/3, clean 1
    assert out["exact"] == pytest.approx(100 / 3)
    assert out["clean_exact"] == pytest.approx(100.0)
    assert out["drop_exact"] == pytest.approx(200 / 3)
    # (a): turk0 answers from the distractor; merged b answers "Oslo", in the appended text
    assert out["answer_in_adv"] == pytest.approx(100 * 2 / 3)
    # (b): turk0 ranks the distractor first; the merged sentence is rank 1 for b
    assert out["adv_at_rank1"] == pytest.approx(100 * 2 / 3)
    assert (out["n_groups"], out["n_adv"], out["n_merged"]) == (3, 3, 1)


def test_merged_sentence_answer_from_original_text_is_not_counted(split):
    nbest = {q: _top("Rome", 0) for q in split.contexts}
    hits = adv.answer_in_adversarial(nbest, split.contexts, split.spans, split.adv_sents)
    assert hits == 0.0


def test_missing_prediction_fails(split):
    with pytest.raises(ValueError, match="no prediction"):
        adv.question_scores(split.data, {"a": "France"})


def test_ranking_only_run_leaves_answer_metrics_empty(split):
    ranking = {q: [[0, 0.0]] for q in split.spans}
    out = adv.score_run(split, ranking=ranking)
    assert out["f1"] is None and out["answer_in_adv"] is None
    assert out["adv_at_rank1"] == pytest.approx(100 / 3)  # only b's merged sentence 0 is the distractor


def test_table_and_tex(split, tmp_path):
    run = tmp_path / "adv" / "r3_deberta_span_s42"
    run.mkdir(parents=True)
    preds = {q: "France" if q.startswith("a") else ("Rome" if q.startswith("b") else "Cats") for q in split.contexts}
    (run / "addsent.predictions.json").write_text(json.dumps(preds))
    (run / "addsent.nbest.json").write_text(json.dumps({q: _top(t, 0) for q, t in preds.items()}))
    systems = (("DeBERTa span (R3)", "r3_deberta_span", "", False),)
    rows = adv.table_rows(split, tmp_path / "adv", tmp_path / "bm25", systems)
    assert len(rows) == 1 and rows[0]["seeds"] == "42" and rows[0]["f1_mean"] == pytest.approx(100.0)
    csv_path, tex_path = adv.write_table(rows, "addsent", tmp_path / "tables")
    assert "DeBERTa span (R3) & 100.0 & 100.0 & 0.0 & 0.0 & --" in tex_path.read_text()
    assert csv_path.read_text().startswith("system,run_id,seeds,n_adv,")


def test_figure_from_tables(split, tmp_path):
    run = tmp_path / "adv" / "r3_deberta_span_s42"
    run.mkdir(parents=True)
    preds = {q: "Cats" for q in split.contexts}
    (run / "addsent.predictions.json").write_text(json.dumps(preds))
    (run / "addsent.nbest.json").write_text(json.dumps({q: _top("Cats", 0) for q in preds}))
    rows = adv.table_rows(split, tmp_path / "adv", tmp_path / "bm25", (("DeBERTa span (R3)", "r3_deberta_span", "", False),))
    csv_path, _ = adv.write_table(rows, "addsent", tmp_path / "tables")
    out = adv.plot_drops({"addsent": adv.read_table(csv_path)}, tmp_path / "fig" / "drop.pdf")
    assert out.is_file() and out.with_suffix(".png").is_file()


def test_unknown_split_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        adv.Split("val", tmp_path)
