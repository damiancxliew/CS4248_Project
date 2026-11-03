"""sentence_ranking (G2, C8): slots match C4, passage score = best window log-prob, full sorted ranking.

C3 records and window arrays are hand-built here; no model or tokenizer.
"""
import numpy as np
import pytest
from datasets import Dataset

from qa.data.sentence_features import add_sentence_fields
from qa.decoding.sentence_ranking import passage_ranking, window_logprobs, window_slots

RECORD = {"id": "q1", "sentences": [[0, 10], [11, 20], [21, 30], [31, 40]], "gold_sent": 2,
          "neg_candidates": [0, 1, 3], "lex_neg": 1, "rand_neg": 3}
EVIDENCE = {"q1": RECORD}
SPECIALS = 2
SPANS_A = [(0, 4), (5, 10), (11, 20), (21, 30)]  # sentences 0, 1, 2
SPANS_B = [(21, 25), (26, 30), (31, 40)]         # sentences 2, 3: sentence 2 is in the overlap


def offsets(*windows, width=8):
    """[N, width, 2]: SPECIALS non-context tokens, one context token per span, then -1 padding."""
    out = np.full((len(windows), width, 2), -1)
    for i, spans in enumerate(windows):
        for t, span in enumerate(spans):
            out[i, SPECIALS + t] = span
    return out


def scores(ranking):
    return {sent: score for sent, score in ranking}


def test_window_slots_are_the_c4_slots():
    rows = [{"example_id": "q1", "input_ids": [0] * (SPECIALS + len(spans)),
             "offset_mapping": [None] * SPECIALS + [list(span) for span in spans]}
            for spans in (SPANS_A, SPANS_B)]
    c4 = add_sentence_fields(Dataset.from_list(rows), EVIDENCE, "model")
    offs = offsets(SPANS_A, SPANS_B)
    slots = [window_slots(offs[i], RECORD["sentences"]).tolist() for i in range(2)]
    assert slots == [[0, 1, 2], [2, 3]]
    for i in range(2):
        assert max(c4[i]["sent_ids"]) + 1 == len(slots[i])
        assert slots[i][c4[i]["sent_label"]] == RECORD["gold_sent"]


def test_passage_score_is_the_best_window_logprob():
    logits = np.array([[2.0, 0.0, 1.0, 9.9], [4.0, 0.0, 9.9, 9.9]])  # 9.9 = slots past the window
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
    ranking = passage_ranking(["q1", "q1"], offsets(SPANS_A, SPANS_B), logits, mask, EVIDENCE)
    a, b = window_logprobs([2.0, 0.0, 1.0]), window_logprobs([4.0, 0.0])
    assert b[0] > a[2]  # sentence 2 scores best in the second window
    assert scores(ranking["q1"]) == pytest.approx({0: a[0], 1: a[1], 2: b[0], 3: b[1]})


def test_ranking_is_sorted_complete_and_ties_go_to_the_earlier_sentence():
    ranking = passage_ranking(["q1"], offsets(SPANS_A), np.array([[1.0, 1.0, 3.0]]),
                              np.array([[1, 1, 1]]), EVIDENCE)["q1"]
    assert [sent for sent, _ in ranking] == [2, 0, 1, 3]
    values = [score for _, score in ranking]
    assert values == sorted(values, reverse=True)
    assert ranking[-1][1] == min(values[:3])  # sentence 3 is in no window: the passage's lowest


def test_plan_rule_window_with_fewer_sentences_gives_an_easier_softmax():
    # Kept on purpose (TEAM_TASKS §2.3 step 7): equal raw scores, but window B shows only 2 sentences.
    logits = np.zeros((2, 3))
    mask = np.array([[1, 1, 1], [1, 1, 0]])
    ranking = scores(passage_ranking(["q1", "q1"], offsets(SPANS_A, SPANS_B), logits, mask, EVIDENCE)["q1"])
    assert ranking[0] == pytest.approx(np.log(1 / 3))
    assert ranking[2] == pytest.approx(np.log(1 / 2))


def test_window_showing_no_sentence_adds_nothing():
    ranking = passage_ranking(["q1"], offsets([]), np.zeros((1, 1)), np.zeros((1, 1)), EVIDENCE)
    assert ranking == {"q1": [[0, 0.0], [1, 0.0], [2, 0.0], [3, 0.0]]}


def test_evidence_that_does_not_match_the_scored_sentences_is_an_error():
    with pytest.raises(ValueError, match="C3 file"):
        passage_ranking(["q1"], offsets(SPANS_A), np.zeros((1, 4)), np.array([[1, 1, 0, 0]]), EVIDENCE)
