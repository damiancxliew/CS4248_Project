"""evidence_labels: sentence splitting and merging of splits that fall inside answers."""
import json
from pathlib import Path

import pytest

from qa.data.evidence_labels import (
    bm25_scores,
    build_evidence,
    gold_sentence,
    lexical_negative,
    merge_straddles,
    negative_candidates,
    random_negative,
    split_sentences,
)

TRAIN = Path(__file__).resolve().parents[1] / "data" / "raw" / "train-v1.1.json"


def test_split_sentences_gives_char_spans_without_surrounding_whitespace():
    context = "Mary appeared in Rome.  Bernadette saw Mary in Lourdes. It was 1858. "
    spans = split_sentences(context)
    assert [context[s:e] for s, e in spans] == [
        "Mary appeared in Rome.",
        "Bernadette saw Mary in Lourdes.",
        "It was 1858.",
    ]


def test_split_sentences_does_not_break_on_common_abbreviations():
    context = "Dr. Smith moved to St. Louis in 1850. He died there."
    spans = split_sentences(context)
    assert [context[s:e] for s, e in spans] == ["Dr. Smith moved to St. Louis in 1850.", "He died there."]


def test_merge_joins_the_sentences_an_answer_crosses():
    context = "He was born in St. Louis in 1850. He moved to Paris."
    spans = [(0, 17), (18, 33), (34, 52)]  # splitter broke after "St."
    answer = (15, 24)  # "St. Louis"
    assert context[slice(*answer)] == "St. Louis"
    merged, flags = merge_straddles(spans, [answer])
    assert merged == [(0, 33), (34, 52)]
    assert flags == [True, False]


def test_merge_leaves_spans_unchanged_when_no_answer_crosses():
    spans = [(0, 17), (18, 33), (34, 52)]
    merged, flags = merge_straddles(spans, [(3, 7), (40, 45)])
    assert merged == spans
    assert flags == [False, False, False]


def test_merge_uses_every_answer_in_the_context():
    spans = [(0, 10), (11, 20), (21, 30), (31, 40)]
    merged, flags = merge_straddles(spans, [(8, 13), (28, 33)])
    assert merged == [(0, 20), (21, 40)]
    assert flags == [True, True]


def test_merge_handles_an_answer_crossing_three_sentences():
    spans = [(0, 10), (11, 20), (21, 30), (31, 40)]
    merged, flags = merge_straddles(spans, [(5, 25)])
    assert merged == [(0, 30), (31, 40)]
    assert flags == [True, False]


LOURDES = ("Mary appeared in Rome. Bernadette saw Mary in Lourdes. "
           "Lourdes is in France. Bernadette later became a nun.")
LOURDES_SPANS = [(0, 22), (23, 54), (55, 76), (77, 107)]


def test_gold_sentence_is_the_one_containing_the_answer():
    start = LOURDES.index("Lourdes")
    assert gold_sentence(LOURDES_SPANS, start, start + len("Lourdes")) == 1


def test_gold_sentence_rejects_an_answer_that_crosses_sentences():
    with pytest.raises(ValueError):
        gold_sentence(LOURDES_SPANS, 40, 60)  # unmerged: starts in sentence 1, ends in sentence 2


def test_negative_candidates_exclude_gold_and_sentences_containing_the_answer():
    # Sentence 2 repeats "Lourdes", so it may be valid evidence: not a negative.
    assert negative_candidates(LOURDES, LOURDES_SPANS, 1, ["Lourdes"]) == [0, 3]


def test_negative_candidates_keep_sentences_that_only_share_words_with_the_answer():
    # "Bernadette" alone is not the answer "Bernadette saw Mary".
    assert negative_candidates(LOURDES, LOURDES_SPANS, 1, ["Bernadette saw Mary"]) == [0, 2, 3]


def test_negative_candidates_check_every_reference_answer():
    assert negative_candidates(LOURDES, LOURDES_SPANS, 1, ["Lourdes", "a nun"]) == [0]


def test_negative_candidates_are_empty_for_a_single_sentence_context():
    assert negative_candidates("Only one sentence.", [(0, 18)], 0, ["one"]) == []


def test_bm25_gives_one_score_per_sentence_and_ranks_the_matching_sentence_first():
    scores = bm25_scores(LOURDES, LOURDES_SPANS, "Where did Bernadette see Mary in Lourdes?")
    assert len(scores) == len(LOURDES_SPANS)
    assert max(range(len(scores)), key=scores.__getitem__) == 1


def test_bm25_separates_the_sentences_of_a_two_sentence_paragraph():
    # Okapi IDF is 0 for a word in 1 of 2 sentences and negative for a word in both,
    # so it cannot tell these apart; the non-negative IDF must.
    context = "Mary appeared in Rome. Bernadette saw Mary in Lourdes."
    scores = bm25_scores(context, [(0, 22), (23, 54)], "Where did Bernadette see Mary?")
    assert all(s >= 0 for s in scores)
    assert scores[1] > scores[0] + 0.5


def test_bm25_is_zero_for_a_sentence_sharing_no_word_with_the_question():
    scores = bm25_scores(LOURDES, LOURDES_SPANS, "Where is France?")
    assert scores[0] == 0.0 and scores[2] > 0.0


def test_bm25_matches_words_case_insensitively_ignoring_punctuation():
    context = "LOURDES, France. Rome."
    assert bm25_scores(context, [(0, 16), (17, 22)], "lourdes?")[0] > 0.0


def test_lexical_negative_is_the_highest_bm25_candidate_not_the_gold():
    # Gold (1) scores highest overall but is not a candidate.
    assert lexical_negative([0.4, 9.0, 2.5, 1.0], [0, 2, 3]) == 2


def test_lexical_negative_breaks_ties_by_earliest_sentence():
    assert lexical_negative([1.0, 9.0, 3.0, 3.0], [0, 2, 3]) == 2


def test_lexical_and_random_negatives_are_none_without_candidates():
    assert lexical_negative([5.0], []) is None
    assert random_negative("q1", []) is None


def test_random_negative_is_a_candidate_fixed_per_question():
    picks = [random_negative(f"q{i}", [0, 2, 3, 5]) for i in range(50)]
    assert set(picks) <= {0, 2, 3, 5}
    assert len(set(picks)) > 1  # varies across questions
    assert picks == [random_negative(f"q{i}", [0, 2, 3, 5]) for i in range(50)]  # reproducible


def _squad(paragraphs):
    """SQuAD-format dict; each paragraph is (context, [(qid, question, [answer texts]), ...])."""
    return {"data": [{"title": "T", "paragraphs": [
        {"context": context, "qas": [
            {"id": qid, "question": question,
             "answers": [{"text": t, "answer_start": context.index(t)} for t in texts]}
            for qid, question, texts in qas]}
        for context, qas in paragraphs]}]}


TINY = _squad([
    (LOURDES, [("q1", "Where did Bernadette see Mary?", ["Lourdes"]),
               ("q2", "What did Bernadette become?", ["a nun"])]),
    ("He was born in St. Louis in 1850. He moved to Paris.",
     [("q3", "Where was he born?", ["St. Louis"])]),
    ("Only one sentence here.", [("q4", "How many sentences?", ["one"])]),
])
C3_KEYS = {"id", "sentences", "gold_sent", "neg_candidates", "lex_neg", "rand_neg", "bm25", "straddle_merged"}


def test_build_evidence_writes_one_c3_record_per_question():
    records, _ = build_evidence(TINY)
    assert [r["id"] for r in records] == ["q1", "q2", "q3", "q4"]
    assert all(set(r) == C3_KEYS for r in records)
    q1 = records[0]
    assert q1["sentences"] == [list(s) for s in LOURDES_SPANS]
    assert (q1["gold_sent"], q1["neg_candidates"]) == (1, [0, 3])
    assert len(q1["bm25"]) == len(q1["sentences"])
    assert q1["lex_neg"] in q1["neg_candidates"] and q1["rand_neg"] in q1["neg_candidates"]


def test_build_evidence_gives_no_negatives_for_a_single_sentence_context():
    records, _ = build_evidence(TINY)
    assert records[2]["straddle_merged"] is False  # pysbd keeps "St. Louis" together
    q4 = records[3]
    assert (q4["sentences"], q4["gold_sent"]) == ([[0, 23]], 0)
    assert (q4["neg_candidates"], q4["lex_neg"], q4["rand_neg"]) == ([], None, None)


def test_build_evidence_flags_a_gold_sentence_formed_by_merging():
    context = "The games are in the form of .ipg files. They are zip archives."
    records, stats = build_evidence(_squad([(context, [("q", "What format?", [".ipg"])])]))
    gold = records[0]["sentences"][records[0]["gold_sent"]]
    assert ".ipg" in context[gold[0]:gold[1]]
    assert records[0]["straddle_merged"] is True
    assert stats["pct_answers_needing_merge"] == 100.0


def test_build_evidence_stats():
    _, stats = build_evidence(TINY)
    assert stats["contexts"] == 3 and stats["questions"] == 4
    assert stats["sentences_per_context"] == pytest.approx((4 + 2 + 1) / 3)
    assert stats["pct_contexts_single_sentence"] == pytest.approx(100 / 3)
    assert stats["pct_questions_no_candidate"] == pytest.approx(25.0)
    # q1 excludes sentence 2 ("Lourdes is in France."): 1 of 3 + 3 + 1 + 0 non-gold sentences.
    assert stats["pct_sentences_excluded"] == pytest.approx(100 / 7)


def test_every_train_answer_lies_in_its_gold_sentence_after_split_and_merge():
    paragraphs = [p for art in json.loads(TRAIN.read_text())["data"] for p in art["paragraphs"]][:300]
    for p in paragraphs:
        context = p["context"]
        answers = [(a["answer_start"], a["answer_start"] + len(a["text"]))
                   for qa in p["qas"] for a in qa["answers"]]
        spans, _ = merge_straddles(split_sentences(context), answers)
        for qa in p["qas"]:
            answer = qa["answers"][0]
            start, end = answer["answer_start"], answer["answer_start"] + len(answer["text"])
            gold = gold_sentence(spans, start, end)
            assert spans[gold][0] <= start and end <= spans[gold][1], (context[start:end], spans[gold])
            negatives = negative_candidates(context, spans, gold, [answer["text"]])
            assert gold not in negatives
            assert all(answer["text"] not in context[slice(*spans[k])] for k in negatives)
