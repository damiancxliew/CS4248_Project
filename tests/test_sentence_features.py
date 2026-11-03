"""sentence_features: C4 sentence fields on hand-built windows (the C2 mock lives in this file)."""
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from damian_mocks import make_squad_file, make_tiny_encoder, squad_examples, squad_features
from qa.data.sentence_features import IGNORE, add_sentence_fields

# Five sentences: gold = 2, sentence 3 repeats the answer, so it is not a candidate.
RECORD = {
    "id": "q1",
    "sentences": [[0, 10], [11, 20], [21, 30], [31, 40], [41, 50]],
    "gold_sent": 2,
    "neg_candidates": [0, 1, 4],
    "lex_neg": 1,
    "rand_neg": 4,
}


def window(token_spans, example_id="q1", specials=2):
    """A C2 window: ``specials`` non-context tokens, then one context token per span."""
    offsets = [None] * specials + [list(span) if span else None for span in token_spans]
    return {"example_id": example_id, "input_ids": [0] * len(offsets), "offset_mapping": offsets}


def features(*windows):
    return Dataset.from_list(list(windows))


def fields(*windows, evidence=None, neg_mode="model"):
    return add_sentence_fields(features(*windows), evidence or {"q1": RECORD}, neg_mode)


WINDOW_A = window([(0, 4), (5, 10), (11, 20), (21, 25), (26, 30)])  # sentences 0, 1, 2
WINDOW_B = window([(21, 25), (26, 30), (31, 40), (41, 50)])  # sentences 2, 3, 4


def test_tokens_get_window_local_slots_and_non_context_tokens_get_minus_one():
    out = fields(WINDOW_A)
    assert out[0]["sent_ids"] == [-1, -1, 0, 0, 1, 2, 2]


def test_input_columns_are_kept_and_all_c4_columns_added():
    out = fields(WINDOW_A)
    assert set(out.column_names) == {"example_id", "input_ids", "offset_mapping", "sent_ids",
                                     "sent_label", "neg_cand_mask", "static_neg_label"}
    assert out[0]["input_ids"] == WINDOW_A["input_ids"]
    assert all(type(value) is int for value in out[0]["neg_cand_mask"])


def test_answer_in_the_overlap_of_two_windows_is_gold_in_both_with_different_slots():
    out = fields(WINDOW_A, WINDOW_B, neg_mode="lexical")
    assert out["sent_ids"][1] == [-1, -1, 0, 0, 1, 2]
    assert out["sent_label"] == [2, 0]
    assert out["neg_cand_mask"] == [[1, 1, 0], [0, 0, 1]]
    assert out["static_neg_label"] == [1, IGNORE]


def test_gold_sentence_cut_by_the_window_edge_still_counts_as_visible():
    cut_at_end = window([(0, 10), (11, 20), (21, 25)])
    cut_at_start = window([(26, 30), (31, 40)])
    out = fields(cut_at_end, cut_at_start)
    assert out["sent_ids"] == [[-1, -1, 0, 1, 2], [-1, -1, 0, 1]]
    assert out["sent_label"] == [2, 0]


def test_gold_sentence_outside_the_window_is_ignored():
    out = fields(window([(31, 40), (41, 50)]))
    assert out[0]["sent_label"] == IGNORE
    assert out[0]["neg_cand_mask"] == [0, 1]


@pytest.mark.parametrize("neg_mode, expected", [
    ("lexical", 1), ("random", 4), ("model", IGNORE), ("none", IGNORE),
])
def test_static_negative_follows_neg_mode(neg_mode, expected):
    everything = window([(0, 10), (11, 20), (21, 30), (31, 40), (41, 50)])
    assert fields(everything, neg_mode=neg_mode)[0]["static_neg_label"] == expected


def test_static_negative_outside_the_window_is_ignored():
    assert fields(window([(0, 10), (21, 30)]), neg_mode="lexical")[0]["static_neg_label"] == IGNORE


def test_no_negative_in_a_single_sentence_context():
    record = {"id": "q1", "sentences": [[0, 10]], "gold_sent": 0, "neg_candidates": [],
              "lex_neg": None, "rand_neg": None}
    out = fields(window([(0, 10)]), evidence={"q1": record}, neg_mode="random")
    assert out[0]["sent_label"] == 0
    assert out[0]["neg_cand_mask"] == [0]
    assert out[0]["static_neg_label"] == IGNORE


def test_slots_skip_a_sentence_with_no_visible_token():
    out = fields(window([(0, 10), (21, 30)]))
    assert out[0]["sent_ids"] == [-1, -1, 0, 1]
    assert out[0]["sent_label"] == 1
    assert out[0]["neg_cand_mask"] == [1, 0]


def test_token_goes_to_the_sentence_it_overlaps_most():
    out = fields(window([(8, 12), (10, 15), (19, 25)]))  # mostly 0, leading space + 1, mostly 2
    assert out[0]["sent_ids"] == [-1, -1, 0, 1, 2]


def test_token_only_on_the_gap_between_sentences_gets_minus_one():
    assert fields(window([(0, 10), (10, 11), (11, 20)]))[0]["sent_ids"] == [-1, -1, 0, -1, 1]


def test_sequence_ids_column_marks_context_when_offsets_are_not_nulled():
    row = {"example_id": "q1", "input_ids": [0] * 4, "offset_mapping": [[0, 0], [0, 3], [0, 10], [11, 20]],
           "sequence_ids": [None, 0, 1, 1]}
    assert fields(row)[0]["sent_ids"] == [-1, -1, 0, 1]


def test_unknown_neg_mode_is_rejected():
    with pytest.raises(ValueError, match="neg_mode"):
        fields(WINDOW_A, neg_mode="hard")


def test_window_without_an_evidence_record_is_an_error():
    with pytest.raises(KeyError, match="q2"):
        fields(window([(0, 10)], example_id="q2"))


def test_real_sliding_windows_from_a_tokenizer(tmp_path):
    """Mocked C2 windows: the answer token lies in the labelled sentence in every window."""
    tokenizer = AutoTokenizer.from_pretrained(make_tiny_encoder(tmp_path / "encoder"))
    examples = squad_examples(make_squad_file(tmp_path / "squad.json", 30))
    windows = squad_features(examples, tokenizer, max_seq_len=40, doc_stride=12)
    assert len(windows) > len(examples)

    evidence = {}
    for ex in examples:
        context, sentences, pos = ex["context"], [], 0
        while pos < len(context):
            end = context.index(".", pos) + 1
            sentences.append([pos, end])
            pos = end + 1
        start = ex["answers"]["answer_start"][0]
        gold = next(i for i, (cs, ce) in enumerate(sentences) if cs <= start < ce)
        others = [i for i in range(len(sentences)) if i != gold]
        evidence[ex["id"]] = {"id": ex["id"], "sentences": sentences, "gold_sent": gold,
                              "neg_candidates": others, "lex_neg": others[0], "rand_neg": others[-1]}

    out = add_sentence_fields(windows, evidence, "model")
    for row in out:
        assert len(row["sent_ids"]) == len(row["input_ids"])
        n_slots = max(row["sent_ids"]) + 1
        assert sorted(set(row["sent_ids"]) - {-1}) == list(range(n_slots))
        assert len(row["neg_cand_mask"]) == n_slots
        if row["start_positions"] != 0:
            assert row["sent_ids"][row["start_positions"]] == row["sent_label"]
        if row["sent_label"] != IGNORE:
            assert row["neg_cand_mask"][row["sent_label"]] == 0
