"""span_decoder vs. the Hugging Face reference post-processing, on a tiny multi-window sample."""
import collections

import numpy as np
import pytest
from datasets import Dataset

from damian_mocks import make_squad_file, make_tiny_encoder, squad_examples, squad_features
from qa.decoding.span_decoder import NOT_CONTEXT, decode, features_to_arrays
from transformers import AutoTokenizer


def hf_reference(examples, features, start_logits, end_logits, n_best=20, max_answer_len=30):
    """Condensed from transformers examples/pytorch/question-answering/utils_qa.py
    (postprocess_qa_predictions, version_2_with_negative=False), Apache-2.0. Kept independent of
    span_decoder: it reads offsets straight from the C2 features (None = not context)."""
    by_example = collections.defaultdict(list)
    for i, qid in enumerate(features["example_id"]):
        by_example[qid].append(i)
    preds, best_scores = {}, {}
    for example in examples:
        prelim = []
        for i in by_example[example["id"]]:
            offsets = features[i]["offset_mapping"]
            s_idx = np.argsort(start_logits[i])[-1 : -n_best - 1 : -1].tolist()
            e_idx = np.argsort(end_logits[i])[-1 : -n_best - 1 : -1].tolist()
            for s in s_idx:
                for e in e_idx:
                    if s >= len(offsets) or e >= len(offsets) or offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_len:
                        continue
                    prelim.append({"offsets": (offsets[s][0], offsets[e][1]),
                                   "score": start_logits[i][s] + end_logits[i][e]})
        prelim = sorted(prelim, key=lambda x: x["score"], reverse=True)[:n_best]
        if prelim:
            o = prelim[0]["offsets"]
            preds[example["id"]] = example["context"][o[0]:o[1]]
            best_scores[example["id"]] = float(prelim[0]["score"])
    return preds, best_scores


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    root = tmp_path_factory.mktemp("decoder")
    tokenizer = AutoTokenizer.from_pretrained(make_tiny_encoder(root / "enc"))
    examples = squad_examples(make_squad_file(root / "squad.json", 30))
    features = squad_features(examples, tokenizer, max_seq_len=32, doc_stride=8)
    return examples, features


def test_matches_hf_reference(sample):
    examples, features = sample
    assert len(features) > len(examples)  # several questions span more than one window
    rng = np.random.default_rng(0)
    for max_answer_len in (30, 2):
        start = rng.normal(size=(len(features), 32)).astype(np.float32)
        end = rng.normal(size=(len(features), 32)).astype(np.float32)
        example_ids, offsets = features_to_arrays(features)
        contexts = dict(zip(examples["id"], examples["context"]))
        preds, nbest = decode(contexts, example_ids, offsets, start, end, max_answer_len=max_answer_len)
        ref_preds, ref_scores = hf_reference(examples, features, start, end, max_answer_len=max_answer_len)
        assert preds == {qid: ref_preds.get(qid, "") for qid in preds}
        for qid, score in ref_scores.items():
            assert nbest[qid][0]["score"] == pytest.approx(score)


def test_rules_on_spans(sample):
    examples, features = sample
    example_ids, offsets = features_to_arrays(features)
    contexts = dict(zip(examples["id"], examples["context"]))
    rng = np.random.default_rng(1)
    start, end = rng.normal(size=(2, len(features), 32))
    _, nbest = decode(contexts, example_ids, offsets, start, end, n_best=5, max_answer_len=3)
    for qid, entries in nbest.items():
        assert 0 < len(entries) <= 5
        assert [e["score"] for e in entries] == sorted((e["score"] for e in entries), reverse=True)
        for e in entries:
            assert e["text"] == contexts[qid][e["start_char"]:e["end_char"]]
            assert len(e["text"].split()) <= 3  # ≤ 3 tokens; every token here is one word


def test_best_span_is_pooled_across_windows():
    contexts = {"q": "aa bb cc dd"}
    offsets = np.full((2, 4, 2), NOT_CONTEXT)
    offsets[0, 1:3] = [[0, 2], [3, 5]]
    offsets[1, 1:3] = [[6, 8], [9, 11]]
    start = np.array([[9.0, 1, 0, 0], [0, 0, 2, 0]])  # window 0 peaks on CLS: not context
    end = np.array([[0.0, 1, 0, 0], [0, 0, 2, 0]])
    preds, _ = decode(contexts, ["q", "q"], offsets, start, end)
    assert preds == {"q": "dd"}


def test_no_valid_span_gives_empty_answer():
    offsets = np.full((1, 3, 2), NOT_CONTEXT)
    preds, nbest = decode({"q": "abc"}, ["q"], offsets, np.zeros((1, 3)), np.zeros((1, 3)))
    assert preds == {"q": ""} and nbest == {"q": []}


def test_features_to_arrays_accepts_sequence_ids_column():
    features = Dataset.from_dict({
        "example_id": ["q"],
        "offset_mapping": [[[0, 0], [0, 4], [0, 0], [0, 3], [4, 7]]],
        "sequence_ids": [[None, 0, None, 1, 1]],
    })
    _, offsets = features_to_arrays(features)
    assert offsets[0, :, 0].tolist() == [NOT_CONTEXT, NOT_CONTEXT, NOT_CONTEXT, 0, 4]
