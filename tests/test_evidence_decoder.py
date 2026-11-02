"""evidence_decoder (G2): λ = 0 equals span_decoder, λ moves answers by sentence rank, CLI files.

Contexts, C3 sentences and window arrays are synthetic: one word per token, no tokenizer.
"""
import json
import shutil

import numpy as np
import pytest

from qa.decoding.evidence_decoder import decode_joint, main, run
from qa.decoding.sentence_ranking import window_slots
from qa.decoding.span_decoder import decode

QUESTION_TOKENS = 3
C6_KEYS = {"text", "start_logit", "end_logit", "sent_idx", "sent_logprob", "score"}


def make_question(qid, n_words, sent_words, window, stride):
    """Context of ``n_words`` one-token words, sentences of ``sent_words`` words, and windows of
    ``window`` context tokens every ``stride`` words: ``(context, C3 record, window offsets)``."""
    words = [f"{qid}w{i}" for i in range(n_words)]
    spans, char = [], 0
    for word in words:
        spans.append((char, char + len(word)))
        char += len(word) + 1
    sentences = [[spans[i][0], spans[min(i + sent_words, n_words) - 1][1]]
                 for i in range(0, n_words, sent_words)]
    firsts = [0]
    while firsts[-1] + window < n_words:
        firsts.append(firsts[-1] + stride)
    windows = []
    for first in firsts:
        offs = np.full((QUESTION_TOKENS + window + 1, 2), -1)
        for t, span in enumerate(spans[first:first + window]):
            offs[QUESTION_TOKENS + t] = span
        windows.append(offs)
    return " ".join(words), {"id": qid, "sentences": sentences}, windows


def sample():
    """Three questions, two of them split across overlapping windows."""
    contexts, evidence, example_ids, offsets = {}, {}, [], []
    for qid, n_words in (("a", 30), ("b", 12), ("c", 45)):
        context, record, windows = make_question(qid, n_words, sent_words=5, window=16, stride=10)
        contexts[qid], evidence[qid] = context, record
        example_ids += [qid] * len(windows)
        offsets += windows
    return contexts, evidence, example_ids, np.stack(offsets)


def pos(word_index):
    return QUESTION_TOKENS + word_index


def test_lambda_zero_matches_the_standard_decoder():
    contexts, evidence, example_ids, offsets = sample()
    assert len(example_ids) > len(contexts)
    rng = np.random.default_rng(0)
    start, end = rng.normal(size=(2, *offsets.shape[:2]))
    ranking = {qid: [[s, -0.5 * s] for s in range(len(rec["sentences"]))] for qid, rec in evidence.items()}
    for max_answer_len in (30, 3):
        ref_preds, ref_nbest = decode(contexts, example_ids, offsets, start, end, max_answer_len=max_answer_len)
        preds, nbest = decode_joint(contexts, example_ids, offsets, start, end, ranking, evidence,
                                    lam=0.0, max_answer_len=max_answer_len)
        assert preds == ref_preds
        for qid, entries in ref_nbest.items():
            fields = ("text", "start_logit", "end_logit", "score")
            assert [tuple(e[f] for f in fields) for e in nbest[qid]] == [tuple(e[f] for f in fields) for e in entries]


def test_lambda_moves_the_answer_into_a_higher_ranked_sentence():
    context, record, windows = make_question("q", 6, sent_words=3, window=10, stride=10)
    offsets = np.stack(windows)
    start = np.full(offsets.shape[:2], -10.0)
    end = np.full(offsets.shape[:2], -10.0)
    start[0, pos(1)] = end[0, pos(1)] = 5.0   # best span "qw1", sentence 0
    start[0, pos(4)] = end[0, pos(4)] = 4.5   # runner-up "qw4", sentence 1
    ranking = {"q": [[1, -0.1], [0, -3.0]]}
    args = ({"q": context}, ["q"], offsets, start, end, ranking, {"q": record})

    assert decode_joint(*args, lam=0.0)[0] == {"q": "qw1"}
    preds, nbest = decode_joint(*args, lam=1.0)
    assert preds == {"q": "qw4"}
    top = nbest["q"][0]
    assert set(top) == C6_KEYS
    assert (top["sent_idx"], top["sent_logprob"]) == (1, -0.1)
    assert top["score"] == pytest.approx(4.5 + 4.5 - 0.1)
    assert [e["score"] for e in nbest["q"]] == sorted((e["score"] for e in nbest["q"]), reverse=True)


def test_span_starting_outside_every_sentence_gets_the_lowest_logprob():
    # Sentences cover words 0-1 and 3-5; word 2 sits in no sentence.
    context, _, windows = make_question("q", 6, sent_words=3, window=10, stride=10)
    record = {"id": "q", "sentences": [[0, 7], [12, 23]]}
    assert [context[s:e] for s, e in record["sentences"]] == ["qw0 qw1", "qw3 qw4 qw5"]
    offsets = np.stack(windows)
    start = np.full(offsets.shape[:2], -10.0)
    end = np.full(offsets.shape[:2], -10.0)
    start[0, pos(2)] = end[0, pos(2)] = 5.0
    preds, nbest = decode_joint({"q": context}, ["q"], offsets, start, end, {"q": [[0, -0.5], [1, -2.0]]},
                                {"q": record}, lam=1.0)
    top = nbest["q"][0]
    assert preds == {"q": "qw2"}
    assert (top["sent_idx"], top["sent_logprob"]) == (None, -2.0)
    assert top["score"] == pytest.approx(10.0 - 2.0)


def write_inputs(tmp_path, split="val"):
    contexts, evidence, example_ids, offsets = sample()
    squad = {"version": "1.1", "data": [{"title": "synthetic", "paragraphs": [
        {"context": contexts[qid], "qas": [{"id": qid, "question": "?", "answers": []}]} for qid in contexts]}]}
    (tmp_path / "squad.json").write_text(json.dumps(squad))
    (tmp_path / "evidence.jsonl").write_text("\n".join(json.dumps(r) for r in evidence.values()))
    slots = [window_slots(offsets[i], evidence[qid]["sentences"]) for i, qid in enumerate(example_ids)]
    width = max(len(s) for s in slots)
    rng = np.random.default_rng(1)
    sent_mask = np.array([[float(j < len(s)) for j in range(width)] for s in slots])
    np.savez_compressed(tmp_path / f"{split}.logits.npz", example_ids=np.array(example_ids), offsets=offsets,
                        start_logits=rng.normal(size=offsets.shape[:2]), end_logits=rng.normal(size=offsets.shape[:2]),
                        sent_logits=rng.normal(size=sent_mask.shape), sent_mask=sent_mask)
    return contexts, evidence


def test_cli_writes_c8_and_joint_c6_files(tmp_path):
    contexts, evidence = write_inputs(tmp_path)
    paths = run(tmp_path / "val.logits.npz", tmp_path / "evidence.jsonl", tmp_path / "squad.json", lam=0.5)
    assert sorted(p.name for p in paths.values()) == [
        "val.joint_lam0.5.nbest.json", "val.joint_lam0.5.predictions.json", "val.sent_rank.json"]
    sent_rank = json.loads(paths["sent_rank"].read_text())
    for qid, record in evidence.items():
        assert sorted(sent for sent, _ in sent_rank[qid]) == list(range(len(record["sentences"])))
        assert [s for _, s in sent_rank[qid]] == sorted((s for _, s in sent_rank[qid]), reverse=True)
    predictions = json.loads(paths["predictions"].read_text())
    nbest = json.loads(paths["nbest"].read_text())
    assert set(predictions) == set(nbest) == set(contexts)
    assert all(set(e) == C6_KEYS for entries in nbest.values() for e in entries)


def test_cli_refuses_test_split_and_span_only_logits(tmp_path):
    write_inputs(tmp_path)
    shutil.copy(tmp_path / "val.logits.npz", tmp_path / "test.logits.npz")
    common = ["--evidence", str(tmp_path / "evidence.jsonl"), "--data-file", str(tmp_path / "squad.json"), "--lam", "1"]
    with pytest.raises(SystemExit, match="run_all.sh"):
        main(["--logits", str(tmp_path / "test.logits.npz"), *common])
    saved = dict(np.load(tmp_path / "val.logits.npz"))
    del saved["sent_logits"], saved["sent_mask"]
    np.savez_compressed(tmp_path / "span.logits.npz", **saved)
    with pytest.raises(ValueError, match="sent_logits"):
        main(["--logits", str(tmp_path / "span.logits.npz"), *common])
