"""Offline C2 tests with real WordPiece and SentencePiece fast tokenizers."""

import json

import numpy as np
import pytest
import sentencepiece as spm
from datasets import Dataset
from transformers import BertTokenizerFast, DebertaV2TokenizerFast

from qa.data.features import build_features
from qa.decoding.span_decoder import decode, features_to_arrays


WORDS = "what is blue sky red green first last café 東京 😀 alpha beta gamma delta . ?"


@pytest.fixture(scope="module", params=["bert", "deberta"])
def tokenizer(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    if request.param == "bert":
        vocab = root / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                     *WORDS.split(), "cafe", "東", "京"]) + "\n", encoding="utf-8")
        return BertTokenizerFast(vocab_file=str(vocab))
    corpus = root / "corpus.txt"
    corpus.write_text((WORDS + "\n") * 20, encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(root / "spm"), vocab_size=80,
        hard_vocab_limit=False, character_coverage=1.0, minloglevel=2,
        user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"],
        pad_id=0, unk_id=1, bos_id=-1, eos_id=-1, pad_piece="[PAD]", unk_piece="[UNK]",
    )
    return DebertaV2TokenizerFast(vocab_file=str(root / "spm.model"))


def example(context, answer=None, qid="q", start=None):
    return {"id": qid, "title": "synthetic", "context": context, "question": "  what ?",
            "answers": {"text": [] if answer is None else [answer],
                        "answer_start": [] if answer is None else
                        [context.index(answer) if start is None else start]}}


def recover(row, context):
    start, end = row["start_positions"], row["end_positions"]
    if row["offset_mapping"][start] is None:
        return None
    return context[row["offset_mapping"][start][0]:row["offset_mapping"][end][1]]


def test_offsets_labels_unicode_and_decoder(tokenizer):
    rows = [example("first café 東京 😀 blue last", text, str(i))
            for i, text in enumerate(["first", "café", "東京", "😀", "blue", "last"])]
    features = build_features(Dataset.from_list(rows), tokenizer, 64, 8)
    assert len(features) == len(rows)
    contexts = {row["id"]: row["context"] for row in rows}
    gold = {row["id"]: row["answers"]["text"][0] for row in rows}
    ids, offsets = features_to_arrays(features)
    starts = np.full(offsets.shape[:2], -100.0)
    ends = starts.copy()
    for i, row in enumerate(features):
        assert len(row["input_ids"]) == 64
        assert recover(row, contexts[row["example_id"]]) == gold[row["example_id"]]
        for sid, span, mask in zip(row["sequence_ids"], row["offset_mapping"], row["attention_mask"]):
            if sid != 1 or not mask:
                assert span is None
        starts[i, row["start_positions"]] = 100
        ends[i, row["end_positions"]] = 100
    predictions, _ = decode(contexts, ids, offsets, starts, ends)
    assert predictions == gold
    # Non-context offsets alone also suffice for the merged decoder.
    assert np.array_equal(offsets, features_to_arrays(features.remove_columns("sequence_ids"))[1])


def test_overlap_and_window_boundaries(tokenizer):
    context = " ".join(["blue"] * 80)
    # Locate real window boundaries for each tokenizer's different token lengths.
    windows = build_features(Dataset.from_list([example(context)]), tokenizer, 32, 12)
    assert len(windows) > 2
    visible = [[o for o in w["offset_mapping"] if o is not None] for w in windows]
    a = visible[1][0][0]
    b = visible[0][-1][1]
    assert a < b
    cases = [(visible[0][0][0], visible[0][-1][1]), (a, b),
             (visible[0][-1][0], visible[0][-1][1]),
             (visible[0][-1][0], visible[1][-1][1])]
    for start, end in cases:
        row = example(context, context[start:end], start=start)
        features = build_features(Dataset.from_list([row]), tokenizer, 32, 12)
        for feature, spans in zip(features, visible):
            contained = spans[0][0] <= start and spans[-1][1] >= end
            if contained:
                assert recover(feature, context) == context[start:end]
            else:
                cls = feature["input_ids"].index(tokenizer.cls_token_id)
                assert feature["start_positions"] == feature["end_positions"] == cls
        if (start, end) == (a, b):
            assert sum(recover(f, context) == context[start:end] for f in features) >= 2


def test_left_padding_uses_actual_cls_and_empty_context(tokenizer):
    previous = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        rows = [example("blue", "blue", "in"), example("", qid="empty"),
                example("blue", qid="unlabelled")]
        features = build_features(Dataset.from_list(rows), tokenizer, 32, 4)
        assert recover(features[0], "blue") == "blue"
        for feature in list(features)[1:]:
            cls = feature["input_ids"].index(tokenizer.cls_token_id)
            assert cls > 0
            assert feature["start_positions"] == feature["end_positions"] == cls
        context = "first " + "blue " * 60
        windows = build_features(Dataset.from_list([example(context, "first")]), tokenizer, 32, 4)
        tail = windows[-1]
        cls = tail["input_ids"].index(tokenizer.cls_token_id)
        assert cls > 0
        assert tail["start_positions"] == tail["end_positions"] == cls
    finally:
        tokenizer.padding_side = previous


def test_defaults_empty_and_batch_sample_mapping(tokenizer):
    rows = [example("blue", "blue", str(i)) for i in range(257)]
    examples = Dataset.from_list(rows)
    features = build_features(examples, tokenizer)
    assert features["example_id"] == examples["id"]
    assert all(len(ids) == 384 for ids in features["input_ids"])
    empty = build_features(examples.select([]), tokenizer)
    assert len(empty) == 0
    assert empty.features == features.features


def test_invalid_inputs_fail_clearly(tokenizer):
    rows = Dataset.from_list([example("blue", "blue")])
    for length, stride in [(0, 0), (32, -1), (32, 32), (5, 4)]:
        with pytest.raises(ValueError):
            build_features(rows, tokenizer, length, stride)
    for answers in [{"text": ["red"], "answer_start": [0]},
                    {"text": ["blue"], "answer_start": []}]:
        bad = example("blue")
        bad["answers"] = answers
        with pytest.raises(ValueError, match="answer"):
            build_features(Dataset.from_list([bad]), tokenizer, 32, 4)


def test_merged_engine_hook(tokenizer, tmp_path):
    from qa.models.span_model import SpanModel
    from qa.predict import build_split
    from qa.train import load_config

    # Exercise the merged X8 loader and X9 features through the real engine hook.
    source = tmp_path / "synthetic.json"
    source.write_text(json.dumps({"data": [{"title": "synthetic", "paragraphs": [
        {"context": "sky is blue", "qas": [{"id": "q", "question": "what ?",
         "answers": [{"text": "blue", "answer_start": 7}]}]}]}]}), encoding="utf-8")
    cfg = load_config(overrides=["data.examples_fn=qa.data.squad_io:load_examples",
                                "data.features_fn=qa.data.features:build_features",
                                "data.max_seq_len=32", "data.doc_stride=4"])
    examples, features = build_split(cfg, source, tokenizer, SpanModel)
    assert features["example_id"] == examples["id"]
    assert recover(features[0], examples[0]["context"]) == "blue"


def test_requires_fast_tokenizer():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="fast tokenizer"):
        build_features(None, SimpleNamespace(is_fast=False))


def test_reference_order_and_intra_token_boundary(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nblue\nred\nwhat\n?\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    row = example("blue red", "blue")
    row["answers"] = {"text": ["blue", "red"], "answer_start": [0, 5]}
    assert recover(build_features(Dataset.from_list([row]), tokenizer, 16, 0)[0], "blue red") == "blue"
    # Character spans inside a single token cannot be recovered exactly. Preserve
    # the true token offsets so later acceptance audits can report this cause.
    row = example("blue", "lu")
    assert recover(build_features(Dataset.from_list([row]), tokenizer, 16, 0)[0], "blue") == "blue"
    row = example("blue red", " red")
    assert recover(build_features(Dataset.from_list([row]), tokenizer, 16, 0)[0], "blue red") == "blue red"


def audit_training_recovery(models, limit=None):
    """Opt-in CPU acceptance audit; run as a script, never during pytest.

    From the repo root: set PYTHONPATH=src, then run this file with --audit.
    Reads only the processed training split; failures and provenance stay ignored.
    """
    import hashlib
    import subprocess
    import time
    from collections import Counter
    from pathlib import Path

    from datasets import disable_progress_bars
    from transformers import AutoTokenizer

    from qa.data.squad_io import load_examples

    root = Path(__file__).resolve().parents[1]
    source = root / "data/processed/train.json"
    examples = load_examples(source)
    manifest = json.loads((source.parent / "split_titles.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 2026
    assert set(examples["title"]) == set(manifest["train"])
    assert not set(examples["title"]) & set(manifest["val"])
    assert len(examples) == 78259, "Acceptance requires the verified X8 training split"
    if limit is not None:
        examples = examples.select(range(min(limit, len(examples))))
    disable_progress_bars()
    report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "git_sha": subprocess.check_output(
                  ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
                  cwd=root, text=True).strip(),
              "working_tree_features_sha256": hashlib.sha256(
                  (root / "src/qa/data/features.py").read_bytes()).hexdigest(),
              "max_seq_len": 384, "doc_stride": 128, "full_split": limit is None,
              "reference": "first", "models": {}}
    for name in models:
        started = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(
            name, use_fast=True, cache_dir=str(source.parent / "tokenizers"))
        revision = (source.parent / "tokenizers" / f"models--{name.replace('/', '--')}"
                    / "refs/main").read_text(encoding="utf-8").strip()
        failures, recovered, windows = [], 0, 0
        for first in range(0, len(examples), 256):
            batch = examples.select(range(first, min(first + 256, len(examples))))
            originals = {row["id"]: row for row in batch}
            candidates = {qid: [] for qid in originals}
            features = build_features(batch, tokenizer)
            windows += len(features)
            for feature in features:
                qid = feature["example_id"]
                text = recover(feature, originals[qid]["context"])
                if text is not None and text not in candidates[qid]:
                    candidates[qid].append(text)
            for qid, original in originals.items():
                gold = original["answers"]["text"][0]
                if gold in candidates[qid]:
                    recovered += 1
                    continue
                cause = ("no_complete_labelled_window" if not candidates[qid] else
                         "annotation_edge_whitespace" if gold != gold.strip() else
                         "token_boundary_or_normalization")
                failures.append({"id": qid, "gold": gold,
                                 "answer_start": original["answers"]["answer_start"][0],
                                 "recovered": candidates[qid], "cause": cause})
            if first % 2560 == 0:
                print(f"{name}: {first + len(batch)}/{len(examples)} questions", flush=True)
        result = {"questions": len(examples), "windows": windows, "recovered": recovered,
                  "exact_percent": 100 * recovered / len(examples),
                  "meets_99_5": recovered / len(examples) >= .995,
                  "tokenizer_class": type(tokenizer).__name__,
                  "revision": revision,
                  "seconds": time.monotonic() - started,
                  "failure_counts": dict(Counter(f["cause"] for f in failures)),
                  "failures": failures}
        report["models"][name] = result
        destination = source.parent / ("x9_recovery.json" if limit is None else "x9_recovery_pilot.json")
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Opt-in X9 training recovery audit")
    parser.add_argument("--audit", action="store_true", required=True)
    parser.add_argument("--limit", type=int, help="Pilot only; does not satisfy acceptance")
    parser.add_argument("--models", nargs="+", default=["bert-base-uncased", "microsoft/deberta-v3-base"])
    args = parser.parse_args()
    audit_training_recovery(args.models, args.limit)
