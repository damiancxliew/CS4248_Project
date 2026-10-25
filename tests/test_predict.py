"""predict.py: C6 files, saved logits, C3 sentence ids, test-split guard. C1/C2/C3 are mocked."""
import json

import numpy as np
import pytest

from damian_mocks import evidence_records, smoke_config, squad_examples
from qa.decoding.span_decoder import decode
from qa.predict import predict, sentence_index
from qa.train import train

C6_NBEST_KEYS = {"text", "start_logit", "end_logit", "sent_idx", "sent_logprob", "score"}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("predict")
    cfg = smoke_config(root, epochs=1)
    train(cfg, log=lambda _: None)
    return root / "outputs/smoke_s42/best", cfg


def test_writes_c6_files_and_logits(trained, tmp_path):
    checkpoint, cfg = trained
    examples = squad_examples(cfg["data"]["val_file"])
    evidence = tmp_path / "evidence_val.jsonl"
    evidence.write_text("\n".join(json.dumps(r) for r in evidence_records(examples).values()))

    paths = predict(checkpoint, "val", out_dir=tmp_path, evidence_file=evidence, device="cpu")
    predictions = json.loads(paths["predictions"].read_text())
    nbest = json.loads(paths["nbest"].read_text())
    assert set(predictions) == set(nbest) == set(examples["id"])
    contexts = dict(zip(examples["id"], examples["context"]))
    n_sentences = {qid: len(rec["sentences"]) for qid, rec in evidence_records(examples).items()}
    for qid, entries in nbest.items():
        assert entries and predictions[qid] == entries[0]["text"]
        for e in entries:
            assert set(e) == C6_NBEST_KEYS and e["sent_logprob"] is None
            assert 0 <= e["sent_idx"] < n_sentences[qid]

    # Decoders can be re-run on the saved logits without the model.
    saved = np.load(paths["logits"])
    redecoded, _ = decode(contexts, saved["example_ids"].tolist(), saved["offsets"],
                          saved["start_logits"], saved["end_logits"])
    assert redecoded == predictions


def test_sent_idx_is_the_sentence_holding_the_span_start():
    evidence = {"sentences": [[0, 10], [11, 20], [21, 30]]}
    assert sentence_index(evidence, 0, 5) == 0
    assert sentence_index(evidence, 12, 25) == 1  # crosses into sentence 2: start decides
    assert sentence_index(evidence, 10, 12) is None  # starts on the gap between sentences


def test_sent_idx_is_null_without_evidence(trained, tmp_path):
    checkpoint, _ = trained
    nbest = json.loads(predict(checkpoint, "val", out_dir=tmp_path, device="cpu")["nbest"].read_text())
    assert all(e["sent_idx"] is None for entries in nbest.values() for e in entries)


def test_any_file_can_be_predicted(trained, tmp_path):
    checkpoint, cfg = trained
    paths = predict(checkpoint, "addsent", data_file=cfg["data"]["train_file"], out_dir=tmp_path, device="cpu")
    assert len(json.loads(paths["predictions"].read_text())) == 150


def test_test_split_needs_the_run_all_flag(trained, tmp_path):
    checkpoint, _ = trained
    with pytest.raises(SystemExit, match="run_all.sh"):
        predict(checkpoint, "test", out_dir=tmp_path, device="cpu")
    paths = predict(checkpoint, "test", out_dir=tmp_path, device="cpu", allow_test=True)
    assert len(json.loads(paths["predictions"].read_text())) == 9
