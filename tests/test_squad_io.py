"""C1 split isolation, repeatability, loader compatibility, and raw-data integrity."""
import copy
import io
import json

import pytest

from qa.data.squad_io import (
    ADVERSARIAL, ROOT, dataset_statistics, download_adversarial, load_examples, load_squad, main,
    prepare_splits, sha256, split_by_article, verify_raw_data,
)


def document(n=10):
    return {"version": "1.1", "data": [
        {"title": f"Article {i}", "paragraphs": [{"context": "Paris is a city.", "qas": [
            {"id": f"{i}-0", "question": "Where is Paris?", "answers": [
                {"text": "Paris", "answer_start": 0}]},
            {"id": f"{i}-1", "question": "What is Paris?", "answers": [
                {"text": "a city", "answer_start": 9}, {"text": "city", "answer_start": 11}]},
        ]}]} for i in range(n)
    ]}


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def qids(data):
    return {q["id"] for a in data["data"] for p in a["paragraphs"] for q in p["qas"]}


def test_split_is_reproducible_disjoint_and_preserves_all_questions():
    original = document()
    before = copy.deepcopy(original)
    train, val, manifest = split_by_article(original)
    assert (train, val, manifest) == split_by_article(original)
    assert original == before
    assert len(manifest["val"]) == 1 and len(manifest["train"]) == 9
    assert not set(manifest["train"]) & set(manifest["val"])
    assert not qids(train) & qids(val)
    assert qids(train) | qids(val) == qids(original)
    assert train["version"] == val["version"] == "1.1"
    reversed_input = {**original, "data": list(reversed(original["data"]))}
    assert split_by_article(reversed_input)[2] == manifest


def test_repeated_titles_stay_together():
    data = document()
    repeated = copy.deepcopy(data["data"][0])
    for qa in repeated["paragraphs"][0]["qas"]:
        qa["id"] += "-repeat"
    data["data"].append(repeated)
    train, val, _ = split_by_article(data)
    counts = [sum(a["title"] == "Article 0" for a in part["data"]) for part in (train, val)]
    assert sorted(counts) == [0, 2]


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        split_by_article(document(), fraction)


def test_split_needs_two_titles():
    with pytest.raises(ValueError):
        split_by_article(document(1))


def test_load_examples_matches_training_hook_and_preserves_references(tmp_path):
    data = document(2)
    data["data"][0]["paragraphs"][0]["context"] += " café"
    path = write(tmp_path / "train.json", data)
    assert load_squad(path) == data
    examples = load_examples(path)
    assert set(examples.column_names) == {"id", "title", "question", "context", "answers"}
    assert len(examples) == 4
    assert examples[1]["answers"] == {"text": ["a city", "city"], "answer_start": [9, 11]}
    for example in examples:
        for text, start in zip(example["answers"]["text"], example["answers"]["answer_start"]):
            assert example["context"][start:start + len(text)] == text
    empty = load_examples(write(tmp_path / "empty.json", {"data": []}))
    assert len(empty) == 0 and empty.column_names == examples.column_names


@pytest.mark.parametrize("error", ["duplicate", "offset", "no_answers"])
def test_loader_rejects_bad_annotations(tmp_path, error):
    data = document(2)
    qa = data["data"][0]["paragraphs"][0]["qas"][0]
    if error == "duplicate":
        qa["id"] = "1-0"
    elif error == "offset":
        qa["answers"][0]["answer_start"] = -1
    else:
        qa["answers"] = []
    with pytest.raises(ValueError):
        load_squad(write(tmp_path / "bad.json", data))


def test_statistics_are_per_context_and_first_answer_per_question():
    stats = dataset_statistics(document(2))
    assert (stats["articles"], stats["contexts"], stats["questions"]) == (2, 2, 4)
    assert stats["context_words"]["mean"] == 4
    assert stats["answer_words"]["mean"] == 1.5
    assert stats["answer_words"]["median"] == 1.5
    assert stats["question_types"] == {"what": 2, "where": 2}
    assert stats["reference_offset_mismatches"] == 0


def test_prepare_outputs_are_byte_reproducible_and_leave_input_unchanged(tmp_path):
    source = write(tmp_path / "source.json", document())
    before = sha256(source)
    output, tables = tmp_path / "processed", tmp_path / "tables"
    first = prepare_splits(source, output, tables)
    outputs = list(output.iterdir()) + list(tables.iterdir())
    contents = {p: p.read_bytes() for p in outputs}
    assert prepare_splits(source, output, tables) == first
    assert all(p.read_bytes() == expected for p, expected in contents.items())
    assert sha256(source) == before
    assert json.loads((output / "split_titles.json").read_text())["source_sha256"] == before
    assert len(load_examples(output / "train.json")) == 18
    assert len(load_examples(output / "val.json")) == 2
    assert "Questions" in (tables / "data_stats.tex").read_text()


def test_prepare_refuses_test_data_and_raw_outputs(tmp_path):
    dev = write(tmp_path / "dev-v1.1.json", document())
    with pytest.raises(ValueError, match="test set"):
        prepare_splits(dev, tmp_path / "out", tmp_path / "tables")
    source = write(tmp_path / "source.json", document())
    with pytest.raises(ValueError, match="read-only"):
        prepare_splits(source, ROOT / "data/raw", tmp_path / "tables")
    with pytest.raises(ValueError, match="overwrite"):
        prepare_splits(write(tmp_path / "train.json", document()), tmp_path, tmp_path / "tables")


def test_checksum_verification_detects_corruption(tmp_path):
    names = ["train-v1.1.json", "dev-v1.1.json"]
    for name in names:
        (tmp_path / name).write_bytes(b"synthetic raw data")
    (tmp_path / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(tmp_path / n)}  {n}" for n in names) + "\n")
    assert set(verify_raw_data(tmp_path)) == set(names)
    (tmp_path / names[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_raw_data(tmp_path)


def test_prepare_cli(tmp_path, capsys):
    path = write(tmp_path / "source.json", document())
    main(["prepare", "--input", str(path), "--output-dir", str(tmp_path / "out"),
          "--tables-dir", str(tmp_path / "tables")])
    assert json.loads(capsys.readouterr().out)["val"]["articles"] == 1


def test_adversarial_download_checks_hash_and_stays_separate(tmp_path, monkeypatch):
    import qa.data.squad_io as squad_io

    fixture = write(tmp_path / "fixture.json", document())
    data = fixture.read_bytes()
    monkeypatch.setattr(squad_io, "ADVERSARIAL", {
        name: (bundle, sha256(fixture)) for name, (bundle, _) in ADVERSARIAL.items()
    })
    monkeypatch.setattr(squad_io.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(data))
    output = tmp_path / "processed"
    provenance = download_adversarial("codalab", output)
    assert provenance["evaluation_only"] is True
    assert set(provenance["files"]) == {"addsent", "addonesent"}
    assert load_squad(output / "addsent.json") == document()
    assert not (output / "train.json").exists()
    # A bad response cannot replace previously validated data.
    monkeypatch.setattr(squad_io.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_adversarial("codalab", output)
    assert (output / "addsent.json").read_bytes() == data


def test_hf_download_requires_pinned_revision(tmp_path, monkeypatch):
    import datasets

    example = {"id": "q1", "title": "Article", "context": "Paris is a city.",
               "question": "Where?", "answers": {"text": ["Paris"], "answer_start": [0]}}
    calls = []

    def fake_load(repo, **kwargs):
        calls.append((repo, kwargs))
        return [example]

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    with pytest.raises(ValueError, match="commit SHA"):
        download_adversarial("hf", tmp_path / "out")
    provenance = download_adversarial("hf", tmp_path / "out", revision="a" * 40)
    assert len(calls) == 2
    assert all(repo == "parquet" for repo, _ in calls)
    assert "a" * 40 in calls[0][1]["data_files"]["validation"]
    assert calls[0][1]["split"] == "validation"
    assert provenance["revision"] == "a" * 40
    assert len(load_examples(tmp_path / "out/addonesent.json")) == 1


def test_loader_matches_merged_training_build_split_hook(tmp_path):
    # Exercise the merged engine's real import/call path; only X9 remains mocked.
    from damian_mocks import make_tiny_encoder
    from transformers import AutoTokenizer

    from qa.models.span_model import SpanModel
    from qa.predict import build_split
    from qa.train import load_config

    source = write(tmp_path / "train.json", document(2))
    tokenizer = AutoTokenizer.from_pretrained(make_tiny_encoder(tmp_path / "encoder"))
    cfg = load_config(overrides=[
        "data.examples_fn=qa.data.squad_io:load_examples",
        "data.features_fn=damian_mocks:squad_features",
        "data.max_seq_len=32", "data.doc_stride=8",
    ])
    examples, features = build_split(cfg, source, tokenizer, SpanModel)
    assert len(examples) == 4
    assert set(features["example_id"]) == set(examples["id"])
    assert {"input_ids", "attention_mask", "offset_mapping", "start_positions",
            "end_positions"} <= set(features.column_names)
