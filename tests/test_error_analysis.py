"""Tests for error_analysis against mocked C3/C6 inputs (no spaCy needed: NER is injected)."""
import csv

import pytest

from qa.eval import error_analysis as ea

MOCK_LABELS = {"1998": "DATE", "Paris": "GPE", "Barack Obama": "PERSON"}


def mock_ner(text):
    return MOCK_LABELS.get(text.strip())


def cat(pred, pred_sent, gold_sent, gold):
    return ea.categorise(pred, pred_sent, gold_sent, gold, mock_ner)


def test_wrong_sentence_beats_everything():
    # prediction overlaps the gold text, but sits in another sentence
    assert cat("Barack", 3, 1, ["Barack Obama"]) == "wrong_sentence"


def test_boundary_when_overlapping_in_gold_sentence():
    assert cat("President Barack Obama", 1, 1, ["Barack Obama"]) == "boundary"


def test_numeric_date_before_entity():
    assert cat("1999", 1, 1, ["1998"]) == "numeric_date"


def test_entity_when_different_span():
    assert cat("London", 1, 1, ["Paris"]) == "entity"


def test_other():
    assert cat("quickly", 1, 1, ["a long answer phrase"]) == "other"


def test_regex_fallback_recognises_numbers():
    assert ea.regex_entity_label("1998") == "CARDINAL"
    assert ea.regex_entity_label("42%") == "CARDINAL"
    assert ea.regex_entity_label("Paris") is None


def test_question_type_and_answer_length():
    assert ea.question_type("In which year did it end?") == "which"
    assert ea.question_type("Name the capital.") == "other"
    assert [ea.answer_length_bin([a]) for a in ("one", "two words", "a b c d")] == ["1", "2-3", "4+"]


QIDS = ["a", "b", "c", "d"]
EVIDENCE = {q: {"gold_sent": 0} for q in QIDS}
GOLD = {"a": ["Paris"], "b": ["1998"], "c": ["Barack Obama"], "d": ["Paris"]}
PREDS = {"a": "Paris", "b": "1999", "c": "Obama", "d": "London"}
PRED_SENT = {"a": 0, "b": 0, "c": 0, "d": 2}


def test_analyse_and_table_are_mutually_exclusive():
    rec = ea.analyse(PREDS, PRED_SENT, EVIDENCE, GOLD, mock_ner)
    assert rec["a"] == {"correct": True, "category": None}
    assert [rec[q]["category"] for q in "bcd"] == ["numeric_date", "boundary", "wrong_sentence"]
    table = ea.category_table(rec)
    assert sum(v["count"] for v in table.values()) == 3
    assert sum(v["share"] for v in table.values()) == pytest.approx(1.0)


def test_slices_and_evidence_conditioned_accuracy():
    rec = ea.analyse(PREDS, PRED_SENT, EVIDENCE, GOLD, mock_ner)
    s = ea.slice_accuracy(rec, {"a": "who", "b": "when", "c": "who", "d": "who"})
    assert s["who"] == {"n": 3, "accuracy": pytest.approx(1 / 3)}
    cond = ea.evidence_conditioned_accuracy(rec, PRED_SENT, EVIDENCE)
    assert cond["top1_correct"] == {"n": 3, "accuracy": pytest.approx(1 / 3)}
    assert cond["top1_wrong"] == {"n": 1, "accuracy": 0.0}


def test_audit_sheet_samples_only_errors_with_blank_label(tmp_path):
    rec = ea.analyse(PREDS, PRED_SENT, EVIDENCE, GOLD, mock_ner)
    questions = {q: f"question {q}?" for q in QIDS}
    path = tmp_path / "audit.csv"
    sample = ea.write_audit_sheet(path, rec, questions, PREDS, GOLD, n=2, seed=1)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 2 and set(sample) <= {"b", "c", "d"}
    assert all(r["human_label"] == "" and r["auto_category"] for r in rows)


def test_plot_category_shares_writes_png(tmp_path):
    rec = ea.analyse(PREDS, PRED_SENT, EVIDENCE, GOLD, mock_ner)
    table = ea.category_table(rec)
    path = tmp_path / "categories.png"
    ea.plot_category_shares({"span-only": table, "full": table}, path)
    assert path.read_bytes().startswith(b"\x89PNG")
