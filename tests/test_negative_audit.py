"""negative_audit: blinded J3 sheet from a mined-negatives dump, and label counting."""
import json

import pytest

from qa.eval.negative_audit import build_sheet, count_labels, main, read_sheet

SENTS = ["S0.", "S1.", "S2.", "S3."]
# q1: three different negatives. q2: mined == lex. q3: no candidates (all null).
DUMP = [
    {"id": "q1", "question": "Q1?", "gold_sent": 0, "mined_neg": 1, "lex_neg": 2, "rand_neg": 3,
     "sentences": SENTS},
    {"id": "q2", "question": "Q2?", "gold_sent": 3, "mined_neg": 0, "lex_neg": 0, "rand_neg": 1,
     "sentences": SENTS},
    {"id": "q3", "question": "Q3?", "gold_sent": 0, "mined_neg": None, "lex_neg": None,
     "rand_neg": None, "sentences": ["Only."]},
]
ANSWERS = {"q1": "a1", "q2": "a2", "q3": "a3"}


def test_one_gold_row_and_one_row_per_distinct_negative():
    sheet, key = build_sheet(DUMP, ANSWERS)
    items = [(r["qid"], r["item"]) for r in sheet]
    assert items == [("q1", "G"), ("q1", "N1"), ("q1", "N2"), ("q1", "N3"),
                     ("q2", "G"), ("q2", "N1"), ("q2", "N2"), ("q3", "G")]
    shared = [v for v in key["q2"]["items"].values() if v["sent"] == 0]
    assert shared == [{"sent": 0, "methods": ["mined", "lex"]}]
    assert key["q3"]["items"] == {}


def test_sheet_text_matches_key_and_hides_methods():
    sheet, key = build_sheet(DUMP, ANSWERS)
    sents = {row["id"]: row["sentences"] for row in DUMP}
    for r in sheet:
        assert "mined" not in json.dumps(r) and "lex" not in json.dumps(r)
        s = sents[r["qid"]]
        if r["item"] == "G":
            assert r["sentence"] == s[key[r["qid"]]["gold_sent"]]
        else:
            assert r["sentence"] == s[key[r["qid"]]["items"][r["item"]]["sent"]]
            assert r["gold_sentence"] == s[key[r["qid"]]["gold_sent"]]
            assert r["answer"] == ANSWERS[r["qid"]]


def test_shuffle_is_seeded():
    assert build_sheet(DUMP, ANSWERS, seed=1) == build_sheet(DUMP, ANSWERS, seed=1)
    orders = {json.dumps(build_sheet(DUMP, ANSWERS, seed=s)[1]["q1"]) for s in range(20)}
    assert len(orders) > 1


def test_negative_equal_to_gold_is_rejected():
    bad = [{**DUMP[0], "lex_neg": 0}]
    with pytest.raises(ValueError, match="gold"):
        build_sheet(bad, ANSWERS)


def label(sheet, key, by_sent):
    """Fill labels: gold rows Y, negatives by sentence index."""
    for r in sheet:
        r["label"] = "Y" if r["item"] == "G" else by_sent[key[r["qid"]]["items"][r["item"]]["sent"]]
    return sheet


def test_shared_sentence_counts_for_both_methods():
    sheet, key = build_sheet(DUMP, ANSWERS)
    counts = count_labels(label(sheet, key, {0: "F", 1: "P", 2: "e", 3: "E"}), key)
    assert counts["gold_correct"] == 3 and counts["gold_wrong"] == []
    m = counts["methods"]
    assert m["mined"] == {"n": 2, "P": 1, "F": 1, "E": 0}  # q1 S1=P, q2 S0=F
    assert m["lex"] == {"n": 2, "P": 0, "F": 1, "E": 1}    # q1 S2=E (lower-case ok), q2 S0=F
    assert m["rand"] == {"n": 2, "P": 1, "F": 0, "E": 1}   # q1 S3=E, q2 S1=P


def test_paired_plausible_mcnemar():
    from qa.eval.negative_audit import paired_plausible
    # 5 questions only mined is P, 1 only rand is P, 2 both: exact p = 2 * 7/64.
    per_q = {f"q{i}": {"mined": "P", "rand": "E"} for i in range(5)}
    per_q["q5"] = {"mined": "E", "rand": "P"}
    per_q["q6"] = per_q["q7"] = {"mined": "P", "rand": "P"}
    per_q["q8"] = {"mined": "P"}  # no rand negative: excluded
    assert paired_plausible(per_q, "mined", "rand") == {"n": 8, "only_a": 5, "only_b": 1,
                                                         "p": round(14 / 64, 4)}
    assert paired_plausible({"q": {"mined": "P", "lex": "P"}}, "mined", "lex")["p"] == 1.0
    sheet, key = build_sheet(DUMP, ANSWERS)
    counts = count_labels(label(sheet, key, {0: "F", 1: "P", 2: "E", 3: "E"}), key)
    assert counts["paired_plausible"]["mined_vs_lex"] == {"n": 2, "only_a": 1, "only_b": 0,
                                                         "p": 1.0}


def test_blank_or_invalid_labels_are_reported():
    sheet, key = build_sheet(DUMP, ANSWERS)
    sheet = label(sheet, key, {0: "P", 1: "P", 2: "P", 3: "P"})
    sheet[1]["label"] = ""
    sheet[0]["label"] = "P"  # gold rows take Y/N
    with pytest.raises(ValueError, match="2 rows need fixing"):
        count_labels(sheet, key)
    with pytest.raises(ValueError, match="row missing"):
        count_labels([r for r in sheet if r["qid"] != "q3"], key)


def test_agreement_with_drafts_is_reported():
    sheet, key = build_sheet(DUMP, ANSWERS)
    sheet = label(sheet, key, {0: "P", 1: "P", 2: "E", 3: "P"})
    for r in sheet:
        r["draft"] = r["label"]
    sheet[1]["draft"] = "F"  # annotator overrode one draft
    a = count_labels(sheet, key)["llm_draft_agreement"]
    assert a["n_drafted"] == len(sheet) and a["kept"] == len(sheet) - 1
    assert a["changed"] == [f"q1 {sheet[1]['item']}: F -> {sheet[1]['label']}"]
    assert count_labels(label(*build_sheet(DUMP, ANSWERS), {0: "P", 1: "P", 2: "P", 3: "P"}),
                        key)["llm_draft_agreement"] is None


def test_cli_round_trip_and_refuses_to_overwrite(tmp_path):
    dump = tmp_path / "dump.jsonl"
    dump.write_text("".join(json.dumps(r) + "\n" for r in DUMP))
    squad = tmp_path / "squad.json"
    squad.write_text(json.dumps({"data": [{"title": "T", "paragraphs": [
        {"context": "S0. S1. S2. S3.", "qas": [
            {"id": q, "question": "?", "answers": [{"text": "S0", "answer_start": 0}]}
            for q in ("q1", "q2", "q3")]}]}]}))
    prefix = str(tmp_path / "audits" / "j3")
    main(["make-sheet", "--dump", str(dump), "--squad", str(squad), "--prefix", prefix])
    with pytest.raises(SystemExit, match="--force"):
        main(["make-sheet", "--dump", str(dump), "--squad", str(squad), "--prefix", prefix])

    sheet = read_sheet(f"{prefix}_sheet.csv")
    key = json.loads((tmp_path / "audits" / "j3_key.json").read_text())
    import csv
    with open(f"{prefix}_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(label(sheet, key, {0: "P", 1: "F", 2: "E", 3: "P"}))
    main(["count", "--prefix", prefix])
    counts = json.loads((tmp_path / "audits" / "j3_counts.json").read_text())
    assert counts["n_questions"] == 3 and counts["methods"]["mined"]["n"] == 2
