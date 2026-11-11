"""Automatic error analysis for extractive QA predictions (work package C8).

Every question the model gets wrong (EM = 0) falls into exactly one category, tested in order:
  1 wrong_sentence  predicted span lies outside the gold sentence
  2 boundary        right sentence, but the span overlaps a gold answer (EM = 0)
  3 numeric_date    gold answer is a number or date
  4 entity          gold answer is a named entity and the prediction is a different span
  5 other

Inputs follow the team contracts (TEAM_TASKS 3.1):
  preds:    {qid: answer_text}                       (C6 predictions)
  pred_sent: {qid: sent_idx of the predicted span}   (top n-best entry of C6 nbest.json)
  evidence: {qid: {"gold_sent": int, ...}}           (C3)
  gold:     {qid: [answer_text, ...]}                (SQuAD references)

Usage:
  python -m qa.eval.error_analysis --predictions outputs/<run_id>/val.joint_lam0.25.predictions.json
      --nbest outputs/<run_id>/val.joint_lam0.25.nbest.json --evidence data/processed/evidence_val.jsonl
      --data-file data/processed/val.json [--sent-rank outputs/<run_id>/val.sent_rank.json] [--spacy]

Writes ``<split>.error_analysis.json`` (category table, question-type and answer-length slices,
evidence-conditioned accuracy) and ``<split>.error_audit.csv`` (the sample for the manual audit).
Evidence-conditioned accuracy uses the C8 top-1 sentence with --sent-rank (full model) and the
sentence of the predicted span without it (span-implied). The test split needs ``--allow-test``.
"""
import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from qa.eval.official import load_official_metrics as _official

CATEGORIES = ("wrong_sentence", "boundary", "numeric_date", "entity", "other")
NUMERIC_DATE_LABELS = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
WH_WORDS = ("what", "who", "whom", "whose", "when", "where", "which", "why", "how")
_NUMERIC_RE = re.compile(r"^[\s$£€~]*\d[\d,.\s]*(%|st|nd|rd|th|s)?(\s+(thousand|million|billion|percent|dollars))?$", re.I)


def spacy_entity_label(model="en_core_web_sm"):
    """Return `entity_label(text) -> NER label | None`: the label if one entity covers the text.

    Needs spaCy and the `model` package. Callers may pass any function with the same shape.
    """
    import spacy

    nlp = spacy.load(model, disable=["parser", "lemmatizer"])

    def entity_label(text):
        ents = nlp(text).ents
        if len(ents) == 1 and len(ents[0].text) >= 0.8 * len(text.strip()):
            return ents[0].label_
        return None

    return entity_label


def regex_entity_label(text):
    """Fallback without spaCy: recognises numbers only (no entity types)."""
    return "CARDINAL" if _NUMERIC_RE.match(text.strip()) else None


def overlaps(gold_answers, pred):
    """True if the prediction shares at least one normalised token with some gold answer."""
    official = _official()
    pred_tokens = set(official.get_tokens(pred))
    return any(pred_tokens & set(official.get_tokens(g)) for g in gold_answers)


def categorise(pred, pred_sent, gold_sent, gold_answers, entity_label):
    """Category for one wrong prediction (see module docstring)."""
    if pred_sent is None or int(pred_sent) != int(gold_sent):  # None: nothing was predicted
        return "wrong_sentence"
    if overlaps(gold_answers, pred):
        return "boundary"
    labels = {entity_label(g) for g in gold_answers} - {None}
    if labels & NUMERIC_DATE_LABELS:
        return "numeric_date"
    if labels:
        return "entity"
    return "other"


def question_type(question):
    """First wh-word found in the question, else 'other'."""
    tokens = re.findall(r"[a-z]+", question.lower())
    return next((t for t in tokens if t in WH_WORDS), "other")


def answer_length_bin(gold_answers):
    n = len((gold_answers or [""])[0].split())
    return "1" if n == 1 else "2-3" if n <= 3 else "4+"


def analyse(preds, pred_sent, evidence, gold, entity_label=regex_entity_label):
    """Per-question records: {qid: {correct, category}} (category None when correct)."""
    official = _official()
    out = {}
    for qid, ev in evidence.items():
        golds = gold[qid] or [""]  # the official script maps empty gold to ['']
        correct = bool(max(official.compute_exact(g, preds[qid]) for g in golds))
        category = None if correct else categorise(
            preds[qid], pred_sent[qid], ev["gold_sent"], golds, entity_label
        )
        out[qid] = {"correct": correct, "category": category}
    return out


def category_table(records):
    """Counts and share of errors per category."""
    errors = [r["category"] for r in records.values() if not r["correct"]]
    n_err = len(errors)
    return {
        c: {"count": errors.count(c), "share": errors.count(c) / n_err if n_err else 0.0}
        for c in CATEGORIES
    }


def slice_accuracy(records, keys):
    """Accuracy per slice value. keys: {qid: slice value} (e.g. question_type, n_windows bin)."""
    groups = defaultdict(list)
    for qid, r in records.items():
        groups[keys[qid]].append(r["correct"])
    return {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(groups.items())}


def evidence_conditioned_accuracy(records, top1_sent, evidence):
    """P(correct | top-1 sentence is gold) vs P(correct | top-1 is wrong).

    top1_sent: {qid: sentence index} from C8 ranking (full model) or span-implied (span-only).
    """
    right, wrong = [], []
    for qid, r in records.items():
        (right if top1_sent[qid] == evidence[qid]["gold_sent"] else wrong).append(r["correct"])

    def acc(xs):
        return {"n": len(xs), "accuracy": sum(xs) / len(xs) if xs else None}

    return {"top1_correct": acc(right), "top1_wrong": acc(wrong)}


def plot_category_shares(tables, path):
    """Horizontal bars of each error category's share of errors, one bar per system.

    tables: {system label: category_table(...)}, e.g. span-only vs. full model.
    """
    from matplotlib.figure import Figure

    height = 0.8 / len(tables)
    fig = Figure(figsize=(6, 3.2))
    ax = fig.subplots()
    for i, (label, table) in enumerate(tables.items()):
        ys = [j + i * height for j in range(len(CATEGORIES))]
        shares = [100 * table[c]["share"] for c in CATEGORIES]
        ax.barh(ys, shares, height=height, label=label)
    ax.set_yticks([j + 0.4 - height / 2 for j in range(len(CATEGORIES))], CATEGORIES)
    ax.invert_yaxis()
    ax.set_xlabel("share of errors (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _safe_cell(value):
    """Prefix a quote so spreadsheets do not evaluate text starting with = + - @ as a formula."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def write_audit_sheet(path, records, questions, preds, gold, n=100, seed=42):
    """CSV of up to n sampled errors for the human audit. `human_label` is left blank."""
    errors = sorted(qid for qid, r in records.items() if not r["correct"])
    sample = random.Random(seed).sample(errors, min(n, len(errors)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question", "gold", "prediction", "auto_category", "human_label"])
        for qid in sample:
            w.writerow([_safe_cell(c) for c in (
                qid, questions[qid], " ||| ".join(gold[qid]), preds[qid], records[qid]["category"], "")])
    return sample


def _read_squad(squad_file):
    """(questions, gold answers) by question id."""
    data = json.loads(Path(squad_file).read_text())["data"]
    qas = [qa for article in data for p in article["paragraphs"] for qa in p["qas"]]
    return ({qa["id"]: qa["question"] for qa in qas},
            {qa["id"]: [a["text"] for a in qa["answers"]] for qa in qas})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--predictions", required=True, help="C6 <split>.*predictions.json")
    parser.add_argument("--nbest", required=True, help="C6 nbest with sent_idx (top entry = the prediction)")
    parser.add_argument("--evidence", required=True, help="C3 evidence_<split>.jsonl")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--sent-rank", help="C8 ranking; without it the span-implied sentence is used")
    parser.add_argument("--spacy", action="store_true", help="NER labels from spaCy (else numbers only)")
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument("--out-dir")
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if Path(args.data_file).name.lower() == "dev-v1.1.json" and not args.allow_test:
        raise SystemExit("Refusing to analyse test (dev-v1.1.json). Only final checkpoints, via "
                         "scripts/run_all.sh, which passes --allow-test.")
    preds = json.loads(Path(args.predictions).read_text())
    nbest = json.loads(Path(args.nbest).read_text())
    evidence = {rec["id"]: rec for rec in map(json.loads, Path(args.evidence).read_text().splitlines()) if rec}
    questions, gold = _read_squad(args.data_file)
    pred_sent = {qid: (entries[0]["sent_idx"] if entries else None) for qid, entries in nbest.items()}
    entity_label = spacy_entity_label() if args.spacy else regex_entity_label

    records = analyse(preds, pred_sent, evidence, gold, entity_label)
    if args.sent_rank:
        ranking = json.loads(Path(args.sent_rank).read_text())
        top1 = {qid: int(ranking[qid][0][0]) for qid in evidence}
    else:
        top1 = {qid: pred_sent[qid] for qid in evidence}
    out = {
        "n": len(records),
        "errors": sum(not r["correct"] for r in records.values()),
        "category_table": category_table(records),
        "question_type": slice_accuracy(records, {q: question_type(questions[q]) for q in records}),
        "answer_length": slice_accuracy(records, {q: answer_length_bin(gold[q]) for q in records}),
        "evidence_conditioned": evidence_conditioned_accuracy(records, top1, evidence),
        "top1": "c8_ranking" if args.sent_rank else "span_implied",
    }
    split = Path(args.predictions).name.split(".")[0]
    out_dir = Path(args.out_dir or Path(args.predictions).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{split}.error_analysis.json"
    report.write_text(json.dumps(out, indent=2))
    audit = out_dir / f"{split}.error_audit.csv"
    write_audit_sheet(audit, records, questions, preds, gold, n=args.audit_size)
    print(json.dumps(out["category_table"], indent=2))
    print(f"report: {report}\naudit: {audit}")


if __name__ == "__main__":
    main()
