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
"""
import csv
import random
import re
from collections import defaultdict

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
