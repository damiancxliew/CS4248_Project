"""J3 label and negative audit: build a blinded annotation sheet, then count the labels.

Input is the model-mined negatives dump (G1, ``evidence_model dump-negatives``): one JSONL row
per question with ``gold_sent``, ``mined_neg``, ``lex_neg``, ``rand_neg`` (C3 sentence indices,
null = none) and the passage ``sentences``.

``make-sheet`` writes two files:
  - ``<prefix>_sheet.csv``: what the human labels. Per question, one ``G`` row (the gold
    sentence; label ``Y`` if it is the right evidence, else ``N``) and one ``N1``, ``N2``, ...
    row per *distinct* negative sentence, in shuffled order, with no method shown, so the
    annotator cannot tell which method chose it. Label each ``P`` (plausible distractor: looks
    relevant, does not answer), ``F`` (false negative: actually answers the question) or ``E``
    (easy: clearly irrelevant).
  - ``<prefix>_key.json``: which method(s) chose each ``N`` row. Don't open it while labelling.
A sentence chosen by two methods is one row, so it gets one label, counted for both methods.

``count`` reads the labelled sheet and the key and writes ``<prefix>_counts.json``: gold
accuracy, and per method the P/F/E counts over the questions that have that negative.

Usage:
  PYTHONPATH=src python -m qa.eval.negative_audit make-sheet \\
      --dump outputs/mined_negatives_train.jsonl --squad data/processed/train.json
  PYTHONPATH=src python -m qa.eval.negative_audit make-page   # j3_label.html: label in a browser
  PYTHONPATH=src python -m qa.eval.negative_audit count [--sheet ~/Downloads/j3_sheet.csv]
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

from qa.data.squad_io import load_squad

METHODS = ("mined", "lex", "rand")
GOLD_LABELS = ("Y", "N")
NEG_LABELS = ("P", "F", "E")
COLUMNS = ("qid", "item", "question", "answer", "gold_sentence", "sentence", "label", "note")
DEFAULT_PREFIX = "results/audits/j3"


def read_dump(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def answers_by_id(squad_path):
    """Question id -> ``" | "``-joined distinct reference answers."""
    out = {}
    for article in load_squad(squad_path)["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                texts = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                out[qa["id"]] = " | ".join(texts)
    return out


def build_sheet(rows, answers, seed=2026):
    """``(sheet_rows, key)``. Negative rows are deduplicated per question and shuffled with
    ``seed`` (one RNG per question id, so a question's order doesn't depend on the others)."""
    sheet, key = [], {}
    for row in rows:
        qid, sentences = row["id"], row["sentences"]
        gold = sentences[row["gold_sent"]]
        base = {"qid": qid, "question": row["question"], "answer": answers.get(qid, ""),
                "label": "", "note": ""}
        sheet.append({**base, "item": "G", "gold_sentence": "", "sentence": gold})

        chosen = {}
        for method in METHODS:
            idx = row[f"{method}_neg"]
            if idx is None:
                continue
            if idx == row["gold_sent"]:
                raise ValueError(f"{qid}: {method} negative is the gold sentence")
            chosen.setdefault(idx, []).append(method)
        order = sorted(chosen)
        random.Random(f"{seed}:{qid}").shuffle(order)

        key[qid] = {"gold_sent": row["gold_sent"], "items": {}}
        for n, idx in enumerate(order, 1):
            item = f"N{n}"
            key[qid]["items"][item] = {"sent": idx, "methods": chosen[idx]}
            sheet.append({**base, "item": item, "gold_sentence": gold, "sentence": sentences[idx]})
    return sheet, key


def write_sheet(sheet, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sheet)


def read_sheet(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_drafts(path):
    """``(qid, item) -> (draft label, reason)`` from a TSV with columns qid, item, draft, reason."""
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["qid"], r["item"]): (r["draft"], r["reason"])
                for r in csv.DictReader(f, delimiter="\t")}


def write_page(sheet, path, drafts=None):
    """A self-contained HTML labelling page (one question per screen) holding the sheet rows,
    but not the key. ``drafts`` (optional) are shown as suggestions the annotator must accept
    or override. It exports the sheet's columns plus ``draft`` and ``reason``."""
    template = (Path(__file__).with_name("negative_audit_page.html")).read_text(encoding="utf-8")
    drafts = drafts or {}
    rows = []
    for r in sheet:
        draft, reason = drafts.get((r["qid"], r["item"]), ("", ""))
        rows.append({**{c: r.get(c, "") for c in COLUMNS}, "draft": draft, "reason": reason})
    data = json.dumps(rows).replace("</", "<\\/")
    Path(path).write_text(template.replace("__DATA__", data), encoding="utf-8")


def paired_plausible(per_question, a, b):
    """Exact two-sided McNemar test on "labelled P", over questions that have both methods'
    negatives. ``only_a`` / ``only_b``: questions where only that method's negative is P."""
    pairs = [(m[a] == "P", m[b] == "P") for m in per_question.values() if a in m and b in m]
    only_a = sum(x and not y for x, y in pairs)
    only_b = sum(y and not x for x, y in pairs)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n": len(pairs), "only_a": only_a, "only_b": only_b, "p": round(p, 4)}


def count_labels(sheet, key):
    """Counts per method over the questions that have that negative. Raises on any blank,
    invalid or unknown label, listing the offending rows."""
    labels, problems = {}, []
    for r in sheet:
        qid, item, label = r["qid"], r["item"], r["label"].strip().upper()
        allowed = GOLD_LABELS if item == "G" else NEG_LABELS
        if qid not in key or (item != "G" and item not in key[qid]["items"]):
            problems.append(f"{qid} {item}: not in the key")
        elif label not in allowed:
            problems.append(f"{qid} {item}: label {r['label']!r}, expected one of {'/'.join(allowed)}")
        labels[(qid, item)] = label
    missing = [f"{qid} {item}: row missing from the sheet"
               for qid, entry in key.items() for item in ["G", *entry["items"]]
               if (qid, item) not in labels]
    if problems or missing:
        raise ValueError(f"{len(problems) + len(missing)} rows need fixing:\n  "
                         + "\n  ".join(problems + missing))

    agreement = None
    drafted = [r for r in sheet if r.get("draft")]
    if drafted:
        changed = [f"{r['qid']} {r['item']}: {r['draft']} -> {r['label'].strip().upper()}"
                   for r in drafted if r["draft"] != r["label"].strip().upper()]
        agreement = {"n_drafted": len(drafted), "kept": len(drafted) - len(changed),
                     "changed": changed}

    gold_ok = sum(labels[(qid, "G")] == "Y" for qid in key)
    methods = {m: {"n": 0, **{lab: 0 for lab in NEG_LABELS}} for m in METHODS}
    per_question = {qid: {} for qid in key}
    for qid, entry in key.items():
        for item, info in entry["items"].items():
            for m in info["methods"]:
                methods[m]["n"] += 1
                methods[m][labels[(qid, item)]] += 1
                per_question[qid][m] = labels[(qid, item)]
    paired = {f"{a}_vs_{b}": paired_plausible(per_question, a, b)
              for a, b in (("mined", "rand"), ("mined", "lex"), ("lex", "rand"))}
    return {
        "n_questions": len(key),
        "gold_correct": gold_ok,
        "gold_wrong": [qid for qid in key if labels[(qid, "G")] == "N"],
        "methods": methods,
        "paired_plausible": paired,
        "label_meaning": {"P": "plausible distractor", "F": "false negative", "E": "easy/irrelevant"},
        "llm_draft_agreement": agreement,
    }


def summary_lines(counts):
    lines = [f"Gold sentence is the right evidence: {counts['gold_correct']}/{counts['n_questions']}"]
    names = {"mined": "model-mined", "lex": "lexical", "rand": "random"}
    for m, c in counts["methods"].items():
        n = c["n"]
        pct = lambda k: f"{100 * c[k] / n:.0f}%" if n else "-"  # noqa: E731
        lines.append(f"{names[m]:>11} (n={n}): plausible {c['P']} ({pct('P')}), "
                     f"false negative {c['F']} ({pct('F')}), easy {c['E']} ({pct('E')})")
    for pair, t in counts.get("paired_plausible", {}).items():
        a_name, b_name = (names[m] for m in pair.split("_vs_"))
        lines.append(f"Plausible, {a_name} vs {b_name} (McNemar, n={t['n']}): "
                     f"only {a_name} {t['only_a']}, only {b_name} {t['only_b']}, p={t['p']}")
    a = counts.get("llm_draft_agreement")
    if a:
        lines.append(f"Kept {a['kept']}/{a['n_drafted']} LLM draft labels; changed {len(a['changed'])}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="J3 label and negative audit.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-sheet", help="blinded annotation sheet + key")
    make.add_argument("--dump", default="outputs/mined_negatives_train.jsonl")
    make.add_argument("--squad", default="data/processed/train.json",
                      help="SQuAD file holding the dumped questions (for the reference answers)")
    make.add_argument("--prefix", default=DEFAULT_PREFIX)
    make.add_argument("--seed", type=int, default=2026)
    make.add_argument("--force", action="store_true", help="overwrite an existing sheet")
    page = commands.add_parser("make-page", help="HTML labelling page from the sheet")
    page.add_argument("--prefix", default=DEFAULT_PREFIX)
    page.add_argument("--drafts", help="TSV of draft labels to show as suggestions "
                      "(default: <prefix>_drafts.tsv if it exists)")
    count = commands.add_parser("count", help="count the labels in a filled-in sheet")
    count.add_argument("--prefix", default=DEFAULT_PREFIX)
    count.add_argument("--sheet", help="labelled CSV (default: <prefix>_sheet.csv)")
    args = parser.parse_args(argv)

    sheet_path, key_path = Path(f"{args.prefix}_sheet.csv"), Path(f"{args.prefix}_key.json")
    if args.command == "make-page":
        out = Path(f"{args.prefix}_label.html")
        drafts_path = Path(args.drafts or f"{args.prefix}_drafts.tsv")
        drafts = read_drafts(drafts_path) if drafts_path.exists() else None
        write_page(read_sheet(sheet_path), out, drafts)
        print(f"Wrote {out}. Open it in a browser.")
    elif args.command == "make-sheet":
        if sheet_path.exists() and not args.force:
            raise SystemExit(f"{sheet_path} exists and may hold labels; pass --force to overwrite.")
        sheet, key = build_sheet(read_dump(args.dump), answers_by_id(args.squad), args.seed)
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        write_sheet(sheet, sheet_path)
        key_path.write_text(json.dumps(key, indent=1) + "\n")
        n_neg = sum(r["item"] != "G" for r in sheet)
        print(f"Wrote {sheet_path} ({len(key)} questions, {n_neg} negative rows) and {key_path}")
    else:
        labelled = Path(args.sheet) if args.sheet else sheet_path
        counts = count_labels(read_sheet(labelled), json.loads(key_path.read_text()))
        out = Path(f"{args.prefix}_counts.json")
        out.write_text(json.dumps(counts, indent=1) + "\n")
        print("\n".join(summary_lines(counts)))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
