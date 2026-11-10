"""Sentence-ranking quality (J6b): Recall@1, Recall@3 and MRR of a C8 ranking against C3 gold.

For each question, the rank of the gold sentence (``gold_sent``, C3) in the system's C8 ranking
``{qid: [[sent_idx, score], ...]}`` gives three scores: Recall@k = 1 if the rank is k or better,
MRR = 1 / rank. A system's metric is the mean over the split's questions. This is the definition
Gokul's ``g3_grid.sh summary`` uses, so the G3 tuning numbers and the ranking table agree.

Systems (TEAM_TASKS §6) and where their ranking comes from, under ``outputs/``:

- BM25 (R1): ``r1_bm25/<split>.sent_rank.json`` (``qa.baselines.bm25_ranker``).
- Span-implied (R3): no ranker; ``span_implied_ranking`` orders sentences by the best span the
  span-only model puts in each, from ``r3_deberta_span_s<seed>/<split>.nbest.json`` (C6; needs
  ``sent_idx``, i.e. ``predict.py --evidence``). Rank 1 is the sentence holding its top span.
- Our ranker (R4, R5, R6, R6b): ``<run_id>_s<seed>/<split>.sent_rank.json`` (``evidence_decoder``).

  python -m qa.eval.evidence_metrics score --ranking outputs/r1_bm25/val.sent_rank.json --split val
  python -m qa.eval.evidence_metrics score ... --scores-out r1_val_r1.json  # for significance.py
  python -m qa.eval.evidence_metrics table --split val  # -> results/tables/ranking_<split>.{csv,tex}

``score`` prints one ranking's metrics; ``--scores-out`` also writes its per-question Recall@1
(``{qid: 0 or 1}``), the input ``qa.eval.significance --metric recall_at_1 --units fraction``
reads. ``table`` scores every system and seed it finds, and reports mean ± sample std over seeds
in percent (MRR × 100 too); systems with no ranking file are left out and named on stderr. The
test set (dev-v1.1.json) is refused unless ``--allow-test`` is passed, as in ``predict.py``.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

EVIDENCE_DIR = "data/processed"
OUTPUT_DIR = "outputs"
TABLE_DIR = "results/tables"
SEEDS = (13, 42, 87)
METRICS = ("recall_at_1", "recall_at_3", "mrr")

# (label, run_id, seeded, kind): kind "rank" reads C8, "span" builds the span-implied ranking from C6.
SYSTEMS = (
    ("BM25 (R1)", "r1_bm25", False, "rank"),
    ("Span-implied (R3)", "r3_deberta_span", True, "span"),
    ("Full model (R4)", "r4_deberta_evidence", True, "rank"),
    ("No negatives (R5)", "r5_deberta_evidence", True, "rank"),
    ("Random negatives (R6)", "r6_deberta_evidence", True, "rank"),
    ("Lexical negatives (R6b)", "r6b_deberta_evidence", True, "rank"),
)


def gold_ranks(ranking, evidence):
    """``{qid: rank of the gold sentence}``, 1 = first, over every question in ``evidence``.

    Refuses a ranking that misses or adds questions (wrong split) or that does not list each
    passage sentence exactly once (C8 covers every sentence), since either would skew the means.
    """
    missing, extra = set(evidence) - set(ranking), set(ranking) - set(evidence)
    if missing or extra:
        raise ValueError(f"ranking and evidence disagree: {len(missing)} question(s) missing "
                         f"(e.g. {sorted(missing)[:3]}), {len(extra)} extra (e.g. {sorted(extra)[:3]})")
    ranks = {}
    for qid, rec in evidence.items():
        order = [int(sent) for sent, _ in ranking[qid]]
        if sorted(order) != list(range(len(rec["sentences"]))):
            raise ValueError(f"{qid}: ranking lists sentences {order}, but the passage has "
                             f"{len(rec['sentences'])}; C8 lists every sentence exactly once")
        ranks[qid] = order.index(int(rec["gold_sent"])) + 1
    return ranks


def per_question(ranks):
    """Per-question scores ``{metric: {qid: score}}``: Recall@k is 0 or 1, MRR is 1 / rank."""
    return {
        "recall_at_1": {qid: int(rank <= 1) for qid, rank in ranks.items()},
        "recall_at_3": {qid: int(rank <= 3) for qid, rank in ranks.items()},
        "mrr": {qid: 1 / rank for qid, rank in ranks.items()},
    }


def summarize(ranks):
    """``{"n", "recall_at_1", "recall_at_3", "mrr"}``, the metrics as fractions (0..1)."""
    if not ranks:
        raise ValueError("no questions to score")
    scores = per_question(ranks)
    return {"n": len(ranks), **{m: mean(scores[m].values()) for m in METRICS}}


def span_implied_ranking(nbest, evidence):
    """C8-style ranking implied by a span-only model's C6 n-best.

    A sentence scores the best ``start_logit + end_logit`` among the n-best spans starting in it
    (the score the span-only decoder picks its answer by), so rank 1 is the sentence holding the
    top span. Sentences with no n-best span follow, in passage order; spans in no sentence
    (``sent_idx`` null) are skipped.
    """
    missing = set(evidence) - set(nbest)
    if missing:
        raise ValueError(f"nbest is missing {len(missing)} question(s), e.g. {sorted(missing)[:3]}")
    ranking = {}
    for qid, rec in evidence.items():
        entries = nbest[qid]
        if entries and all(e["sent_idx"] is None for e in entries):
            raise ValueError(f"{qid}: no n-best span has a sent_idx; was predict.py run with --evidence?")
        best = {}
        for e in entries:
            if e["sent_idx"] is not None:
                sent, score = int(e["sent_idx"]), e["start_logit"] + e["end_logit"]
                best[sent] = max(score, best.get(sent, float("-inf")))
        scored = sorted(best, key=lambda s: (-best[s], s))
        rest = [s for s in range(len(rec["sentences"])) if s not in best]
        ranking[qid] = [[s, float(best[s])] for s in scored] + [[s, None] for s in rest]
    return ranking


def read_evidence(path):
    with open(path) as f:
        return {rec["id"]: rec for rec in map(json.loads, f) if rec}


def read_ranking(path, kind, evidence):
    """The C8 ranking stored at ``path`` (``kind`` "rank") or implied by its C6 n-best ("span")."""
    data = json.loads(Path(path).read_text())
    return span_implied_ranking(data, evidence) if kind == "span" else data


def system_files(run_id, seeded, kind, split, output_dir):
    """``[(seed or None, path)]`` of the system's ranking files that exist."""
    name = f"{split}.nbest.json" if kind == "span" else f"{split}.sent_rank.json"
    dirs = [(seed, f"{run_id}_s{seed}") for seed in SEEDS] if seeded else [(None, run_id)]
    paths = [(seed, Path(output_dir) / d / name) for seed, d in dirs]
    return [(seed, path) for seed, path in paths if path.is_file()]


def table_rows(split, evidence, output_dir=OUTPUT_DIR, systems=SYSTEMS):
    """One row per system with any ranking file: mean and sample std over seeds, in percent."""
    rows = []
    for label, run_id, seeded, kind in systems:
        found = system_files(run_id, seeded, kind, split, output_dir)
        if not found:
            print(f"{label}: no {split} ranking under {output_dir}/{run_id}*; left out", file=sys.stderr)
            continue
        runs = [summarize(gold_ranks(read_ranking(path, kind, evidence), evidence)) for _, path in found]
        row = {"system": label, "run_id": run_id, "seeds": " ".join(str(s) for s, _ in found if s is not None),
               "n": runs[0]["n"]}
        for m in METRICS:
            values = [100 * run[m] for run in runs]
            row[f"{m}_mean"] = mean(values)
            row[f"{m}_std"] = stdev(values) if len(values) > 1 else None
        rows.append(row)
    return rows


def write_table(rows, split, out_dir=TABLE_DIR):
    """``ranking_<split>.csv`` and ``ranking_<split>.tex``; returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = out_dir / f"ranking_{split}.csv", out_dir / f"ranking_{split}.tex"
    fields = ["system", "run_id", "seeds", "n", *(f"{m}_{s}" for m in METRICS for s in ("mean", "std"))]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def cell(row, m):
        sd = row[f"{m}_std"]
        return f"{row[f'{m}_mean']:.2f}" + ("" if sd is None else f" $\\pm$ {sd:.2f}")

    lines = [f"% Generated by qa.eval.evidence_metrics table --split {split}; percent, "
             "mean $\\pm$ sample std over seeds (none for a single run).",
             r"\begin{tabular}{lccc}", r"\hline",
             r"System & Recall@1 & Recall@3 & MRR \\", r"\hline",
             *(f"{row['system']} & " + " & ".join(cell(row, m) for m in METRICS) + r" \\" for row in rows),
             r"\hline", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")
    return csv_path, tex_path


def _refuse_test(split, allow_test):
    if split == "test" and not allow_test:
        raise SystemExit("Refusing to score the test set (dev-v1.1.json) without --allow-test; "
                         "final runs go through scripts/run_all.sh.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "table"):
        sub = commands.add_parser(name)
        sub.add_argument("--split", required=True, help="val, test, addsent, addonesent")
        sub.add_argument("--evidence-dir", default=EVIDENCE_DIR, help="holds evidence_<split>.jsonl")
        sub.add_argument("--allow-test", action="store_true", help="final runs only (run_all.sh)")
    score = commands.choices["score"]
    score.add_argument("--ranking", required=True, help="C8 sent_rank.json, or C6 nbest.json with --span-implied")
    score.add_argument("--span-implied", action="store_true", help="--ranking is a span-only model's nbest.json")
    score.add_argument("--scores-out", help="write per-question Recall@1 here, for significance.py")
    table = commands.choices["table"]
    table.add_argument("--output-dir", default=OUTPUT_DIR)
    table.add_argument("--table-dir", default=TABLE_DIR)
    args = parser.parse_args(argv)
    _refuse_test(args.split, args.allow_test)
    evidence = read_evidence(Path(args.evidence_dir) / f"evidence_{args.split}.jsonl")

    if args.command == "score":
        ranks = gold_ranks(read_ranking(args.ranking, "span" if args.span_implied else "rank", evidence), evidence)
        if args.scores_out:
            Path(args.scores_out).write_text(json.dumps(per_question(ranks)["recall_at_1"]) + "\n")
        print(json.dumps({"split": args.split, "ranking": args.ranking, **summarize(ranks)}, indent=2))
    else:
        rows = table_rows(args.split, evidence, args.output_dir)
        if not rows:
            raise SystemExit(f"No ranking files for {args.split} under {args.output_dir}/")
        for path in write_table(rows, args.split, args.table_dir):
            print(path)


if __name__ == "__main__":
    main()
