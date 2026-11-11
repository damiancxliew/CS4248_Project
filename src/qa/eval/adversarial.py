"""Adversarial robustness (J5): every final system on AddSent / AddOneSent.

Each file holds the 1,000 original dev questions plus adversarial variants (``<id>-high-conf-turk<k>``)
whose passage is the original with one distractor sentence appended at the end. For one run:

- **EM / F1 (worst case):** official per-question scores, grouped by original id; a group scores the
  minimum over the original and its variants (Jia & Liang, 2017), averaged over the 1,000 groups.
- **Clean EM / F1 and drop:** the same 1,000 originals without a distractor; drop = clean − worst case.
- **(a) answer from the distractor:** share of adversarial questions whose predicted answer (top C6
  n-best entry) lies in the appended sentence.
- **(b) distractor ranked first:** for systems with a sentence ranking, share of adversarial questions
  whose rank-1 sentence is the appended one. Rankers use their C8 file; the span-only R3 uses its
  span-implied ranking (``evidence_metrics.span_implied_ranking`` on its C6 n-best, as in the J6
  table). R7 is R4's checkpoints decoded at λ = 0, so its ranker is R4's and (b) is left out (as R4).

The appended sentence is located by character offset, not by trusting the splitter: in a few passages
(92 of 2,560 in AddSent) the C3 splitter merges it with the original's last sentence. Its sentence is
the C3 sentence holding its first character. That sentence counts for (b) whether merged or not; for
(a), a merged sentence counts only if the predicted text occurs in the appended part.

Inputs, under ``outputs/adversarial/<run_id>_s<seed>/`` (``predict.py --evidence`` +
``evidence_decoder.py``): ``<split>.<pred>.predictions.json``, ``<split>.<pred>.nbest.json`` and, for
rankers, ``<split>.sent_rank.json``. BM25 (R1) rankings come from ``outputs/r1_bm25/``.

  python -m qa.eval.adversarial score --split addsent --run-dir outputs/adversarial/r4_deberta_evidence_s42 \\
      --predictions joint_lam0.25
  python -m qa.eval.adversarial table --split addsent   # -> results/tables/adversarial_<split>.{csv,tex}
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from qa.eval.evidence_metrics import span_implied_ranking
from qa.eval.official import load_official_metrics

DATA_DIR = "data/processed"
ADV_DIR = "outputs/adversarial"
BM25_DIR = "outputs/r1_bm25"
TABLE_DIR = "results/tables"
SEEDS = (13, 42, 87)
SPLITS = ("addsent", "addonesent")

# (label, run_id, predictions name or None, ranking): ranking None = none, "rank" = the run's C8
# <split>.sent_rank.json, "span" = span-implied from the run's span-decoder <split>.nbest.json.
SYSTEMS = (
    ("BM25 (R1)", "r1_bm25", None, "rank"),
    ("BERT span (R2)", "r2_bert_span", "", None),
    ("DeBERTa span (R3)", "r3_deberta_span", "", "span"),
    ("Full model (R4)", "r4_deberta_evidence", "joint_lam0.25", "rank"),
    ("No negatives (R5)", "r5_deberta_evidence", "joint_lam0.25", "rank"),
    ("Random negatives (R6)", "r6_deberta_evidence", "joint_lam0.25", "rank"),
    ("Lexical negatives (R6b)", "r6b_deberta_evidence", "joint_lam0.25", "rank"),
    ("No evidence decoding (R7)", "r7_deberta_evidence", "joint_lam0", None),  # R4's ranker
)
RANKINGS = (None, "rank", "span")
METRICS = ("exact", "f1", "clean_exact", "clean_f1", "drop_exact", "drop_f1",
           "answer_in_adv", "adv_at_rank1")


def base_id(qid):
    """Original question id: adversarial variants are ``<id>-<suffix>``."""
    return qid.split("-", 1)[0]


def read_squad(path):
    """``{qid: context}`` and the SQuAD ``data`` list."""
    data = json.loads(Path(path).read_text())["data"]
    contexts = {qa["id"]: p["context"] for a in data for p in a["paragraphs"] for qa in p["qas"]}
    return contexts, data


def adversarial_spans(contexts):
    """``{adv qid: (start, end)}``, the appended sentence's characters in that question's context.

    Each variant's context must be its original's context followed by the distractor.
    """
    spans = {}
    for qid, ctx in contexts.items():
        if qid == base_id(qid):
            continue
        orig = contexts.get(base_id(qid))
        if orig is None or not ctx.startswith(orig) or len(ctx) == len(orig):
            raise ValueError(f"{qid}: context is not its original's context plus an appended sentence")
        start = len(orig) + (len(ctx[len(orig):]) - len(ctx[len(orig):].lstrip()))
        spans[qid] = (start, len(ctx))
    return spans


def adversarial_sentences(spans, evidence):
    """``{adv qid: (sent_idx, merged)}``: the C3 sentence holding the distractor's first character,
    and whether that sentence also holds original text."""
    out = {}
    for qid, (start, _) in spans.items():
        sents = evidence[qid]["sentences"]
        idx = next((k for k, (s, e) in enumerate(sents) if s <= start < e), None)
        if idx is None:
            raise ValueError(f"{qid}: no C3 sentence holds the appended sentence (start {start})")
        out[qid] = (idx, sents[idx][0] < start)
    return out


def question_scores(data, predictions):
    """Official per-question ``(exact, f1)`` dicts, each score 0..1."""
    missing = {qa["id"] for a in data for p in a["paragraphs"] for qa in p["qas"]} - predictions.keys()
    if missing:
        raise ValueError(f"{len(missing)} questions have no prediction, e.g. {sorted(missing)[:3]}")
    return load_official_metrics().get_raw_scores(data, predictions)


def group_worst(scores, qids):
    """``{original id: minimum score over the original and its variants}``."""
    groups = defaultdict(list)
    for qid in qids:
        groups[base_id(qid)].append(scores[qid])
    if any(b not in scores for b in groups):
        raise ValueError("every group needs its original (clean) question")
    return {b: min(v) for b, v in groups.items()}


def worst_case(scores, qids):
    """Mean over original-id groups of the group's minimum score; also the originals' mean."""
    worst = group_worst(scores, qids)
    return mean(worst.values()), mean(scores[b] for b in worst), len(worst)


def answer_in_adversarial(nbest, contexts, spans, adv_sents):
    """Share of adversarial questions whose top n-best answer lies in the appended sentence."""
    hits = 0
    for qid, (start, end) in spans.items():
        top = nbest[qid][0] if nbest.get(qid) else None
        if top is None or top["sent_idx"] is None:
            continue
        idx, merged = adv_sents[qid]
        if int(top["sent_idx"]) == idx and (not merged or top["text"] in contexts[qid][start:end]):
            hits += 1
    return hits / len(spans)


def adversarial_at_rank1(ranking, adv_sents):
    """Share of adversarial questions whose rank-1 sentence is the appended one."""
    missing = adv_sents.keys() - ranking.keys()
    if missing:
        raise ValueError(f"ranking is missing {len(missing)} adversarial question(s), e.g. {sorted(missing)[:3]}")
    return mean(int(int(ranking[qid][0][0]) == idx) for qid, (idx, _) in adv_sents.items())


class Split:
    """The split's data, evidence and distractor positions, loaded once."""

    def __init__(self, split, data_dir=DATA_DIR):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.name = split
        self.contexts, self.data = read_squad(Path(data_dir) / f"{split}.json")
        with open(Path(data_dir) / f"evidence_{split}.jsonl") as f:
            self.evidence = {r["id"]: r for r in map(json.loads, f) if r}
        self.spans = adversarial_spans(self.contexts)
        self.adv_sents = adversarial_sentences(self.spans, self.evidence)


def score_run(split, predictions=None, nbest=None, ranking=None):
    """Metrics for one run, in percent; entries a run cannot give (no answers, no ranking) are None."""
    out = dict.fromkeys(METRICS)
    if predictions is not None:
        exact, f1 = question_scores(split.data, predictions)
        qids = list(split.contexts)
        for name, scores in (("exact", exact), ("f1", f1)):
            adv, clean, n = worst_case(scores, qids)
            out[name], out[f"clean_{name}"], out[f"drop_{name}"] = 100 * adv, 100 * clean, 100 * (clean - adv)
        out["n_groups"] = n
        out["answer_in_adv"] = 100 * answer_in_adversarial(nbest, split.contexts, split.spans, split.adv_sents)
    if ranking is not None:
        out["adv_at_rank1"] = 100 * adversarial_at_rank1(ranking, split.adv_sents)
    out["n_adv"] = len(split.spans)
    out["n_merged"] = sum(merged for _, merged in split.adv_sents.values())
    return out


def _load(path):
    return json.loads(Path(path).read_text())


def run_files(run_dir, split, pred_name):
    """``(predictions, nbest)`` paths for a prediction name ("" = span decoder)."""
    stem = f"{split}." + (f"{pred_name}." if pred_name else "")
    return Path(run_dir) / f"{stem}predictions.json", Path(run_dir) / f"{stem}nbest.json"


def ranking_file(run_dir, split, ranking):
    """Where a run's sentence ranking comes from (None if it has none)."""
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}, got {ranking!r}")
    if ranking is None:
        return None
    return Path(run_dir) / (f"{split}.sent_rank.json" if ranking == "rank" else f"{split}.nbest.json")


def load_ranking(split, run_dir, ranking):
    path = ranking_file(run_dir, split.name, ranking)
    if path is None:
        return None
    return span_implied_ranking(_load(path), split.evidence) if ranking == "span" else _load(path)


def score_dir(split, run_dir, pred_name, ranking):
    pred_path, nbest_path = run_files(run_dir, split.name, pred_name) if pred_name is not None else (None, None)
    return score_run(split,
                     _load(pred_path) if pred_path else None,
                     _load(nbest_path) if nbest_path else None,
                     load_ranking(split, run_dir, ranking))


def table_rows(split, adv_dir=ADV_DIR, bm25_dir=BM25_DIR, systems=SYSTEMS):
    """One row per system with outputs: mean and sample std over seeds."""
    rows = []
    for label, run_id, pred_name, ranking in systems:
        if run_id == "r1_bm25":
            dirs = [(None, Path(bm25_dir))]
        else:
            dirs = [(s, Path(adv_dir) / f"{run_id}_s{s}") for s in SEEDS]

        def needed(d):
            paths = [run_files(d, split.name, pred_name)[0]] if pred_name is not None else []
            return paths + ([ranking_file(d, split.name, ranking)] if ranking else [])

        found = [(s, d) for s, d in dirs if all(p.is_file() for p in needed(d))]
        if not found:
            print(f"{label}: no {split.name} outputs under {dirs[0][1].parent}; left out", file=sys.stderr)
            continue
        runs = [score_dir(split, d, pred_name, ranking) for _, d in found]
        row = {"system": label, "run_id": run_id,
               "seeds": " ".join(str(s) for s, _ in found if s is not None), "n_adv": runs[0]["n_adv"]}
        for m in METRICS:
            values = [r[m] for r in runs if r[m] is not None]
            row[f"{m}_mean"] = mean(values) if values else None
            row[f"{m}_std"] = stdev(values) if len(values) > 1 else None
        rows.append(row)
    return rows


def write_table(rows, split_name, out_dir=TABLE_DIR):
    """``adversarial_<split>.csv`` and ``.tex``; returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = out_dir / f"adversarial_{split_name}.csv", out_dir / f"adversarial_{split_name}.tex"
    fields = ["system", "run_id", "seeds", "n_adv", *(f"{m}_{s}" for m in METRICS for s in ("mean", "std"))]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def cell(row, m):
        value, sd = row[f"{m}_mean"], row[f"{m}_std"]
        if value is None:
            return "--"
        return f"{value:.1f}" + ("" if sd is None else f" $\\pm$ {sd:.1f}")

    shown = ("clean_f1", "f1", "drop_f1", "answer_in_adv", "adv_at_rank1")
    lines = [f"% Generated by qa.eval.adversarial table --split {split_name}; percent, mean $\\pm$ sample "
             "std over seeds. F1 is worst case over each original's group; (a) answer in the distractor, "
             "(b) distractor ranked first, both over adversarial questions. (b) for R3 is its span-implied "
             "ranking; R7 uses R4's ranker, so its (b) is R4's.",
             r"\begin{tabular}{lccccc}", r"\hline",
             r"System & Clean F1 & Adv. F1 & Drop & (a) & (b) \\", r"\hline",
             *(f"{row['system']} & " + " & ".join(cell(row, m) for m in shown) + r" \\" for row in rows),
             r"\hline", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")
    return csv_path, tex_path


SERIES = (("addsent", "AddSent", "#2a78d6", ""), ("addonesent", "AddOneSent", "#eb6834", "////"))


def read_table(path):
    """Rows of ``adversarial_<split>.csv`` with numbers parsed (empty cells -> None)."""
    with open(path, newline="") as stream:
        return [{k: (v if k in ("system", "run_id", "seeds") else (float(v) if v else None))
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def plot_drops(tables, out_path):
    """Horizontal bars of worst-case F1 drop (clean − adversarial) per system, one series per split,
    seed std as error bars. ``tables`` maps split -> rows; systems without F1 (BM25) are skipped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [r["system"] for r in tables[SERIES[0][0]] if r["drop_f1_mean"] is not None]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(systems) + 0.9))
    height, gap = 0.36, 0.04
    for k, (split, label, color, hatch) in enumerate(SERIES):
        by_name = {r["system"]: r for r in tables.get(split, [])}
        ys, xs, errs = [], [], []
        for i, name in enumerate(systems):
            row = by_name.get(name)
            if row and row["drop_f1_mean"] is not None:
                ys.append(i + (k - 0.5) * (height + gap))
                xs.append(row["drop_f1_mean"])
                errs.append(row["drop_f1_std"] or 0.0)
        ax.barh(ys, xs, height=height, color=color, hatch=hatch, edgecolor="white", linewidth=0,
                xerr=errs, error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#52514e"}, label=label)
    ax.set_yticks(range(len(systems)), systems)
    ax.invert_yaxis()
    ax.set_xlabel("Worst-case F1 drop from clean (points)")
    ax.xaxis.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#a3a29c")
    ax.tick_params(axis="y", length=0, colors="#0b0b0b")
    ax.tick_params(axis="x", colors="#52514e")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if out_path.suffix != ".png":
        fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    figure = commands.add_parser("figure", help="F1-drop figure from the adversarial_<split>.csv tables")
    figure.add_argument("--table-dir", default=TABLE_DIR)
    figure.add_argument("--out", default="results/figures/adversarial_f1_drop.pdf")
    for name in ("score", "table"):
        sub = commands.add_parser(name)
        sub.add_argument("--split", required=True, choices=SPLITS)
        sub.add_argument("--data-dir", default=DATA_DIR)
    score = commands.choices["score"]
    score.add_argument("--run-dir", required=True)
    score.add_argument("--predictions", default="", help='prediction name, e.g. "joint_lam0.25"; "" = span decoder')
    score.add_argument("--no-answers", action="store_true", help="ranking only (BM25)")
    score.add_argument("--ranked", action="store_true", help="also score <split>.sent_rank.json")
    score.add_argument("--span-implied", action="store_true",
                       help="also score the span-implied ranking from <split>.nbest.json (span-only R3)")
    score.add_argument("--group-scores-out",
                       help="write per-group worst-case F1 {original id: 0..1} here, for significance.py")
    table = commands.choices["table"]
    table.add_argument("--adv-dir", default=ADV_DIR)
    table.add_argument("--bm25-dir", default=BM25_DIR)
    table.add_argument("--table-dir", default=TABLE_DIR)
    args = parser.parse_args(argv)
    if args.command == "figure":
        tables = {s: read_table(Path(args.table_dir) / f"adversarial_{s}.csv")
                  for s, *_ in SERIES if (Path(args.table_dir) / f"adversarial_{s}.csv").is_file()}
        if SERIES[0][0] not in tables:
            raise SystemExit(f"No adversarial_{SERIES[0][0]}.csv under {args.table_dir}/; run `table` first")
        print(plot_drops(tables, args.out))
        return
    split = Split(args.split, args.data_dir)

    if args.command == "score":
        if args.ranked and args.span_implied:
            raise SystemExit("--ranked and --span-implied are exclusive")
        pred = None if args.no_answers else args.predictions
        ranking = "rank" if args.ranked else ("span" if args.span_implied else None)
        if args.group_scores_out:
            if pred is None:
                raise SystemExit("--group-scores-out needs predictions")
            _, f1 = question_scores(split.data, _load(run_files(args.run_dir, args.split, pred)[0]))
            worst = group_worst(f1, list(split.contexts))
            Path(args.group_scores_out).write_text(json.dumps(worst) + "\n")
        print(json.dumps({"split": args.split, "run_dir": args.run_dir,
                          **score_dir(split, args.run_dir, pred, ranking)}, indent=2))
    else:
        rows = table_rows(split, args.adv_dir, args.bm25_dir)
        if not rows:
            raise SystemExit(f"No {args.split} outputs under {args.adv_dir}/")
        for path in write_table(rows, args.split, args.table_dir):
            print(path)


if __name__ == "__main__":
    main()
