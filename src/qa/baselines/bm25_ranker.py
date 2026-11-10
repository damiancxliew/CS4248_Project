"""BM25 sentence ranker (R1): the C3 ``bm25`` scores as a C8 ranking, for every question of a split.

No model and no new scores: ``evidence_labels.py`` (J1) already scores each passage sentence
against its question. This ranks them, best first, with ties going to the earlier sentence (the
rule ``sentence_ranking.py`` uses for the model ranker), so every C8 producer orders ties alike.

C8: ``outputs/<run_id>/<split>.sent_rank.json``, ``{qid: [[sent_idx, score], ...]}`` covering every
passage sentence; ``score`` = raw BM25. Clemen's sequential decoder (R9a) reads it.

  python -m qa.baselines.bm25_ranker --split val

Prints Recall@1, Recall@3 and MRR from ``qa.eval.evidence_metrics`` (J6b), the code that scores
every ranker in the ranking table. The test set
(dev-v1.1.json) is refused unless ``--allow-test`` is passed, as in ``predict.py``.
"""
import argparse
import json
from pathlib import Path

from qa.eval.evidence_metrics import gold_ranks, read_evidence, summarize

RUN_ID = "r1_bm25"
EVIDENCE_DIR = "data/processed"
OUTPUT_DIR = "outputs"


def rank_sentences(bm25):
    """``[[sent_idx, score], ...]`` best first; ties go to the earlier sentence."""
    order = sorted(range(len(bm25)), key=lambda i: (-bm25[i], i))
    return [[i, float(bm25[i])] for i in order]


def rank_split(evidence):
    """C8 ranking per question; ``evidence`` maps question id -> C3 record."""
    return {qid: rank_sentences(rec["bm25"]) for qid, rec in evidence.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", required=True, help="val, test, addsent, addonesent, train")
    parser.add_argument("--evidence-dir", default=EVIDENCE_DIR, help="holds evidence_<split>.jsonl")
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--allow-test", action="store_true", help="final runs only (run_all.sh)")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to rank the test set (dev-v1.1.json) without --allow-test; "
                         "final runs go through scripts/run_all.sh.")

    evidence = read_evidence(Path(args.evidence_dir) / f"evidence_{args.split}.jsonl")
    ranking = rank_split(evidence)
    out = Path(args.out_dir) / args.run_id / f"{args.split}.sent_rank.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ranking) + "\n")

    metrics = summarize(gold_ranks(ranking, evidence))
    print(json.dumps({"split": args.split, **metrics, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
