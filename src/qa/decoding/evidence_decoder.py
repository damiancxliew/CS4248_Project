"""Joint decoding (README §3.4, G2): span score + λ · log p(sentence holding the span's start).

Candidates are exactly the standard decoder's (``span_decoder._window_candidates``: top
``n_best`` starts × ends per window, in the context, ``start <= end``, ``<= max_answer_len``
tokens). Each is rescored ``start_logit + end_logit + λ · sent_logprob`` and pooled across the
question's windows. ``sent_logprob`` is the C8 passage-level log-probability of the sentence
holding the span's start (``predict.sentence_index``, the C6 rule); a start in no sentence gets the
passage's lowest. λ = 0 reproduces ``span_decoder.decode``.

Usage:
  python -m qa.decoding.evidence_decoder --logits outputs/<run_id>/val.logits.npz
      --evidence data/processed/evidence_val.jsonl --data-file data/processed/val.json --lam 0.5

Writes next to the logits (or to ``--out-dir``): ``<split>.sent_rank.json`` (C8), and
``<split>.joint_lam<λ>.predictions.json`` / ``.nbest.json`` (C6, ``score`` = joint score), so
span-only outputs are never overwritten. The test split needs ``--allow-test`` (``run_all.sh``).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qa.decoding.sentence_ranking import passage_ranking
from qa.decoding.span_decoder import _window_candidates
from qa.predict import read_evidence, sentence_index


def decode_joint(contexts, example_ids, offsets, start_logits, end_logits, ranking, evidence, lam,
                 n_best=20, max_answer_len=30):
    """Return ``(predictions, nbest)`` in C6 format; ``ranking`` is C8, ``evidence`` C3 by id."""
    windows = defaultdict(list)
    for i, qid in enumerate(example_ids):
        windows[qid].append(i)

    predictions, nbest = {}, {}
    for qid, feature_indices in windows.items():
        logprobs = dict(ranking[qid])
        floor = min(logprobs.values(), default=0.0)
        candidates = []
        for i in feature_indices:
            for c in _window_candidates(offsets[i], start_logits[i], end_logits[i], n_best, max_answer_len):
                sent = sentence_index(evidence[qid], c["start_char"], c["end_char"])
                logprob = logprobs[sent] if sent is not None else floor
                candidates.append({**c, "sent_idx": sent, "sent_logprob": logprob,
                                   "score": c["score"] + lam * logprob})
        candidates.sort(key=lambda c: c["score"], reverse=True)
        context = contexts[qid]
        entries = [
            {
                "text": context[c["start_char"]:c["end_char"]],
                "start_logit": c["start_logit"],
                "end_logit": c["end_logit"],
                "sent_idx": c["sent_idx"],
                "sent_logprob": c["sent_logprob"],
                "score": c["score"],
            }
            for c in candidates[:n_best]
        ]
        nbest[qid] = entries
        predictions[qid] = entries[0]["text"] if entries else ""
    return predictions, nbest


def read_contexts(squad_file):
    data = json.loads(Path(squad_file).read_text())["data"]
    return {qa["id"]: p["context"] for article in data for p in article["paragraphs"] for qa in p["qas"]}


def run(logits_file, evidence_file, data_file, lam, out_dir=None, n_best=20, max_answer_len=30,
        allow_test=False):
    """Write the C8 ranking and the joint-decoding C6 files for one split; returns their paths."""
    logits_file = Path(logits_file)
    split = logits_file.name.removesuffix(".logits.npz")
    if split == "test" and not allow_test:
        raise SystemExit("Refusing to decode test (dev-v1.1.json). Only final checkpoints, via "
                         "scripts/run_all.sh, which passes --allow-test.")
    saved = np.load(logits_file)
    if "sent_logits" not in saved.files:
        raise ValueError(f"{logits_file} has no sent_logits: decode an evidence-model run")
    example_ids = saved["example_ids"].tolist()
    evidence = read_evidence(evidence_file)
    ranking = passage_ranking(example_ids, saved["offsets"], saved["sent_logits"], saved["sent_mask"] > 0,
                              evidence)
    predictions, nbest = decode_joint(read_contexts(data_file), example_ids, saved["offsets"],
                                      saved["start_logits"], saved["end_logits"], ranking, evidence, lam,
                                      n_best, max_answer_len)

    out_dir = Path(out_dir or logits_file.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{split}.joint_lam{lam:g}"
    paths = {
        "sent_rank": out_dir / f"{split}.sent_rank.json",
        "predictions": out_dir / f"{tag}.predictions.json",
        "nbest": out_dir / f"{tag}.nbest.json",
    }
    paths["sent_rank"].write_text(json.dumps(ranking))
    paths["predictions"].write_text(json.dumps(predictions, indent=1))
    paths["nbest"].write_text(json.dumps(nbest))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--logits", required=True, help="<split>.logits.npz from predict.py (evidence model)")
    parser.add_argument("--evidence", required=True, help="C3 evidence_<split>.jsonl the features used")
    parser.add_argument("--data-file", required=True, help="the SQuAD-format file the logits come from")
    parser.add_argument("--lam", type=float, required=True, help="weight of log p(sentence); 0 = span-only")
    parser.add_argument("--out-dir")
    parser.add_argument("--n-best", type=int, default=20)
    parser.add_argument("--max-answer-len", type=int, default=30)
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    paths = run(args.logits, args.evidence, args.data_file, args.lam, args.out_dir, args.n_best,
                args.max_answer_len, args.allow_test)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
