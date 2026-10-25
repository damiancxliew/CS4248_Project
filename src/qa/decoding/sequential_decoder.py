"""Rank-then-read (sequential) decoding: restrict answer spans to the top-k ranked sentences.

Inputs follow the team contracts (TEAM_TASKS 3.1):
  C6 nbest:     {qid: [{text, start_logit, end_logit, sent_idx, sent_logprob, score}, ...]}
  C8 sent_rank: {qid: [[sent_idx, score], ...]} sorted best-first, covering every sentence
  C3 evidence:  {"id", "gold_sent", ...} one record per question

The reader picks, among the n-best spans lying in the kept sentences, the one with the
highest start_logit + end_logit (no sentence term; that is the joint decoder's job).
"""
import importlib.util
from pathlib import Path

K_ALL = "all"
DEFAULT_KS = (1, 2, 3, K_ALL)


def top_k_sentences(ranking, k):
    """Sentence indices kept for budget k; k = 'all' keeps every ranked sentence."""
    ranked = [int(s) for s, _ in ranking]
    return set(ranked) if k == K_ALL else set(ranked[:k])


def decode_question(nbest, ranking, k):
    """Best span text among n-best candidates inside the top-k sentences, or None.

    None means no candidate falls in the kept sentences (the reader has nothing to read).
    """
    kept = top_k_sentences(ranking, k)
    in_kept = [c for c in nbest if c["sent_idx"] in kept]
    if not in_kept:
        return None
    return max(in_kept, key=lambda c: c["start_logit"] + c["end_logit"])["text"]


def decode(nbest, sent_rank, k):
    """{qid: answer_text} for budget k. Questions with nothing readable get ''."""
    preds = {}
    for qid, cands in nbest.items():
        text = decode_question(cands, sent_rank[qid], k)
        preds[qid] = "" if text is None else text
    return preds


def recall_at_k(sent_rank, evidence, k):
    """Fraction of questions whose gold sentence is in the top-k: the EM/F1-ceiling of rank-then-read."""
    hits = [
        evidence[qid]["gold_sent"] in top_k_sentences(sent_rank[qid], k)
        for qid in evidence
    ]
    return sum(hits) / len(hits)


def _load_official_metrics():
    path = Path(__file__).resolve().parents[1] / "eval" / "evaluate-v2.0.py"
    spec = importlib.util.spec_from_file_location("squad_official_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error_split(preds, sent_rank, evidence, gold_answers, k):
    """Split questions into correct / ranker error / reader error at budget k.

    ranker error: gold sentence not in the top-k (the reader cannot recover it)
    reader error: gold sentence kept but the chosen span is not an exact match
    gold_answers: {qid: [answer_text, ...]}
    """
    official = _load_official_metrics()
    counts = {"correct": 0, "ranker_error": 0, "reader_error": 0}
    for qid, ev in evidence.items():
        kept = top_k_sentences(sent_rank[qid], k)
        exact = max(official.compute_exact(g, preds[qid]) for g in gold_answers[qid])
        if exact:
            counts["correct"] += 1
        elif ev["gold_sent"] not in kept:
            counts["ranker_error"] += 1
        else:
            counts["reader_error"] += 1
    n = len(evidence)
    return {**counts, "n": n, **{f"{name}_rate": c / n for name, c in counts.items()}}


def sweep(nbest, sent_rank, evidence, gold_answers, ks=DEFAULT_KS):
    """Run every k on one split (use val to pick k); returns {k: {recall_at_k, exact, error split}}."""
    official = _load_official_metrics()
    results = {}
    for k in ks:
        preds = decode(nbest, sent_rank, k)
        split = error_split(preds, sent_rank, evidence, gold_answers, k)
        f1 = sum(
            max(official.compute_f1(g, preds[qid]) for g in gold_answers[qid])
            for qid in evidence
        ) / len(evidence)
        results[k] = {
            "recall_at_k": recall_at_k(sent_rank, evidence, k),
            "exact": split["correct_rate"],
            "f1": f1,
            **split,
        }
    return results
