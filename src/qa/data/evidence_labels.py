"""Evidence annotations (contract C3): sentence spans of each context, merged around answers.

Contexts are split with pysbd (rule-based, deterministic). On the 87.6k train answers it puts a
sentence boundary inside 0.12% of answers vs. 0.17% for spaCy's parser, at about half the time
(``scripts/compare_splitters.py``). Sentences that an answer crosses are merged, using every
answer of every question in the context, so a context has one sentence list shared by all its
questions and every answer lies in exactly one sentence.

Spans are character offsets ``(start, end)`` into the context, end exclusive. The gold sentence
is the one containing ``answers[0]`` (train and val have exactly one answer); negative candidates
exclude sentences containing any reference answer.

  python -m qa.data.evidence_labels --input data/processed/train.json --split train

BM25 scores each sentence against the question with IDF computed over the paragraph's own
sentences (the reader's input). Okapi's IDF ``log((N - n + 0.5) / (n + 0.5))`` is zero or negative
for a word in half or more of the sentences, which on SQuAD's ~5-sentence paragraphs discards
useful question words; we use Lucene's non-negative ``log(1 + (N - n + 0.5) / (n + 0.5))``.
"""
import argparse
import functools
import json
import math
import random
import re
from pathlib import Path

import pysbd

BM25_K1 = 1.5
BM25_B = 0.75
RANDOM_NEG_SEED = 2026


@functools.lru_cache(maxsize=1)
def _segmenter():
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


def split_sentences(context):
    """Sentence spans of ``context``, with surrounding whitespace trimmed.

    pysbd's spans include the whitespace that follows a sentence; trimming it makes
    ``context[start:end]`` the sentence text exactly.
    """
    spans = []
    for sent in _segmenter().segment(context):
        start, end = sent.start, sent.end
        while start < end and context[start].isspace():
            start += 1
        while end > start and context[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def merge_straddles(spans, answers):
    """Merge each run of consecutive sentences that a single answer ``(start, end)`` overlaps.

    Returns ``(spans, merged)``: the new sentence spans, and per span whether it was formed by
    merging (the source of C3's ``straddle_merged``).
    """
    if not spans:
        return [], []
    join = [False] * (len(spans) - 1)  # join[k]: glue sentence k to sentence k + 1
    for a, b in answers:
        hit = [k for k, (s, e) in enumerate(spans) if s < b and a < e]
        for k in range(hit[0], hit[-1]) if hit else ():
            join[k] = True

    merged, flags = [spans[0]], [False]
    for k, span in enumerate(spans[1:]):
        if join[k]:
            merged[-1] = (merged[-1][0], span[1])
            flags[-1] = True
        else:
            merged.append(span)
            flags.append(False)
    return merged, flags


def gold_sentence(spans, start, end):
    """Index of the sentence containing the answer ``[start, end)``.

    Raises ``ValueError`` if no single sentence contains it (spans not merged around answers).
    """
    for k, (s, e) in enumerate(spans):
        if s <= start and end <= e:
            return k
    raise ValueError(f"answer [{start}, {end}) is not inside one sentence: {spans}")


def negative_candidates(context, spans, gold, answer_texts):
    """Non-gold sentences that contain none of ``answer_texts`` (exact, case-sensitive).

    A sentence that repeats the answer may be valid evidence, so treating it as a negative
    would teach the ranker a false negative. Pass every reference answer (test has several).
    """
    return [
        k for k, (s, e) in enumerate(spans)
        if k != gold and not any(text in context[s:e] for text in answer_texts)
    ]


def _tokens(text):
    return re.findall(r"\w+", text.lower())


def bm25_scores(context, spans, question, k1=BM25_K1, b=BM25_B):
    """BM25 score of each sentence against ``question``; the paragraph's sentences are the corpus."""
    docs = [_tokens(context[s:e]) for s, e in spans]
    avgdl = sum(map(len, docs)) / len(docs) if docs else 0.0
    if avgdl == 0:
        return [0.0] * len(docs)
    n_docs = len(docs)
    doc_freq = {}
    for doc in docs:
        for word in set(doc):
            doc_freq[word] = doc_freq.get(word, 0) + 1
    query = _tokens(question)
    scores = []
    for doc in docs:
        norm = k1 * (1 - b + b * len(doc) / avgdl)
        score = 0.0
        for word in query:
            tf = doc.count(word)
            if tf:
                n = doc_freq[word]
                idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
                score += idf * tf * (k1 + 1) / (tf + norm)
        scores.append(score)
    return scores


def lexical_negative(bm25, candidates):
    """The candidate with the highest BM25 score (earliest on ties), or ``None`` if there is none."""
    return max(candidates, key=lambda k: (bm25[k], -k), default=None)


def random_negative(qid, candidates, seed=RANDOM_NEG_SEED):
    """A random candidate, fixed per question id (independent of file order), or ``None``."""
    if not candidates:
        return None
    return random.Random(f"{seed}:{qid}").choice(candidates)


def _answer_spans(answers):
    return [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in answers]


def build_evidence(squad):
    """C3 records (one per question, in file order) and dataset statistics for a SQuAD-format dict.

    ``straddle_merged`` is true when the question's gold sentence was formed by merging.
    """
    records = []
    n_ctx = n_sent = n_single = n_ctx_merged = 0
    n_q = n_straddled = n_non_gold = n_excluded = n_no_cand = 0
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            raw = split_sentences(context)
            spans, merged = merge_straddles(
                raw, [span for qa in paragraph["qas"] for span in _answer_spans(qa["answers"])])
            n_ctx += 1
            n_sent += len(spans)
            n_single += len(spans) == 1
            n_ctx_merged += any(merged)
            for qa in paragraph["qas"]:
                start, end = _answer_spans(qa["answers"][:1])[0]
                gold = gold_sentence(spans, start, end)
                candidates = negative_candidates(context, spans, gold, [a["text"] for a in qa["answers"]])
                bm25 = bm25_scores(context, spans, qa["question"])
                records.append({
                    "id": qa["id"],
                    "sentences": [list(span) for span in spans],
                    "gold_sent": gold,
                    "neg_candidates": candidates,
                    "lex_neg": lexical_negative(bm25, candidates),
                    "rand_neg": random_negative(qa["id"], candidates),
                    "bm25": bm25,
                    "straddle_merged": merged[gold],
                })
                n_q += 1
                n_straddled += sum(s < end and start < e for s, e in raw) > 1
                n_non_gold += len(spans) - 1
                n_excluded += len(spans) - 1 - len(candidates)
                n_no_cand += not candidates

    def pct(num, den):
        return 100 * num / den if den else 0.0

    stats = {
        "contexts": n_ctx,
        "questions": n_q,
        "sentences_per_context": n_sent / n_ctx if n_ctx else 0.0,
        "pct_answers_needing_merge": pct(n_straddled, n_q),
        "pct_contexts_with_merge": pct(n_ctx_merged, n_ctx),
        "pct_sentences_excluded": pct(n_excluded, n_non_gold),
        "pct_contexts_single_sentence": pct(n_single, n_ctx),
        "pct_questions_no_candidate": pct(n_no_cand, n_q),
    }
    return records, stats


def write_evidence(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="SQuAD-format JSON file")
    parser.add_argument("--split", required=True, help="train, val, test, addsent, ...")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args(argv)
    with open(args.input) as f:
        records, stats = build_evidence(json.load(f))
    out = Path(args.out_dir)
    write_evidence(records, out / f"evidence_{args.split}.jsonl")
    (out / f"evidence_{args.split}.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
