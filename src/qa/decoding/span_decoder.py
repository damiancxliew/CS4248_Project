"""Standard n-best span decoding over sliding windows (README §3.4 with λ = 0, k = all).

Follows the Hugging Face reference post-processing (``utils_qa.postprocess_qa_predictions``,
SQuAD v1 mode): per window, take the ``n_best`` highest start and end logits, keep pairs that
lie in the context with ``start <= end`` and length ``<= max_answer_len`` tokens, score them
``start_logit + end_logit``, pool the candidates of all windows of a question, and map the
best one back to text through the offset mapping.

Decoders work on arrays so they can be re-run on the logits ``predict.py`` saves:
- ``example_ids``: one question id per window (feature).
- ``offsets``: int array [N, T, 2] of character offsets into the context; ``-1`` marks
  tokens that are not context (question, special and padding tokens).
- ``start_logits`` / ``end_logits``: float arrays [N, T].
"""
from collections import defaultdict

import numpy as np

NOT_CONTEXT = -1


def features_to_arrays(features):
    """Example ids and the [N, T, 2] offsets array from C2 features.

    Context tokens are those with ``sequence_ids == 1`` when the features carry that column;
    otherwise those whose offset is not None (the HF convention of nulling the offsets of
    non-context tokens in evaluation features).
    """
    example_ids = list(features["example_id"])
    offset_rows = features["offset_mapping"]
    sequence_ids = features["sequence_ids"] if "sequence_ids" in features.column_names else None
    max_len = max(len(row) for row in offset_rows)
    offsets = np.full((len(offset_rows), max_len, 2), NOT_CONTEXT, dtype=np.int64)
    for i, row in enumerate(offset_rows):
        for t, span in enumerate(row):
            in_context = span is not None if sequence_ids is None else sequence_ids[i][t] == 1
            if in_context and span is not None:
                offsets[i, t] = span
    return example_ids, offsets


def decode(contexts, example_ids, offsets, start_logits, end_logits, n_best=20, max_answer_len=30):
    """Return ``(predictions, nbest)``.

    predictions: ``{qid: answer_text}`` (official format; ``""`` if no valid span exists).
    nbest: ``{qid: [{text, start_logit, end_logit, score, start_char, end_char}, ...]}``,
    best first, at most ``n_best`` entries.
    """
    windows = defaultdict(list)
    for i, qid in enumerate(example_ids):
        windows[qid].append(i)

    predictions, nbest = {}, {}
    for qid, feature_indices in windows.items():
        candidates = []
        for i in feature_indices:
            candidates.extend(
                _window_candidates(offsets[i], start_logits[i], end_logits[i], n_best, max_answer_len)
            )
        candidates.sort(key=lambda c: c["score"], reverse=True)
        context = contexts[qid]
        entries = [
            {**c, "text": context[c["start_char"]:c["end_char"]]} for c in candidates[:n_best]
        ]
        nbest[qid] = entries
        predictions[qid] = entries[0]["text"] if entries else ""
    return predictions, nbest


def _window_candidates(offsets, start_logits, end_logits, n_best, max_answer_len):
    start_indexes = np.argsort(start_logits)[-1 : -n_best - 1 : -1]
    end_indexes = np.argsort(end_logits)[-1 : -n_best - 1 : -1]
    out = []
    for s in start_indexes:
        for e in end_indexes:
            if offsets[s, 0] == NOT_CONTEXT or offsets[e, 0] == NOT_CONTEXT:
                continue
            if e < s or e - s + 1 > max_answer_len:
                continue
            out.append({
                "start_logit": float(start_logits[s]),
                "end_logit": float(end_logits[e]),
                "score": float(start_logits[s] + end_logits[e]),
                "start_char": int(offsets[s, 0]),
                "end_char": int(offsets[e, 1]),
            })
    return out
