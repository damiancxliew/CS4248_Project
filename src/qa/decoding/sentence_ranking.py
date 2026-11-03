"""Passage-level sentence ranking (C8, G2): window sentence scores -> one ranking per question.

Works on what ``predict.py`` saves for the evidence model (``offsets``, ``sent_logits``,
``sent_mask``) plus the C3 sentences. Slot j of a window is its j-th visible passage sentence in
passage order, recovered from the window's offsets with ``token_sentences``: the rule C4 uses.

Each window's sentence logits are log-softmaxed over its visible slots, and a sentence's passage
score is its best log-probability over the windows that show it (TEAM_TASKS §2.3 step 7). A
window that shows fewer sentences gives them an easier softmax; only paragraphs split across
windows are affected.

C8: ``{qid: [[sent_idx, score], ...]}``, best first (ties: earlier sentence first), covering every
passage sentence; ``score`` = passage-level log-probability. A sentence no window shows gets the
passage's lowest score.
"""
import numpy as np

from qa.data.sentence_features import NOT_IN_SENTENCE, token_sentences


def window_slots(offsets, sentences):
    """Passage sentence of each slot of one window: ``offsets`` [T, 2] (-1 = not context)."""
    ids = token_sentences(offsets, sentences)
    return np.unique(ids[ids != NOT_IN_SENTENCE])


def window_logprobs(logits):
    """Log-softmax over one window's visible slots."""
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - shifted.max()
    return shifted - np.log(np.exp(shifted).sum())


def passage_ranking(example_ids, offsets, sent_logits, sent_mask, evidence):
    """C8 ranking per question; ``evidence`` maps question id -> C3 record."""
    best = {}
    for i, qid in enumerate(example_ids):
        slots = window_slots(offsets[i], evidence[qid]["sentences"])
        n_scored = int(np.count_nonzero(sent_mask[i]))
        if n_scored != len(slots):
            raise ValueError(f"{qid}: window {i} has {n_scored} scored sentences but its offsets show "
                             f"{len(slots)}; is this the C3 file the features were built with?")
        scores = best.setdefault(qid, {})
        if not len(slots):
            continue
        for sent, logprob in zip(slots.tolist(), window_logprobs(sent_logits[i, :len(slots)])):
            scores[sent] = max(float(logprob), scores.get(sent, -np.inf))
    return {qid: _ranked(scores, len(evidence[qid]["sentences"])) for qid, scores in best.items()}


def _ranked(scores, n_sentences):
    floor = min(scores.values(), default=0.0)
    full = [(sent, scores.get(sent, floor)) for sent in range(n_sentences)]
    return [[sent, score] for sent, score in sorted(full, key=lambda item: (-item[1], item[0]))]
