"""Sentence features (contract C4): project C3 evidence annotations onto the C2 windows.

``add_sentence_fields`` adds four window-local columns. Slots number the sentences visible in a
window, 0..S-1 in passage order, so one passage sentence can have a different slot per window:

- ``sent_ids``: slot of each token, ``-1`` outside the context or on the gap between sentences.
- ``sent_label``: slot of the gold sentence, ``-100`` if it is not visible.
- ``neg_cand_mask``: 1 per slot [S] whose sentence is a negative candidate.
- ``static_neg_label``: slot of ``lex_neg`` / ``rand_neg`` under ``lexical`` / ``random``, ``-100``
  if it is not visible and under the other modes.

A token belongs to the sentence it overlaps most, since tokenizers may include a leading space in
a token's offsets. A sentence cut by the window edge counts as visible if any token of it is.
"""
import numpy as np

from qa.decoding.span_decoder import features_to_arrays

NEG_MODES = ("model", "lexical", "random", "none")
STATIC_NEG_KEYS = {"lexical": "lex_neg", "random": "rand_neg"}
IGNORE = -100
NOT_IN_SENTENCE = -1


def token_sentences(offsets, sentences):
    """Passage sentence index of each token; ``offsets`` [T, 2] uses -1 for non-context tokens."""
    result = []
    for token_start, token_end in offsets.tolist():
        best_sentence, best_overlap = NOT_IN_SENTENCE, 0
        for i, (sent_start, sent_end) in enumerate(sentences):
            overlap = min(token_end, sent_end) - max(token_start, sent_start)
            if overlap > best_overlap:
                best_sentence, best_overlap = i, overlap
        result.append(best_sentence)
    return np.array(result)


def window_fields(offsets, record, neg_mode):
    """The four C4 values of one window."""
    passage_ids = token_sentences(offsets, record["sentences"])
    in_sentence = passage_ids != NOT_IN_SENTENCE
    visible = np.unique(passage_ids[in_sentence])
    slot = {sentence: i for i, sentence in enumerate(visible.tolist())}
    candidates = set(record["neg_candidates"])
    static_key = STATIC_NEG_KEYS.get(neg_mode)
    return {
        "sent_ids": np.where(in_sentence, np.searchsorted(visible, passage_ids), -1).tolist(),
        "sent_label": slot.get(record["gold_sent"], IGNORE),
        "neg_cand_mask": [int(sentence in candidates) for sentence in slot],
        "static_neg_label": slot.get(record[static_key], IGNORE) if static_key else IGNORE,
    }


def add_sentence_fields(features, evidence, neg_mode):
    """Return the C2 ``features`` with the C4 columns; ``evidence`` maps question id -> C3 record."""
    if neg_mode not in NEG_MODES:
        raise ValueError(f"neg_mode must be one of {NEG_MODES}, got {neg_mode!r}")
    example_ids, offsets = features_to_arrays(features)
    lengths = [len(row) for row in features["input_ids"]]
    rows = [
        window_fields(offsets[i, :lengths[i]], evidence[qid], neg_mode)
        for i, qid in enumerate(example_ids)
    ]
    for column in ("sent_ids", "sent_label", "neg_cand_mask", "static_neg_label"):
        features = features.add_column(column, [row[column] for row in rows])
    return features
