"""Evidence-guided span extractor: span head + sentence-ranker head (README §2.2 row 5, G1).

Subclasses ``SpanModel``. For each sentence visible in a window, the ranker scores
``s_j = Linear(dropout(mean of the sentence's token states))``. Training loss (TEAM_TASKS §2.3):

    L = (L_start + L_end) / 2 + alpha * L_rank + beta * L_neg

- ``L_rank``: listwise cross-entropy over the window's sentences against the gold sentence.
- ``L_neg``: margin ``max(0, margin - s_gold + s_neg)``. ``neg_mode`` picks ``s_neg``:
  ``model`` mines the highest-scoring candidate under the current head (online, from
  ``neg_cand_mask``); ``lexical`` / ``random`` use the fixed ``static_neg_label``; ``none`` has
  no term. Both losses skip windows whose gold sentence is not visible; ``L_neg`` also skips
  windows without a negative.

Config (``model:`` section): ``class: qa.models.evidence_model:EvidenceModel`` plus
``neg_mode``, ``alpha``, ``beta`` and ``margin``.

C4 columns (Jun Yip, ``sentence_features.py``), all window-local: ``sent_ids`` per token
(-1 outside the context), ``sent_label``, ``neg_cand_mask`` [S] and ``static_neg_label``
(``-100`` = none). C5 adds ``sent_logits`` [B, S] (``MASKED_LOGIT`` on slots with no visible
token) and ``sent_mask`` [B, S].
"""
import torch
import torch.nn.functional as F
from torch import nn

from qa.models.span_model import SpanModel

NEG_MODES = ("model", "lexical", "random", "none")
IGNORE = -100
MASKED_LOGIT = -1e4  # same pad value as predict.LOGIT_PAD


class EvidenceModel(SpanModel):
    input_keys = SpanModel.input_keys + ("sent_ids",)
    label_keys = SpanModel.label_keys + ("sent_label", "neg_cand_mask", "static_neg_label")
    pad_values = {**SpanModel.pad_values, "sent_ids": -1, "neg_cand_mask": 0}

    def __init__(self, encoder, neg_mode, alpha, beta, margin, **cfg):
        if neg_mode not in NEG_MODES:
            raise ValueError(f"neg_mode must be one of {NEG_MODES}, got {neg_mode!r}")
        super().__init__(encoder, **cfg)
        self.neg_mode = neg_mode
        self.alpha, self.beta, self.margin = alpha, beta, margin
        self.ranker = nn.Linear(encoder.config.hidden_size, 1)

    @classmethod
    def prepare_features(cls, features, model_cfg):
        raise NotImplementedError(
            "EvidenceModel needs the C4 sentence fields: add_sentence_fields (Jun Yip, J2) called "
            "with the split's C3 evidence file, which prepare_features is not given yet (Damian, D1)."
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, sent_ids=None,
                start_positions=None, end_positions=None, sent_label=None, neg_cand_mask=None,
                static_neg_label=None, **_):
        out = super().forward(input_ids, attention_mask, token_type_ids, start_positions, end_positions)
        n_slots = int(sent_ids.max()) + 1
        if neg_cand_mask is not None:
            n_slots = max(n_slots, neg_cand_mask.size(1))
            neg_cand_mask = F.pad(neg_cand_mask.bool(), (0, n_slots - neg_cand_mask.size(1)))
        pooled, sent_mask = pool_sentences(out["hidden_states"], sent_ids, max(n_slots, 1))
        scores = self.ranker(self.dropout(pooled)).squeeze(-1)
        out["sent_logits"] = scores.masked_fill(~sent_mask, MASKED_LOGIT)
        out["sent_mask"] = sent_mask
        if "loss" in out and sent_label is not None:
            neg = negative_index(self.neg_mode, out["sent_logits"], sent_mask, sent_label,
                                 neg_cand_mask, static_neg_label)
            l_rank, l_neg = ranking_losses(out["sent_logits"], sent_label, neg, self.margin)
            out["loss"] = out["loss"] + self.alpha * l_rank + self.beta * l_neg
        return out


def pool_sentences(hidden, sent_ids, n_slots):
    """Mean token state per window-local sentence: ``(pooled [B, S, H], mask [B, S])``.
    A sentence cut by the window edge is pooled over its visible tokens only."""
    slots = torch.arange(n_slots, device=sent_ids.device)
    member = (sent_ids.unsqueeze(1) == slots.view(1, -1, 1)).to(hidden.dtype)  # [B, S, T]
    counts = member.sum(-1)
    pooled = (member @ hidden) / counts.clamp(min=1).unsqueeze(-1)
    return pooled, counts > 0


def negative_index(neg_mode, sent_logits, sent_mask, sent_label, neg_cand_mask=None,
                   static_neg_label=None):
    """Window-local index of each window's negative sentence, ``IGNORE`` where there is none.

    ``model``: argmax of the current scores over the candidates (visible, non-gold, in
    ``neg_cand_mask``); ``lexical`` / ``random``: ``static_neg_label``; ``none``: no negative.
    Also used to dump the mined negatives for the J3 audit.
    """
    if neg_mode == "none":
        return torch.full_like(sent_label, IGNORE)
    if neg_mode in ("lexical", "random"):
        if static_neg_label is None:
            raise ValueError(f"neg_mode={neg_mode!r} needs the C4 static_neg_label column")
        return static_neg_label
    if neg_cand_mask is None:
        raise ValueError("neg_mode='model' needs the C4 neg_cand_mask column")
    slots = torch.arange(sent_logits.size(1), device=sent_logits.device)
    cand = neg_cand_mask.bool() & sent_mask & (slots != sent_label.unsqueeze(1))
    best = sent_logits.masked_fill(~cand, float("-inf")).argmax(-1)
    return torch.where(cand.any(-1), best, IGNORE)


def ranking_losses(sent_logits, sent_label, neg_index, margin):
    """``(L_rank, L_neg)``, each the mean over the windows it applies to, or 0 if none do."""
    zero = sent_logits.new_zeros(())
    has_gold = sent_label != IGNORE
    l_rank = F.cross_entropy(sent_logits[has_gold], sent_label[has_gold]) if has_gold.any() else zero
    use = has_gold & (neg_index != IGNORE)
    if not use.any():
        return l_rank, zero
    logits = sent_logits[use]
    s_gold = logits.gather(1, sent_label[use].unsqueeze(1)).squeeze(1)
    s_neg = logits.gather(1, neg_index[use].unsqueeze(1)).squeeze(1)
    return l_rank, F.relu(margin - s_gold + s_neg).mean()
