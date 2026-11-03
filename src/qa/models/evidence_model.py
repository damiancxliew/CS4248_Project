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
``neg_mode``, ``alpha``, ``beta``, ``margin`` and optionally ``evidence_dir`` (default
``data/processed``), where ``prepare_features`` finds the split's C3 ``evidence_{split}.jsonl``.

C4 columns (Jun Yip, ``sentence_features.py``), all window-local: ``sent_ids`` per token
(-1 outside the context), ``sent_label``, ``neg_cand_mask`` [S] and ``static_neg_label``
(``-100`` = none). C5 adds ``sent_logits`` [B, S] (``MASKED_LOGIT`` on slots with no visible
token) and ``sent_mask`` [B, S].

Mined negatives for the J3 audit, from a trained checkpoint:
  python -m qa.models.evidence_model dump-negatives --checkpoint outputs/<run_id>/best --split train
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qa.data.sentence_features import NOT_IN_SENTENCE, add_sentence_fields, token_sentences
from qa.decoding.span_decoder import features_to_arrays
from qa.models.span_model import SpanModel
from qa.predict import (
    amp_precision,
    import_object,
    load_checkpoint,
    read_evidence,
    resolve_device,
    run_inference,
)

NEG_MODES = ("model", "lexical", "random", "none")
IGNORE = -100
MASKED_LOGIT = -1e4  # same pad value as predict.LOGIT_PAD
EVIDENCE_DIR = "data/processed"


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
    def prepare_features(cls, features, model_cfg, split=None, data_file=None):
        """Add the C4 sentence fields from the split's C3 file (``add_sentence_fields``, J2)."""
        path = evidence_path(model_cfg, split, data_file)
        return add_sentence_fields(features, read_evidence(path), model_cfg["neg_mode"])

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


def evidence_path(model_cfg, split, data_file=None):
    """``<evidence_dir>/evidence_{split}.jsonl``; the error names the command that builds it."""
    if split is None:
        raise ValueError("EvidenceModel needs the split name to find its C3 file evidence_{split}.jsonl")
    path = Path(model_cfg.get("evidence_dir", EVIDENCE_DIR)) / f"evidence_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: PYTHONPATH=src python -m qa.data.evidence_labels "
            f"--input {data_file or '<squad file>'} --split {split} --out-dir {path.parent}")
    return path


def passage_negatives(example_ids, offsets, evidence, mined):
    """Each question's first mined negative (window slot, ``IGNORE`` = none) as a C3 sentence index.

    Slot j is the window's j-th visible sentence in passage order, as in C4 (``token_sentences``).
    """
    out = {}
    for i, qid in enumerate(example_ids):
        if mined[i] == IGNORE or qid in out:
            continue
        ids = token_sentences(offsets[i], evidence[qid]["sentences"])
        out[qid] = int(np.unique(ids[ids != NOT_IN_SENTENCE])[mined[i]])
    return out


def dump_mined_negatives(checkpoint, split, out_file, qids=None, n=100, seed=2026, data_file=None,
                         device="auto"):
    """Write each question's model-mined negative (one JSONL row) for the J3 audit.

    ``qids``: the questions to dump; default ``n`` sampled from the split with ``seed``, listed in
    ``<out_file stem>.qids.txt`` so the audit uses the same ones. A question's negative comes from
    its first window that shows the gold sentence and a candidate (``mined_neg`` null if none).
    Sentence indices are C3 passage sentences. Returns the rows.
    """
    if split == "test":
        raise SystemExit("The J3 audit uses train questions; refusing to dump the test set.")
    device = resolve_device(device)
    model, tokenizer, cfg = load_checkpoint(checkpoint, device)
    data_file = data_file or cfg["data"][f"{split}_file"]
    examples = import_object(cfg["data"]["examples_fn"])(data_file)
    if qids is None:
        qids = random.Random(seed).sample(sorted(examples["id"]), min(n, len(examples)))
    wanted = set(qids)
    examples = examples.filter(lambda ex: ex["id"] in wanted)
    features = import_object(cfg["data"]["features_fn"])(
        examples, tokenizer, cfg["data"]["max_seq_len"], cfg["data"]["doc_stride"])
    features = type(model).prepare_features(features, cfg["model"], split=split, data_file=data_file)
    outputs = run_inference(model, features, cfg["train"]["eval_batch_size"], device,
                            amp_precision(device, cfg["train"]["precision"]))

    sent_logits = torch.tensor(outputs["sent_logits"])
    width = sent_logits.size(1)
    cand = torch.tensor([row + [0] * (width - len(row)) for row in features["neg_cand_mask"]])
    mined = negative_index("model", sent_logits, torch.tensor(outputs["sent_mask"] > 0),
                           torch.tensor(features["sent_label"]), cand).tolist()
    evidence = read_evidence(evidence_path(cfg["model"], split, data_file))
    example_ids, offsets = features_to_arrays(features)
    mined_neg = passage_negatives(example_ids, offsets, evidence, mined)

    by_id = {ex["id"]: ex for ex in examples}
    rows = [
        {
            "id": qid,
            "question": by_id[qid]["question"],
            "gold_sent": evidence[qid]["gold_sent"],
            "mined_neg": mined_neg.get(qid),
            "lex_neg": evidence[qid]["lex_neg"],
            "rand_neg": evidence[qid]["rand_neg"],
            "sentences": [by_id[qid]["context"][s:e] for s, e in evidence[qid]["sentences"]],
        }
        for qid in qids
    ]
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out_file.with_suffix(".qids.txt").write_text("\n".join(qids) + "\n")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evidence-model utilities.")
    commands = parser.add_subparsers(dest="command", required=True)
    dump = commands.add_parser("dump-negatives", help="model-mined negatives for the J3 audit")
    dump.add_argument("--checkpoint", required=True, help="e.g. outputs/<run_id>/best")
    dump.add_argument("--split", default="train")
    dump.add_argument("--out", help="default: <run dir>/mined_negatives_<split>.jsonl")
    dump.add_argument("--qids", help="file with one question id per line (default: sample --n)")
    dump.add_argument("--n", type=int, default=100)
    dump.add_argument("--seed", type=int, default=2026)
    dump.add_argument("--data-file", help="SQuAD-format file (default: data.<split>_file in the config)")
    dump.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    qids = Path(args.qids).read_text().split() if args.qids else None
    out = args.out or Path(args.checkpoint).parent / f"mined_negatives_{args.split}.jsonl"
    rows = dump_mined_negatives(args.checkpoint, args.split, out, qids, args.n, args.seed,
                                args.data_file, args.device)
    print(f"{len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
