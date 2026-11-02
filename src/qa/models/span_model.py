"""Span extractor: pretrained encoder + Linear(hidden -> 2) start/end head (README §2.2, row 4).

This is also the interface every model class plugged into ``train.py`` follows, so
Gokul's evidence model can subclass it and be selected with ``model.class`` in a config:

- ``build(model_cfg, encoder_config=None)``: construct from the ``model:`` config section.
  Every key in that section reaches ``__init__`` unchanged (e.g. ``neg_mode``, ``alpha``).
  With ``encoder_config`` (a transformers config object) the encoder is built without
  downloading weights; ``train.py`` uses this when loading a checkpoint.
- ``self.encoder``: the transformers encoder; its config is saved with each checkpoint.
- ``input_keys``: feature columns the model consumes; ``train.py`` batches only these
  plus ``label_keys``. Subclasses extend both (e.g. ``sent_ids``, ``sent_label``).
- ``pad_values``: padding value per column when features differ in length.
- ``prepare_features(features, model_cfg, split=None, data_file=None)``: hook to add
  model-specific columns to the C2 features (e.g. C4 ``add_sentence_fields``). ``split``
  is ``train``, ``val``, ``test`` or a ``predict.py --split`` name (e.g. ``addsent``), so a
  model can load per-split extras such as ``evidence_{split}.jsonl``. Identity here.
- ``forward(**batch)`` returns a dict with the C5 outputs ``start_logits`` and
  ``end_logits`` [B, T] (subclasses may add ``sent_logits`` [B, S_max] and ``sent_mask``),
  ``hidden_states`` [B, T, H], and ``loss`` when the label columns are present.
"""
from torch import nn
from transformers import AutoModel


class SpanModel(nn.Module):
    input_keys = ("input_ids", "attention_mask", "token_type_ids")
    label_keys = ("start_positions", "end_positions")
    pad_values = {"input_ids": 0, "attention_mask": 0, "token_type_ids": 0}

    def __init__(self, encoder, dropout=None, **model_cfg):
        super().__init__()
        self.encoder = encoder
        self.model_cfg = model_cfg
        hidden = encoder.config.hidden_size
        p = encoder.config.hidden_dropout_prob if dropout is None else dropout
        self.dropout = nn.Dropout(p)
        self.qa_outputs = nn.Linear(hidden, 2)

    @classmethod
    def build(cls, model_cfg, encoder_config=None):
        cfg = dict(model_cfg)
        name = cfg.pop("encoder")
        cfg.pop("class", None)
        cfg.pop("tokenizer", None)
        if encoder_config is None:
            encoder = AutoModel.from_pretrained(name)
        else:
            encoder = AutoModel.from_config(encoder_config)
        return cls(encoder, **cfg)

    @classmethod
    def prepare_features(cls, features, model_cfg, split=None, data_file=None):
        return features

    def encode(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # DeBERTa-v3 has no segment embeddings (type_vocab_size = 0); BERT uses them.
        if token_type_ids is not None and self.encoder.config.type_vocab_size > 0:
            kwargs["token_type_ids"] = token_type_ids
        return self.encoder(**kwargs).last_hidden_state

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None, **_):
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        logits = self.qa_outputs(self.dropout(hidden))
        start_logits, end_logits = (x.squeeze(-1) for x in logits.split(1, dim=-1))
        out = {"start_logits": start_logits, "end_logits": end_logits, "hidden_states": hidden}
        if start_positions is not None and end_positions is not None:
            out["loss"] = span_loss(start_logits, end_logits, start_positions, end_positions)
        return out


def span_loss(start_logits, end_logits, start_positions, end_positions):
    """(CE(start) + CE(end)) / 2. Positions outside the window are ignored, as in BERT's QA head;
    C2 already maps out-of-window answers to CLS (index 0), which is a valid target."""
    seq_len = start_logits.size(1)
    start = start_positions.clamp(0, seq_len)
    end = end_positions.clamp(0, seq_len)
    ce = nn.CrossEntropyLoss(ignore_index=seq_len)
    return (ce(start_logits, start) + ce(end_logits, end)) / 2


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
