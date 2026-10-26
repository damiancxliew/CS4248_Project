"""Test-only mocks for Damian's D1/D2 tests (TEAM_TASKS §3.1 contracts, docs/tasks/damian.md §5.3).

Until squad_io.py (X8) and features.py (X9) are merged, these stand in for them:
- ``squad_examples``: C1-style SQuAD file -> flat examples.
- ``squad_features``: C2 features, built the Hugging Face reference way (sliding windows,
  offsets nulled outside the context, out-of-window answers -> CLS).
Also a tiny offline tokenizer + encoder, a synthetic SQuAD file, a C3 evidence mock, and a
toy evidence-style model that checks a new model plugs into train.py through config alone.
Nothing here is imported by src/.
"""
import json
import random

import torch
from datasets import Dataset
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizerFast

from qa.models.span_model import SpanModel

THINGS = ["sky", "grass", "sun", "sea", "snow", "rose", "coal", "lemon", "leaf", "cloud"]
COLOURS = ["blue", "green", "yellow", "white", "red", "black", "orange", "grey", "pink", "brown"]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "what", "is", "the", "colour", "of",
         "?", ".", "and", "today", *THINGS, *COLOURS]


def make_tiny_encoder(path):
    """Write a 2-layer random BERT + WordPiece tokenizer to ``path`` (no downloads)."""
    vocab = path / "vocab.txt"
    path.mkdir(parents=True, exist_ok=True)
    vocab.write_text("\n".join(VOCAB) + "\n")
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(path)
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    BertModel(config).save_pretrained(path)
    return path


def make_squad_file(path, n_questions, seed=0):
    """Synthetic SQuAD v1.1 file; each context has 6-9 sentences "the X is Y ." so that
    small windows split it, and each question asks for one sentence's colour."""
    rng = random.Random(seed)
    paragraphs, qid = [], 0
    while qid < n_questions:
        things = rng.sample(THINGS, rng.randint(6, 9))
        colours = [rng.choice(COLOURS) for _ in things]
        sentences = [f"the {t} is {c} ." for t, c in zip(things, colours)]
        context = " ".join(sentences)
        qas = []
        for t, c in rng.sample(list(zip(things, colours)), 3):
            start = context.index(f"the {t} is {c}") + len(f"the {t} is ")
            qas.append({"id": f"q{qid}", "question": f"what is the colour of the {t} ?",
                        "answers": [{"text": c, "answer_start": start}]})
            qid += 1
        paragraphs.append({"context": context, "qas": qas})
    data = {"version": "1.1", "data": [{"title": "colours", "paragraphs": paragraphs}]}
    path.write_text(json.dumps(data))
    return path


def smoke_config(root, model_class="qa.models.span_model:SpanModel", epochs=2, **model_keys):
    """A full train.py config on synthetic data with the tiny encoder, CPU only."""
    from qa.train import load_config

    encoder = make_tiny_encoder(root / "encoder")
    cfg = load_config(overrides=[
        "run_id=smoke", "system=tiny_span", "variant=span_only", "device=cpu",
        f"output_dir={root / 'outputs'}",
        f"model.class={model_class}", f"model.encoder={encoder}",
        f"data.train_file={make_squad_file(root / 'train.json', 150, seed=1)}",
        f"data.val_file={make_squad_file(root / 'val.json', 45, seed=2)}",
        f"data.test_file={make_squad_file(root / 'test.json', 9, seed=3)}",
        "data.examples_fn=damian_mocks:squad_examples",
        "data.features_fn=damian_mocks:squad_features",
        "data.max_seq_len=40", "data.doc_stride=12",
        f"train.epochs={epochs}", "train.learning_rate=1e-3", "train.per_device_batch_size=16",
        "train.grad_accum_steps=2", "train.precision=fp32", "train.eval_batch_size=64",
    ])
    cfg["model"].update(model_keys)
    return cfg


def squad_examples(squad_file):
    """Mock of the X8 loader: flat examples in the HF ``squad`` schema."""
    rows = []
    for article in json.loads(open(squad_file).read())["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                rows.append({
                    "id": qa["id"], "title": article["title"], "context": para["context"],
                    "question": qa["question"],
                    "answers": {"text": [a["text"] for a in qa["answers"]],
                                "answer_start": [a["answer_start"] for a in qa["answers"]]},
                })
    return Dataset.from_list(rows)


def squad_features(examples, tokenizer, max_seq_len, doc_stride):
    """Mock of X9 (C2): HF reference windowing; one dataset for training and evaluation."""
    enc = tokenizer(examples["question"], examples["context"], truncation="only_second",
                    max_length=max_seq_len, stride=doc_stride, return_overflowing_tokens=True,
                    return_offsets_mapping=True, padding="max_length")
    sample_map = enc.pop("overflow_to_sample_mapping")
    starts, ends, example_ids, offsets = [], [], [], []
    for i, offs in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        example = examples[sample_map[i]]
        answer_start = example["answers"]["answer_start"][0]
        answer_end = answer_start + len(example["answers"]["text"][0])
        ctx = [t for t, s in enumerate(seq_ids) if s == 1]
        if offs[ctx[0]][0] > answer_start or offs[ctx[-1]][1] < answer_end:
            starts.append(0)
            ends.append(0)
        else:
            starts.append(next(t for t in ctx if offs[t][1] > answer_start))
            ends.append(next(t for t in reversed(ctx) if offs[t][0] < answer_end))
        example_ids.append(example["id"])
        offsets.append([list(o) if s == 1 else None for o, s in zip(offs, seq_ids)])
    return Dataset.from_dict({
        "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
        "token_type_ids": enc["token_type_ids"], "offset_mapping": offsets,
        "example_id": example_ids, "start_positions": starts, "end_positions": ends,
    })


def evidence_records(examples):
    """Mock C3 records (sentence char offsets + gold sentence) for the synthetic contexts."""
    out = {}
    for ex in examples:
        sentences, pos = [], 0
        while pos < len(ex["context"]):
            end = ex["context"].index(".", pos) + 1
            sentences.append([pos, end])
            pos = end + 1
        start = ex["answers"]["answer_start"][0]
        gold = next(i for i, (cs, ce) in enumerate(sentences) if cs <= start < ce)
        out[ex["id"]] = {"id": ex["id"], "sentences": sentences, "gold_sent": gold}
    return out


class ToyEvidenceModel(SpanModel):
    """Stands in for Gokul's evidence model: extra config key, extra feature column,
    extra C5 output. It must train and predict without any change to train.py."""

    input_keys = SpanModel.input_keys + ("toy_sent_ids",)
    pad_values = {**SpanModel.pad_values, "toy_sent_ids": -1}
    seen_neg_modes = []

    def __init__(self, encoder, neg_mode, **cfg):
        super().__init__(encoder, **cfg)
        self.neg_mode = neg_mode
        ToyEvidenceModel.seen_neg_modes.append(neg_mode)
        self.ranker = nn.Linear(encoder.config.hidden_size, 1)

    @classmethod
    def prepare_features(cls, features, model_cfg):
        # Crude "sentence id" per token: count of '.' tokens seen so far in the window.
        def add(row):
            ids, sent = [], 0
            for offset, token in zip(row["offset_mapping"], row["input_ids"]):
                ids.append(-1 if offset is None else sent)
                sent += token == VOCAB.index(".")
            return {"toy_sent_ids": ids}
        return features.map(add)

    def forward(self, toy_sent_ids, **batch):
        out = super().forward(**batch)
        scores = self.ranker(out["hidden_states"]).squeeze(-1)  # token-level stand-in
        out["sent_logits"] = scores
        out["sent_mask"] = toy_sent_ids >= 0
        if "loss" in out:
            out["loss"] = out["loss"] + 0.0 * scores.sum()
        return out
