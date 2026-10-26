"""Run a trained checkpoint on any split and write C6 predictions plus raw logits (D2).

Usage:
  python -m qa.predict --checkpoint outputs/r3_deberta_span_s42/best --split val
  python -m qa.predict --checkpoint ... --data-file data/processed/addsent.json --split addsent
  python -m qa.predict --checkpoint ... --split val --evidence data/processed/evidence_val.jsonl

Writes to ``<out-dir>`` (default: the checkpoint's run directory):
- ``<split>.predictions.json``: C6 official format ``{qid: answer_text}``.
- ``<split>.nbest.json``: C6 ``{qid: [{text, start_logit, end_logit, sent_idx, sent_logprob, score}]}``.
  ``sent_idx`` is filled from the C3 evidence file when ``--evidence`` is given, else null;
  ``sent_logprob`` is null for the span decoder.
- ``<split>.logits.npz``: ``example_ids``, ``offsets`` [N, T, 2] (-1 = not context),
  ``start_logits``, ``end_logits`` [N, T], plus any other C5 output the model returns
  (e.g. ``sent_logits``). Decoders can be swept on this without re-running the model.

The test split (``dev-v1.1.json``) is only for checkpoints in ``configs/final_checkpoints.yaml``
and only through ``scripts/run_all.sh``, which passes ``--allow-test``.

This module also holds the inference helpers ``train.py`` uses.
"""
import argparse
import contextlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoTokenizer

from qa.decoding.span_decoder import decode, features_to_arrays

LOGIT_PAD = -1e4  # value for window positions past a short batch; never decoded (offset -1)
EXTRA_OUTPUTS = ("sent_logits", "sent_mask")


def import_object(path):
    """``"package.module:Name"`` -> the object."""
    module, _, name = path.partition(":")
    if not name:
        raise ValueError(f"expected 'module:name', got {path!r}")
    return getattr(importlib.import_module(module), name)


def build_split(cfg, data_file, tokenizer, model_cls):
    """Examples and features for one SQuAD-format file, through the configured C1/C2 code."""
    data = cfg["data"]
    for key in ("examples_fn", "features_fn"):
        if not data.get(key):
            raise ValueError(
                f"data.{key} is not set. It must point at squad_io.py / features.py "
                "(Xin Hui, X8/X9) once they are merged."
            )
    examples = import_object(data["examples_fn"])(data_file)
    features = import_object(data["features_fn"])(
        examples, tokenizer, data["max_seq_len"], data["doc_stride"]
    )
    return examples, model_cls.prepare_features(features, cfg["model"])


def collate(rows, keys, pad_values):
    """Stack the given columns; pad list-valued columns to the longest row in the batch."""
    batch = {}
    for key in keys:
        if key not in rows[0]:
            continue
        values = [row[key] for row in rows]
        if isinstance(values[0], (list, tuple)):
            width = max(len(v) for v in values)
            pad = pad_values.get(key, 0)
            values = [list(v) + [pad] * (width - len(v)) for v in values]
        batch[key] = torch.tensor(values)
    return batch


def batches(features, keys, pad_values, batch_size, shuffle=False, generator=None):
    columns = [k for k in keys if k in features.column_names]
    dataset = features.select_columns(columns)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda rows: collate(rows, columns, pad_values),
    )


def amp_precision(device, precision):
    """Mixed precision only on CUDA; CPU/MPS runs (tests, local debugging) use fp32."""
    return precision if device.type == "cuda" else "fp32"


def autocast(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def run_inference(model, features, batch_size, device, precision="fp32"):
    """All C5 outputs for every feature, as numpy arrays padded to the longest window."""
    model.eval()
    chunks = {}
    for batch in batches(features, model.input_keys, model.pad_values, batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device, precision):
            out = model(**batch)
        for key in ("start_logits", "end_logits", *EXTRA_OUTPUTS):
            if key in out:
                chunks.setdefault(key, []).append(out[key].float().cpu().numpy())
    return {key: _concat_padded(parts, LOGIT_PAD if key.endswith("logits") else 0)
            for key, parts in chunks.items()}


def _concat_padded(parts, pad):
    width = max(p.shape[1] for p in parts)
    padded = [np.pad(p, ((0, 0), (0, width - p.shape[1])), constant_values=pad) for p in parts]
    return np.concatenate(padded)


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, tokenizer, cfg, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    model.encoder.config.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path)
    (path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def load_checkpoint(path, device="cpu"):
    """``(model, tokenizer, cfg)`` from a directory written by ``save_checkpoint``."""
    path = Path(path)
    cfg = yaml.safe_load((path / "config.yaml").read_text())
    model_cls = import_object(cfg["model"]["class"])
    model = model_cls.build(cfg["model"], encoder_config=AutoConfig.from_pretrained(path / "encoder"))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model.to(device), tokenizer, cfg


def sentence_index(evidence, start_char, end_char):
    """Index of the C3 sentence containing the span start, or None."""
    for idx, (cs, ce) in enumerate(evidence["sentences"]):
        if cs <= start_char < ce:
            return idx
    return None


def to_c6_nbest(nbest, evidence=None):
    """Span-decoder n-best -> C6 entries; ``sent_idx`` comes from C3 when given."""
    out = {}
    for qid, entries in nbest.items():
        ev = evidence.get(qid) if evidence else None
        out[qid] = [
            {
                "text": e["text"],
                "start_logit": e["start_logit"],
                "end_logit": e["end_logit"],
                "sent_idx": sentence_index(ev, e["start_char"], e["end_char"]) if ev else None,
                "sent_logprob": None,
                "score": e["score"],
            }
            for e in entries
        ]
    return out


def read_evidence(path):
    with open(path) as f:
        return {rec["id"]: rec for rec in map(json.loads, f) if rec}


def predict(checkpoint, split, data_file=None, out_dir=None, evidence_file=None,
            device="auto", allow_test=False, batch_size=None):
    """Write the C6 files and logits for one split; returns the output paths."""
    if split == "test" and not allow_test:
        raise SystemExit(
            "Refusing to predict on test (dev-v1.1.json). Only final checkpoints, via "
            "scripts/run_all.sh, which passes --allow-test."
        )
    device = resolve_device(device)
    model, tokenizer, cfg = load_checkpoint(checkpoint, device)
    data_file = data_file or cfg["data"][f"{split}_file"]
    examples, features = build_split(cfg, data_file, tokenizer, type(model))
    outputs = run_inference(
        model, features, batch_size or cfg["train"]["eval_batch_size"], device,
        amp_precision(device, cfg["train"]["precision"]),
    )
    example_ids, offsets = features_to_arrays(features)
    contexts = dict(zip(examples["id"], examples["context"]))
    predictions, nbest = decode(
        contexts, example_ids, offsets, outputs["start_logits"], outputs["end_logits"],
        n_best=cfg["decode"]["n_best"], max_answer_len=cfg["decode"]["max_answer_len"],
    )
    evidence = read_evidence(evidence_file) if evidence_file else None

    out_dir = Path(out_dir or Path(checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": out_dir / f"{split}.predictions.json",
        "nbest": out_dir / f"{split}.nbest.json",
        "logits": out_dir / f"{split}.logits.npz",
    }
    paths["predictions"].write_text(json.dumps(predictions, indent=1))
    paths["nbest"].write_text(json.dumps(to_c6_nbest(nbest, evidence)))
    np.savez_compressed(paths["logits"], example_ids=np.array(example_ids), offsets=offsets, **outputs)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, help="train, val, test, or a name for --data-file")
    parser.add_argument("--data-file", help="SQuAD-format file (default: data.<split>_file in the config)")
    parser.add_argument("--evidence", help="C3 evidence_<split>.jsonl, to fill sent_idx in nbest.json")
    parser.add_argument("--out-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    paths = predict(args.checkpoint, args.split, args.data_file, args.out_dir, args.evidence,
                    args.device, args.allow_test, args.batch_size)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
