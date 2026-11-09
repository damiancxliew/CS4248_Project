"""Efficiency table (D5): parameters, training GPU-hours and inference throughput per system.

Every system is timed in one process on the same split, GPU, batch size (``--batch-size``,
default 64 = ``train.eval_batch_size``) and precision (``train.precision``: bf16 on CUDA):

  python -m qa.eval.efficiency time --out results/efficiency/timings.json   # on the GPU
  python -m qa.eval.efficiency table                                         # anywhere, CPU

On the SoC cluster: ``configs/efficiency/d5_slurm.sbatch``.

Stages, each timed on its own (median of ``--repeats`` runs, after one warm-up pass):

- ``segment``: sentence splitting of the split's unique contexts (``evidence_labels.split_sentences``,
  the C3 splitter). Only the evidence model needs it at inference, so it is 0 for span-only models.
- ``features``: C2 windows (tokenisation) plus the model's ``prepare_features`` (the C4 sentence
  fields for the evidence model, built from the precomputed C3 file).
- ``forward``: ``predict.run_inference`` over every window: batching, encoder and heads, copy to host.
- ``decode``: post-processing into answers. ``span``: the standard decoder (λ = 0). ``joint``: the
  C8 ranking + ``evidence_decoder.decode_joint`` at the config's λ. ``sequential``: the C8 ranking
  + span n-best + ``sequential_decoder.decode`` at budget ``--k``.

``model_qps`` = questions / (forward + decode); ``e2e_qps`` = questions / all four stages.

Weights do not change the timing, so by default each model is built from its config (pretrained
encoder, freshly initialised heads) and no checkpoint is needed; ``--checkpoint NAME=DIR`` times a
trained one instead. The table takes training GPU-hours from the C7 records in ``results/runs/``
(mean over the seeds of the same system and variant). Timing uses val; the test split is refused.
"""
import argparse
import csv
import datetime
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from qa.data.evidence_labels import split_sentences
from qa.decoding import sequential_decoder
from qa.decoding.evidence_decoder import decode_joint
from qa.decoding.sentence_ranking import passage_ranking
from qa.decoding.span_decoder import decode as span_decode
from qa.decoding.span_decoder import features_to_arrays
from qa.models.evidence_model import evidence_path
from qa.models.span_model import count_params_m
from qa.predict import (
    amp_precision,
    import_object,
    load_checkpoint,
    read_evidence,
    resolve_device,
    run_inference,
    to_c6_nbest,
)
from qa.train import ROOT, device_name, git_sha, load_config

DECODERS = ("span", "joint", "sequential")
EVIDENCE_DECODERS = ("joint", "sequential")

# name -> (config, decoders). The evidence model's "span" row is R7 (λ = 0): the ranker head is
# computed but unused, which isolates its forward cost from the decoding cost.
SYSTEMS = {
    "bert_span": ("configs/baselines/bert_base.yaml", ("span",)),
    "deberta_span": ("configs/deberta_span/span_only.yaml", ("span",)),
    "deberta_evidence": ("configs/ablations/r4_full.yaml", ("span", "joint", "sequential")),
    "deberta_large_span": ("configs/deberta_span/large_span_only.yaml", ("span",)),
}
BASELINE = ("deberta_span_only", "span")  # the row the ranker-head overhead is measured against

# (C7 system, decoder) -> (CSV label, LaTeX label)
LABELS = {
    ("bert_span_only", "span"): ("BERT-base span-only (R2)", r"BERT-base span-only (R2)"),
    ("deberta_span_only", "span"): ("DeBERTa-v3-base span-only (R3)",
                                    r"DeBERTa-v3-base span-only (R3)"),
    ("deberta_evidence", "span"): ("Evidence model, lambda = 0 (R7)",
                                   r"Evidence model, $\lambda = 0$ (R7)"),
    ("deberta_evidence", "joint"): ("Evidence model, joint (R4)", r"Evidence model, joint (R4)"),
    ("deberta_evidence", "sequential"): ("Evidence model, rank-then-read",
                                         r"Evidence model, rank-then-read"),
    ("deberta_large_span_only", "span"): ("DeBERTa-v3-large span-only (R8)",
                                          r"DeBERTa-v3-large span-only (R8)"),
}


# --- timing ----------------------------------------------------------------------------------

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, repeats, device, warmup=True):
    """``(last result, [seconds per run])``; one untimed warm-up call first unless ``warmup=False``."""
    if warmup:
        fn()
    result, seconds = None, []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        result = fn()
        _sync(device)
        seconds.append(time.perf_counter() - start)
    return result, seconds


def build_model(cfg, checkpoint, device):
    """``(model, tokenizer, cfg)``: a trained checkpoint, or the config's pretrained encoder."""
    if checkpoint:
        model, tokenizer, ckpt_cfg = load_checkpoint(checkpoint, device)
        return model, tokenizer, ckpt_cfg
    model_cls = import_object(cfg["model"]["class"])
    model = model_cls.build(cfg["model"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"].get("tokenizer") or cfg["model"]["encoder"])
    return model, tokenizer, cfg


def time_system(name, cfg, decoders, split="val", checkpoint=None, batch_size=64, repeats=3,
                limit=None, k=1, device="auto"):
    """Time one model on ``split``; one result row per decoder (they share the forward pass)."""
    if split == "test":
        raise SystemExit("Refusing to time on test (dev-v1.1.json); efficiency is measured on val.")
    unknown = set(decoders) - set(DECODERS)
    if unknown:
        raise ValueError(f"unknown decoders {sorted(unknown)}; choose from {DECODERS}")
    device = resolve_device(device)
    torch.manual_seed(0)
    model, tokenizer, cfg = build_model(cfg, checkpoint, device)
    model.eval()
    model_cls = type(model)
    is_evidence = "sent_ids" in model_cls.input_keys
    if not is_evidence and set(decoders) & set(EVIDENCE_DECODERS):
        raise ValueError(f"{name}: joint/sequential decoding needs an evidence model")

    data = cfg["data"]
    data_file = data[f"{split}_file"]
    examples = import_object(data["examples_fn"])(data_file)
    if limit:
        examples = examples.select(range(min(limit, len(examples))))
    contexts = dict(zip(examples["id"], examples["context"]))
    unique_contexts = list(dict.fromkeys(examples["context"]))
    features_fn = import_object(data["features_fn"])

    def segment():
        return [split_sentences(context) for context in unique_contexts]

    def build_features():
        windows = features_fn(examples, tokenizer, data["max_seq_len"], data["doc_stride"])
        return model_cls.prepare_features(windows, cfg["model"], split=split, data_file=data_file)

    stage = {"segment": [0.0] * repeats}
    if is_evidence:
        _, stage["segment"] = timed(segment, repeats, device)
    features, stage["features"] = timed(build_features, repeats, device)

    precision = amp_precision(device, cfg["train"]["precision"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    outputs, stage["forward"] = timed(
        lambda: run_inference(model, features, batch_size, device, precision), repeats, device)
    peak_gb = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None

    evidence = read_evidence(evidence_path(cfg["model"], split, data_file)) if is_evidence else None
    n_best, max_len = cfg["decode"]["n_best"], cfg["decode"]["max_answer_len"]
    lam = cfg["decode"].get("lam", 0.0)

    def decoder(kind):
        def run():
            example_ids, offsets = features_to_arrays(features)
            start, end = outputs["start_logits"], outputs["end_logits"]
            if kind == "span":
                return span_decode(contexts, example_ids, offsets, start, end, n_best, max_len)[0]
            ranking = passage_ranking(example_ids, offsets, outputs["sent_logits"],
                                      outputs["sent_mask"] > 0, evidence)
            if kind == "joint":
                return decode_joint(contexts, example_ids, offsets, start, end, ranking, evidence,
                                    lam, n_best, max_len)[0]
            _, nbest = span_decode(contexts, example_ids, offsets, start, end, n_best, max_len)
            return sequential_decoder.decode(to_c6_nbest(nbest, evidence), ranking, k)
        return run

    n_questions = len(examples)
    rows = []
    for kind in decoders:
        predictions, decode_runs = timed(decoder(kind), repeats, device, warmup=False)
        if len(predictions) != n_questions:
            raise RuntimeError(f"{name}/{kind}: {len(predictions)} answers for {n_questions} questions")
        runs = {**stage, "decode": decode_runs}
        median = {key: statistics.median(values) for key, values in runs.items()}
        model_s = median["forward"] + median["decode"]
        total_s = model_s + median["segment"] + median["features"]
        rows.append({
            "name": name,
            "system": cfg["system"],
            "variant": cfg["variant"],
            "decoder": kind,
            "lam": lam if kind == "joint" else 0.0 if kind == "span" else None,
            "k": k if kind == "sequential" else None,
            "encoder": cfg["model"]["encoder"],
            "checkpoint": str(checkpoint) if checkpoint else None,
            "params_M": round(count_params_m(model), 3),
            "n_questions": n_questions,
            "n_windows": len(features),
            "batch_size": batch_size,
            "precision": precision,
            **{f"{key}_s": round(value, 4) for key, value in median.items()},
            "runs_s": {key: [round(v, 4) for v in values] for key, values in runs.items()},
            "model_qps": round(n_questions / model_s, 2),
            "e2e_qps": round(n_questions / total_s, 2),
            "peak_mem_gb": round(peak_gb, 2) if peak_gb is not None else None,
        })
    del model, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def measure(systems, split="val", checkpoints=None, batch_size=64, repeats=3, limit=None, k=1,
            device="auto", log=print):
    """Time every system in ``systems`` ({name: (config path or dict, decoders)})."""
    checkpoints = checkpoints or {}
    resolved = resolve_device(device)
    rows = []
    for name, (config, decoders) in systems.items():
        cfg = load_config(ROOT / config) if isinstance(config, (str, Path)) else config
        log(f"timing {name} ({', '.join(decoders)}) on {split}")
        rows += time_system(name, cfg, decoders, split, checkpoints.get(name), batch_size, repeats,
                            limit, k, device)
        log(f"  {rows[-1]['n_questions']} questions: " + ", ".join(
            f"{r['decoder']} {r['model_qps']} q/s" for r in rows if r["name"] == name))
    return {
        "meta": {
            "split": split,
            "gpu": device_name(resolved),
            "batch_size": batch_size,
            "repeats": repeats,
            "limit": limit,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "host": platform.node(),
            "git_sha": git_sha(),
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        },
        "rows": rows,
    }


# --- table -----------------------------------------------------------------------------------

def train_hours(runs_dir, system, variant):
    """``(mean, sample std or None, seed count)`` of ``train_gpu_hours`` over the C7 val records."""
    hours = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        record = json.loads(path.read_text())
        if (record.get("system"), record.get("variant"), record.get("split")) == (system, variant, "val") \
                and record.get("train_gpu_hours") is not None:
            hours.append(record["train_gpu_hours"])
    if not hours:
        return None, None, 0
    return statistics.mean(hours), statistics.stdev(hours) if len(hours) > 1 else None, len(hours)


def table_rows(timings, runs_dir="results/runs"):
    """One table row per timed (system, decoder), with training hours and the overheads."""
    rows = timings["rows"]
    base = next((r for r in rows if (r["system"], r["decoder"]) == BASELINE), None)
    out = []
    for r in rows:
        mean_h, std_h, n_h = train_hours(runs_dir, r["system"], r["variant"])
        label = LABELS.get((r["system"], r["decoder"]), (f"{r['system']} ({r['decoder']})",) * 2)
        out.append({
            "label": label[0],
            "label_tex": label[1],
            "system": r["system"],
            "variant": r["variant"],
            "decoder": r["decoder"],
            "params_M": r["params_M"],
            "train_gpu_hours_mean": None if mean_h is None else round(mean_h, 3),
            "train_gpu_hours_std": None if std_h is None else round(std_h, 3),
            "train_seeds": n_h,
            "segment_s": r["segment_s"],
            "features_s": r["features_s"],
            "forward_s": r["forward_s"],
            "decode_s": r["decode_s"],
            "decode_ms_per_q": round(1000 * r["decode_s"] / r["n_questions"], 3),
            "model_qps": r["model_qps"],
            "e2e_qps": r["e2e_qps"],
            "forward_vs_base_pct": None if base is None else round(
                100 * (r["forward_s"] / base["forward_s"] - 1), 1),
            "peak_mem_gb": r["peak_mem_gb"],
            "n_questions": r["n_questions"],
            "gpu": timings["meta"]["gpu"],
        })
    return out


def _fmt(value, digits):
    return "--" if value is None else f"{value:.{digits}f}"


def write_table(rows, meta, out_dir="results/tables", stem="efficiency"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.tex"
    fields = [key for key in rows[0] if key != "label_tex"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "% Generated by python -m qa.eval.efficiency table; do not edit by hand.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"System & Params (M) & Train GPU-h & Forward (s) & Decode (ms/q) & q/s & q/s (end-to-end) \\",
        r"\midrule",
    ]
    for r in rows:
        hours = _fmt(r["train_gpu_hours_mean"], 2)
        if r["train_gpu_hours_std"] is not None:
            hours += rf" $\pm$ {r['train_gpu_hours_std']:.2f}"
        lines.append(" & ".join([
            r["label_tex"], _fmt(r["params_M"], 1), hours, _fmt(r["forward_s"], 1),
            _fmt(r["decode_ms_per_q"], 2), _fmt(r["model_qps"], 0), _fmt(r["e2e_qps"], 0),
        ]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"% {meta['split']}, {rows[0]['n_questions']} questions, batch {meta['batch_size']}, "
        f"{meta['gpu']}, median of {meta['repeats']} runs.",
    ]
    tex_path.write_text("\n".join(lines) + "\n")
    return csv_path, tex_path


# --- CLI -------------------------------------------------------------------------------------

def _parse_checkpoints(items):
    out = {}
    for item in items or ():
        name, sep, path = item.partition("=")
        if not sep or name not in SYSTEMS:
            raise SystemExit(f"--checkpoint expects NAME=DIR with NAME in {sorted(SYSTEMS)}, got {item!r}")
        out[name] = path
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    t = commands.add_parser("time", help="time every system on one GPU and write the timings JSON")
    t.add_argument("--systems", nargs="+", choices=sorted(SYSTEMS),
                   help="default: every system whose config exists")
    t.add_argument("--checkpoint", action="append", metavar="NAME=DIR",
                   help="time a trained checkpoint instead of the pretrained encoder")
    t.add_argument("--split", default="val")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--repeats", type=int, default=3)
    t.add_argument("--limit", type=int, help="first N questions only (quick checks)")
    t.add_argument("--k", type=int, default=1, help="sentence budget for sequential decoding")
    t.add_argument("--device", default="auto")
    t.add_argument("--out", default="results/efficiency/timings.json")

    tab = commands.add_parser("table", help="timings JSON + C7 records -> efficiency CSV and LaTeX")
    tab.add_argument("--timings", default="results/efficiency/timings.json")
    tab.add_argument("--runs-dir", default="results/runs")
    tab.add_argument("--out-dir", default="results/tables")

    args = parser.parse_args(argv)
    if args.command == "time":
        names = args.systems or [n for n, (cfg, _) in SYSTEMS.items() if (ROOT / cfg).exists()]
        timings = measure({n: SYSTEMS[n] for n in names}, args.split, _parse_checkpoints(args.checkpoint),
                          args.batch_size, args.repeats, args.limit, args.k, args.device)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(timings, indent=1) + "\n")
        print(f"timings: {out}")
    else:
        timings = json.loads(Path(args.timings).read_text())
        rows = table_rows(timings, args.runs_dir)
        for path in write_table(rows, timings["meta"], args.out_dir):
            print(f"table: {path}")


if __name__ == "__main__":
    main()
