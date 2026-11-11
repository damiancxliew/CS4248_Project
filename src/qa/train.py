"""Train any span-style QA model from a YAML config, select the best epoch on val (D1).

Usage:
  python -m qa.train --config configs/deberta_span/span_only.yaml --seed 42
  python -m qa.train --config ... --set train.learning_rate=2e-5 --set train.epochs=3
  python -m qa.train --config ... --seed 13 --record     # also writes results/runs/<run_id>.json

Config: ``configs/base.yaml``, overlaid by ``--config``, overlaid by ``--set key.path=value``
(values parsed as YAML). The model class comes from ``model.class`` ("module:Class", see
``qa.models.span_model`` for the interface), so new models need no change here. Every key
under ``model:`` reaches the model unchanged.

Each epoch the model is evaluated on val with the standard span decoder (λ = 0) and the
official EM/F1; the best-F1 epoch is saved to ``<output_dir>/<run_id>/best/``. Evidence
weights such as λ are tuned afterwards on saved logits. The C7 run record for that epoch
is written to ``<output_dir>/<run_id>/run.json`` (and to ``results/runs/`` with ``--record``).
Optimiser: AdamW, linear decay with warm-up (TEAM_TASKS §2.2).
"""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from qa.decoding.span_decoder import decode, features_to_arrays
from qa.models.span_model import count_params_m
from qa.predict import (
    amp_precision,
    autocast,
    batches,
    build_split,
    import_object,
    resolve_device,
    run_inference,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_EVAL = ROOT / "src/qa/eval/evaluate-v2.0.py"


# --- config -------------------------------------------------------------------------------

def load_config(config_path=None, base_path=ROOT / "configs/base.yaml", overrides=()):
    cfg = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
    if config_path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {})
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        set_dotted(cfg, key, parse_value(value))
    return cfg


def parse_value(text):
    """YAML scalar, except that '2e-5' is a float (PyYAML needs '2.0e-5')."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def deep_merge(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def set_dotted(cfg, dotted, value):
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def check_no_test_leak(cfg):
    test = Path(cfg["data"]["test_file"]).resolve()
    for split in ("train", "val"):
        if Path(cfg["data"][f"{split}_file"]).resolve() == test:
            raise ValueError(f"data.{split}_file is the test set ({test}); tune on val only.")


# --- reproducibility and records -----------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def device_name(device):
    return torch.cuda.get_device_name(device) if device.type == "cuda" else device.type


def run_record(cfg, run_id, metrics, params_m, train_hours, device):
    """C7 run record (ranking fields and infer_qps are filled by J6 / D5, not here)."""
    return {
        "run_id": run_id,
        "system": cfg["system"],
        "variant": cfg["variant"],
        "seed": cfg["seed"],
        "split": "val",
        "exact": metrics["exact"],
        "f1": metrics["f1"],
        "recall_at_1": None,
        "recall_at_3": None,
        "mrr": None,
        "n": metrics["n"],
        "params_M": round(params_m, 3),
        "train_gpu_hours": round(train_hours, 4) if device.type == "cuda" else None,
        "gpu": device_name(device),
        "infer_qps": None,
        "git_sha": git_sha(),
        "config": cfg,
    }


# --- evaluation ----------------------------------------------------------------------------

def _official_module():
    spec = importlib.util.spec_from_file_location("squad_official_eval", OFFICIAL_EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_scores(squad_file, predictions):
    """EM/F1 (0-100) from the unmodified official script on a SQuAD-format file.

    Interim until run_eval.py (Xin Hui, X2) exposes a callable API; it computes the same thing.
    """
    official = _official_module()
    dataset = json.loads(Path(squad_file).read_text(encoding="utf-8"))["data"]
    qids = {qa["id"] for article in dataset for p in article["paragraphs"] for qa in p["qas"]}
    missing = qids - set(predictions)
    if missing:  # the official script skips missing qids instead of scoring them 0
        raise ValueError(f"{len(missing)} questions have no prediction, e.g. {sorted(missing)[:3]}")
    with contextlib.redirect_stdout(io.StringIO()):
        exact_raw, f1_raw = official.get_raw_scores(dataset, predictions)
    scores = official.make_eval_dict(exact_raw, f1_raw)
    return {"exact": scores["exact"], "f1": scores["f1"], "n": scores["total"]}


def evaluate(model, examples, features, squad_file, cfg, device):
    outputs = run_inference(model, features, cfg["train"]["eval_batch_size"], device,
                            amp_precision(device, cfg["train"]["precision"]))
    example_ids, offsets = features_to_arrays(features)
    contexts = dict(zip(examples["id"], examples["context"]))
    predictions, _ = decode(contexts, example_ids, offsets, outputs["start_logits"],
                            outputs["end_logits"], n_best=cfg["decode"]["n_best"],
                            max_answer_len=cfg["decode"]["max_answer_len"])
    return official_scores(squad_file, predictions)


# --- training ------------------------------------------------------------------------------

def make_optimizer(model, train_cfg):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": train_cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=train_cfg["learning_rate"])


def check_precision(device, precision):
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"train.precision must be fp32, bf16 or fp16, got {precision!r}")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU has no bf16 (e.g. T4). Set train.precision to fp16, or fp32 "
                         "for DeBERTa-v3 if fp16 overflows (TEAM_TASKS §2.2).")


def train(cfg, record=False, log=print):
    """Train one seed. Returns the C7 run record of the best val epoch."""
    for key in ("run_id", "system", "variant"):
        if not cfg.get(key):
            raise ValueError(f"config key {key!r} is required (set it in the experiment config)")
    check_no_test_leak(cfg)
    run_id = f"{cfg['run_id']}_s{cfg['seed']}"
    run_dir = Path(cfg["output_dir"]) / run_id
    tcfg = cfg["train"]

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))
    check_precision(device, tcfg["precision"])
    precision = amp_precision(device, tcfg["precision"])

    model_cls = import_object(cfg["model"]["class"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"].get("tokenizer") or cfg["model"]["encoder"])
    model = model_cls.build(cfg["model"]).to(device)
    _, train_features = build_split(cfg, cfg["data"]["train_file"], tokenizer, model_cls, "train")
    val_examples, val_features = build_split(cfg, cfg["data"]["val_file"], tokenizer, model_cls, "val")

    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = batches(train_features, model_cls.input_keys + model_cls.label_keys, model_cls.pad_values,
                     tcfg["per_device_batch_size"], shuffle=True, generator=generator)
    accum = tcfg["grad_accum_steps"]
    total_steps = math.ceil(len(loader) / accum) * tcfg["epochs"]
    optimizer = make_optimizer(model, tcfg)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    wandb_run = _start_wandb(cfg, run_id)

    log(f"[{run_id}] {len(train_features)} train / {len(val_features)} val windows, "
        f"{total_steps} optimiser steps on {device_name(device)} ({precision})")
    best, step, start = None, 0, time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device, precision):
                loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {i} "
                                         f"(fp16 overflow? try bf16 or fp32)")
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % tcfg["log_every"] == 0:
                    log(f"[{run_id}] step {step}/{total_steps} loss {loss.item():.4f}")
                    if wandb_run:
                        wandb_run.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)

        metrics = evaluate(model, val_examples, val_features, cfg["data"]["val_file"], cfg, device)
        log(f"[{run_id}] epoch {epoch}: val EM {metrics['exact']:.2f} F1 {metrics['f1']:.2f}")
        if wandb_run:
            wandb_run.log({"val/exact": metrics["exact"], "val/f1": metrics["f1"], "epoch": epoch}, step=step)
        if best is None or metrics["f1"] > best["f1"]:
            best = {**metrics, "epoch": epoch}
            save_checkpoint(model, tokenizer, cfg, run_dir / "best")

    log(f"[{run_id}] best epoch {best['epoch']}: val F1 {best['f1']:.2f}")
    result = run_record(cfg, run_id, best, count_params_m(model), (time.time() - start) / 3600, device)
    write_json(run_dir / "run.json", result)
    if record:
        write_json(ROOT / "results/runs" / f"{run_id}.json", result)
    if wandb_run:
        wandb_run.finish()
    return result


def _start_wandb(cfg, run_id):
    if not cfg.get("wandb", {}).get("enabled"):
        return None
    import wandb  # optional; not in requirements.txt

    return wandb.init(project=cfg["wandb"]["project"], name=run_id, config=cfg)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", help="experiment config, overlaid on configs/base.yaml")
    parser.add_argument("--base", default=ROOT / "configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--record", action="store_true", help="also write results/runs/<run_id>.json")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.base, args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    result = train(cfg, record=args.record)
    print(json.dumps({k: result[k] for k in ("run_id", "exact", "f1", "n")}))


if __name__ == "__main__":
    main()
