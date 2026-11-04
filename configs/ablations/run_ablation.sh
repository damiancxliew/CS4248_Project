#!/usr/bin/env bash
# G4: train, score and record the evidence-model ablation runs (R4-R7, R6b). Owner: Gokul.
#
# One config per row of the ablation table (configs/ablations/*.yaml). For each config and seed:
# train (the epoch is picked on val at lambda 0, as for every model), predict val, decode jointly
# at the config's decode.lam (G2) and score with the official script, writing the C7 record
# results/runs/<run_id>_s<seed>.json. Configs with `reuses: <run_id>` (R7) are not trained: they
# re-decode that run's saved val logits for the same seed into outputs/<run_id>_s<seed>/.
# Checkpoints are kept (final systems). The test set is never read (scripts/run_all.sh, C9).
#
#   bash configs/ablations/run_ablation.sh list                  # the configs and what they change
#   bash configs/ablations/run_ablation.sh run configs/ablations/r5_no_negatives.yaml 42
#   bash configs/ablations/run_ablation.sh run configs/ablations/r4_full.yaml \
#       configs/ablations/r7_no_evidence_decoding.yaml all     # seeds 13, 42, 87 in turn
#   bash configs/ablations/run_ablation.sh summary               # val records per config
#
# On the SoC cluster: configs/ablations/ablation_slurm.sbatch (one array task per seed).
# Extra train.py overrides go after `--`, e.g. `run <config> 42 -- --set train.precision=fp32`.
# OUTPUT_DIR (default outputs) and RESULTS_DIR (default results/runs) move the outputs.
#
# Not train.py --record: that scores the picked epoch at lambda 0 (span-only), which for these
# configs is R7's number, not theirs. Records here are made at each config's own lambda.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
output_dir="${OUTPUT_DIR:-outputs}"
results_dir="${RESULTS_DIR:-results/runs}"
seeds=(13 42 87)
overrides=()

usage() {
    echo "usage: $0 list | run <config.yaml>... <seed|all> [-- train.py overrides...] | summary" >&2
    exit 2
}

run_config() {
    local cfg="$1" seed="$2" info run_id system variant lam reuses val_file evidence_dir
    local name run_dir source_dir out_dir

    # The config with overrides applied; stop before training if an input is missing.
    info="$("$python_bin" - "$cfg" "$seed" "$output_dir" ${overrides[@]+"${overrides[@]}"} <<'EOF'
import json
import sys
from pathlib import Path

from qa.train import load_config

cfg_file, seed, output_dir, args = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
cfg = load_config(cfg_file, overrides=[args[i + 1] for i, arg in enumerate(args) if arg == "--set"])
reuses = cfg.get("reuses")
if reuses:
    # R7: decode the saved logits of the run it reuses, with the data and model settings of that run.
    source = output_dir / f"{reuses}_s{seed}"
    if not (source / "run.json").is_file() or not (source / "val.logits.npz").is_file():
        sys.exit(f"{cfg_file} re-decodes {source}/val.logits.npz, which is missing: run the "
                 f"{reuses} config for seed {seed} first (same OUTPUT_DIR).")
    trained = json.loads((source / "run.json").read_text())["config"]
    for key in ("class", "encoder", "neg_mode", "alpha", "beta", "margin"):
        if trained["model"].get(key) != cfg["model"].get(key):
            sys.exit(f"{source} was trained with model.{key} = {trained['model'].get(key)!r}, "
                     f"but {cfg_file} says {cfg['model'].get(key)!r}.")
    data, evidence_dir = trained["data"], Path(trained["model"].get("evidence_dir", "data/processed"))
    needed = {evidence_dir / "evidence_val.jsonl": "the C3 file the run was trained with"}
else:
    data, evidence_dir = cfg["data"], Path(cfg["model"].get("evidence_dir", "data/processed"))
    needed = {}
    for split in ("train", "val"):
        needed[data[f"{split}_file"]] = "PYTHONPATH=src python -m qa.data.squad_io prepare"
        needed[evidence_dir / f"evidence_{split}.jsonl"] = (
            f"PYTHONPATH=src python -m qa.data.evidence_labels --input {data[f'{split}_file']} "
            f"--split {split} --out-dir {evidence_dir}")
missing = [f"  {path}\n    build it with: {command}" for path, command in needed.items()
           if not Path(path).is_file()]
if missing:
    sys.exit("Missing input files:\n" + "\n".join(missing))
for line in (cfg["run_id"], cfg["system"], cfg["variant"], f"{float(cfg['decode']['lam']):g}",
             reuses or "-", data["val_file"], evidence_dir):
    print(line)
EOF
)"
    { read -r run_id; read -r system; read -r variant; read -r lam; read -r reuses
      read -r val_file; read -r evidence_dir; } <<< "$info"
    name="${run_id}_s${seed}"
    out_dir="$output_dir/$name"

    if [[ "$reuses" == - ]]; then
        "$python_bin" -m qa.train --config "$cfg" --seed "$seed" --set "output_dir=$output_dir" \
            ${overrides[@]+"${overrides[@]}"}
        "$python_bin" -m qa.predict --checkpoint "$out_dir/best" --split val --out-dir "$out_dir"
        source_dir="$out_dir"
    else
        source_dir="$output_dir/${reuses}_s${seed}"
        echo "[$name] no training: decoding $source_dir/val.logits.npz at lambda $lam"
    fi
    "$python_bin" -m qa.decoding.evidence_decoder --logits "$source_dir/val.logits.npz" \
        --evidence "$evidence_dir/evidence_val.jsonl" --data-file "$val_file" --lam "$lam" \
        --out-dir "$out_dir" > /dev/null

    mkdir -p "$results_dir"
    rm -f "$results_dir/$name.json"  # run_eval never overwrites; a rerun replaces this seed's record
    "$python_bin" - "$run_id" "$system" "$variant" "$lam" "$reuses" "$source_dir" "$out_dir" \
        "$val_file" "$results_dir/$name.json" <<'EOF'
import json
import sys
from pathlib import Path

from qa.eval.run_eval import run_eval

run_id, system, variant, lam, reuses, source_dir, out_dir, val_file, out_file = sys.argv[1:]
metadata = json.loads((Path(source_dir) / "run.json").read_text())
config = metadata["config"]
config.setdefault("decode", {})["lam"] = float(lam)  # the record names the lambda it was scored at
if reuses != "-":  # same model and training fields; only the row and the decoding differ
    metadata["system"], metadata["variant"] = system, variant
    config.update(run_id=run_id, system=system, variant=variant, reuses=reuses)
record = run_eval(val_file, Path(out_dir) / f"val.joint_lam{lam}.predictions.json", metadata, "val",
                  Path(out_file).stem, out_file)
print(f"[{record['run_id']}] val EM {record['exact']:.2f} F1 {record['f1']:.2f} (lambda {lam}) -> {out_file}")
EOF
}

case "${1:-}" in
    list)
        "$python_bin" - configs/ablations/*.yaml <<'EOF'
import sys
from pathlib import Path

import yaml

print(f"{'config':<30} {'run_id':<22} {'variant':<22} {'neg_mode':<8} {'beta':>4} {'lambda':>6}")
for path in sys.argv[1:]:
    cfg = yaml.safe_load(open(path))
    model = cfg["model"]
    note = f"  decode-only, reuses {cfg['reuses']}" if cfg.get("reuses") else ""
    print(f"{Path(path).name:<30} {cfg['run_id']:<22} {cfg['variant']:<22} {model['neg_mode']:<8} "
          f"{model['beta']:>4g} {cfg['decode']['lam']:>6g}{note}")
EOF
        ;;
    run)
        shift
        configs=()
        seed_arg=""
        while (( $# )); do
            case "$1" in
                --) shift; overrides=("$@"); break ;;
                *.yaml) configs+=("$1") ;;
                *) [[ -z "$seed_arg" ]] || usage; seed_arg="$1" ;;
            esac
            shift
        done
        (( ${#configs[@]} )) && [[ -n "$seed_arg" ]] || usage
        if [[ "$seed_arg" == all ]]; then
            run_seeds=("${seeds[@]}")
        elif [[ "$seed_arg" =~ ^[0-9]+$ ]]; then
            run_seeds=("$seed_arg")
        else
            echo "seed must be a number or 'all', got '$seed_arg'" >&2
            exit 2
        fi
        if dirty="$(git status --porcelain --untracked-files=no 2>/dev/null)" && [[ -n "$dirty" ]]; then
            echo "Warning: tracked files differ from the commit, so each record's git_sha will end in" \
                 "-dirty (restore them, e.g. git checkout -- results/tables/):" >&2
            echo "$dirty" >&2
        fi
        for seed in "${run_seeds[@]}"; do
            for cfg in "${configs[@]}"; do
                run_config "$cfg" "$seed"
            done
        done
        ;;
    summary)
        "$python_bin" - "$results_dir" configs/ablations/*.yaml <<'EOF'
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import yaml

results_dir, seeds = Path(sys.argv[1]), (13, 42, 87)
print("val records per ablation config; std = sample std (ddof 1), as in make_tables.py")
for path in sys.argv[2:]:
    cfg = yaml.safe_load(open(path))
    records = [json.loads(p.read_text()) for p in sorted(results_dir.glob(f"{cfg['run_id']}_s*.json"))]
    print(f"\n{cfg['run_id']} ({cfg['variant']})")
    if not records:
        print("  no records yet")
        continue
    for r in sorted(records, key=lambda r: r["seed"]):
        print(f"  seed {r['seed']:>2}: F1 {r['f1']:.2f}  EM {r['exact']:.2f}  {r['gpu']}  {r['git_sha']}")
    for metric in ("f1", "exact"):
        values = [r[metric] for r in records]
        spread = f" ± {stdev(values):.2f}" if len(values) > 1 else ""
        print(f"  {'F1' if metric == 'f1' else 'EM'} {mean(values):.2f}{spread} over {len(values)} seed(s)")
    warnings = []
    missing = sorted(set(seeds) - {r["seed"] for r in records})
    if missing:
        warnings.append(f"missing seeds {missing}")
    for field in ("gpu", "git_sha"):
        if len({r[field] for r in records}) > 1:
            warnings.append(f"seeds differ in {field}, so make_tables.py splits them into separate rows")
    if any(str(r["git_sha"]).endswith("-dirty") for r in records):
        warnings.append("git_sha ends in -dirty (tracked files were changed when it ran)")
    if any(r["config"]["decode"]["lam"] != cfg["decode"]["lam"] for r in records):
        warnings.append("a record was scored at a different lambda than the config")
    for warning in warnings:
        print(f"  warning: {warning}")
EOF
        ;;
    *)
        sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
