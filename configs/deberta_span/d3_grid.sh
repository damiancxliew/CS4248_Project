#!/usr/bin/env bash
# D3: DeBERTa-v3-base span-only hyper-parameter search on val (seed 42). Owner: Damian.
#
# Grid: learning_rate {1e-5, 2e-5, 3e-5} x epochs {2, 3} x warmup_ratio {0.06, 0.10} = 12 runs.
# Each run trains on data/processed/train.json and selects its best epoch on val.json.
# The test set is never read (train.py refuses it).
#
#   bash configs/deberta_span/d3_grid.sh list          # show the 12 runs with their index
#   bash configs/deberta_span/d3_grid.sh run 3         # run one (index 0-11)
#   bash configs/deberta_span/d3_grid.sh run all       # run all 12 one after another
#   bash configs/deberta_span/d3_grid.sh summary       # table of finished runs, best val F1 first
#
# Slurm array (one GPU per task):  sbatch --array=0-11 --gres=gpu:1 --wrap \
#   "bash configs/deberta_span/d3_grid.sh run \$SLURM_ARRAY_TASK_ID"
# Extra train.py overrides can be passed after the index, e.g. --set train.precision=fp16 on a T4.
# Damian picks the setting and writes it to configs/deberta_span/best.yaml.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
output_dir="${OUTPUT_DIR:-outputs}"

grid=()
for lr in 1e-5 2e-5 3e-5; do
    for ep in 2 3; do
        for wu in 0.06 0.10; do
            grid+=("$lr $ep $wu")
        done
    done
done

run_id() {  # 2e-5 3 0.06 -> d3_lr2e-5_ep3_wu06
    echo "d3_lr$1_ep$2_wu${3#0.}"
}

run_one() {
    local index="$1"; shift
    if ! [[ "$index" =~ ^[0-9]+$ ]] || (( index >= ${#grid[@]} )); then
        echo "index must be 0-$(( ${#grid[@]} - 1 )), got '$index'" >&2
        exit 2
    fi
    read -r lr ep wu <<< "${grid[$index]}"
    if [[ ! -f data/processed/train.json || ! -f data/processed/val.json ]]; then
        echo "Missing data/processed split. Run: PYTHONPATH=src $python_bin -m qa.data.squad_io prepare" >&2
        exit 1
    fi
    "$python_bin" -m qa.train --config configs/deberta_span/span_only.yaml --seed 42 \
        --set "run_id=$(run_id "$lr" "$ep" "$wu")" --set "output_dir=$output_dir" \
        --set "train.learning_rate=$lr" --set "train.epochs=$ep" --set "train.warmup_ratio=$wu" \
        "$@"
}

case "${1:-}" in
    list)
        for i in "${!grid[@]}"; do
            read -r lr ep wu <<< "${grid[$i]}"
            printf '%2d  %s\n' "$i" "$(run_id "$lr" "$ep" "$wu")_s42"
        done
        ;;
    run)
        [[ $# -ge 2 ]] || { echo "usage: $0 run <index|all> [train.py --set overrides...]" >&2; exit 2; }
        target="$2"; shift 2
        if [[ "$target" == all ]]; then
            for i in "${!grid[@]}"; do run_one "$i" "$@"; done
        else
            run_one "$target" "$@"
        fi
        ;;
    summary)
        "$python_bin" - "$output_dir" <<'EOF'
import json, sys
from pathlib import Path

rows = []
for path in sorted(Path(sys.argv[1]).glob("d3_*_s42/run.json")):
    r = json.loads(path.read_text())
    t = r["config"]["train"]
    rows.append((r["f1"], r["exact"], t["learning_rate"], t["epochs"], t["warmup_ratio"],
                 r["train_gpu_hours"], r["run_id"]))
if not rows:
    sys.exit(f"No finished D3 runs under {sys.argv[1]}/")
print(f"{'val F1':>7} {'val EM':>7} {'lr':>7} {'ep':>3} {'warmup':>6} {'GPU-h':>6}  run_id")
for f1, em, lr, ep, wu, hours, run_id in sorted(rows, reverse=True):
    gpu_h = f"{hours:.2f}" if hours is not None else "-"
    print(f"{f1:7.2f} {em:7.2f} {lr:7.0e} {ep:3d} {wu:6.2f} {gpu_h:>6}  {run_id}")
print(f"\n{len(rows)}/12 runs finished. Best epoch per run is chosen on val F1.")
EOF
        ;;
    *)
        sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
