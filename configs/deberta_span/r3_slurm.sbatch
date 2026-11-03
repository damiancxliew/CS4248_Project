#!/usr/bin/env bash
# D4 / R3: DeBERTa-v3-base span-only, frozen D3 setting, seeds 13, 42, 87 (one array task each).
# Owner: Damian. Submit from the repository root; setup notes are in d3_slurm.sbatch:
#
#   mkdir -p logs
#   sbatch configs/deberta_span/r3_slurm.sbatch                         # all three seeds
#   sbatch --gpus=h100-96 --array=2 configs/deberta_span/r3_slurm.sbatch  # one seed, another GPU type
#
# Each task writes outputs/r3_deberta_span_s<seed>/ and, with --record, results/runs/r3_deberta_span_s<seed>.json.
# Run from a git checkout (or a copy with .git) so the record's git_sha is filled.

#SBATCH --job-name=r3-seeds
#SBATCH --array=0-2
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpj0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/r3_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"

seeds=(13 42 87)
seed="${seeds[${SLURM_ARRAY_TASK_ID:?run inside a Slurm array, or set SLURM_ARRAY_TASK_ID}]}"
echo "job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID} seed $seed on $(hostname) at $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
fi
"$python_bin" -m qa.train --config configs/deberta_span/span_only.yaml --seed "$seed" --record "$@"
echo "finished at $(date)"
