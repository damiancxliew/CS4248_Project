#!/usr/bin/env bash
# D6 / R8: DeBERTa-v3-large span-only, seed 42, lr 3e-5 (the frozen D3 value) and 2e-5 (one array
# task each). Owner: Damian. Submit from the repository root; setup notes are in d3_slurm.sbatch.
# Download microsoft/deberta-v3-large into the Hugging Face cache on a compute node first.
#
#   mkdir -p logs
#   sbatch configs/deberta_span/r8_slurm.sbatch
#
# Each task writes outputs/r8_deberta_large_span_lr<lr>_s42/ (val only; no --record, the chosen
# run's run.json is copied to results/runs/ afterwards).

#SBATCH --job-name=r8-large
#SBATCH --array=0-1
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpj0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/r8_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"

lrs=(3e-5 2e-5)
lr="${lrs[${SLURM_ARRAY_TASK_ID:?run inside a Slurm array, or set SLURM_ARRAY_TASK_ID}]}"
echo "job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID} lr $lr on $(hostname) at $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
fi
"$python_bin" -m qa.train --config configs/deberta_span/large_span_only.yaml --seed 42 \
    --set "run_id=r8_deberta_large_span_lr$lr" --set "train.learning_rate=$lr" "$@"
echo "finished at $(date)"
