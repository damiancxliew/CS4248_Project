#!/usr/bin/env bash
# G3 grid on the SoC Compute Cluster: one Slurm array task per grid point (0-11), one GPU each.
# Owner: Gokul. Cluster notes (GPU types, the 3-hour cap, xgpj0, the offline Hugging Face cache)
# are in configs/deberta_span/d3_slurm.sbatch. Submit from the repository root:
#
#   mkdir -p logs
#   sbatch configs/evidence/g3_slurm.sbatch                              # all 12 runs
#   sbatch --array=4,6 configs/evidence/g3_slurm.sbatch                  # only some indices
#   sbatch --gpus=h100-96 --array=6-11 configs/evidence/g3_slurm.sbatch  # another GPU type
#
# Before the first submit, on the login node: the splits and both C3 files must exist
# (bash configs/evidence/g3_grid.sh run <i> names the command for any missing one), and the
# pretrained checkpoint must be in the Hugging Face cache (see d3_slurm.sbatch).
# Each task trains outputs/g3_<setting>_s42/, predicts val, and writes 5 records (one per lambda)
# to results/runs/g3_grid/. Arguments after the script name are passed to train.py as overrides.
# Run from a git checkout so each record's git_sha is filled.

#SBATCH --job-name=g3-grid
#SBATCH --array=0-11
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpj0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/g3_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1  # step/epoch lines reach the log as they happen

echo "job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-?} on $(hostname) at $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
fi
bash configs/evidence/g3_grid.sh run "${SLURM_ARRAY_TASK_ID:?run inside a Slurm array, or set SLURM_ARRAY_TASK_ID}" "$@"
echo "finished at $(date)"
