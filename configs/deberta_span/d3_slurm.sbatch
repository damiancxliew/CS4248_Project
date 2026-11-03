#!/usr/bin/env bash
# D3 grid on the SoC Compute Cluster: one Slurm array task per grid point (0-11), one GPU each.
# Owner: Damian. Submit from the repository root:
#
#   mkdir -p logs
#   sbatch configs/deberta_span/d3_slurm.sbatch                         # all 12 runs
#   sbatch --array=4,6 configs/deberta_span/d3_slurm.sbatch             # only some indices
#   sbatch --gpus=h100-96 configs/deberta_span/d3_slurm.sbatch          # another GPU type
#
# SoC Slurm refuses jobs that do not request a GPU, and caps jobs at 3 hours. bf16 needs
# compute capability >= 8.0, so request A100/H100 by name (a100-80, a100-40, h100-96);
# --gpus=1 alone may land on a Titan V or T4. xgpj0 (an a100-80 node) is aarch64, so an x86 .venv
# cannot run there. Per-user limits allow only one a100-80 at a time; spread the array over GPU
# types to run in parallel, e.g. `sbatch --gpus=h100-96 --array=3,7 ...`. Arguments after the
# script name are passed to train.py as overrides, e.g. `... d3_slurm.sbatch --set train.per_device_batch_size=8`.
#
# Compute nodes read the pretrained checkpoint from the Hugging Face cache (HF_HUB_OFFLINE=1),
# so download it once on the login node first:
#   .venv/bin/python -c "from transformers import AutoModel, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('microsoft/deberta-v3-base'); AutoModel.from_pretrained('microsoft/deberta-v3-base')"

#SBATCH --job-name=d3-grid
#SBATCH --array=0-11
#SBATCH --gpus=a100-80
#SBATCH --exclude=xgpj0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/d3_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1  # step/epoch lines reach the log as they happen

echo "job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-?} on $(hostname) at $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
fi
bash configs/deberta_span/d3_grid.sh run "${SLURM_ARRAY_TASK_ID:?run inside a Slurm array, or set SLURM_ARRAY_TASK_ID}" "$@"
echo "finished at $(date)"
