#!/usr/bin/env bash
# G3 grid on the SoC Compute Cluster: one Slurm array task per grid point (0-11), one GPU each.
# Owner: Gokul. Cluster notes (GPU types, the 3-hour cap, xgpj0, the offline Hugging Face cache)
# are in configs/deberta_span/d3_slurm.sbatch. Submit from the repository root:
#
#   mkdir -p logs
#   sbatch configs/evidence/g3_slurm.sbatch                    # all 12 runs, at most 3 at a time
#   sbatch --array=0 configs/evidence/g3_slurm.sbatch          # one run (a first check)
#   sbatch --array=1-11%3 configs/evidence/g3_slurm.sbatch     # the rest; keep the %3 when overriding
#
# Keep every task on h100-47 (H100 NVL MIG 3g.47gb, R3's GPU): the same setting moves about 0.2 F1
# across GPU types, the pick's tie margin. Some h100-47 nodes hand out a whole H100 instead
# (xgpi15 on 4 Nov, so it is excluded), so a task stops before training unless torch sees
# EXPECT_GPU (default "MIG 3g.47gb"; EXPECT_GPU= skips the check). Resubmit a stopped index with
# its node added, e.g. `sbatch --exclude=xgpj0,xgpi15,xgpi12 --array=5 ...`.
# At most 3 tasks at a time also caps the disk at about three models (0.74 GB each), since each
# task deletes its model once scored (g3_grid.sh).
#
# Before the first submit: the splits and both C3 files must exist (bash configs/evidence/g3_grid.sh
# run <i> names the command for any missing one), and the pretrained checkpoint must be in the
# Hugging Face cache (see d3_slurm.sbatch). Install and download on a compute node
# (srun --gpus=1 --time=00:30:00 --pty bash): the login node can't load torch's CUDA libraries.
# Each task trains outputs/g3_<setting>_s42/, predicts val, and writes 5 records (one per lambda)
# to results/runs/g3_grid/. Arguments after the script name are passed to train.py as overrides.
# Run from a git checkout so each record's git_sha is filled.

#SBATCH --job-name=g3-grid
#SBATCH --array=0-11%3
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpj0,xgpi15
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
expect_gpu="${EXPECT_GPU-MIG 3g.47gb}"
if [[ -n "$expect_gpu" ]]; then  # the name torch reports is the one the records carry
    python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
    gpu_name="$("$python_bin" -c 'import torch; print(torch.cuda.get_device_name(0))')"
    if [[ "$gpu_name" != *"$expect_gpu"* ]]; then
        echo "torch sees '$gpu_name' on $(hostname -s), not '$expect_gpu': stopping before training." >&2
        echo "Resubmit this index with --exclude=xgpj0,xgpi15,$(hostname -s) (see the header)." >&2
        exit 1
    fi
    echo "GPU: $gpu_name"
fi
bash configs/evidence/g3_grid.sh run "${SLURM_ARRAY_TASK_ID:?run inside a Slurm array, or set SLURM_ARRAY_TASK_ID}" "$@"
echo "finished at $(date)"
