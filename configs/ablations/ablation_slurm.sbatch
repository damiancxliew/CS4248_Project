#!/usr/bin/env bash
# Ablation runs (R4-R7, R6b) on the SoC Compute Cluster: one Slurm array task per seed (13, 42, 87).
# Owner: Gokul. Submit all three seeds at once, from the repository root of a git checkout:
#
#   mkdir -p logs
#   sbatch configs/ablations/ablation_slurm.sbatch configs/ablations/r5_no_negatives.yaml    # Xin Hui
#   sbatch configs/ablations/ablation_slurm.sbatch configs/ablations/r6_random_negatives.yaml   # Jun Yip
#   sbatch configs/ablations/ablation_slurm.sbatch configs/ablations/r6b_lexical_negatives.yaml # Clemen
#   sbatch configs/ablations/ablation_slurm.sbatch configs/ablations/r4_full.yaml \
#       configs/ablations/r7_no_evidence_decoding.yaml                                          # Gokul
#   sbatch --array=1 configs/ablations/ablation_slurm.sbatch ...   # rerun one seed (index 1 = 42)
#
# Each task trains <run_id>_s<seed> into outputs/ (checkpoint kept, about 0.74 GB) and writes its
# val record to results/runs/ (configs/ablations/run_ablation.sh; `summary` checks the three).
# make_tables.py puts the seeds in one row only if GPU and git_sha match, hence one submit (one
# commit, clean `git status`) and every task on h100-47 (H100 NVL MIG 3g.47gb, R3's GPU). Some
# h100-47 nodes hand out a whole H100 instead (xgpi15 on 4 Nov, excluded), so a task stops before
# training unless torch sees EXPECT_GPU (default "MIG 3g.47gb"; EXPECT_GPU= skips the check).
# Resubmit a stopped seed with its node added, e.g. `sbatch --exclude=xgpj0,xgpi15,xgpi12 --array=0 ...`.
# Arguments after `--` go to train.py, e.g. `... r5_no_negatives.yaml -- --set train.precision=fp32`.
#
# Before the first submit, once per cluster account (more in configs/deberta_span/d3_slurm.sbatch):
#   - A Python 3.12 .venv with requirements.txt, built on a compute node (srun --gpus=1 --pty bash):
#     the login node can't load torch's CUDA libraries. Set TMPDIR=~/tmp for pip, since /tmp on
#     the login node has a 50 MB per-user quota.
#   - data/processed/{train,val}.json and evidence_{train,val}.jsonl (run_ablation.sh names the
#     command for a missing one) and microsoft/deberta-v3-base in the Hugging Face cache.
#   - A clean `git status`: squad_io prepare rewrites results/tables/data_stats.csv (restore it
#     with git checkout -- results/tables/), otherwise each record's git_sha ends in -dirty.

#SBATCH --job-name=ablation
#SBATCH --array=0-2
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpj0,xgpi15
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/abl_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1  # step/epoch lines reach the log as they happen

seeds=(13 42 87)
seed="${seeds[${SLURM_ARRAY_TASK_ID:?run inside a Slurm array, or set SLURM_ARRAY_TASK_ID}]}"
[[ $# -gt 0 ]] || { echo "usage: sbatch $0 <config.yaml>... [-- train.py overrides]" >&2; exit 2; }
echo "job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID} seed $seed on $(hostname) at $(date): $*"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
fi
expect_gpu="${EXPECT_GPU-MIG 3g.47gb}"
if [[ -n "$expect_gpu" ]]; then  # the name torch reports is the one the records carry
    python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
    gpu_name="$("$python_bin" -c 'import torch; print(torch.cuda.get_device_name(0))')"
    if [[ "$gpu_name" != *"$expect_gpu"* ]]; then
        echo "torch sees '$gpu_name' on $(hostname -s), not '$expect_gpu': stopping before training." >&2
        echo "Resubmit this seed with --exclude=xgpj0,xgpi15,$(hostname -s) (see the header)." >&2
        exit 1
    fi
    echo "GPU: $gpu_name"
fi

configs=()
while (( $# )) && [[ "$1" != -- ]]; do
    configs+=("$1")
    shift
done
bash configs/ablations/run_ablation.sh run "${configs[@]}" "$seed" "$@"
echo "finished at $(date)"
