#!/usr/bin/env bash
# C9: the final test run. Scores the 18 frozen checkpoints of configs/final_checkpoints.yaml on the
# test set, then runs the sequential pipelines, the ranking and adversarial tables and the error
# analysis, through scripts/run_all.sh. Owner: Clemen (the test-set gatekeeper). One GPU, one job.
#
# This is the only job that touches test (dev-v1.1.json). Submit it once, from a clean checkout of
# `main` after the freeze PR is merged. Nothing in it chooses anything on test: lambda and k come
# from configs/final_checkpoints.yaml, set on val.
#
# Before submitting, on the login node (compute nodes are offline):
#   git checkout main && git pull && git status --short          # clean
#   mkdir -p logs
#   bash scripts/run_all.sh verify                                  # 18 ok, exit 0
#   bash scripts/run_all.sh data                                    # val/test/AddSent/AddOneSent evidence files
#   .venv/bin/python -c "import spacy; spacy.load('en_core_web_sm')"   # the errors stage needs the model
# then:
#   sbatch configs/final_test.sbatch
#
# Stops at the first failure (set -e). Records are never overwritten: run_all.sh skips a run whose
# results/runs/test/*.json exists, so a resubmit after a failure continues where it stopped (delete a
# record to re-score that run). After the job: `bash scripts/run_all.sh tables` and
# `bash scripts/run_all.sh efficiency` (CPU, login node), then check results/runs/test/ holds
# 18 (R2-R6b) + 3 (R7) + 6 (R9a, R9b) = 27 records.

#SBATCH --job-name=final-test
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpj0
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/final_test_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "git: $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | head -1)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
fi

# The test records reuse each training record's GPU and git_sha, so the GPU used here is not recorded
# and any node but the aarch64 xgpj0 (excluded above) works.
for stage in verify test sequential ranking adversarial errors; do
    echo "=== run_all.sh $stage at $(date) ==="
    bash scripts/run_all.sh "$stage"
done
echo "finished at $(date)"
