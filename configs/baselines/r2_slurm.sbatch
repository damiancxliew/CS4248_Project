#!/usr/bin/env bash
# C4 / R2: BERT-base span-only baseline, frozen D3 setting, seeds 13, 42, 87 (one array task each).
# Owner: Clemen. Same job as configs/deberta_span/r3_slurm.sbatch with bert_base.yaml, and the GPU
# check of ablation_slurm.sbatch: make_tables.py puts the seeds in one row only if GPU and git_sha
# match, hence every task on h100-47 (H100 NVL MIG 3g.47gb, R3's GPU) from one clean `main` checkout.
#
#   mkdir -p logs
#   sbatch configs/baselines/r2_slurm.sbatch                           # all three seeds
#   sbatch --array=1 configs/baselines/r2_slurm.sbatch                 # one seed (index 1 = 42)
#   sbatch --exclude=xgpj0,xgpi15,xgpi17 --array=1,2 configs/baselines/r2_slurm.sbatch
#
# Each task writes outputs/r2_bert_span_s<seed>/ and, with --record, results/runs/r2_bert_span_s<seed>.json.
# A task stops before training unless torch sees EXPECT_GPU (default "MIG 3g.47gb"; EXPECT_GPU= skips
# the check); resubmit that seed with the node added to --exclude.
#
# Before the first submit (setup notes for the .venv, data and a clean `git status` are in
# configs/deberta_span/d3_slurm.sbatch and configs/ablations/ablation_slurm.sbatch):
#   - data/processed/{train,val}.json from `python -m qa.data.squad_io prepare` (no evidence files needed).
#   - bert-base-uncased in the Hugging Face cache, fetched once on the login node. Compute nodes are
#     offline, and importing torch or the Rust downloader fails on the login node, hence:
#       HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; \
#         print(snapshot_download('bert-base-uncased', max_workers=1, \
#         ignore_patterns=['*.h5','*.msgpack','*.onnx','*.ot','*.bin']))"
#     (keeps model.safetensors, which transformers loads first; the .bin copy would double the quota use).

#SBATCH --job-name=r2-seeds
#SBATCH --array=0-2
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpj0,xgpi15
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/r2_%A_%a.out

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
expect_gpu="${EXPECT_GPU-MIG 3g.47gb}"
if [[ -n "$expect_gpu" ]]; then  # the name torch reports is the one the records carry
    gpu_name="$("$python_bin" -c 'import torch; print(torch.cuda.get_device_name(0))')"
    if [[ "$gpu_name" != *"$expect_gpu"* ]]; then
        echo "torch sees '$gpu_name' on $(hostname -s), not '$expect_gpu': stopping before training." >&2
        echo "Resubmit this seed with --exclude=xgpj0,xgpi15,$(hostname -s) (see the header)." >&2
        exit 1
    fi
    echo "GPU: $gpu_name"
fi
"$python_bin" -m qa.train --config configs/baselines/bert_base.yaml --seed "$seed" --record "$@"
echo "finished at $(date)"
