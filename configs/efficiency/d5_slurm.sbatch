#!/usr/bin/env bash
# D5: time every system on one GPU for the efficiency table (src/qa/eval/efficiency.py). Owner: Damian.
# Submit from the repository root of a clean git checkout; setup notes are in
# configs/deberta_span/d3_slurm.sbatch. Once per account, on a compute node (srun -p normal --pty bash):
#   - bert-base-uncased, microsoft/deberta-v3-base and -large in the Hugging Face cache;
#   - data/processed/val.json (squad_io prepare). evidence_val.jsonl is built below if missing.
#
#   mkdir -p logs
#   sbatch configs/efficiency/d5_slurm.sbatch                       # all systems
#   sbatch configs/efficiency/d5_slurm.sbatch --systems bert_span   # extra args go to `efficiency time`
#
# Writes results/efficiency/timings.json; then `python -m qa.eval.efficiency table` (CPU) builds
# results/tables/efficiency.{csv,tex}. Same GPU type as every training run (h100-47 = H100 NVL MIG
# 3g.47gb); some h100-47 nodes hand out a whole H100, so the job stops unless torch sees EXPECT_GPU.

#SBATCH --job-name=d5-efficiency
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpj0,xgpi15
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/d5_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1  # configs contain non-ASCII (λ, ±); batch jobs may start with an ASCII locale
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
expect_gpu="${EXPECT_GPU-MIG 3g.47gb}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
gpu="$("$python_bin" -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "gpu: $gpu"
if [[ -n "$expect_gpu" && "$gpu" != *"$expect_gpu"* ]]; then
    echo "expected a GPU matching '$expect_gpu'; resubmit with this node excluded (EXPECT_GPU= skips)" >&2
    exit 1
fi
if [[ ! -f data/processed/evidence_val.jsonl ]]; then
    "$python_bin" -m qa.data.evidence_labels --input data/processed/val.json --split val
fi
"$python_bin" -m qa.eval.efficiency time --out results/efficiency/timings.json "$@"
echo "finished at $(date)"
