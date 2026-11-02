#!/usr/bin/env bash
# Verify committed SQuAD data; optionally fetch evaluation-only adversarial data.
# Usage: bash scripts/download_data.sh --check-only
#        bash scripts/download_data.sh --source codalab
#        bash scripts/download_data.sh --source hf --revision <full-dataset-commit-sha>
# HF uses a commit from refs/convert/parquet, not the legacy loader on main.
# Verified revision: 0eb158aad5fd6cf738033016948d2fd7d30bed8a
# No source is selected automatically. QA_PYTHON may select a Python executable.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${QA_PYTHON:-}" ]]; then
    qa_python="$QA_PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    qa_python="$repo_root/.venv/bin/python"
elif [[ -x "$repo_root/.venv/Scripts/python.exe" ]]; then
    qa_python="$repo_root/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    qa_python=python3
else
    qa_python=python
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "--check-only" && $# -eq 1 ]]; then
    exec "$qa_python" -m qa.data.squad_io check --raw-dir "$repo_root/data/raw"
fi
if [[ $# -eq 0 ]]; then
    echo "Choose --check-only or --source codalab|hf (HF also needs --revision SHA)." >&2
    exit 2
fi
exec "$qa_python" -m qa.data.squad_io download "$@"
