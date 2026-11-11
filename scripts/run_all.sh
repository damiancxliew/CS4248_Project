#!/usr/bin/env bash
# Reproduce the test-set results from the frozen checkpoints (C9). The only script that touches test.
#
# Usage: bash scripts/run_all.sh data       # check SQuAD files, fetch AddSent/AddOneSent, build C3 evidence (login node: needs internet)
#        bash scripts/run_all.sh verify     # sha256 of every checkpoint (6 systems x 3 seeds) in configs/final_checkpoints.yaml
#        bash scripts/run_all.sh test       # predict + decode + score each frozen checkpoint on test
#        bash scripts/run_all.sh sequential # R9a (BM25 + R3) and R9b (R4's ranker + reader); needs decode.k
#        bash scripts/run_all.sh ranking    # Recall@1/3 and MRR on test -> results/tables/ranking_test.*   (J6b)
#        bash scripts/run_all.sh adversarial # AddSent/AddOneSent for every system -> results/tables/adversarial_*, figure   (J5)
#        bash scripts/run_all.sh errors     # C8 error analysis of R3 (span-implied) and R4 (C8 ranking), seed 42
#        bash scripts/run_all.sh efficiency # efficiency table from the committed timings (D5); imports torch, so run it on a compute node, not the login node
#        bash scripts/run_all.sh tables     # results/tables/test/*.csv|tex from the test records
#        bash scripts/run_all.sh all        # data, verify, test, sequential, ranking, adversarial, errors, efficiency, tables
#
# Needs a GPU for `test` and `adversarial` (same cluster setup as the training sbatch files; run them
# on a compute node, e.g. via configs/final_test.sbatch). `data` needs internet, so run it on a login node first.
# Not wired in: significance tests on the test records (Xin Hui's `qa.eval.significance` needs
# per-question score files that no stage writes yet). `all` says so at the end.
# QA_PYTHON may select a Python executable.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -n "${QA_PYTHON:-}" ]]; then
    qa_python="$QA_PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    qa_python="$repo_root/.venv/bin/python"
else
    qa_python=python3
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false

manifest=configs/final_checkpoints.yaml
test_runs=results/runs/test

# One line per frozen checkpoint and seed: run_id kind seed checkpoint record sha256 (sha256 "null" when unset).
entries() {
    "$qa_python" - "$manifest" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
for c in m["checkpoints"]:
    for seed in m["seeds"]:
        stem = f"{c['run_id']}_s{seed}"
        print(c["run_id"], c["kind"], seed, f"outputs/{stem}/best", f"results/runs/{stem}.json",
              c["sha256"].get(seed) or "null")
PY
}

manifest_value() {  # manifest_value decode.lam
    "$qa_python" - "$manifest" "$1" <<'PY'
import sys, yaml
value = yaml.safe_load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

cmd_data() {
    bash scripts/download_data.sh --check-only
    "$qa_python" -m qa.data.squad_io prepare
    "$qa_python" -m qa.data.evidence_labels --input data/processed/val.json --split val
    "$qa_python" -m qa.data.evidence_labels --input data/raw/dev-v1.1.json --split test
    if [[ ! -f data/processed/addsent.json || ! -f data/processed/addonesent.json ]]; then
        bash scripts/download_data.sh --source codalab  # AddSent / AddOneSent, evaluation only
    fi
    for split in addsent addonesent; do
        "$qa_python" -m qa.data.evidence_labels --input "data/processed/$split.json" --split "$split"
    done
}

cmd_verify() {
    local failed=0 run_id kind seed checkpoint record sha actual
    while read -r run_id kind seed checkpoint record sha; do
        if [[ "$sha" == "null" ]]; then
            echo "MISSING  ${run_id}_s$seed: no sha256 in $manifest" >&2; failed=1; continue
        fi
        if [[ ! -f "$checkpoint/model.pt" ]]; then
            echo "MISSING  ${run_id}_s$seed: $checkpoint/model.pt not found" >&2; failed=1; continue
        fi
        actual="$("$qa_python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$checkpoint/model.pt")"
        if [[ "$actual" != "$sha" ]]; then
            echo "MISMATCH ${run_id}_s$seed: $actual" >&2; failed=1
        else
            echo "ok       ${run_id}_s$seed"
        fi
    done < <(entries)
    return "$failed"
}

# score <run_id> <seed-record> <predictions file> <out run_id> [variant]
# A re-decoding that reuses another run's record (R9a, R9b) passes its own variant, so make_tables.py
# does not pool it with the run it borrows from.
score() {
    local out="$test_runs/$4.json" meta
    if [[ -e "$out" ]]; then echo "skip $out (exists; delete it to re-score)"; return; fi
    # run_eval copies every field except the scores from the metadata. The val record's ranking
    # fields (recall_at_1, recall_at_3, mrr) describe val, so they are nulled here; the test ranking
    # is in results/tables/ranking_test.csv. A re-decoding also gets its own variant (see above).
    meta="$(mktemp)"
    "$qa_python" - "$2" "$meta" "${5:-}" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
for key in ("recall_at_1", "recall_at_3", "mrr"):
    record[key] = None
if sys.argv[3]:
    record["variant"] = sys.argv[3]
json.dump(record, open(sys.argv[2], "w"))
PY
    "$qa_python" -m qa.eval.run_eval --data-file data/raw/dev-v1.1.json --predictions "$3" \
        --metadata "$meta" --split test --run-id "$4" --out-file "$out" --allow-test
    rm -f "$meta"
}

# The table scripts write CRLF line endings; the committed tables use LF.
normalize_eol() {
    "$qa_python" - <<'PY'
from pathlib import Path
for path in Path("results/tables").rglob("*"):
    if path.suffix in (".csv", ".tex"):
        data = path.read_bytes()
        if b"\r\n" in data:
            path.write_bytes(data.replace(b"\r\n", b"\n"))
PY
}

cmd_test() {
    cmd_verify
    local lam run_id kind seed checkpoint record sha dir
    lam="$(manifest_value decode.lam)"
    mkdir -p "$test_runs"
    while read -r run_id kind seed checkpoint record sha; do
        dir="$(dirname "$checkpoint")"
        if [[ "$kind" == "span" ]]; then
            "$qa_python" -m qa.predict --checkpoint "$checkpoint" --split test --allow-test \
                --evidence data/processed/evidence_test.jsonl
            score "$run_id" "$record" "$dir/test.predictions.json" "${run_id}_s${seed}_test"
        else
            "$qa_python" -m qa.predict --checkpoint "$checkpoint" --split test --allow-test \
                --evidence data/processed/evidence_test.jsonl
            for l in "$lam" 0; do
                "$qa_python" -m qa.decoding.evidence_decoder --logits "$dir/test.logits.npz" \
                    --evidence data/processed/evidence_test.jsonl --data-file data/raw/dev-v1.1.json \
                    --lam "$l" --allow-test
            done
            score "$run_id" "$record" "$dir/test.joint_lam${lam}.predictions.json" "${run_id}_s${seed}_test"
            if [[ "$run_id" == "r4_deberta_evidence" ]]; then  # R7 = R4 decoded with lambda 0
                score "$run_id" results/runs/r7_deberta_evidence_s${seed}.json \
                    "$dir/test.joint_lam0.predictions.json" "r7_deberta_evidence_s${seed}_test"
            fi
        fi
    done < <(entries)
}

cmd_sequential() {
    local k seed dir
    k="$(manifest_value decode.k)"
    if [[ "$k" == "None" ]]; then echo "decode.k is not frozen in $manifest (set it on val first)" >&2; return 1; fi
    mkdir -p "$test_runs"
    "$qa_python" -m qa.baselines.bm25_ranker --split test --allow-test  # R1: outputs/r1_bm25/test.sent_rank.json
    for seed in $(manifest_value seeds | tr -d '[],'); do
        # R9a: BM25 top-k sentences, then R3's reader
        dir=outputs/r3_deberta_span_s${seed}
        "$qa_python" -m qa.decoding.sequential_decoder --nbest "$dir/test.nbest.json" \
            --sent-rank outputs/r1_bm25/test.sent_rank.json --evidence data/processed/evidence_test.jsonl \
            --data-file data/raw/dev-v1.1.json --k "$k" --allow-test
        score r3_deberta_span results/runs/r3_deberta_span_s${seed}.json \
            "$dir/test.sequential_k${k}.predictions.json" "r9a_sequential_k${k}_s${seed}_test" "sequential_bm25_k${k}"
        # R9b: R4's own top-k sentences, then R4's reader
        dir=outputs/r4_deberta_evidence_s${seed}
        "$qa_python" -m qa.decoding.sequential_decoder --nbest "$dir/test.joint_lam$(manifest_value decode.lam).nbest.json" \
            --sent-rank "$dir/test.sent_rank.json" --evidence data/processed/evidence_test.jsonl \
            --data-file data/raw/dev-v1.1.json --k "$k" --allow-test
        score r4_deberta_evidence results/runs/r4_deberta_evidence_s${seed}.json \
            "$dir/test.sequential_k${k}.predictions.json" "r9b_sequential_k${k}_s${seed}_test" "sequential_model_k${k}"
    done
}

cmd_errors() {  # seed-42 outputs only (analysis_seed), like the ranking and adversarial scripts
    local lam seed
    lam="$(manifest_value decode.lam)"
    seed="$(manifest_value analysis_seed)"
    "$qa_python" -m qa.eval.error_analysis --spacy --allow-test --data-file data/raw/dev-v1.1.json \
        --evidence data/processed/evidence_test.jsonl \
        --predictions outputs/r3_deberta_span_s${seed}/test.predictions.json \
        --nbest outputs/r3_deberta_span_s${seed}/test.nbest.json
    local dir=outputs/r4_deberta_evidence_s${seed}
    "$qa_python" -m qa.eval.error_analysis --spacy --allow-test --data-file data/raw/dev-v1.1.json \
        --evidence data/processed/evidence_test.jsonl --sent-rank "$dir/test.sent_rank.json" \
        --predictions "$dir/test.joint_lam${lam}.predictions.json" --nbest "$dir/test.joint_lam${lam}.nbest.json"
}

cmd_ranking() {
    "$qa_python" -m qa.baselines.bm25_ranker --split test --allow-test
    "$qa_python" -m qa.eval.evidence_metrics table --split test --allow-test
    normalize_eol
}

# Every system and seed on AddSent / AddOneSent. Inputs for qa.eval.adversarial and, through
# --output-dir, for the ranking tables on the same splits. R7 is R4's logits decoded at lambda 0.
cmd_adversarial() {
    local lam run_id kind seed checkpoint record sha split out
    lam="$(manifest_value decode.lam)"
    while read -r run_id kind seed checkpoint record sha; do
        for split in addsent addonesent; do
            out="outputs/adversarial/${run_id}_s${seed}"
            "$qa_python" -m qa.predict --checkpoint "$checkpoint" --split "$split" \
                --data-file "data/processed/$split.json" --evidence "data/processed/evidence_$split.jsonl" \
                --out-dir "$out"
            if [[ "$kind" == "evidence" ]]; then
                "$qa_python" -m qa.decoding.evidence_decoder --logits "$out/$split.logits.npz" \
                    --evidence "data/processed/evidence_$split.jsonl" --data-file "data/processed/$split.json" \
                    --lam "$lam" --out-dir "$out"
                if [[ "$run_id" == "r4_deberta_evidence" ]]; then
                    "$qa_python" -m qa.decoding.evidence_decoder --logits "$out/$split.logits.npz" \
                        --evidence "data/processed/evidence_$split.jsonl" --data-file "data/processed/$split.json" \
                        --lam 0 --out-dir "outputs/adversarial/r7_deberta_evidence_s${seed}"
                fi
            fi
        done
    done < <(entries)
    for split in addsent addonesent; do
        "$qa_python" -m qa.baselines.bm25_ranker --split "$split"                              # outputs/r1_bm25
        "$qa_python" -m qa.baselines.bm25_ranker --split "$split" --out-dir outputs/adversarial
        "$qa_python" -m qa.eval.adversarial table --split "$split"
        "$qa_python" -m qa.eval.evidence_metrics table --split "$split" --output-dir outputs/adversarial
    done
    "$qa_python" -m qa.eval.adversarial figure
    normalize_eol
}

cmd_efficiency() {
    "$qa_python" -m qa.eval.efficiency table
}

cmd_tables() {
    "$qa_python" scripts/make_tables.py --runs-dir "$test_runs" --out-dir results/tables/test
    "$qa_python" scripts/make_tables.py --literature --out-dir results/tables
    normalize_eol
}

case "${1:-}" in
    data) cmd_data ;;
    verify) cmd_verify ;;
    test) cmd_test ;;
    sequential) cmd_sequential ;;
    ranking) cmd_ranking ;;
    adversarial) cmd_adversarial ;;
    errors) cmd_errors ;;
    efficiency) cmd_efficiency ;;
    tables) cmd_tables ;;
    all)
        cmd_data; cmd_test; cmd_sequential; cmd_ranking; cmd_adversarial; cmd_errors; cmd_efficiency; cmd_tables
        echo "Not wired in: significance tests on the test records (see the header of this script)." >&2
        ;;
    *) sed -n '2,22p' "$0" >&2; exit 2 ;;
esac
