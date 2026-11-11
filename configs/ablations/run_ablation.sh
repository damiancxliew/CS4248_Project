#!/usr/bin/env bash
# G4: train, score and record the evidence-model ablation runs (R4-R7, R6b). Owner: Gokul.
#
# One config per row of the ablation table (configs/ablations/*.yaml). For each config and seed:
# train (the epoch is picked on val at lambda 0, as for every model), predict val, decode jointly
# at the config's decode.lam (G2) and score with the official script, writing the C7 record
# results/runs/<run_id>_s<seed>.json. Configs with `reuses: <run_id>` (R7) are not trained: they
# re-decode that run's saved val logits for the same seed into outputs/<run_id>_s<seed>/.
# Checkpoints are kept (final systems). The test set is never read (scripts/run_all.sh, C9).
#
#   bash configs/ablations/run_ablation.sh list                  # the configs and what they change
#   bash configs/ablations/run_ablation.sh run configs/ablations/r5_no_negatives.yaml 42
#   bash configs/ablations/run_ablation.sh run configs/ablations/r4_full.yaml \
#       configs/ablations/r7_no_evidence_decoding.yaml all     # seeds 13, 42, 87 in turn
#   bash configs/ablations/run_ablation.sh summary               # val records per config
#   bash configs/ablations/run_ablation.sh table                 # G5 table and tests against R4
#
# On the SoC cluster: configs/ablations/ablation_slurm.sbatch (one array task per seed).
# Extra train.py overrides go after `--`, e.g. `run <config> 42 -- --set train.precision=fp32`.
# OUTPUT_DIR (default outputs), RESULTS_DIR (default results/runs) and TABLES_DIR (default
# results/tables) move the outputs.
#
# Not train.py --record: that scores the picked epoch at lambda 0 (span-only), which for these
# configs is R7's number, not theirs. Records here are made at each config's own lambda.
#
# G5 `table`: one row per ablation (R4, R3, R5, R6, R6b, R7) plus BM25 (R1) for ranking quality.
# EM and F1 are mean +- sample std over seeds 13/42/87 from the records; Recall@1 and MRR come
# from J6's ranking_val.csv. Each row is tested against R4 on seed 42: F1 per question from the
# official script on OUTPUT_DIR/<run_id>_s42's val answers, through qa.eval.significance (X3), and
# Recall@1 from J6's ranking_val_significance.json. Writes TABLES_DIR/ablations_val.{csv,tex},
# ablations_val_macros.tex (the numbers section 5.2 quotes) and ablations_val_significance.json; a
# row whose seed-42 answers are missing is marked pending. The CSV and macros also carry each row's
# test means, read from RESULTS_DIR/test/ (C9) and J6's ranking_test.csv; the test set is not rescored.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
output_dir="${OUTPUT_DIR:-outputs}"
results_dir="${RESULTS_DIR:-results/runs}"
seeds=(13 42 87)
overrides=()

usage() {
    echo "usage: $0 list | run <config.yaml>... <seed|all> [-- train.py overrides...] | summary | table" >&2
    exit 2
}

run_config() {
    local cfg="$1" seed="$2" info run_id system variant lam reuses val_file evidence_dir
    local name run_dir source_dir out_dir

    # The config with overrides applied; stop before training if an input is missing.
    info="$("$python_bin" - "$cfg" "$seed" "$output_dir" ${overrides[@]+"${overrides[@]}"} <<'EOF'
import json
import sys
from pathlib import Path

from qa.train import load_config

cfg_file, seed, output_dir, args = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
cfg = load_config(cfg_file, overrides=[args[i + 1] for i, arg in enumerate(args) if arg == "--set"])
reuses = cfg.get("reuses")
if reuses:
    # R7: decode the saved logits of the run it reuses, with the data and model settings of that run.
    source = output_dir / f"{reuses}_s{seed}"
    if not (source / "run.json").is_file() or not (source / "val.logits.npz").is_file():
        sys.exit(f"{cfg_file} re-decodes {source}/val.logits.npz, which is missing: run the "
                 f"{reuses} config for seed {seed} first (same OUTPUT_DIR).")
    trained = json.loads((source / "run.json").read_text())["config"]
    for key in ("class", "encoder", "neg_mode", "alpha", "beta", "margin"):
        if trained["model"].get(key) != cfg["model"].get(key):
            sys.exit(f"{source} was trained with model.{key} = {trained['model'].get(key)!r}, "
                     f"but {cfg_file} says {cfg['model'].get(key)!r}.")
    data, evidence_dir = trained["data"], Path(trained["model"].get("evidence_dir", "data/processed"))
    needed = {evidence_dir / "evidence_val.jsonl": "the C3 file the run was trained with"}
else:
    data, evidence_dir = cfg["data"], Path(cfg["model"].get("evidence_dir", "data/processed"))
    needed = {}
    for split in ("train", "val"):
        needed[data[f"{split}_file"]] = "PYTHONPATH=src python -m qa.data.squad_io prepare"
        needed[evidence_dir / f"evidence_{split}.jsonl"] = (
            f"PYTHONPATH=src python -m qa.data.evidence_labels --input {data[f'{split}_file']} "
            f"--split {split} --out-dir {evidence_dir}")
missing = [f"  {path}\n    build it with: {command}" for path, command in needed.items()
           if not Path(path).is_file()]
if missing:
    sys.exit("Missing input files:\n" + "\n".join(missing))
for line in (cfg["run_id"], cfg["system"], cfg["variant"], f"{float(cfg['decode']['lam']):g}",
             reuses or "-", data["val_file"], evidence_dir):
    print(line)
EOF
)"
    { read -r run_id; read -r system; read -r variant; read -r lam; read -r reuses
      read -r val_file; read -r evidence_dir; } <<< "$info"
    name="${run_id}_s${seed}"
    out_dir="$output_dir/$name"

    if [[ "$reuses" == - ]]; then
        "$python_bin" -m qa.train --config "$cfg" --seed "$seed" --set "output_dir=$output_dir" \
            ${overrides[@]+"${overrides[@]}"}
        "$python_bin" -m qa.predict --checkpoint "$out_dir/best" --split val --out-dir "$out_dir"
        source_dir="$out_dir"
    else
        source_dir="$output_dir/${reuses}_s${seed}"
        echo "[$name] no training: decoding $source_dir/val.logits.npz at lambda $lam"
    fi
    "$python_bin" -m qa.decoding.evidence_decoder --logits "$source_dir/val.logits.npz" \
        --evidence "$evidence_dir/evidence_val.jsonl" --data-file "$val_file" --lam "$lam" \
        --out-dir "$out_dir" > /dev/null

    mkdir -p "$results_dir"
    rm -f "$results_dir/$name.json"  # run_eval never overwrites; a rerun replaces this seed's record
    "$python_bin" - "$run_id" "$system" "$variant" "$lam" "$reuses" "$source_dir" "$out_dir" \
        "$val_file" "$results_dir/$name.json" <<'EOF'
import json
import sys
from pathlib import Path

from qa.eval.run_eval import run_eval

run_id, system, variant, lam, reuses, source_dir, out_dir, val_file, out_file = sys.argv[1:]
metadata = json.loads((Path(source_dir) / "run.json").read_text())
config = metadata["config"]
config.setdefault("decode", {})["lam"] = float(lam)  # the record names the lambda it was scored at
if reuses != "-":  # same model and training fields; only the row and the decoding differ
    metadata["system"], metadata["variant"] = system, variant
    config.update(run_id=run_id, system=system, variant=variant, reuses=reuses)
record = run_eval(val_file, Path(out_dir) / f"val.joint_lam{lam}.predictions.json", metadata, "val",
                  Path(out_file).stem, out_file)
print(f"[{record['run_id']}] val EM {record['exact']:.2f} F1 {record['f1']:.2f} (lambda {lam}) -> {out_file}")
EOF
}

case "${1:-}" in
    list)
        "$python_bin" - configs/ablations/*.yaml <<'EOF'
import sys
from pathlib import Path

import yaml

print(f"{'config':<30} {'run_id':<22} {'variant':<22} {'neg_mode':<8} {'beta':>4} {'lambda':>6}")
for path in sys.argv[1:]:
    cfg = yaml.safe_load(open(path))
    model = cfg["model"]
    note = f"  decode-only, reuses {cfg['reuses']}" if cfg.get("reuses") else ""
    print(f"{Path(path).name:<30} {cfg['run_id']:<22} {cfg['variant']:<22} {model['neg_mode']:<8} "
          f"{model['beta']:>4g} {cfg['decode']['lam']:>6g}{note}")
EOF
        ;;
    run)
        shift
        configs=()
        seed_arg=""
        while (( $# )); do
            case "$1" in
                --) shift; overrides=("$@"); break ;;
                *.yaml) configs+=("$1") ;;
                *) [[ -z "$seed_arg" ]] || usage; seed_arg="$1" ;;
            esac
            shift
        done
        (( ${#configs[@]} )) && [[ -n "$seed_arg" ]] || usage
        if [[ "$seed_arg" == all ]]; then
            run_seeds=("${seeds[@]}")
        elif [[ "$seed_arg" =~ ^[0-9]+$ ]]; then
            run_seeds=("$seed_arg")
        else
            echo "seed must be a number or 'all', got '$seed_arg'" >&2
            exit 2
        fi
        if dirty="$(git status --porcelain --untracked-files=no 2>/dev/null)" && [[ -n "$dirty" ]]; then
            echo "Warning: tracked files differ from the commit, so each record's git_sha will end in" \
                 "-dirty (restore them, e.g. git checkout -- results/tables/):" >&2
            echo "$dirty" >&2
        fi
        for seed in "${run_seeds[@]}"; do
            for cfg in "${configs[@]}"; do
                run_config "$cfg" "$seed"
            done
        done
        ;;
    summary)
        "$python_bin" - "$results_dir" configs/ablations/*.yaml <<'EOF'
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import yaml

results_dir, seeds = Path(sys.argv[1]), (13, 42, 87)
print("val records per ablation config; std = sample std (ddof 1), as in make_tables.py")
for path in sys.argv[2:]:
    cfg = yaml.safe_load(open(path))
    records = [json.loads(p.read_text()) for p in sorted(results_dir.glob(f"{cfg['run_id']}_s*.json"))]
    print(f"\n{cfg['run_id']} ({cfg['variant']})")
    if not records:
        print("  no records yet")
        continue
    for r in sorted(records, key=lambda r: r["seed"]):
        print(f"  seed {r['seed']:>2}: F1 {r['f1']:.2f}  EM {r['exact']:.2f}  {r['gpu']}  {r['git_sha']}")
    for metric in ("f1", "exact"):
        values = [r[metric] for r in records]
        spread = f" ± {stdev(values):.2f}" if len(values) > 1 else ""
        print(f"  {'F1' if metric == 'f1' else 'EM'} {mean(values):.2f}{spread} over {len(values)} seed(s)")
    warnings = []
    missing = sorted(set(seeds) - {r["seed"] for r in records})
    if missing:
        warnings.append(f"missing seeds {missing}")
    for field in ("gpu", "git_sha"):
        if len({r[field] for r in records}) > 1:
            warnings.append(f"seeds differ in {field}, so make_tables.py splits them into separate rows")
    if any(str(r["git_sha"]).endswith("-dirty") for r in records):
        warnings.append("git_sha ends in -dirty (tracked files were changed when it ran)")
    if any(r["config"]["decode"]["lam"] != cfg["decode"]["lam"] for r in records):
        warnings.append("a record was scored at a different lambda than the config")
    for warning in warnings:
        print(f"  warning: {warning}")
EOF
        ;;
    table)
        "$python_bin" - "$results_dir" "$output_dir" "${TABLES_DIR:-results/tables}" <<'EOF'
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean, stdev

from qa.eval.run_eval import _official_module
from qa.eval.significance import compare_scores

results_dir, output_dir, tables_dir = (Path(arg) for arg in sys.argv[1:4])
SEEDS, SEED, ALPHA, BASE, BM25 = (13, 42, 87), 42, 0.01, "r4_deberta_evidence", "r1_bm25"
ROWS = (  # TEAM_TASKS G5: full vs. -ranking vs. -negatives vs. random vs. lexical vs. -evidence decoding
    (BASE, "R4 full model", "Rfour"),  # (run_id, label as in the paper's tables, LaTeX macro tag)
    ("r3_deberta_span", "R3 span-only", "Rthree"),
    ("r5_deberta_evidence", "R5 no negatives", "Rfive"),
    ("r6_deberta_evidence", "R6 random negatives", "Rsix"),
    ("r6b_deberta_evidence", "R6b lexical negatives", "Rsixb"),
    ("r7_deberta_evidence", "R7 evidence weight 0", "Rseven"),
    (BM25, "R1 BM25 ranker", "Rone"),  # a ranker with no reader: ranking columns only
)
RANKING_FROM = {"r7_deberta_evidence": BASE}  # R7 re-decodes R4's models, so it ranks as R4 does
METRICS = ("em", "f1", "recall_at_1", "mrr")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(run_id):
    found = {}
    for path in results_dir.glob(f"{run_id}_s*.json"):
        record = json.loads(path.read_text())
        if record["split"] == "val":
            found[record["seed"]] = record
    if sorted(found) != list(SEEDS):
        sys.exit(f"{results_dir}/{run_id}_s*.json: need val records for seeds {SEEDS}, found {sorted(found)}")
    return found


def answers(record):
    """(file, {qid: answer}) of the run's val answers under OUTPUT_DIR; answers is None if absent."""
    run_dir = output_dir / record["run_id"]
    lam = record["config"].get("decode", {}).get("lam")
    if lam is not None:  # evidence model: the joint decoding its record was scored with
        path = run_dir / f"val.joint_lam{float(lam):g}.predictions.json"
        return path, json.loads(path.read_text()) if path.is_file() else None
    path, nbest = run_dir / "val.predictions.json", run_dir / "val.nbest.json"
    if path.is_file():
        return path, json.loads(path.read_text())
    if nbest.is_file():  # span-only: the C6 n-best is sorted best first, so its top text is the answer
        return nbest, {qid: entries[0]["text"] if entries else ""
                       for qid, entries in json.loads(nbest.read_text()).items()}
    return path, None


def f1_per_question(record, path, preds):
    missing = set(qids) - set(preds)
    if missing:
        sys.exit(f"{path} has no answer for {len(missing)} val question(s), e.g. {sorted(missing)[:3]}")
    f1 = official.get_raw_scores(dataset, preds)[1]
    f1 = {qid: f1[qid] for qid in qids}
    score = 100 * mean(f1.values())
    if abs(score - record["f1"]) > 1e-6:
        sys.exit(f"{path} scores val F1 {score:.4f}, but {record['run_id']} records {record['f1']:.4f}: "
                 "is it that run's file?")
    return f1


def number(text):
    return float(text) if text not in (None, "") else None


ranking_file, ranking_tests_file = tables_dir / "ranking_val.csv", tables_dir / "ranking_val_significance.json"
ranking = {}
if ranking_file.is_file():
    with ranking_file.open(newline="") as stream:
        ranking = {row["run_id"]: row for row in csv.DictReader(stream)}
else:
    print(f"No {ranking_file} (J6: qa.eval.evidence_metrics table): ranking columns left empty", file=sys.stderr)
ranking_tests = {}  # run_id -> (sign that turns the stored difference into run_id - R4, comparison)
if ranking_tests_file.is_file():
    for test in json.loads(ranking_tests_file.read_text())["comparisons"]:
        pair = (test["system_a"], test["system_b"])
        if test["metric"] == "recall_at_1" and f"{BASE}_s{SEED}" in pair:
            other, sign = (pair[1], -1) if pair[0] == f"{BASE}_s{SEED}" else (pair[0], 1)
            ranking_tests[other.removesuffix(f"_s{SEED}")] = (sign, test)

base = records(BASE)
dataset = json.loads(Path(base[SEED]["config"]["data"]["val_file"]).read_text(encoding="utf-8"))["data"]
qids = [qa["id"] for article in dataset for paragraph in article["paragraphs"] for qa in paragraph["qas"]]
official = _official_module()
base_file, base_answers = answers(base[SEED])
if base_answers is None:
    sys.exit(f"{base_file} is missing: every row is tested against R4 seed {SEED}. "
             "Set OUTPUT_DIR to the folder that holds the seed-42 outputs.")
base_f1 = f1_per_question(base[SEED], base_file, base_answers)

rows, f1_tests, changed = [], [], {}
for run_id, label, _ in ROWS:
    row = {"system": label, "run_id": run_id, "seeds": "", "n": None, "ranking_from": None,
           "f1_test": "n/a", **{f"{m}_{s}": None for m in METRICS for s in ("mean", "std")},
           **{f"{m}_{s}": None for m in ("f1", "recall_at_1") for s in ("diff", "ci_low", "ci_high", "p")},
           **{f"test_{m}_{s}": None for m in METRICS for s in ("mean", "std")}}
    if run_id != BM25:
        recs = records(run_id)
        row.update(seeds=" ".join(map(str, SEEDS)), n=recs[SEED]["n"])
        for metric, field in (("em", "exact"), ("f1", "f1")):
            values = [recs[seed][field] for seed in SEEDS]
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean(values), stdev(values)
        if run_id == BASE:
            row["f1_test"] = "reference"
        else:
            path, preds = answers(recs[SEED])
            entry = {"system": f"{run_id}_s{SEED}", "file": path.relative_to(output_dir).as_posix()}
            if preds is None:
                row["f1_test"] = "pending"
                f1_tests.append({**entry, "status": "pending"})
                print(f"{label}: no {path}, so its F1 test against R4 is pending", file=sys.stderr)
            else:
                test = compare_scores(f1_per_question(recs[SEED], path, preds), base_f1, metric="f1",
                                      units="fraction", seed=2026)
                low, high = test["bootstrap"]["ci"]
                row.update(f1_test="done", f1_diff=100 * test["difference"], f1_ci_low=100 * low,
                           f1_ci_high=100 * high, f1_p=test["randomisation"]["p_value"])
                changed[run_id] = sum(preds[qid] != base_answers[qid] for qid in qids)
                f1_tests.append({**entry, "status": "done", "sha256": sha256(path),
                                 "answers_differ_from_reference": changed[run_id],
                                 "reference": f"{BASE}_s{SEED}",
                                 "reference_file": base_file.relative_to(output_dir).as_posix(),
                                 "reference_sha256": sha256(base_file), **test})
    source = RANKING_FROM.get(run_id, run_id)
    if source in ranking:
        row["ranking_from"] = source
        for metric in ("recall_at_1", "mrr"):
            row[f"{metric}_mean"] = number(ranking[source][f"{metric}_mean"])
            row[f"{metric}_std"] = number(ranking[source][f"{metric}_std"])
    if run_id in ranking_tests:
        sign, test = ranking_tests[run_id]
        scale = 100 if test["units"] == "fraction" else 1
        low, high = sorted(sign * scale * value for value in test["bootstrap"]["ci"])
        row.update(recall_at_1_diff=sign * scale * test["difference"], recall_at_1_ci_low=low,
                   recall_at_1_ci_high=high, recall_at_1_p=test["randomisation"]["p_value"])
    rows.append(row)

# Test (official dev, C9's frozen checkpoints): the same rows, so section 5.2 can say whether the val
# pattern holds. Read only from the committed records and J6's ranking_test.csv; nothing is rescored.
test_ranking_file = tables_dir / "ranking_test.csv"
test_ranking = {}
if test_ranking_file.is_file():
    with test_ranking_file.open(newline="") as stream:
        test_ranking = {row["run_id"]: row for row in csv.DictReader(stream)}
for row in rows:
    found = {}
    for path in (results_dir / "test").glob(f"{row['run_id']}_s*_test.json"):
        record = json.loads(path.read_text())
        if record["split"] == "test":
            found[record["seed"]] = record
    if sorted(found) == list(SEEDS):
        for metric, field in (("em", "exact"), ("f1", "f1")):
            values = [found[seed][field] for seed in SEEDS]
            row[f"test_{metric}_mean"], row[f"test_{metric}_std"] = mean(values), stdev(values)
    source = RANKING_FROM.get(row["run_id"], row["run_id"])
    if source in test_ranking:
        for metric in ("recall_at_1", "mrr"):
            row[f"test_{metric}_mean"] = number(test_ranking[source][f"{metric}_mean"])
            row[f"test_{metric}_std"] = number(test_ranking[source][f"{metric}_std"])
if not any(row["test_f1_mean"] is not None for row in rows):
    print(f"No test records under {results_dir}/test: test macros left out", file=sys.stderr)

tables_dir.mkdir(parents=True, exist_ok=True)
csv_path, tex_path = tables_dir / "ablations_val.csv", tables_dir / "ablations_val.tex"
json_path = tables_dir / "ablations_val_significance.json"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def spread(row, metric):
    if metric in ("recall_at_1", "mrr") and row["ranking_from"] not in (None, row["run_id"]):
        return "= R4"
    value, std = row[f"{metric}_mean"], row[f"{metric}_std"]
    if value is None:
        return "--"
    return f"{value:.2f}" + ("" if std is None else f" ({std:.2f})")  # mean (std), as in the paper's other tables


def signed(value):
    return "0.00" if abs(value) < 0.005 else f"{value:+.2f}".replace("-", "$-$")  # no "-0.00"


def p_text(p):  # math mode, after the p: $p\ablRfivePFOne$ -> p = 0.17
    return "<0.001" if p < 0.001 else f"={p:.3f}" if p < 0.01 else f"={p:.2f}"


def mark(row, metric):
    """$^{*}$ if the seed-42 test against R4 has p < ALPHA, $^{\\dagger}$ if the F1 test is pending."""
    if metric == "f1" and row["f1_test"] == "pending":
        return "$^{\\dagger}$"
    p = row[f"{metric}_p"]
    return "$^{*}$" if p is not None and p < ALPHA else ""


lines = [  # one ACL column: EM, MRR, the seed-42 differences and the test means are in the CSV and macros
    "% Generated by configs/ablations/run_ablation.sh table; val, percent. F1 and Recall@1: mean (sample "
    f"std) over seeds 13, 42 and 87. $^{{*}}$: differs from the full model (R4) on seed 42, p < {ALPHA:g}, "
    "paired randomisation test (qa.eval.significance); $^{\\dagger}$: that F1 test is pending. R3's ranking "
    "is span-implied; R7 re-decodes R4's models, so its ranking is R4's.",
    r"\begin{tabular}{lcc}", r"\hline",
    r"System & F1 & Recall@1 \\", r"\hline",
    *(f"{row['system']} & {spread(row, 'f1')}{mark(row, 'f1')} & "
      f"{spread(row, 'recall_at_1')}{mark(row, 'recall_at_1')} \\\\" for row in rows),
    r"\hline", r"\end{tabular}",
]
tex_path.write_text("\n".join(lines) + "\n")

# Numbers the section 5.2 text quotes, so none is copied by hand: \abl<tag><what>, e.g. \ablRfiveDeltaFOne.
macros = {"NQuestions": f"{len(qids):,}".replace(",", "{,}")}
done = [row for row in rows if row["f1_test"] == "done"]
if done:
    macros["MaxAbsDeltaFOne"] = f"{max(abs(row['f1_diff']) for row in done):.2f}"
    macros["MinPFOne"] = p_text(min(row["f1_p"] for row in done))
for row, (_, _, tag) in zip(rows, ROWS):
    for metric, name in (("f1", "FOne"), ("recall_at_1", "ROne")):
        if row[f"{metric}_diff"] is not None:
            macros[f"{tag}Delta{name}"] = signed(row[f"{metric}_diff"])
            macros[f"{tag}Delta{name}Abs"] = f"{abs(row[f'{metric}_diff']):.2f}"
            macros[f"{tag}P{name}"] = p_text(row[f"{metric}_p"])
    if row["run_id"] in changed:
        macros[f"{tag}ChangedAnswers"] = f"{changed[row['run_id']]:,}".replace(",", "{,}")
em_means = [row["em_mean"] for row in rows if row["em_mean"] is not None]
macros["EmMin"], macros["EmMax"] = f"{min(em_means):.2f}", f"{max(em_means):.2f}"
# Test: \ablTest<tag>FOne / \ablTest<tag>ROne, plus how far any ablation lands from R4.
tested = [row for row in rows if row["test_f1_mean"] is not None]
for row, (_, _, tag) in zip(rows, ROWS):
    if row["test_f1_mean"] is not None:
        macros[f"Test{tag}FOne"] = f"{row['test_f1_mean']:.2f}"
    if row["test_recall_at_1_mean"] is not None and row["run_id"] not in RANKING_FROM:
        macros[f"Test{tag}ROne"] = f"{row['test_recall_at_1_mean']:.2f}"
if rows[0]["test_f1_mean"] is not None and len(tested) > 1:
    macros["TestMaxAbsDiffFOne"] = f"{max(abs(r['test_f1_mean'] - rows[0]['test_f1_mean']) for r in tested[1:]):.2f}"
    macros["TestMaxStdFOne"] = f"{max(r['test_f1_std'] for r in tested):.2f}"
rankers = [row["test_recall_at_1_mean"] for row in rows  # the trained rankers, not R3's implied one or BM25
           if row["run_id"] not in (BM25, "r3_deberta_span", *RANKING_FROM) and row["test_recall_at_1_mean"] is not None]
if rankers:
    macros["TestROneMin"], macros["TestROneMax"] = f"{min(rankers):.2f}", f"{max(rankers):.2f}"
macros_path = tables_dir / "ablations_val_macros.tex"
macros_path.write_text("% Generated by configs/ablations/run_ablation.sh table: numbers quoted in section 5.2.\n"
                       + "".join(f"\\newcommand{{\\abl{name}}}{{{value}}}\n" for name, value in macros.items()))
json_path.write_text(json.dumps({
    "note": f"Val, seed {SEED}, every row against {BASE}_s{SEED}; difference = row - R4 (A - B). F1: per-question "
            "official F1 of each run's val answers (paths relative to OUTPUT_DIR), qa.eval.significance."
            "compare_scores, seed 2026. Recall@1: J6's comparisons, copied as stored (their A/B order).",
    "alpha": ALPHA,
    "f1": f1_tests,
    "recall_at_1": {"source": ranking_tests_file.name,
                    "sha256": sha256(ranking_tests_file) if ranking_tests_file.is_file() else None,
                    "comparisons": [test for _, test in ranking_tests.values()]},
}, indent=2) + "\n")

def show(value, spec):
    return "--" if value is None else format(value, spec)


print(f"val, seed-{SEED} tests against R4; * = p < {ALPHA:g}")
print(f"{'system':<27} {'F1 (3 seeds)':>14} {'dF1':>8} {'p':>7} {'Recall@1':>9} {'dR@1':>7} {'p':>7}")
for row in rows:
    f1 = "--" if row["f1_mean"] is None else f"{row['f1_mean']:.2f} +- {row['f1_std']:.2f}"
    d_f1 = "pending" if row["f1_test"] == "pending" else show(row["f1_diff"], "+.2f")
    print(f"{row['system']:<27} {f1:>14} {d_f1:>8} {show(row['f1_p'], '.4f'):>7}"
          f" {show(row['recall_at_1_mean'], '.2f'):>9} {show(row['recall_at_1_diff'], '+.2f'):>7}"
          f" {show(row['recall_at_1_p'], '.4f'):>7}")
for path in (csv_path, tex_path, macros_path, json_path):
    print(path)
EOF
        ;;
    *)
        sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
