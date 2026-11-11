#!/usr/bin/env bash
# G3: evidence-model tuning on val (seed 42, D3 hyper-parameters frozen, neg_mode = model). Owner: Gokul.
#
# Grid: alpha {0.1, 0.3, 1.0} x beta {0.5, 1.0} x margin {0.5, 1.0} = 12 training runs. Each run
# selects its best epoch on val with span-only decoding (lambda = 0), then its val logits are
# decoded jointly at lambda {0, 0.25, 0.5, 1, 2} (inference only) and scored with the official
# script: one C7 record per (run, lambda) in results/runs/g3_grid/. The test set is never read.
#
#   bash configs/evidence/g3_grid.sh list          # show the 12 runs with their index
#   bash configs/evidence/g3_grid.sh run 3         # train, predict, decode and score one (index 0-11)
#   bash configs/evidence/g3_grid.sh run all       # all 12 one after another
#   bash configs/evidence/g3_grid.sh summary       # table of finished runs and the pick
#   bash configs/evidence/g3_grid.sh curve         # the lambda figure for paper section 5.2
#
# On the SoC cluster: configs/evidence/g3_slurm.sbatch (one array task per index).
# Extra train.py overrides can be passed after the index, e.g. --set train.precision=fp16 on a T4.
# OUTPUT_DIR (default outputs) and RESULTS_DIR (default results/runs/g3_grid) move the outputs.
# A run's model (outputs/<run>/best/, about 0.74 GB) is deleted once its 5 records are written:
# nothing later loads a G3 model (R4 retrains from best.yaml), and val.logits.npz stays, so any
# lambda can still be decoded. KEEP_CHECKPOINT=1 keeps it.
#
# Pick rule, fixed before the search: the highest val F1 over every (run, lambda). Runs whose best
# F1 is within 0.2 of it count as tied (the same D3 setting gave 88.49 and 88.32 on two GPUs), and
# among those the highest Recall@1 of the run's sentence ranking wins. A run's lambda is its
# best-F1 one (equal F1: the smaller lambda). Gokul confirms the pick for configs/evidence/best.yaml.
#
# `curve` plots val F1 against lambda for every run in RESULTS_DIR: the best.yaml pick in colour,
# the other runs in grey, and span-only R3 seed 42 as a dashed line when its record exists.
# Writes FIGURES_DIR (default results/figures)/lambda_curve.{pdf,png} and the numbers section 5.2
# quotes to TABLES_DIR (default results/tables)/lambda_curve_macros.tex.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${QA_PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
output_dir="${OUTPUT_DIR:-outputs}"
results_dir="${RESULTS_DIR:-results/runs/g3_grid}"
keep_checkpoint="${KEEP_CHECKPOINT:-0}"
lambdas=(0 0.25 0.5 1 2)

grid=()
for alpha in 0.1 0.3 1.0; do
    for beta in 0.5 1.0; do
        for margin in 0.5 1.0; do
            grid+=("$alpha $beta $margin")
        done
    done
done

run_id() {  # 0.3 1.0 0.5 -> g3_a0.3_b1.0_m0.5
    echo "g3_a$1_b$2_m$3"
}

run_one() {
    local index="$1"; shift
    if ! [[ "$index" =~ ^[0-9]+$ ]] || (( index >= ${#grid[@]} )); then
        echo "index must be 0-$(( ${#grid[@]} - 1 )), got '$index'" >&2
        exit 2
    fi
    read -r alpha beta margin <<< "${grid[$index]}"
    local name run_dir paths val_file evidence_dir
    name="$(run_id "$alpha" "$beta" "$margin")_s42"
    run_dir="$output_dir/$name"
    local overrides=(--set "run_id=$(run_id "$alpha" "$beta" "$margin")" --set "output_dir=$output_dir"
                     --set "model.alpha=$alpha" --set "model.beta=$beta" --set "model.margin=$margin" "$@")

    # The files this run reads, overrides applied; stop before training if one is missing.
    paths="$("$python_bin" - "${overrides[@]}" <<'EOF'
import sys
from pathlib import Path

from qa.train import load_config

args = sys.argv[1:]
cfg = load_config("configs/evidence/evidence.yaml",
                  overrides=[args[i + 1] for i, arg in enumerate(args) if arg == "--set"])
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
print(data["val_file"])
print(evidence_dir)
EOF
)"
    val_file="${paths%%$'\n'*}"
    evidence_dir="${paths#*$'\n'}"

    "$python_bin" -m qa.train --config configs/evidence/evidence.yaml --seed 42 "${overrides[@]}"
    "$python_bin" -m qa.predict --checkpoint "$run_dir/best" --split val --out-dir "$run_dir"

    mkdir -p "$results_dir"
    rm -f "$results_dir/${name}"_lam*.json  # run_eval never overwrites; a rerun replaces this run's records
    for lam in "${lambdas[@]}"; do
        "$python_bin" -m qa.decoding.evidence_decoder --logits "$run_dir/val.logits.npz" \
            --evidence "$evidence_dir/evidence_val.jsonl" --data-file "$val_file" --lam "$lam" > /dev/null
        "$python_bin" - "$run_dir" "$lam" "$val_file" "$results_dir/${name}_lam$lam.json" <<'EOF'
import json
import sys
from pathlib import Path

from qa.eval.run_eval import run_eval

run_dir, lam, val_file, out_file = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
metadata = json.loads((run_dir / "run.json").read_text())
metadata["config"].setdefault("decode", {})["lam"] = float(lam)  # each record names its lambda
record = run_eval(val_file, run_dir / f"val.joint_lam{lam}.predictions.json", metadata, "val",
                  out_file.stem, out_file)
print(f"[{record['run_id']}] val EM {record['exact']:.2f} F1 {record['f1']:.2f}")
EOF
    done
    if [[ "$keep_checkpoint" != 1 ]]; then
        rm -rf "$run_dir/best"  # scored, so the model is no longer needed (see the header)
        echo "[$name] deleted $run_dir/best (KEEP_CHECKPOINT=1 keeps it)"
    fi
}

case "${1:-}" in
    list)
        for i in "${!grid[@]}"; do
            read -r alpha beta margin <<< "${grid[$i]}"
            printf '%2d  %s\n' "$i" "$(run_id "$alpha" "$beta" "$margin")_s42"
        done
        ;;
    run)
        [[ $# -ge 2 ]] || { echo "usage: $0 run <index|all> [train.py --set overrides...]" >&2; exit 2; }
        target="$2"; shift 2
        if [[ "$target" == all ]]; then
            for i in "${!grid[@]}"; do run_one "$i" "$@"; done
        else
            run_one "$target" "$@"
        fi
        ;;
    summary)
        "$python_bin" - "$results_dir" "$output_dir" "${lambdas[@]}" <<'EOF'
import json
import sys
from pathlib import Path

from qa.predict import read_evidence

TIE_F1 = 0.2  # runs this close to the best F1 count as tied (seen across GPUs for one D3 setting)
results_dir, output_dir, lambdas = Path(sys.argv[1]), Path(sys.argv[2]), [float(x) for x in sys.argv[3:]]

runs = {}
for path in sorted(results_dir.glob("g3_*_lam*.json")):
    record = json.loads(path.read_text())
    run = runs.setdefault(record["run_id"].rsplit("_lam", 1)[0], {"record": record, "f1": {}, "em": {}})
    lam = record["config"]["decode"]["lam"]
    run["f1"][lam], run["em"][lam] = record["f1"], record["exact"]
if not runs:
    sys.exit(f"No finished G3 runs under {results_dir}/")


def ranking_quality(name, cfg):
    """Recall@1 and MRR of the run's C8 val ranking against the C3 gold sentence (a tuning aid;
    the reported ranking metrics come from J6)."""
    rank_file = output_dir / name / "val.sent_rank.json"
    evidence_file = Path(cfg["model"].get("evidence_dir", "data/processed")) / "evidence_val.jsonl"
    if not rank_file.is_file() or not evidence_file.is_file():
        return None, None
    evidence, ranking = read_evidence(evidence_file), json.loads(rank_file.read_text())
    ranks = [[sent for sent, _ in ranking[qid]].index(evidence[qid]["gold_sent"]) + 1 for qid in ranking]
    return sum(rank == 1 for rank in ranks) / len(ranks), sum(1 / rank for rank in ranks) / len(ranks)


rows = []
for name, run in runs.items():
    model = run["record"]["config"]["model"]
    best_lam = max(run["f1"], key=lambda lam: (run["f1"][lam], -lam))  # ties: smaller lambda
    recall1, mrr = ranking_quality(name, run["record"]["config"])
    rows.append({"name": name, "alpha": model["alpha"], "beta": model["beta"], "margin": model["margin"],
                 "f1": run["f1"][best_lam], "em": run["em"][best_lam], "lam": best_lam, "curve": run["f1"],
                 "recall1": recall1, "mrr": mrr, "gpu_h": run["record"]["train_gpu_hours"]})
rows.sort(key=lambda row: row["f1"], reverse=True)

fmt = lambda value, spec: "-" if value is None else format(value, spec)
print("val F1 by lambda (joint decoding); best = the run's highest; R@1 / MRR = its sentence ranking")
print(f"{'alpha':>5} {'beta':>4} {'m':>4} " + " ".join(f"{f'F1@{lam:g}':>7}" for lam in lambdas)
      + f" {'best':>4} {'EM':>6} {'R@1':>6} {'MRR':>6} {'GPU-h':>5}  run_id")
for row in rows:
    print(f"{row['alpha']:5.1f} {row['beta']:4.1f} {row['margin']:4.1f} "
          + " ".join(f"{fmt(row['curve'].get(lam), '7.2f'):>7}" for lam in lambdas)
          + f" {row['lam']:4g} {row['em']:6.2f} {fmt(row['recall1'], '6.3f'):>6} {fmt(row['mrr'], '6.3f'):>6}"
          f" {fmt(row['gpu_h'], '5.2f'):>5}  {row['name']}")

incomplete = [row["name"] for row in rows if len(row["curve"]) < len(lambdas)]
print(f"\n{len(rows)}/12 runs finished." + (f" Missing lambda records: {', '.join(incomplete)}" if incomplete else ""))
tied = [row for row in rows if row["f1"] >= rows[0]["f1"] - TIE_F1]
if any(row["recall1"] is None for row in tied):
    pick = rows[0]
    print(f"No sentence ranking under {output_dir}/ for every tied run, so this pick ignores the "
          "Recall@1 tie-break; run summary where the outputs are.")
else:
    pick = max(tied, key=lambda row: (row["recall1"], row["f1"]))
print(f"{len(tied)} run(s) within {TIE_F1} F1 of the best ({rows[0]['f1']:.2f}).")
print(f"Pick: alpha {pick['alpha']:g}, beta {pick['beta']:g}, margin {pick['margin']:g}, lambda {pick['lam']:g}"
      f" -> val F1 {pick['f1']:.2f} / EM {pick['em']:.2f}, Recall@1 {fmt(pick['recall1'], '.3f')} ({pick['name']})")
if len(rows) < 12 or incomplete:
    print("Not final: the grid is incomplete.")
EOF
        ;;
    curve)
        "$python_bin" - "$results_dir" "${FIGURES_DIR:-results/figures}" "${TABLES_DIR:-results/tables}" <<'EOF'
import json
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir, figures_dir, tables_dir = (Path(arg) for arg in sys.argv[1:4])
best = yaml.safe_load(Path("configs/evidence/best.yaml").read_text())
pick = tuple(float(best["model"][key]) for key in ("alpha", "beta", "margin"))
pick_lam = float(best["decode"]["lam"])
span_only = Path("results/runs/r3_deberta_span_s42.json")  # same seed and GPU type as the grid

curves = {}  # (alpha, beta, margin) -> {lambda: val F1}
for path in sorted(results_dir.glob("g3_*_lam*.json")):
    record = json.loads(path.read_text())
    model = record["config"]["model"]
    key = tuple(float(model[name]) for name in ("alpha", "beta", "margin"))
    curves.setdefault(key, {})[float(record["config"]["decode"]["lam"])] = record["f1"]
if pick not in curves or pick_lam not in curves[pick]:
    sys.exit(f"No record in {results_dir}/ for the best.yaml pick (alpha, beta, margin = {pick}, "
             f"lambda {pick_lam:g})")

span_only_f1 = json.loads(span_only.read_text())["f1"] if span_only.is_file() else None
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(3.1, 1.65))
for key, curve in curves.items():
    if key != pick:
        lams = sorted(curve)
        ax.plot(lams, [curve[lam] for lam in lams], color="#a3a29c", linewidth=0.8, alpha=0.8, zorder=1)
ax.plot([], [], color="#a3a29c", linewidth=0.8, label=f"Other grid runs ({len(curves) - 1})")
lams = sorted(curves[pick])
alpha, beta, margin = pick
ax.plot(lams, [curves[pick][lam] for lam in lams], color="#2a78d6", linewidth=1.8, marker="o",
        markersize=3, zorder=3, label=f"Pick ($\\alpha$ {alpha:g}, $\\beta$ {beta:g}, $m$ {margin:g})")
ax.plot([pick_lam], [curves[pick][pick_lam]], marker="o", markersize=7, markerfacecolor="none",
        markeredgecolor="#2a78d6", zorder=4)
if span_only_f1 is not None:
    ax.axhline(span_only_f1, color="#52514e", linewidth=1, linestyle="--", zorder=2, label="Span-only (R3)")
ax.set_xticks(lams, [f"{lam:g}" for lam in lams])
ax.set_xlabel(r"$\lambda$ (weight of $\log p$(sentence) in decoding)")
ax.set_ylabel("Val F1 (seed 42)")
ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#a3a29c")
ax.tick_params(colors="#52514e")
ax.legend(loc="lower left", fontsize=7, facecolor="white", edgecolor="none", framealpha=0.9)
fig.tight_layout()
figures_dir.mkdir(parents=True, exist_ok=True)
out = figures_dir / "lambda_curve.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
gains = [max(curve.values()) - curve[0.0] for curve in curves.values() if 0.0 in curve]
bests = [max(curve.values()) for curve in curves.values()]
print(f"{len(curves)} runs; pick F1 {curves[pick][pick_lam]:.2f} at lambda {pick_lam:g}; "
      f"best lambda gains at most {max(gains):.2f} F1 over lambda 0")
macros = {"Runs": str(len(curves)), "PickAlpha": f"{alpha:g}", "PickBeta": f"{beta:g}", "PickMargin": f"{margin:g}",
          "Pick": f"{pick_lam:g}", "PickFOne": f"{curves[pick][pick_lam]:.2f}",
          "MaxGain": f"{max(gains):.2f}", "RunSpread": f"{max(bests) - min(bests):.2f}"}
if span_only_f1 is not None:
    macros["SpanOnlyFOne"] = f"{span_only_f1:.2f}"
tables_dir.mkdir(parents=True, exist_ok=True)
macros_path = tables_dir / "lambda_curve_macros.tex"
macros_path.write_text("% Generated by configs/evidence/g3_grid.sh curve: numbers quoted in section 5.2.\n"
                       + "".join(f"\\newcommand{{\\lam{name}}}{{{value}}}\n" for name, value in macros.items()))
for path in (out, out.with_suffix(".png"), macros_path):
    print(path)
EOF
        ;;
    *)
        sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
