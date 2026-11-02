"""Generate C7 mean/sample-standard-deviation CSV and LaTeX tables.

Usage (repository environment, PYTHONPATH=src):
  python scripts/make_tables.py --runs-dir results/runs --out-dir results/tables

Groups retain system, variant, split, question count, parameter count, GPU,
git SHA and resolved config except top-level seed/run_id. Thus different
encoders, decoding settings, data paths and training settings never pool.
Null metrics stay unavailable; singleton standard deviations are undefined.
Outputs include seed counts and IDs, without claiming three-seed completeness.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from qa.eval.run_eval import METRICS, validate_record

GROUP_FIELDS = ("system", "variant", "split", "n", "params_M", "gpu", "git_sha")


def aggregate_records(records):
    groups, run_ids = {}, set()
    for record in records:
        validate_record(record)
        if record["run_id"] in run_ids:
            raise ValueError(f"duplicate run_id: {record['run_id']}")
        run_ids.add(record["run_id"])
        cfg = {k: v for k, v in record["config"].items() if k not in ("seed", "run_id")}
        identity = {k: record[k] for k in GROUP_FIELDS}
        identity["config"] = cfg
        key = json.dumps(identity, sort_keys=True, allow_nan=False)
        groups.setdefault(key, []).append(record)
    rows = []
    for key, group in sorted(groups.items()):
        seeds = [r["seed"] for r in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate seed in a compatible group; select one run per seed")
        row = json.loads(key)
        row["config"] = json.dumps(row["config"], sort_keys=True, ensure_ascii=False)
        row.update(seed_count=len(group), seeds=json.dumps(sorted(seeds, key=str)),
                   run_ids=json.dumps(sorted(r["run_id"] for r in group)))
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            if values and len(values) != len(group):
                raise ValueError(f"mixed null/present {metric} within a group")
            row[f"{metric}_mean"] = mean(values) if values else None
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else None
        rows.append(row)
    return rows


def _tex(value):
    escape = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
              "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
              "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escape.get(c, c) for c in str(value))


def make_tables(runs_dir="results/runs", out_dir="results/tables"):
    paths = sorted(Path(runs_dir).glob("*.json"))
    if not paths:
        raise ValueError("no C7 run records; no experiment tables generated")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rows = aggregate_records(records)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [*GROUP_FIELDS, "config", "seed_count", "seeds", "run_ids",
              *(f"{m}_{s}" for m in METRICS for s in ("mean", "std"))]
    with (out_dir / "run_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["% Generated; sample std (ddof=1), NA = unavailable/undefined.",
             r"\begin{tabular}{llllr" + "r" * len(METRICS) + "}", r"\hline",
             "Group & System & Variant & Split & Seeds & " +
             " & ".join(_tex(m) for m in METRICS) + r" \\", r"\hline"]
    for idx, row in enumerate(rows, 1):
        cells = [str(idx), *(_tex(row[k]) for k in ("system", "variant", "split")),
                 str(row["seed_count"])]
        for metric in METRICS:
            avg, sd = row[f"{metric}_mean"], row[f"{metric}_std"]
            cells.append("NA" if avg is None else
                         f"{avg:.2f} $\\pm$ " + ("NA" if sd is None else f"{sd:.2f}"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    # Full group identities let identical-looking labels be traced to the CSV.
    lines += [f"% Group {i}: " + _tex(json.dumps(row, sort_keys=True, ensure_ascii=False))
              for i, row in enumerate(rows, 1)]
    (out_dir / "run_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args(argv)
    rows = make_tables(args.runs_dir, args.out_dir)
    print(f"Generated {len(rows)} compatible groups in {args.out_dir}")


if __name__ == "__main__":
    main()
