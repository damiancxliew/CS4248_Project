"""Paired question-level comparisons; local X3 interface, not a team contract.

API::

    compare_scores(a, b, metric="f1", units="fraction", seed=2026)

``a`` and ``b`` are nonempty mappings from unique, nonempty string question IDs
to numeric scores for ONE metric. IDs must match exactly; sorted IDs align pairs
independently of insertion order. CLI inputs are JSON objects of the same shape,
e.g. {"q1": 1, "q2": 0.5}; duplicate JSON keys are rejected. Python mappings
cannot retain duplicate keys: callers must reject duplicates before constructing
them. Raw predictions (C6), C7 aggregates and C8 rankings are not score inputs.

Metrics: ``em``, ``f1``, ``recall_at_1``. Explicit units are ``fraction`` (0..1)
or ``percent`` (0..100). EM/Recall@1 must be binary (0 or the unit maximum).
F1 may be fractional. Both systems must use the declared units; no scale is
inferred. Reported means, difference and interval retain those units (percentage
points for percent). Positive difference = mean(A) - mean(B), favouring A.

Bootstrap: sample n question pairs with replacement, independently for each of
10,000 default resamples. The interval is the equal-tailed percentile interval
of resampled mean differences, using NumPy's linear quantile interpolation;
default confidence is 0.95. This is not a bootstrap p-value or a BCa interval.
Randomisation: independently swap A/B within each question with probability 1/2
(sign-flip paired differences). The two-sided statistic is abs(mean(A-B)). Count
ties as extreme, with tolerance 8*float64_epsilon*mean(abs(A-B)) for rounding.
Monte Carlo p = (extreme_count + 1)/(n_randomisations + 1), never zero.

Independent NumPy PCG64 streams spawned from a nonnegative integer seed make
results reproducible for the same inputs/options and NumPy version. Changing
the bootstrap count does not change the randomisation draws. Memory is
O(n_questions + n_resamples): one resample at a time, never a resamples-by-n
matrix. The question is the resampling unit; passage/article clustering and
variation across training seeds are not modelled. No multiple-testing correction
is applied. Callers must choose comparisons and account for multiplicity.

CLI (PowerShell, repository root)::

    $env:PYTHONPATH = 'src'
    .venv/Scripts/python.exe -m qa.eval.significance --help
    .venv/Scripts/python.exe -m qa.eval.significance --system-a <a.json> --system-b <b.json> --metric f1 --units fraction --seed 2026

JSON goes to stdout (redirect to a caller-selected file if desired). Output
includes input paths and SHA-256 hashes for CLI provenance. Keep synthetic
inputs/outputs in temporary directories. This module reads scores only, never
SQuAD data or model checkpoints. Producers must agree with Clemen/J6/X2 on
score export, ranking units, split/run provenance and seed pairing before real
integration; it neither modifies run_all.sh nor depends on unmerged X2 code.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _align(system_a, system_b, metric, units):
    if metric not in ("em", "f1", "recall_at_1"):
        raise ValueError("metric must be em, f1 or recall_at_1")
    if units not in ("fraction", "percent"):
        raise ValueError("units must be fraction or percent")
    maximum = 1.0 if units == "fraction" else 100.0
    for scores in (system_a, system_b):
        if not isinstance(scores, Mapping) or not scores:
            raise ValueError("Each system must be a nonempty question-ID mapping")
        if any(not isinstance(qid, str) or not qid.strip() for qid in scores):
            raise ValueError("Question IDs must be nonempty strings")
        for value in scores.values():
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not 0 <= value <= maximum or not np.isfinite(value)):
                raise ValueError(f"Scores must be finite numbers in [0, {maximum}]")
            if metric != "f1" and value not in (0, maximum):
                raise ValueError("EM and Recall@1 scores must be binary")
    if system_a.keys() != system_b.keys():
        raise ValueError("Question IDs must match exactly between systems")
    ids = sorted(system_a)
    return (np.array([system_a[qid] for qid in ids], dtype=np.float64),
            np.array([system_b[qid] for qid in ids], dtype=np.float64))


def compare_scores(system_a, system_b, *, metric, units, n_resamples=10_000,
                   n_randomisations=10_000, confidence=0.95, seed=2026):
    """Return JSON-serialisable paired bootstrap CI and randomisation p-value.

    See the module docstring for input validation, units, statistical definitions,
    limitations and CLI examples. Invalid inputs/options raise ValueError.
    The inputs are not mutated; all randomness is local to this call.
    """
    _positive_integer(n_resamples, "n_resamples")
    _positive_integer(n_randomisations, "n_randomisations")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not 0 < confidence < 1 or not np.isfinite(confidence)):
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    a, b = _align(system_a, system_b, metric, units)
    delta = a - b
    observed = float(delta.mean())
    bootstrap_seed, randomisation_seed = np.random.SeedSequence(seed).spawn(2)
    bootstrap_rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    randomisation_rng = np.random.Generator(np.random.PCG64(randomisation_seed))
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        means[i] = delta[bootstrap_rng.integers(0, len(delta), size=len(delta))].mean()
    tail = (1 - confidence) / 2
    interval = np.quantile(means, [tail, 1 - tail], method="linear").tolist()
    tolerance = 8 * np.finfo(np.float64).eps * float(np.abs(delta).mean())
    extreme = 0
    for _ in range(n_randomisations):
        signs = 2 * randomisation_rng.integers(0, 2, size=len(delta)) - 1
        statistic = abs(float(np.mean(delta * signs)))
        extreme += int(statistic >= abs(observed) - tolerance)
    return {
        "metric": metric, "units": units, "n": len(delta),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "difference": observed, "difference_direction": "A - B", "seed": seed,
        "rng": "PCG64 / SeedSequence.spawn(2)",
        "bootstrap": {"n_resamples": n_resamples, "confidence": confidence,
                      "method": "paired percentile; linear quantiles", "ci": interval},
        "randomisation": {"n_randomisations": n_randomisations,
                          "alternative": "two-sided", "extreme_count": extreme,
                          "tie_tolerance": tolerance,
                          "p_value": (extreme + 1) / (n_randomisations + 1),
                          "p_value_formula": "(extreme_count + 1) / (n_randomisations + 1)"},
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key / question ID: {key!r}")
        result[key] = value
    return result


def main(argv=None):
    """Read two local score JSON objects and write one comparison to stdout."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system-a", type=Path, required=True)
    parser.add_argument("--system-b", type=Path, required=True)
    parser.add_argument("--metric", choices=("em", "f1", "recall_at_1"), required=True)
    parser.add_argument("--units", choices=("fraction", "percent"), required=True)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--n-randomisations", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        inputs, provenance = [], {}
        for label, path in (("system_a", args.system_a), ("system_b", args.system_b)):
            raw = path.read_bytes()
            inputs.append(json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object))
            provenance[label] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        result = compare_scores(
            *inputs, metric=args.metric, units=args.units, seed=args.seed,
            n_resamples=args.n_resamples, n_randomisations=args.n_randomisations,
            confidence=args.confidence,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    result["inputs"] = provenance
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
