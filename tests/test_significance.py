"""Synthetic CPU checks only; CLI files live under pytest's temporary directory."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from qa.eval.significance import compare_scores, main


def compare(a, b, **options):
    defaults = dict(metric="f1", units="fraction", n_resamples=1000,
                    n_randomisations=2000, seed=42)
    defaults.update(options)
    return compare_scores(a, b, **defaults)


@pytest.mark.parametrize("metric", ["em", "f1", "recall_at_1"])
def test_identical_systems(metric):
    scores = {"q1": 0, "q2": 1, "q3": 1}
    result = compare_scores(scores, scores, metric=metric, units="fraction")
    assert result["difference"] == 0
    assert result["bootstrap"]["ci"] == [0, 0]
    assert result["bootstrap"]["n_resamples"] == 10_000
    assert result["randomisation"]["p_value"] == 1


def test_constant_gain_and_direction():
    a = {str(i): 0.75 for i in range(20)}
    b = {str(i): 0.25 for i in range(20)}
    forward, reverse = compare(a, b), compare(b, a)
    assert forward["mean_a"] == 0.75
    assert forward["mean_b"] == 0.25
    assert forward["difference"] == 0.5
    assert forward["bootstrap"]["ci"] == [0.5, 0.5]
    assert reverse["bootstrap"]["ci"] == [-0.5, -0.5]
    assert reverse["difference"] == -0.5
    assert forward["randomisation"]["p_value"] < 0.01
    assert forward["randomisation"]["p_value"] == reverse["randomisation"]["p_value"]


def test_reordered_ids_reproducibility_and_units():
    a = {"z": 0.75, "a": 1, "m": 0}
    b = {"a": 0.5, "m": 1, "z": 0.5}
    result = compare(a, b)
    assert compare(dict(reversed(list(a.items()))), b) == result
    assert compare(a, dict(reversed(list(b.items())))) == result
    assert compare(a, b) == result
    assert result["difference"] == pytest.approx(-1 / 12)
    percent = compare({k: v * 100 for k, v in a.items()},
                      {k: v * 100 for k, v in b.items()}, units="percent")
    assert percent["difference"] == pytest.approx(100 * result["difference"])
    assert percent["bootstrap"]["ci"] == pytest.approx(np.array(result["bootstrap"]["ci"]) * 100)
    assert percent["randomisation"]["p_value"] == result["randomisation"]["p_value"]
    for seed in (13, 87):
        assert compare(a, b, seed=seed) == compare(a, b, seed=seed)
    assert compare(a, b, n_resamples=17)["randomisation"] == result["randomisation"]
    assert a == {"z": 0.75, "a": 1, "m": 0}


def test_randomisation_against_enumeration_including_ties():
    a = {str(i): 1 for i in range(4)}
    b = {str(i): 0 for i in range(4)}
    # Only all-positive and all-negative of 16 swaps are as extreme: p=1/8.
    exact = sum(abs(sum(s)) >= 4 for s in itertools.product((-1, 1), repeat=4)) / 16
    result = compare(a, b, n_randomisations=20_000)
    randomisation = result["randomisation"]
    assert randomisation["p_value"] == pytest.approx(exact, abs=0.01)
    assert randomisation["p_value"] == (randomisation["extreme_count"] + 1) / 20_001
    # One pair always ties under a two-sided test, even for a nonzero difference.
    assert compare({"q": 1}, {"q": 0})["randomisation"]["p_value"] == 1
    # Equal means do not imply identical per-question scores.
    assert compare({"a": 1, "b": 0}, {"a": 0, "b": 1})["randomisation"]["p_value"] == 1


def test_bootstrap_interval_against_discrete_distribution():
    # Differences [0, 1]: exact n=2 bootstrap means have masses .25/.5/.25.
    result = compare({"a": 0, "b": 1}, {"a": 0, "b": 0}, n_resamples=10_000)
    assert result["bootstrap"]["ci"] == [0, 1]
    assert result["difference"] == 0.5
    # Verify linear interpolation of a nontrivial finite Monte Carlo sample.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(42).spawn(2)[0]))
    means = [np.array([0., 1.])[rng.integers(0, 2, size=2)].mean() for _ in range(7)]
    small = compare({"a": 0, "b": 1}, {"a": 0, "b": 0}, n_resamples=7, confidence=0.6)
    assert small["bootstrap"]["ci"] == pytest.approx(np.quantile(means, [.2, .8], method="linear"))


@pytest.mark.parametrize("a,b", [
    ({}, {}), ({"a": 1}, {"b": 1}), ({"a": 1}, {"a": 1, "b": 0}),
    ([("a", 1), ("a", 0)], {"a": 1}), ({None: 1}, {None: 1}),
    ({"": 1}, {"": 1}), ({" ": 1}, {" ": 1}),
    *[({"a": value}, {"a": 0}) for value in
      [float("nan"), float("inf"), -float("inf"), -1, 1.1, True, "1", None, [1], 10**1000]],
])
def test_invalid_scores_and_ids(a, b):
    with pytest.raises(ValueError):
        compare(a, b)
    with pytest.raises(ValueError):
        compare(b, a)


@pytest.mark.parametrize("options", [
    {"metric": "accuracy"}, {"units": "auto"}, {"metric": "em"},
    {"metric": "recall_at_1"}, {"n_resamples": 0}, {"n_resamples": True},
    {"n_randomisations": -1}, {"n_randomisations": 1.5},
    {"confidence": 0}, {"confidence": 1}, {"confidence": float("nan")},
    {"confidence": True}, {"seed": -1}, {"seed": None}, {"seed": True},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        compare({"q": 0.5}, {"q": 0}, **options)


def test_percent_binary_metrics():
    for metric in ("em", "recall_at_1"):
        assert compare({"a": 100}, {"a": 0}, metric=metric, units="percent")["difference"] == 100
        with pytest.raises(ValueError, match="binary"):
            compare({"a": 1}, {"a": 0}, metric=metric, units="percent")


def test_cli_and_duplicate_json_ids(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('{"q1": 1, "q2": 0.5}', encoding="utf-8")
    b.write_text('{"q2": 0, "q1": 0.5}', encoding="utf-8")
    args = ["--system-a", str(a), "--system-b", str(b), "--metric", "f1",
            "--units", "fraction", "--n-resamples", "20", "--n-randomisations", "30"]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    run = subprocess.run([sys.executable, "-m", "qa.eval.significance", *args],
                         capture_output=True, text=True, env=env, check=True)
    result = json.loads(run.stdout)
    assert result["difference"] == 0.5
    assert result["inputs"]["system_a"]["sha256"] == hashlib.sha256(a.read_bytes()).hexdigest()
    for invalid in ('{"q": 1, "q": 0}', '{"q": NaN}', '[]', '{"q": null}', '{'):
        a.write_text(invalid, encoding="utf-8")
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
        assert capsys.readouterr().out == ""


def test_resampling_memory_shapes(monkeypatch):
    # Instrument RNG calls to catch accidentally allocating [resamples, n].
    original = np.random.Generator
    shapes = []

    class Recorder:
        def __init__(self, bit_generator):
            self.generator = original(bit_generator)

        def integers(self, *args, **kwargs):
            shapes.append(kwargs["size"])
            return self.generator.integers(*args, **kwargs)

    monkeypatch.setattr(np.random, "Generator", Recorder)
    scores = {str(i): i % 2 for i in range(10_000)}
    compare(scores, scores, n_resamples=3, n_randomisations=4)
    assert shapes == [10_000] * 7
