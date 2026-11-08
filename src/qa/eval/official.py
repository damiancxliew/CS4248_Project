"""Cached loader for the official SQuAD evaluator (note: python cannot read hyphen and 2.0)."""
import importlib.util
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_official_metrics():
    path = Path(__file__).resolve().parent / "evaluate-v2.0.py"
    spec = importlib.util.spec_from_file_location("squad_official_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
