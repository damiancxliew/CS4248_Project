import sys
from pathlib import Path

# Make `import qa...` work when running plain `pytest -q` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
