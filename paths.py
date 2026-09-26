"""Every path the package uses. The data lives in $PACT_DATA (default: ../pact-data-v1)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PACT_DATA", ROOT.parent / "pact-data-v1")).resolve()
CHECKPOINTS = ROOT / "checkpoints" / "shipped"
OUT = ROOT / "out"


def need(pth: str) -> Path:
    path = DATA / pth
    if not path.exists():
        raise FileNotFoundError(f"missing data file: {pth}; see README (Quickstart)")
    return path
