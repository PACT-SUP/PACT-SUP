"""Check the unpacked data against SHA256SUMS.

python fetch_data.py --verify [DIR]  check every file against SHA256SUMS (default: $PACT_DATA)
"""

import hashlib
import sys
from pathlib import Path

import paths


def check(root):
    """Files listed in SHA256SUMS that are missing or whose sha256 differs."""
    bad = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if not (root / name).exists():
            bad.append(f"MISSING {name}")
        elif (
            hashlib.file_digest(open(root / name, "rb"), "sha256").hexdigest() != digest
        ):
            bad.append(f"BAD {name}")
    return bad


if __name__ == "__main__":
    if sys.argv[1:2] != ["--verify"]:
        sys.exit(__doc__)
    root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else paths.DATA
    bad = check(root)
    print("\n".join(bad) or f"OK: every file in {root} matches SHA256SUMS")
    sys.exit(1 if bad else 0)
