"""PACT paper supplement.

python run.py verify [--checkpoints shipped|retrained] [--only PART ...]
python run.py report [--checkpoints shipped|retrained]
python run.py train  [--lp] [--cleansplit] [--seeds I ...] [--epochs N] [--out DIR]
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import paths

LP = [f"lp{i}" for i in range(6)]
CLEANSPLIT = [f"cleansplit{i}" for i in range(5)]
SD_TOLERANCE = 2
RELATIVE_TOLERANCE = 0.05


def lookup(outputs, part, path):
    value = outputs[part]
    for key in path.split("/"):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def status(entry, outputs, identical):
    """(recomputed value, status) of one printed number."""
    part, rule, scale = entry["part"], entry.get("rule"), entry.get("scale", 1)
    if part not in outputs:
        return None, "not run"
    try:
        value = lookup(outputs, part, entry["path"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None, "error"
    if value is None:
        return None, "not run"
    value *= scale
    token = entry["printed"]
    printed = float(token.replace("{,}", "").replace(",", ""))
    digits = len(token.split(".")[1]) if "." in token else 0
    if rule == "<=":
        ok = value <= printed
    elif rule == ">=":
        ok = value >= printed
    elif rule == "about":  # "about 60": equal once rounded to the printed zeros
        ok = round(value, len(token.rstrip("0")) - len(token)) == printed
    else:
        ok = f"{value:.{digits}f}" == f"{printed:.{digits}f}"
    if identical:
        return value, "PASS" if ok else "FAIL"
    if rule is None and not ok:
        if "sd" in entry:
            width = SD_TOLERANCE * lookup(outputs, part, entry["sd"]) * scale
        else:
            width = RELATIVE_TOLERANCE * abs(printed)
        ok = abs(value - printed) <= width
    return value, "consistent" if ok else "differs"


def used(kind):
    """The checkpoints verify reads. Training CleanSplit is optional, so retrained folds count only when all five are there."""
    folder = paths.ROOT / "checkpoints" / kind
    if kind == "shipped" or all((folder / f"{name}.pt").is_file() for name in CLEANSPLIT):
        return LP + CLEANSPLIT
    return LP


def checkpoints_identical(kind):
    shipped, other = (
        paths.ROOT / "checkpoints" / "shipped",
        paths.ROOT / "checkpoints" / kind,
    )
    return all(
        (other / f"{name}.pt").read_bytes() == (shipped / f"{name}.pt").read_bytes()
        for name in used(kind)
    )


def write_report(rows, out, title):
    (out / "report.json").write_text(json.dumps(rows, indent=1))
    counts = Counter(row["status"] for row in rows)
    first = {"FAIL": 0, "error": 0, "differs": 1, "not run": 3}
    lines = [
        f"# {title}",
        "",
        ", ".join(f"{s}: {n}" for s, n in counts.most_common()),
        "",
        "| table / section | paper line | printed | reproduced | part | status |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda row: (first.get(row["status"], 2), row["line"])):
        got = "" if row["got"] is None else f"{row['got']:.6g}"
        lines.append(
            f"| {row['where']} | {row['line']} | {row['printed']} | {got} | {row['part']} | {row['status']} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return counts


def report(kind, parts=None):
    """Compare every printed number with the saved outputs of `parts` and write out/<kind>/report.md."""
    from pact import paper

    out = paths.OUT / kind
    outputs = {}
    for part in parts or paper.PORTS:
        if (out / f"{part}.json").is_file():
            outputs[part] = json.loads((out / f"{part}.json").read_text())
    identical = kind == "shipped" or checkpoints_identical(kind)
    title = f"PACT paper numbers, {kind} checkpoints"
    if not identical:
        title += " (not byte-identical: consistency check)"
    rows = []
    for entry in json.loads(
        (paths.ROOT / "expected" / "paper_values.json").read_text()
    ):
        got, result = status(entry, outputs, identical)
        rows.append(
            {
                "id": f"L{entry['line']}:{entry['printed']}",
                "line": entry["line"],
                "where": entry["where"],
                "printed": entry["printed"],
                "got": got,
                "part": entry["part"],
                "status": result,
            }
        )
    counts = write_report(rows, out, title)
    print(
        title + ": " + ", ".join(f"{s} {n}" for s, n in counts.most_common()),
        f"-> {out / 'report.md'}",
    )


def missing_files(kind):
    sums = paths.DATA / "SHA256SUMS"
    if not sums.is_file():
        return [str(sums)]
    names = [line.split("  ", 1)[1] for line in sums.read_text().splitlines()]
    missing = [name for name in names if not (paths.DATA / name).exists()]
    return missing + [
        f"checkpoints/{kind}/{name}.pt"
        for name in used(kind)
        if not (paths.CHECKPOINTS / f"{name}.pt").is_file()
    ]


def verify(args):
    missing = missing_files(args.checkpoints)
    if missing:
        sys.exit(
            f"missing {len(missing)} file(s): {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
            f" (data: {paths.DATA}); download the data (README, Quickstart) or run run.py train"
        )
    from pact import paper

    parts = args.only or list(paper.PORTS)
    unknown = sorted(set(parts) - set(paper.PORTS))
    if unknown:
        sys.exit(f"unknown part(s) {unknown}; choose from {list(paper.PORTS)}")
    out = paths.OUT / args.checkpoints
    out.mkdir(parents=True, exist_ok=True)
    for part in parts:
        began = time.time()
        try:
            result = paper.PORTS[part]()
        except Exception:
            traceback.print_exc()
            result = {"error": traceback.format_exc(limit=1)}
        (out / f"{part}.json").write_text(
            json.dumps(result, indent=1, default=lambda x: x.item())
        )
        print(f"{part}: {time.time() - began:.0f}s", flush=True)
    report(args.checkpoints, parts)


def train(args):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import torch

    if not torch.cuda.is_available():
        sys.exit("run.py train requires a CUDA GPU (none is visible)")
    from pact.train import train as train_one

    datasets = [name for name in ("lp", "cleansplit") if getattr(args, name)] or ["lp"]
    jobs = [
        (name, i)
        for name in datasets
        for i in (args.seeds or range(6 if name == "lp" else 5))
    ]
    folder = Path(args.out) if args.out else None
    with ProcessPoolExecutor(len(jobs), mp_context=get_context("spawn")) as pool:
        for path in pool.map(
            partial(train_one, epochs=args.epochs, folder=folder), *zip(*jobs)
        ):
            print("saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACT paper supplement")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("verify", "recompute every paper number and compare (CPU)"),
        ("report", "rebuild out/<checkpoints>/report.md from saved outputs"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument(
            "--checkpoints", choices=("shipped", "retrained"), default="shipped"
        )
        if name == "verify":
            command.add_argument("--only", nargs="+", metavar="PART")
    command = commands.add_parser("train", help="retrain PACT (one CUDA GPU)")
    command.add_argument(
        "--lp", action="store_true", help="LP-PDBbind, seeds 0-5 (the default)"
    )
    command.add_argument(
        "--cleansplit",
        action="store_true",
        help="CleanSplit, folds 0-4; heavy, so only on request (--lp --cleansplit trains both)",
    )
    command.add_argument("--seeds", type=int, nargs="+")
    command.add_argument("--epochs", type=int, default=400)
    command.add_argument("--out")
    args = parser.parse_args()
    if args.command != "train":
        paths.CHECKPOINTS = paths.ROOT / "checkpoints" / args.checkpoints
    if args.command == "verify":
        verify(args)
    elif args.command == "report":
        report(args.checkpoints)
    else:
        train(args)
