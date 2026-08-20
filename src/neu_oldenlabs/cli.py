from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_study
from .io import load_cage_cached
from .pipeline import run_study
from .report import build_study_report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neu-oldenlabs")
    sub = p.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="parse one cage export to a Parquet table")
    ingest.add_argument("export")
    ingest.add_argument("--cage", type=int, required=True)
    ingest.add_argument("--output", required=True)

    for name in ("qc", "profile", "compare", "study-report"):
        cmd = sub.add_parser(name)
        cmd.add_argument("config")
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--cache", default=None)
    return p


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output)
    if args.command == "ingest":
        frame = load_cage_cached(Path(args.export), args.cage, output)
        output.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output / f"cage{args.cage}.parquet", index=False)
        print(f"wrote {output / f'cage{args.cage}.parquet'} ({len(frame)} rows)")
        return

    study = load_study(args.config)
    cache = Path(args.cache) if args.cache else None
    if args.command == "study-report":
        print(build_study_report(study, output, cache_dir=cache))
        return
    result = run_study(study, output, cache_dir=cache)
    if args.command == "qc":
        print(result["qc"]["rack"].to_string(index=False))
    elif args.command == "profile":
        print(result["profile"].to_string(index=False))
    else:
        print(result["scan"].to_string(index=False) if len(result["scan"])
              else "no cage contains both genotypes; no comparison computed")


if __name__ == "__main__":
    main()
