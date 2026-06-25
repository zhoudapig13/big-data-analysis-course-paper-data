"""Run the CFPB analysis pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_step(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf", "official-api", "local-csv"], default="hf")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--local-path", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch_cmd = [
        sys.executable,
        str(SCRIPTS / "01_fetch_data.py"),
        "--source",
        args.source,
        "--start",
        args.start,
        "--end",
        args.end,
        "--max-records",
        str(args.max_records),
    ]
    if args.force:
        fetch_cmd.append("--force")
    for path in args.local_path:
        fetch_cmd.extend(["--local-path", path])

    run_step(fetch_cmd)
    run_step([sys.executable, str(SCRIPTS / "02_clean_features.py")])
    run_step([sys.executable, str(SCRIPTS / "03_profile_and_models.py")])


if __name__ == "__main__":
    main()
