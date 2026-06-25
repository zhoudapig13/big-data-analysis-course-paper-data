"""Fetch CFPB complaint data from accessible public sources.

The official CFPB API/full CSV is supported, but the current network may
return 403 from consumerfinance.gov. The default path therefore downloads two
public Hugging Face mirrors/subsets and writes a combined raw CSV. The cleaning
script is source-agnostic, so the same downstream pipeline can be rerun on the
official full CSV later.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

HF_SOURCES = [
    {
        "name": "hf_aciborowska_30k",
        "url": "https://huggingface.co/datasets/aciborowska/customers-complaints/resolve/main/data/train-00000-of-00001-a5763026b6750ff3.parquet",
        "format": "parquet",
        "note": "30k CFPB-derived records with full original-style fields.",
    },
    {
        "name": "hf_claritystorm_1k",
        "url": "https://huggingface.co/datasets/claritystorm/cfpb-consumer-complaints/resolve/main/sample_1000.csv",
        "format": "csv",
        "note": "1k CFPB-derived records with clean CSV structure and 2023-2025 examples.",
    },
]

OFFICIAL_API_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    "search/api/v1/"
)


def read_remote_table(url: str, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(url)
    if fmt == "csv":
        return pd.read_csv(url, low_memory=False)
    raise ValueError(f"Unsupported format: {fmt}")


def fetch_hf_sources(force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    manifest = []

    for source in HF_SOURCES:
        local_path = RAW_DIR / f"{source['name']}.csv"
        if local_path.exists() and not force:
            df = pd.read_csv(local_path, low_memory=False)
            status = "cached"
        else:
            df = read_remote_table(source["url"], source["format"])
            df.to_csv(local_path, index=False, encoding="utf-8")
            status = "downloaded"

        df["raw_source"] = source["name"]
        frames.append(df)
        manifest.append(
            {
                "name": source["name"],
                "url": source["url"],
                "format": source["format"],
                "note": source["note"],
                "status": status,
                "rows": int(len(df)),
                "local_path": str(local_path),
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined_path = RAW_DIR / "cfpb_combined_raw.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8")

    with (RAW_DIR / "source_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "mode": "hf",
                "rows": int(len(combined)),
                "combined_path": str(combined_path),
                "sources": manifest,
            },
            fh,
            indent=2,
        )

    return combined


def extract_hits(payload: dict) -> list[dict]:
    hits = payload.get("hits", {})
    if isinstance(hits, dict):
        records = hits.get("hits", [])
    elif isinstance(hits, list):
        records = hits
    else:
        records = []

    out = []
    for item in records:
        if isinstance(item, dict) and "_source" in item:
            out.append(item["_source"])
        elif isinstance(item, dict):
            out.append(item)
    return out


def fetch_official_api(
    start: str,
    end: str,
    max_records: int,
    only_narratives: bool = True,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """Fetch a sample from the official CFPB search API."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/125 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.consumerfinance.gov/data-research/consumer-complaints/search/",
    }
    base_params = {
        "date_received_min": start,
        "date_received_max": end,
        "size": 100,
        "sort": "created_date_desc",
        "no_aggs": "true",
        "no_highlight": "true",
    }
    if only_narratives:
        base_params["has_narrative"] = "yes"

    records = []
    offset = 0
    while len(records) < max_records:
        params = dict(base_params)
        params["frm"] = offset
        response = requests.get(
            OFFICIAL_API_URL, headers=headers, params=params, timeout=45
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Official CFPB API returned {response.status_code}: "
                f"{response.text[:300]}"
            )
        batch = extract_hits(response.json())
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        time.sleep(sleep_seconds)

    df = pd.DataFrame(records[:max_records])
    path = RAW_DIR / "cfpb_official_api_raw.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

    with (RAW_DIR / "source_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "mode": "official-api",
                "rows": int(len(df)),
                "official_api_url": OFFICIAL_API_URL,
                "start": start,
                "end": end,
                "only_narratives": only_narratives,
                "local_path": str(path),
            },
            fh,
            indent=2,
        )
    return df


def fetch_local_csv(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    manifest = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        df = pd.read_csv(path, low_memory=False)
        df["raw_source"] = path.name
        frames.append(df)
        manifest.append({"path": str(path), "rows": int(len(df))})
    combined = pd.concat(frames, ignore_index=True, sort=False)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "cfpb_combined_raw.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8")
    with (RAW_DIR / "source_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "mode": "local-csv",
                "rows": int(len(combined)),
                "combined_path": str(out_path),
                "sources": manifest,
            },
            fh,
            indent=2,
        )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["hf", "official-api", "local-csv"],
        default="hf",
        help="Data source mode.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload HF data.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--include-no-narrative", action="store_true")
    parser.add_argument(
        "--local-path",
        action="append",
        default=[],
        help="One or more local CSV files for --source local-csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "hf":
        df = fetch_hf_sources(force=args.force)
    elif args.source == "official-api":
        df = fetch_official_api(
            start=args.start,
            end=args.end,
            max_records=args.max_records,
            only_narratives=not args.include_no_narrative,
        )
    else:
        if not args.local_path:
            raise SystemExit("--local-path is required for --source local-csv")
        df = fetch_local_csv(args.local_path)

    print(f"Fetched {len(df):,} rows")
    print(f"Raw data directory: {RAW_DIR}")


if __name__ == "__main__":
    main()
