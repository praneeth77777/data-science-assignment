#!/usr/bin/env python3
"""Download the CC BY 4.0 UCI Online Retail archive."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "online-retail.zip"
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(args.output)
    workbook = args.output / "Online Retail.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError("Expected Online Retail.xlsx was not present in the archive")
    print(f"Downloaded: {workbook} ({workbook.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
