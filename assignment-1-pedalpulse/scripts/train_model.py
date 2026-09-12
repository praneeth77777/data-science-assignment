#!/usr/bin/env python3
"""Train the locked deployment candidate without evaluating the held-out test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pedalpulse.modeling import save_bundle, train_deployment_bundle  # noqa: E402
from pedalpulse.paths import DEFAULT_MODEL, DEFAULT_MODEL_METADATA, DEFAULT_TRAIN_DATA  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Training data not found: {args.data}")
    started = time.perf_counter()
    raw_hash = sha256(args.data)
    raw = pd.read_csv(args.data)
    bundle = train_deployment_bundle(raw, input_sha256=raw_hash)
    bundle["metadata"]["fit_runtime_seconds"] = time.perf_counter() - started
    save_bundle(bundle, args.model, args.metadata)
    print(json.dumps({
        "status": "trained",
        "candidate": bundle["metadata"]["candidate"],
        "training_rows": bundle["metadata"]["training_rows"],
        "locked_test_rows_used_for_training": 0,
        "locked_test_metrics_computed": False,
        "model_path": str(args.model),
        "metadata_path": str(args.metadata),
    }, indent=2))


if __name__ == "__main__":
    main()
