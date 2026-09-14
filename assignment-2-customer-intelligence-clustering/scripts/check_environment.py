#!/usr/bin/env python3
"""Report SegmentForge runtime, dependencies, files, output permissions, and port."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib", "openpyxl", "streamlit", "pytest"]


def writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".segmentforge_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def port_available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main() -> None:
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    raw = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
    model = PROJECT_ROOT / "artifacts" / "models" / "segmentforge_clusterer.joblib"
    output = PROJECT_ROOT / "artifacts"
    actions = []
    if sys.version_info < (3, 12):
        actions.append("Install Python 3.12 and recreate .venv.")
    if missing:
        actions.append("Install requirements.txt using the active interpreter.")
    if not raw.is_file():
        actions.append("Run scripts/download_data.py before reproducing the analysis.")
    if not model.is_file():
        actions.append("Run scripts/run_pipeline.py to build the model artifacts.")
    result = {
        "python_version": sys.version.split()[0], "active_interpreter": sys.executable,
        "supported_python": sys.version_info >= (3, 12), "missing_dependencies": missing,
        "raw_workbook": {"path": str(raw), "exists": raw.is_file()},
        "model": {"path": str(model), "exists": model.is_file()},
        "artifacts_writable": writable(output), "port_8502_available": port_available(8502),
        "corrective_actions": actions,
        "status": "PASS" if not missing and sys.version_info >= (3, 12) and model.is_file() else "REVIEW",
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
