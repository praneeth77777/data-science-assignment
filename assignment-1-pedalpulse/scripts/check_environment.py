#!/usr/bin/env python3
"""Check whether a fresh PedalPulse environment is ready to run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "pedalpulse_rf.joblib"
EXPECTED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "streamlit": "streamlit",
    "nbformat": "nbformat",
    "nbclient": "nbclient",
    "pytest": "pytest",
}


def writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix="pedalpulse_", delete=True):
            pass
        return True
    except OSError:
        return False


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--allow-missing-data", action="store_true")
    parser.add_argument("--allow-missing-model", action="store_true")
    args = parser.parse_args()

    missing = [name for name, module in EXPECTED_IMPORTS.items() if importlib.util.find_spec(module) is None]
    outputs = [PROJECT_ROOT / "artifacts", PROJECT_ROOT / "artifacts" / "models"]
    checks = {
        "python_version": sys.version.split()[0],
        "supported_python": sys.version_info[:2] in {(3, 11), (3, 12)},
        "active_interpreter": sys.executable,
        "missing_dependencies": missing,
        "dataset": {"path": str(args.data), "exists": args.data.is_file()},
        "model": {"path": str(args.model), "exists": args.model.is_file()},
        "writable_output_directories": {str(path): writable(path) for path in outputs},
        "port": {"number": args.port, "available": port_available(args.port)},
    }
    failures = []
    if not checks["supported_python"]:
        failures.append("Install Python 3.11 or 3.12 and recreate .venv.")
    if missing:
        failures.append("Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    if not checks["dataset"]["exists"] and not args.allow_missing_data:
        failures.append("Download Kaggle train.csv and place it in data\\raw\\train.csv.")
    if not checks["model"]["exists"] and not args.allow_missing_model:
        failures.append("Run scripts\\train_model.py after placing train.csv.")
    if not all(checks["writable_output_directories"].values()):
        failures.append("Move the repository to a writable folder or correct directory permissions.")
    if not checks["port"]["available"]:
        failures.append(f"Port {args.port} is busy; stop that process or select another Streamlit port.")
    checks["corrective_actions"] = failures
    checks["status"] = "PASS" if not failures else "FAIL"
    print(json.dumps(checks, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
