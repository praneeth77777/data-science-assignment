"""Project-relative paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_WORKBOOK = RAW_DIR / "Online Retail.xlsx"
ARTIFACTS = PROJECT_ROOT / "artifacts"
TABLES = ARTIFACTS / "tables"
FIGURES = ARTIFACTS / "figures"
REPORTS = ARTIFACTS / "reports"
EXPERIMENTS = ARTIFACTS / "experiments"
MODELS = ARTIFACTS / "models"
CONFIG = PROJECT_ROOT / "configs" / "autoresearch.json"


def ensure_artifact_dirs() -> None:
    for directory in (TABLES, FIGURES, REPORTS, EXPERIMENTS, MODELS):
        directory.mkdir(parents=True, exist_ok=True)
