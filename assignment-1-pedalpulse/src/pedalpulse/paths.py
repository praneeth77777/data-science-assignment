"""Project-relative paths shared by scripts, tests, and the dashboard."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
TABLES_DIR = ARTIFACTS_DIR / "tables"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODELS_DIR = ARTIFACTS_DIR / "models"
DEFAULT_TRAIN_DATA = DATA_DIR / "train.csv"
DEFAULT_MODEL = MODELS_DIR / "pedalpulse_rf.joblib"
DEFAULT_MODEL_METADATA = MODELS_DIR / "pedalpulse_rf_metadata.json"
