#!/usr/bin/env python3
"""Build and execute the narrative PedalPulse notebook from frozen results."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import os
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "PedalPulse_CRISP_DM.ipynb"
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "train.csv"


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


def build_notebook():
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
    notebook["cells"] = [
        markdown("""
# PedalPulse: a CRISP-DM bike-sharing demand project

This executed notebook is a concise, reproducible narrative over the modular package and frozen evaluation artifacts. It performs data-quality and leakage checks, displays saved analysis results, and deliberately **does not recompute locked-test metrics**. The selected model was already evaluated once.
        """),
        markdown("""
## 1. Business understanding

The imaginary operator needs one-hour-ahead, system-wide demand estimates for rebalancing, staffing, and capacity planning. Success was defined before model selection as at least 10% RMSLE and 5% MAE improvement over a rolling seasonal-naive baseline. Predictions are decision support, not automatic service-allocation rules.
        """),
        code("""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "pedalpulse").is_dir():
            return candidate
    raise RuntimeError("Run this notebook from within assignment-1-pedalpulse")

PROJECT_ROOT = find_project_root(Path.cwd().resolve())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
TABLES = PROJECT_ROOT / "artifacts" / "tables"
FIGURES = PROJECT_ROOT / "artifacts" / "figures"
DATA_PATH = Path(os.environ.get("PEDALPULSE_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "train.csv"))

from pedalpulse.data import chronological_split, safe_predictor_frame

pd.set_option("display.max_columns", 20)
print("Reproducible project imports: PASS")
        """),
        markdown("""
## 2. Data understanding

The raw Kaggle training file is loaded without mutation. The following cell reports its structure and validates the target identity that creates the central leakage risk.
        """),
        code("""
if not DATA_PATH.is_file():
    raise FileNotFoundError("Download Kaggle train.csv and configure PEDALPULSE_DATA_PATH or place it in data/raw/train.csv")

raw = pd.read_csv(DATA_PATH)
timestamps = pd.to_datetime(raw["datetime"])
summary = pd.DataFrame({
    "measure": ["rows", "columns", "duplicate rows", "missing cells", "timestamp start", "timestamp end"],
    "value": [len(raw), raw.shape[1], raw.duplicated().sum(), raw.isna().sum().sum(), timestamps.min(), timestamps.max()],
})
display(summary)
print("count equals casual + registered for every row:", bool((raw["count"] == raw["casual"] + raw["registered"]).all()))
display(raw.dtypes.rename("dtype").to_frame())
        """),
        markdown("""
## 3. Data preparation and leakage control

Calendar and cyclical features are generated inside a scikit-learn pipeline. `safe_predictor_frame` explicitly selects the permitted predictor contract, so `casual`, `registered`, and `count` cannot enter preprocessing.
        """),
        code("""
splits = chronological_split(raw)
split_summary = pd.DataFrame([
    {
        "split": name,
        "rows": len(frame),
        "start": pd.to_datetime(frame["datetime"]).min(),
        "end": pd.to_datetime(frame["datetime"]).max(),
    }
    for name, frame in splits.items()
])
display(split_summary)
safe_columns = safe_predictor_frame(splits["train"]).columns.tolist()
print("Permitted raw predictors:", safe_columns)
print("Leakage fields in permitted predictors:", sorted({"casual", "registered", "count"}.intersection(safe_columns)))
        """),
        markdown("""
## 4. Exploratory data analysis

The executed EDA found strong hour and calendar structure. Working-day commuting peaks are observed evidence; explanations about why riders behave this way remain hypotheses rather than causal findings.
        """),
        code("""
display(Image(filename=str(FIGURES / "temporal_demand_patterns.png"), width=1000))
display(Image(filename=str(FIGURES / "hour_weekday_heatmap.png"), width=900))
        """),
        markdown("""
**Interpretation.** Demand varies strongly by hour and weekday. The heatmap shows weekday morning/evening peaks and a different weekend profile. These patterns justify temporal features and a chronological validation design.
        """),
        markdown("""
## 5. Outliers, clustering, and feature selection

The training-only IQR diagnostic flagged 258 high-demand periods, but many coincide with legitimate evening commuting demand. K-Means k=2 was selected under a minimum 5% cluster-size rule and did not use the demand target.
        """),
        code("""
display(Image(filename=str(FIGURES / "outlier_sensitivity.png"), width=850))
display(Image(filename=str(FIGURES / "cluster_pca_projection.png"), width=850))
display(pd.read_csv(TABLES / "cluster_operational_personas.csv"))
display(pd.read_csv(TABLES / "feature_selection_ablation.csv").sort_values("rmsle").head(6))
        """),
        markdown("""
**Interpretation.** Log transformation had the best RMSLE in the fixed-model outlier sensitivity study, while deletion was not justified. The PCA display is a projection, not proof of complete separation. Domain-selected features led the fixed-HGB ablation, and removing temporal features caused the largest deterioration.
        """),
        markdown("""
## 6. Temporal modeling

Five expanding temporal folds compared dummy, seasonal-naive, linear, Random Forest, and gradient-boosting candidates. XGBoost was skipped because it was unavailable; no result is invented for it.
        """),
        code("""
family_results = pd.read_csv(TABLES / "model_family_winners.csv")
display(family_results[["family", "candidate", "mae_mean", "mae_std", "rmsle_mean", "rmsle_std", "r2_mean"]])
display(Image(filename=str(FIGURES / "model_family_comparison.png"), width=900))
        """),
        markdown("""
**Interpretation.** Random Forest had the lowest mean development RMSLE, but its fold variation is substantial. The model was selected on development evidence rather than complexity or locked-test performance.
        """),
        markdown("""
## 7. Locked evaluation and explainability

This section reads the immutable metrics produced by the single locked-test evaluation. It does not call a scoring function or refit a model.
        """),
        code("""
final_metrics = pd.read_csv(TABLES / "final_locked_test_metrics.csv")
audit = pd.read_csv(TABLES / "final_leakage_reproducibility_audit.csv")
display(final_metrics)
display(audit)
display(Image(filename=str(FIGURES / "residuals_vs_predictions.png"), width=850))
display(Image(filename=str(FIGURES / "permutation_importance.png"), width=850))
        """),
        markdown("""
**Interpretation.** The Random Forest achieved RMSLE 0.3739 versus 0.5255 for seasonal naive, meeting the predeclared improvement criteria. Positive residuals show underprediction in the later quarter. Cyclical hour features dominate permutation importance, but correlated inputs can redistribute importance. SHAP was unavailable and was not claimed.
        """),
        markdown("""
## 8. Deployment and limitations

The dashboard uses a development-only trained artifact and project-relative paths. The final test contains only season code 4, the data cover 2011–2012, and the tree-spread interval is not calibrated. Deployment therefore requires drift monitoring, newer data, and human oversight.

The raw Kaggle data are intentionally excluded from publication. See `README.md`, `artifacts/reports/MODEL_CARD.md`, `artifacts/reports/DATA_CARD.md`, and `../docs/WINDOWS_SETUP.md`.
        """),
    ]
    return notebook


def execute_in_process(notebook) -> None:
    """Execute cells without a Jupyter socket and store standard notebook outputs."""

    namespace = {"__name__": "__main__"}
    execution_count = 0
    original_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = []
            displayed = []

            def capture_display(*objects, **_kwargs):
                displayed.extend(objects)

            namespace["display"] = capture_display
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(compile(cell.source, f"notebook-cell-{execution_count}", "exec"), namespace)
                namespace["display"] = capture_display
            except Exception as exc:
                cell.outputs.append(nbformat.v4.new_output(
                    "error",
                    ename=type(exc).__name__,
                    evalue=str(exc),
                    traceback=traceback.format_exc().splitlines(),
                ))
                raise
            text_output = stdout.getvalue()
            if text_output:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text_output))
            for obj in displayed:
                data = {"text/plain": repr(obj)}
                output_metadata = {}
                html = getattr(obj, "_repr_html_", lambda: None)()
                if html:
                    data["text/html"] = html
                png = getattr(obj, "_repr_png_", lambda: None)()
                if isinstance(png, tuple):
                    png, image_metadata = png
                    output_metadata["image/png"] = image_metadata
                if isinstance(png, bytes):
                    data["image/png"] = base64.b64encode(png).decode("ascii")
                elif isinstance(png, str):
                    data["image/png"] = png
                cell.outputs.append(nbformat.v4.new_output("display_data", data=data, metadata=output_metadata))
    finally:
        os.chdir(original_cwd)
    notebook.metadata["pedalpulse_execution"] = {
        "status": "completed",
        "engine": "in-process fallback",
        "code_cells_executed": execution_count,
        "locked_test_metrics_recomputed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--execution-mode", choices=["auto", "kernel", "in-process"], default="auto")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbformat.write(notebook, args.output)
    if args.no_execute:
        print(f"Created unexecuted notebook: {args.output}")
        return
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    previous = os.environ.get("PEDALPULSE_DATA_PATH")
    os.environ["PEDALPULSE_DATA_PATH"] = str(args.data.resolve())
    try:
        if args.execution_mode == "in-process":
            execute_in_process(notebook)
        else:
            try:
                client = NotebookClient(
                    notebook,
                    timeout=300,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(PROJECT_ROOT)}},
                )
                client.execute()
                notebook.metadata["pedalpulse_execution"] = {
                    "status": "completed",
                    "engine": "Jupyter kernel",
                    "locked_test_metrics_recomputed": False,
                }
            except RuntimeError:
                if args.execution_mode == "kernel":
                    raise
                execute_in_process(notebook)
        nbformat.write(notebook, args.output)
    finally:
        if previous is None:
            os.environ.pop("PEDALPULSE_DATA_PATH", None)
        else:
            os.environ["PEDALPULSE_DATA_PATH"] = previous
    print(f"Created and executed notebook: {args.output}")


if __name__ == "__main__":
    main()
