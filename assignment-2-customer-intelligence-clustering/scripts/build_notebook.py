#!/usr/bin/env python3
"""Build and execute the compact SegmentForge CRISP-DM notebook."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import os
import traceback
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# SegmentForge: CRISP-DM Customer Intelligence Clustering\n\nThis notebook reads frozen, reproducible outputs from the executed pipeline. Raw customer transactions are not embedded."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import Image, display\nROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\nTABLES = ROOT / 'artifacts' / 'tables'\nFIGURES = ROOT / 'artifacts' / 'figures'\nMODELS = ROOT / 'artifacts' / 'models'\nprint('Project root resolved from the current notebook context')"),
        nbf.v4.new_markdown_cell("## Business and data understanding\n\nThe objective is descriptive customer segmentation for analysis and controlled hypotheses. It is not causal targeting or customer-lifetime-value prediction."),
        nbf.v4.new_code_cell("quality = pd.read_csv(TABLES / 'raw_quality_summary.csv')\nexclusions = pd.read_csv(TABLES / 'exclusion_audit.csv')\ndisplay(quality, exclusions)\ndisplay(Image(filename=str(FIGURES / 'data_quality_overview.png')))"),
        nbf.v4.new_markdown_cell("## Data preparation\n\nRecency, Frequency, and Monetary features use valid positive purchases and a fixed snapshot date. Cancellations are retained as a separate diagnostic feature. Identifiers are excluded from clustering."),
        nbf.v4.new_code_cell("summary = pd.read_csv(TABLES / 'customer_feature_summary.csv')\ndisplay(summary.loc[summary['Unnamed: 0'].isin(['recency_days','frequency','monetary'])])\ndisplay(Image(filename=str(FIGURES / 'rfm_distributions.png')))"),
        nbf.v4.new_markdown_cell("## Modeling and bounded AutoResearch\n\nThe search changes one configuration component at a time and evaluates cohesion, separation, stability, balance, interpretability, and complexity. Rejected trials remain auditable."),
        nbf.v4.new_code_cell("leaderboard = pd.read_csv(TABLES / 'autoresearch_leaderboard.csv')\ncols = ['trial_id','algorithm','n_clusters','feature_view','transformation','scaler','outlier_policy','silhouette','stability_ari_mean','objective']\ndisplay(leaderboard[cols].head(10))\ndisplay(Image(filename=str(FIGURES / 'autoresearch_search.png')))"),
        nbf.v4.new_markdown_cell("## Locked evaluation and interpretation\n\nThe audit population was not used during hill climbing. Internal clustering metrics are not classification accuracy."),
        nbf.v4.new_code_cell("metrics = pd.read_csv(TABLES / 'locked_solution_metrics.csv')\nprofiles = pd.read_csv(TABLES / 'cluster_profiles.csv')\ndisplay(metrics, profiles[['cluster','persona','customers','customer_share','revenue_share','evidence']])\ndisplay(Image(filename=str(FIGURES / 'cluster_pca_projection.png')))\ndisplay(Image(filename=str(FIGURES / 'cluster_profile_heatmap.png')))"),
        nbf.v4.new_markdown_cell("## Deployment and limitations\n\nThe Streamlit dashboard exposes frozen aggregate artifacts, experiment history, masked customer lookup, quality evidence, and model metadata. The data represent one retailer in 2010–2011; segments can drift and do not prove intent or campaign impact."),
        nbf.v4.new_code_cell("metadata = json.loads((MODELS / 'segmentforge_metadata.json').read_text())\nmetadata"),
    ]
    return nb


def execute_in_process(notebook) -> None:
    """Execute code cells without Jupyter sockets and preserve notebook outputs."""
    namespace = {"__name__": "__main__"}
    original_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    count = 0
    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            cell.execution_count = count
            cell.outputs = []
            displayed = []

            def capture_display(*objects, **_kwargs):
                displayed.extend(objects)

            namespace["display"] = capture_display
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(compile(cell.source, f"segmentforge-cell-{count}", "exec"), namespace)
            except Exception as exc:
                cell.outputs.append(nbf.v4.new_output(
                    "error", ename=type(exc).__name__, evalue=str(exc),
                    traceback=traceback.format_exc().splitlines(),
                ))
                raise
            if stdout.getvalue():
                cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=stdout.getvalue()))
            for obj in displayed:
                data = {"text/plain": repr(obj)}
                html = getattr(obj, "_repr_html_", lambda: None)()
                if html:
                    data["text/html"] = html
                png = getattr(obj, "_repr_png_", lambda: None)()
                if isinstance(png, tuple):
                    png = png[0]
                if isinstance(png, bytes):
                    data["image/png"] = base64.b64encode(png).decode("ascii")
                elif isinstance(png, str):
                    data["image/png"] = png
                cell.outputs.append(nbf.v4.new_output("display_data", data=data, metadata={}))
    finally:
        os.chdir(original_cwd)
    notebook.metadata["segmentforge_execution"] = {
        "status": "completed", "engine": "in-process fallback", "code_cells_executed": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "notebooks" / "SegmentForge_CRISP_DM.ipynb")
    parser.add_argument("--execution-mode", choices=["auto", "kernel", "in-process"], default="auto")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    if args.execution_mode == "in-process":
        execute_in_process(notebook)
        executed = notebook
    else:
        try:
            executed = NotebookClient(
                notebook, timeout=180, kernel_name="python3",
                resources={"metadata": {"path": str(PROJECT_ROOT)}},
            ).execute()
            executed.metadata["segmentforge_execution"] = {"status": "completed", "engine": "Jupyter kernel"}
        except RuntimeError:
            if args.execution_mode == "kernel":
                raise
            execute_in_process(notebook)
            executed = notebook
    nbf.write(executed, args.output)
    errors = [output for cell in executed.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.output_type == "error"]
    print(f"Executed {sum(cell.cell_type == 'code' for cell in executed.cells)} code cells; errors={len(errors)}; output={args.output}")


if __name__ == "__main__":
    main()
