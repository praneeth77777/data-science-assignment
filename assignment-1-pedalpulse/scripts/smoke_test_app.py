#!/usr/bin/env python3
"""Exercise every Streamlit page without opening a network socket."""

from __future__ import annotations

import argparse
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "app.py"
PAGES = [
    "Demand prediction",
    "EDA",
    "Clusters & outliers",
    "Model performance",
    "Explainability",
    "Reproducibility",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    app = AppTest.from_file(str(APP), default_timeout=args.timeout).run()
    checked = []
    prediction_rendered = False
    for page in PAGES:
        app.sidebar.radio[0].set_value(page)
        app.run()
        if app.exception:
            messages = "\n".join(str(exception.value) for exception in app.exception)
            raise RuntimeError(f"Streamlit page failed: {page}\n{messages}")
        if page == "Demand prediction":
            app.button[0].click().run()
            if app.exception or not app.metric:
                messages = "\n".join(str(exception.value) for exception in app.exception)
                raise RuntimeError(f"Prediction interaction failed.\n{messages}")
            prediction_rendered = True
        checked.append(page)
    print(
        f"PASS: Streamlit rendered {len(checked)} pages and the prediction interaction "
        f"completed={prediction_rendered}, with no uncaught exceptions."
    )


if __name__ == "__main__":
    main()
