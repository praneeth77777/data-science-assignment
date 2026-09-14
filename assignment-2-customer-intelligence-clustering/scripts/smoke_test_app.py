#!/usr/bin/env python3
"""Render every SegmentForge dashboard page without a browser socket."""

from __future__ import annotations

import argparse
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "Executive Overview", "Data Explorer", "Segment Explorer", "Customer Lookup",
    "Model Lab", "Explainability", "Quality & Drift", "Deployment Admin",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=args.timeout).run()
    checked = []
    for page in PAGES:
        app.sidebar.radio[0].set_value(page)
        app.run()
        if app.exception:
            detail = "\n".join(str(item.value) for item in app.exception)
            raise RuntimeError(f"Page failed: {page}\n{detail}")
        checked.append(page)
    print(f"PASS: rendered {len(checked)} SegmentForge pages with no uncaught exceptions.")


if __name__ == "__main__":
    main()
