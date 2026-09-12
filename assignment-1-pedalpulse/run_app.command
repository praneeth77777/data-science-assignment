#!/bin/bash
set -u

cd "$(dirname "$0")" || exit 1

PROJECT_PYTHON="./.venv/bin/python"
if [ ! -x "$PROJECT_PYTHON" ]; then
  echo "ERROR: .venv was not found."
  echo "Run these commands from this project directory:"
  echo "  python3.12 -m venv .venv"
  echo "  ./.venv/bin/python -m pip install --upgrade pip"
  echo "  ./.venv/bin/python -m pip install -r requirements.txt"
  echo
  read -r -p "Press Return to close..." _
  exit 1
fi

echo "Starting PedalPulse at http://localhost:8501"
exec "$PROJECT_PYTHON" -m streamlit run app.py \
  --server.port 8501 \
  --server.address localhost \
  --browser.gatherUsageStats false
