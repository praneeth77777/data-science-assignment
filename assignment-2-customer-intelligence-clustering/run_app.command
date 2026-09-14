#!/bin/bash
set -u
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv is missing. Follow README.md."
  read -r -p "Press Return to close..." _
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py --server.port 8502 --server.address localhost --browser.gatherUsageStats false
