@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv was not found.
  echo Run: py -3.12 -m venv .venv
  echo Then install: .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

echo Starting PedalPulse at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
