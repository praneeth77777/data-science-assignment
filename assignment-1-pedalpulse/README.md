# PedalPulse

An end-to-end CRISP-DM project for hourly bike-sharing demand prediction, usage-pattern discovery, outlier analysis, clustering, feature selection, temporal model validation, explainability, and reproducible deployment.

## Key safeguards

- `casual` and `registered` are forbidden inputs because they sum to the target `count`.
- Validation uses five expanding chronological folds; no random train/test shuffle is used.
- Preprocessing is fitted on training data only.
- The selected model was evaluated once on the locked October–December 2012 period.
- Raw Kaggle data are not committed or bundled.

## Project layout

```text
assignment-1-pedalpulse/
├── app.py                         # Streamlit dashboard
├── requirements.txt
├── run_app.command                # macOS launcher
├── run_app.cmd                    # Windows launcher
├── notebooks/
│   └── PedalPulse_CRISP_DM.ipynb
├── scripts/
│   ├── build_notebook.py
│   ├── check_environment.py
│   ├── smoke_test_app.py
│   └── train_model.py
├── src/pedalpulse/                # data, features, pipelines, validation, inference
├── tests/
├── artifacts/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   └── models/
└── data/raw/                      # user-supplied Kaggle CSV files
```

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py --allow-missing-data
python -m pytest
python -m streamlit run app.py
```

Open `http://localhost:8501`.

After downloading Kaggle `train.csv` to `data/raw/train.csv`, the locked deployment candidate can be rebuilt without scoring the held-out test:

```bash
python scripts/train_model.py --data data/raw/train.csv
python scripts/build_notebook.py --data data/raw/train.csv
```

For a complete from-source analysis rerun, use a new output directory and execute the stages in order. The final command intentionally recomputes the locked evaluation, so do not use its results for additional tuning:

```bash
python scripts/audit_data.py --train data/raw/train.csv --test data/raw/test.csv --sample-submission data/raw/sampleSubmission.csv --output artifacts/raw-audit
python scripts/generate_chunk34.py --train data/raw/train.csv --output artifacts/reproduced-chunk34
python scripts/run_chunk56.py --train data/raw/train.csv --output artifacts/reproduced-chunk567
python scripts/run_chunk7.py --train data/raw/train.csv --selected-config artifacts/reproduced-chunk567/tables/selected_candidate_config.json --output artifacts/reproduced-chunk567
```

macOS users should follow [../docs/MACOS_SETUP.md](../docs/MACOS_SETUP.md). Windows users should follow [../docs/WINDOWS_SETUP.md](../docs/WINDOWS_SETUP.md); neither guide requires activating the virtual environment.

## Dashboard pages

- Demand prediction with an uncalibrated tree-spread range.
- Aggregate EDA figures and evidence-focused interpretations.
- Outlier sensitivity and non-target-leaking cluster personas.
- Development and final performance comparisons.
- Permutation importance, partial dependence, and residual evidence.
- Model metadata, cards, and leakage/reproducibility audit.

## Results and limitations

The final Random Forest achieved RMSLE 0.3739 and MAE 57.409, versus 0.5255 and 62.737 for seasonal naive. Positive final residuals reveal underprediction in the later period; current real-world performance remains unknown. See `artifacts/reports/MODEL_CARD.md` and `artifacts/reports/DATA_CARD.md`.
