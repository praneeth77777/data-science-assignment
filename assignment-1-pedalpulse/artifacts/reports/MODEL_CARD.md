# PedalPulse Model Card

## Model details

- Candidate: `rf_raw_depth18_leaf1`
- Family: Random Forest
- Target scale: raw
- Random seed: 42
- Output: predicted system-wide hourly rental count
- Prediction contract: one-hour-ahead operational planning

## Training and evaluation

- Development rows: 9,519, ending 2012-09-19
- Locked test rows: 1,367, covering 2012-10-01 through 2012-12-19
- `casual` and `registered` were excluded from every model input.
- Locked-test evaluation was performed once after candidate selection.

| Metric | Final model | Seasonal naive |
|---|---:|---:|
| MAE | 57.409 | 62.737 |
| RMSE | 83.410 | 106.176 |
| RMSLE | 0.3739 | 0.5255 |
| R² | 0.8327 | 0.7289 |

## Uncertainty

The 10th–90th percentile range across individual Random Forest trees covered **61.67%** of locked-test outcomes and had mean width **127.67**. This is a heuristic ensemble spread, not a calibrated prediction interval.

## Explainability

The highest locked-test permutation importance was `numeric__hour_cos`. Partial-dependence diagnostics were generated for temperature, humidity, and windspeed. These summaries are associative and can be distorted by correlated predictors. SHAP was not run because the package was unavailable.

## Intended use

Support system-level rebalancing, staffing, and capacity planning. Predictions should inform human decisions rather than automatically deny service or allocate resources without monitoring.

## Limitations and risks

- The locked-period mean residual (`actual − predicted`) was +49.97, indicating systematic underprediction that requires monitoring before operational use.
- Historical data cover only 2011–2012 and may not represent current mobility behavior.
- No station location, dock inventory, trip purpose, event calendar, or unmet-demand field is available.
- The model predicts observed rentals, which may be constrained by historical bike availability.
- Rare severe-weather performance cannot be reliably estimated.
- Drift, outages, policy changes, and new mobility alternatives can reduce accuracy.
