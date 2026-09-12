# PedalPulse Data Preparation Report

**CRISP-DM phase:** Data Preparation
**Status:** Deterministic feature and preprocessing components implemented and executed. No supervised model was fitted.

## Locked chronological partitions

| Partition | Rows | Minimum timestamp | Maximum timestamp |
|---|---:|---|---|
| Train | 8,151 | 2011-01-01 00:00:00 | 2012-06-19 23:00:00 |
| Validation | 1,368 | 2012-07-01 00:00:00 | 2012-09-19 23:00:00 |
| Untouched labeled test | 1,367 | 2012-10-01 00:00:00 | 2012-12-19 23:00:00 |

The boundaries were chosen before model evaluation. Detailed EDA uses only train plus validation; the locked test target is neither summarized nor passed into preprocessing. The Kaggle competition test remains inference-only. Later cross-validation will occur inside the development period without accessing locked-test outcomes for feature or model decisions.

## Features implemented

- Parsed and strictly validated timestamp.
- Hour, weekday, month, year, weekend, and working-day rush-hour flag.
- Sine/cosine encodings for hour, weekday, and month.
- Explicit zero flags for `atemp`, `humidity`, and `windspeed`.
- Original shared exogenous predictors retained.

Rush hour is defined deterministically as 07:00–09:59 or 16:00–19:59 on a published working day. This is a declared operational feature definition, not an observed causal result.

## Leakage and ordering controls

- `safe_predictor_frame` selects only the nine shared predictor fields.
- Schema validation raises an exception if `count`, `casual`, or `registered` reaches the feature pipeline.
- Duplicate, malformed, or out-of-order timestamps raise exceptions.
- The chronological split verifies strictly ordered, non-overlapping periods.

## Pipeline design

The scikit-learn `Pipeline` performs calendar feature engineering, suspicious-zero handling, and a `ColumnTransformer`. Continuous and cyclical fields receive median imputation followed by standardization. Categorical fields receive most-frequent imputation followed by one-hot encoding with unknown-category tolerance.

Both candidate strategies produced **74** finite output columns and were fitted only on the training partition before transforming validation and test partitions.

## Suspicious-value comparison

1. `preserve_with_flags`: retain zeros and add explicit zero indicators.
2. `replace_with_train_median_and_flags`: replace flagged zeros using nonzero medians learned from training only, while retaining the indicators.

Measured training-only replacement values were: **atemp=22.725, humidity=61, windspeed=12.998**. No strategy has been declared superior; later temporal validation must compare them. Preserving flagged zeros is the conservative default until that sensitivity experiment is complete.

## Determinism

No randomized transformation is used. Feature definitions, column lists, category handling, and split boundaries are explicit. The executed unit suite checks schema failure, leakage prevention, timestamp ordering, feature values, training-only suspicious-value medians, deterministic transformation, and split non-overlap.
