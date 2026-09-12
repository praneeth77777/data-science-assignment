# PedalPulse Data Card

## Dataset

Kaggle Bike Sharing Demand competition data: https://www.kaggle.com/competitions/bike-sharing-demand

The student supplied `train.csv`. Redistribution permission has not been established, so raw data are excluded from publication artifacts; reproduction instructions must require users to download the dataset themselves.

## Unit of observation

One system-wide hourly record containing timestamp, calendar/weather fields, `casual`, `registered`, and total `count`.

## Target and leakage

`count` is the target and equals `casual + registered` for every training row. The component fields are forbidden model inputs.

## Splits

- Development: 9,519 rows through 2012-09-19.
- Locked test: 1,367 rows from 2012-10-01 through 2012-12-19.
- Kaggle competition test: target-free and used only for future inference/submission demonstrations.

## Known quality issues

No ordinary missing cells or duplicated timestamps were measured. Suspicious values include concentrated zero humidity, frequent zero windspeed, rare weather code 4, and gaps in the hourly calendar. These were flagged rather than silently deleted.

## Representation and ethics

The data do not contain protected demographic attributes, but that does not eliminate fairness concerns. Service allocation based on historical demand can reinforce geographic or access inequities that cannot be audited without station and neighborhood information.
