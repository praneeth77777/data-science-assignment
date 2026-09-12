# PedalPulse Raw Data Dictionary

Competition meanings are documented semantics; observed ranges and counts come from the attached raw files.

| field | raw_train_dtype | expected_logical_type | competition_documentation_meaning | expected_values_or_rule | model_role | forecast_time_availability | leakage_status | present_in_competition_test | observed_raw_train |
|---|---|---|---|---|---|---|---|---|---|
| datetime | object | timestamp text; parsed diagnostically | Hourly date and time | parseable, unique, chronologically orderable | temporal key | Known before prediction | No | True | 2011-01-01 00:00:00 to 2012-12-19 23:00:00; 10886 unique |
| season | int64 | categorical integer | 1=spring, 2=summer, 3=fall, 4=winter | {1,2,3,4} | candidate predictor | Calendar-derived and known | No | True | min=1, max=4, unique=4, missing=0 |
| holiday | int64 | binary integer | Whether the day is a holiday | {0,1} | candidate predictor | Known from holiday calendar | No | True | min=0, max=1, unique=2, missing=0 |
| workingday | int64 | binary integer | Whether the day is neither weekend nor holiday | {0,1} | candidate predictor | Known from calendar | No | True | min=0, max=1, unique=2, missing=0 |
| weather | int64 | categorical integer | 1=clear/few clouds; 2=mist/cloudy; 3=light precipitation; 4=severe precipitation/fog | {1,2,3,4} | candidate predictor | Requires target-hour weather forecast in deployment | No direct leakage | True | min=1, max=4, unique=4, missing=0 |
| temp | float64 | continuous float | Temperature in degrees Celsius | >= 0 for this audit | candidate predictor | Requires a forecast-time value | No | True | min=0.82, max=41.0, unique=49, missing=0 |
| atemp | float64 | continuous float | Feels-like temperature in degrees Celsius | >= 0 for this audit | candidate predictor | Requires a forecast-time value | No | True | min=0.76, max=45.455, unique=60, missing=0 |
| humidity | int64 | integer percentage | Relative humidity | 0 to 100 | candidate predictor | Requires a forecast-time value | No | True | min=0, max=100, unique=89, missing=0 |
| windspeed | float64 | continuous float | Wind speed; unit not stated in supplied CSV | >= 0 | candidate predictor | Requires a forecast-time value | No | True | min=0.0, max=56.9969, unique=28, missing=0 |
| casual | int64 | nonnegative integer count | Rentals by non-registered users | integer >= 0 | outcome component; audit only | Known only after/during target hour | PROHIBITED: direct target leakage | False | min=0, max=367, unique=309, missing=0 |
| registered | int64 | nonnegative integer count | Rentals by registered users | integer >= 0 | outcome component; audit only | Known only after/during target hour | PROHIBITED: direct target leakage | False | min=0, max=886, unique=731, missing=0 |
| count | int64 | nonnegative integer count | Total rentals | integer >= 0 and casual + registered | supervised target | Unknown at prediction time | Target; never contemporaneous input | False | min=1, max=977, unique=822, missing=0 |

Source: https://www.kaggle.com/competitions/bike-sharing-demand/data
