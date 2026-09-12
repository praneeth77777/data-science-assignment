# PedalPulse Assumptions and Decisions Log

**CRISP-DM phase:** Business Understanding
**Status vocabulary:** `Locked decision`, `Planning constraint`, `To validate`, `Unresolved`

No entry may be changed silently. A revision must retain the previous statement and record the trigger, date, rationale, affected artifacts, and whether earlier results must be rerun.

## Decisions

| ID | Decision | Rationale | Status | Validation or change trigger |
|---|---|---|---|---|
| PP-D01 | Predict total rental `count` for the next clock hour at the end of the current hour. | Provides a concrete intraday operational horizon. | Locked decision | Change only if the raw timestamp structure makes it infeasible; revise before modeling. |
| PP-D02 | Treat output as an expected system-wide aggregate, not a station-level recommendation. | The expected dataset has no station inventory or geographic placement fields. | Locked decision | Verify granularity in Chunk 2; pause if the uploaded data differ. |
| PP-D03 | Calendar features for the target hour are allowed; target-hour weather is treated as a forecast-weather proxy. | Calendar is known, but realized weather is not known perfectly in advance. | Locked decision | Add a deployment-realism sensitivity experiment after schema verification. |
| PP-D04 | `count` is the target. `casual` and `registered` are prohibited from predictor preprocessing, selection, clustering, inference, and model fitting. | The asserted identity makes both direct target leakage. | Locked decision | Verify the identity in Chunk 2; automated leakage tests remain mandatory regardless. |
| PP-D05 | Sort by timestamp, never shuffle, reserve the latest defensible period as untouched test data, and use expanding temporal validation on earlier data. | Simulates learning from the past and predicting the future. | Rule locked; cutoffs unresolved | Determine exact cutoffs from timestamp coverage before examining model performance. |
| PP-D06 | Primary seasonal baseline is exact `count` at \(t-168\) hours, followed by exact \(t-24\), fold-training same-hour/working-day median, then fold-training global median. | Creates a full-coverage, transparent, history-only benchmark despite possible gaps. | Locked decision | Chunk 2 audits timestamp continuity; later tests verify every source timestamp precedes its target. |
| PP-D07 | Report fallback tier and pure weekly-lag coverage for every temporal split. | Prevents fallback behavior from being hidden inside a single metric. | Locked decision | Implement with baseline evaluation. |
| PP-D08 | Required metrics are MAE, RMSE, RMSLE, and \(R^2\); mean fold RMSLE is primary. | Covers operational, peak-sensitive, relative, and descriptive error. | Locked decision | Metric tests must use identical paired evaluation rows. |
| PP-D09 | Minimum validation success is at least 10.0% lower mean RMSLE, 5.0% lower mean MAE, and lower RMSLE in a strict majority of folds versus seasonal naive. | Predeclares a material improvement and prevents moving the goalposts. | Locked decision | Do not revise after observing validation or test metrics. |
| PP-D10 | The latest test period remains sealed until preprocessing, features, model family, hyperparameters, prediction transformation, and explanation approach are locked. Evaluate once. | Preserves an honest future-period estimate. | Locked decision | Accidental access becomes a documented protocol breach; do not tune afterward. |
| PP-D11 | Clusters are formed from standardized contextual features without `count`, `casual`, or `registered`; demand may be summarized only after cluster formation. | Prevents target segmentation from masquerading as unsupervised discovery. | Locked decision | Audit the clustering feature list and selection process. |
| PP-D12 | Rare observations are not removed merely because they reduce model scores. Domain validity precedes IQR/Isolation Forest analysis and training-only sensitivity experiments. | Legitimate peaks may be operationally important. | Locked decision | Removal requires a documented invalidity rule independent of test results. |
| PP-D13 | Preprocessing, imputation, scaling, feature selection, and suspicious-value handling are fitted within training folds. | Prevents validation information from affecting learned transformations. | Locked decision | Pipeline and unit-test audit. |
| PP-D14 | Random seeds, configurations, package versions, runtime, hardware, and split boundaries are recorded. Use repository-relative `pathlib` paths. | Supports deterministic reruns and the Windows handoff. | Locked decision | Clean-clone verification in Chunk 8. |
| PP-D15 | Raw Kaggle files remain outside Git unless redistribution permission is explicitly verified; publish download and placement instructions. | Protects licensing, provenance, and repository hygiene. | Locked decision | Revisit only after an explicit license review. |
| PP-D16 | Before a GitHub write, verify authenticated identity `praneeth77777` and whether `data-science-assignment` already exists; never overwrite it. | Protects the user's account and existing work. | Locked decision | Checks occur only at an authorized GitHub checkpoint. |
| PP-D17 | The user personally records, narrates, edits, and uploads both videos. Assistance is limited to structure, cue cards, and checklists. | Preserves student authorship and follows the assignment instructions. | Locked decision | Keep pending URL placeholders until real links are supplied and verified. |

## Assumptions and unresolved items

| ID | Assumption or unresolved item | Current status | How it will be checked | Consequence if false or unresolved |
|---|---|---|---|---|
| PP-A01 | The user will upload the official Kaggle `train.csv`, or an already-configured Kaggle API will be available. | To validate | Confirm before Chunk 2 data work. | Data Understanding cannot begin. |
| PP-A02 | The uploaded file has the expected Bike Sharing Demand schema. | To validate | Raw schema and provenance audit in Chunk 2; record a checksum when practical. | Stop and clarify rather than adapting silently. |
| PP-A03 | `count = casual + registered` for every row. | To validate | Row-level equality test in Chunk 2. | Document violations; both component columns remain forbidden predictors. |
| PP-A04 | Timestamps have sufficient coverage for chronological splitting and seasonal lags. | To validate | Coverage, duplicates, gaps, and frequency audit in Chunk 2. | Revise split dates or fallback mechanics with a logged decision. |
| PP-A05 | Recorded weather fields are realized conditions, not archived operational forecasts. | Planning assumption | Verify dataset documentation and columns; dataset itself cannot prove forecast availability. | State likely optimism and run a feature-availability sensitivity comparison. |
| PP-A06 | Earlier hourly `count` becomes available before each next-hour forecast. | Planning assumption | Confirm as part of the operational contract; no live system exists to verify it. | Lag features and rolling baseline may be inappropriate. |
| PP-A07 | Recorded rentals are an imperfect proxy for desired demand because supply constraints may suppress trips. | Planning assumption | Cannot be resolved without station inventory and unserved-demand data. | Interpret predictions as rentals, not unconstrained demand. |
| PP-A08 | Station inventory, station geography, bike availability, outages, events, prices, transit disruptions, and routing data are unavailable. | To validate | Confirm actual schema and supplied files in Chunk 2. | Restrict operational recommendations and error explanations. |
| PP-A09 | No operator cost matrix, intervention threshold, staffing rule, or pilot KPI has been supplied. | Unresolved | Revisit only if the user provides operational parameters. | Do not claim cost savings or optimize a business-cost function. |
| PP-A10 | Dataset redistribution permission has not been verified. | Unresolved | Review official terms before publication. | Raw data stay uncommitted. |
| PP-A11 | A CPU-only workflow of approximately 2–4 cores, at most 8 GB RAM, and roughly two CPU-hours for Part 1 is acceptable. | Planning constraint | Record actual environment and runtimes during execution. | Reduce search spaces or explanation sample sizes transparently. |
| PP-A12 | XGBoost and full-dataset SHAP may be omitted if unavailable, incompatible, or outside the compute budget. | Planning constraint | Dependency and runtime checks in modeling/evaluation chunks. | Use the specified scikit-learn alternatives and disclose the choice. |
| PP-A13 | The historical dataset may not generalize to another city, a newer period, or changed fleet/policy conditions. | Unresolved | Examine temporal error and drift evidence; external validation is unavailable. | Limit deployment claims and recommend new-data validation. |
| PP-A14 | The expanded Part 2 replication scope may differ from the Assignment 2 described in the linked guidance. | Unresolved | Obtain exact scope approval before executing Part 2. | May require a supplementary AI-assisted application demonstration. |
| PP-A15 | The student's own conversation export must be submitted separately from the professor's example conversation. | Planning requirement | Confirm publication/submission checklist before finalization. | Avoid treating the professor's transcript as the student's required evidence. |

## Test-set and change-control rule

Iterations may revise Business Understanding, Data Understanding, preparation, features, or validation using development data. The final test period does not become another iteration source. If test results fail the predeclared criteria, record the failure and recommend future work; do not return to tuning and retest.

## Measured-result status

This log contains decisions, assumptions, constraints, and validation plans only. No dataset statistic, baseline score, model metric, test outcome, runtime, GitHub state, application state, publication link, or video result has been measured.
