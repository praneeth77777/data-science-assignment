# SegmentForge Implementation Plan and Completion Record

## Completed phases

1. Business scope, stakeholders, unsupported claims, and research questions.
2. Raw workbook fingerprint and pre-cleaning quality audit.
3. Declared treatment of duplicates, missing identifiers, cancellations, and nonpositive lines.
4. Customer RFM and behavioral feature engineering with a fixed snapshot.
5. Deterministic discovery and locked customer-audit partition.
6. K-Means, Gaussian Mixture, and Agglomerative candidate families.
7. Bounded 42-trial hill climbing with one-change mutations and complete trial logging.
8. Cluster-size constraints, multiple internal metrics, and subsample ARI stability.
9. One locked audit evaluation and post-selection persona construction.
10. Model bundle, metadata, tables, figures, reports, executed notebook, tests, and eight-page dashboard.

## Change-control rules

- Do not optimize against the locked audit after viewing its metrics.
- Any change to data rules, features, objective weights, or constraints creates a new experiment version.
- Preserve failed and rejected trials.
- Never convert descriptive segments into causal or profitability claims without additional evidence.
- Production drift thresholds remain pending until a real comparison period and business tolerance exist.
