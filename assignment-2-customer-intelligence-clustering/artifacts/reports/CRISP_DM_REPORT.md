# SegmentForge CRISP-DM Report

## Business understanding

Segment customers from historical purchase behavior to support analysis and controlled campaign hypotheses. The system does not claim causality, profitability lift, or individual intent.

## Data understanding

The raw workbook contained 541,909 transaction lines. Raw defects and cancellations were documented before preparation. See `raw_quality_summary.csv` and `exclusion_audit.csv`.

## Data preparation

Exact duplicates and rows without CustomerID were excluded from customer aggregation. Monetary behavior uses completed positive-quantity, positive-price invoices. Cancellation behavior is retained as a separate rate. A fixed snapshot one day after the final valid purchase defines recency.

## Modeling and AutoResearch

The bounded hill climber executed 42 trials and changed one configuration component per mutation. The locked winner was `kmeans` with k=4, `rfm` features, `yeo_johnson`, `standard`, and `keep`. The winning composite objective was 0.8128. All component metrics and rejected/failed trials remain visible.

## Evaluation

Discovery silhouette was 0.3331; locked-audit silhouette was 0.3383. Discovery stability ARI was 0.9888. These are internal clustering diagnostics, not predictive accuracy.

## Deployment

The Streamlit admin dashboard reads frozen aggregate artifacts and a versioned model bundle. It exposes quality, segments, experiments, explainability, and reproducibility without embedding the raw workbook.

## Runtime

Complete analysis runtime: 55.52 seconds on Linux-6.18.44-x86_64-with-glibc2.39 with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, and scikit-learn 1.8.0.
