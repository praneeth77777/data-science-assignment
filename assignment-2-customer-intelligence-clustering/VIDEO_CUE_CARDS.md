# SegmentForge student video cue cards

Use these prompts to explain the project in your own words. Do not read them as a script.

## Suggested 10–14 minute flow

1. **Problem and CRISP-DM — 60 seconds**
   - Why customer segmentation is descriptive rather than predictive accuracy.
   - Business questions supported and unsupported.

2. **Dataset and raw audit — 90 seconds**
   - Show UCI source and license.
   - Explain 541,909 transaction rows and eight fields.
   - Show missing IDs, cancellations, duplicates, and nonpositive values.

3. **Feature engineering — 75 seconds**
   - Explain Recency, Frequency, and Monetary in plain language.
   - CustomerID groups records but never enters the clustering matrix.
   - Explain the fixed snapshot date and why future transactions cannot leak backward.

4. **Candidate algorithms — 90 seconds**
   - K-Means: centroid baseline, compact Euclidean groups.
   - GMM: probabilistic elliptical components.
   - Agglomerative: hierarchical Ward grouping.
   - Explain scaling, transformations, and outlier sensitivity.

5. **Bounded AutoResearch — 90 seconds**
   - Show `configs/autoresearch.json`.
   - One mutation changes one decision.
   - 42 trials: 34 valid, 8 rejected, 0 failed in the corrected run.
   - Composite objective is a search aid, not scientific truth.

6. **Selection and locked audit — 90 seconds**
   - Winner: K-Means, k=4, RFM, Yeo–Johnson, StandardScaler, keep outliers.
   - Discovery silhouette 0.3331; audit 0.3383.
   - Stability ARI 0.9888 ± 0.0025.
   - Explain why these are not accuracy scores.

7. **Personas — 75 seconds**
   - Show profile heatmap and revenue contribution.
   - Explain all four segments using median R/F/M.
   - Historical revenue share does not prove future lifetime value.

8. **Dashboard — 2 minutes**
   - Visit all eight pages.
   - Demonstrate masked customer lookup and Model Lab rejected trials.
   - Show quality/drift page and its not-measured production warning.

9. **Engineering and tests — 60 seconds**
   - Explain `src`, `scripts`, `tests`, `artifacts`, notebook, requirements, and model cards.
   - Run pytest and the dashboard smoke test.

10. **Lessons and limitations — 60 seconds**
    - Stability and audit matter more than a pretty PCA plot.
    - One historical retailer cannot justify universal customer types.
    - Actions must be tested through controlled experiments.

## Results card

- Customers segmented: 4,338
- Discovery/audit: 3,443 / 895
- Search trials: 42
- Winner: K-Means, four clusters
- Discovery/audit silhouette: 0.3331 / 0.3383
- Stability ARI: 0.9888 ± 0.0025
- Dashboard pages: eight
- Tests: use only the count from your own executed terminal
