# SegmentForge Model Card

## Model

- Algorithm: `kmeans`
- Clusters: 4
- Feature view: `rfm`
- Transformation: `yeo_johnson`
- Scaler: `standard`
- Outlier policy: `keep`
- Random seed: 42

## Population and validation

- Customer feature rows: 4,338
- Discovery customers: 3,443
- Locked audit customers: 895
- AutoResearch trials: 42 total; 34 valid; 8 rejected; 0 failed
- Dataset SHA-256: `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`

| Metric | Discovery | Locked audit |
|---|---:|---:|
| Silhouette | 0.3331 | 0.3383 |
| Calinski-Harabasz | 2872.26 | 741.30 |
| Davies-Bouldin | 1.0184 | 1.0219 |
| Minimum cluster share | 21.49% | 23.13% |
| Maximum cluster share | 29.33% | 27.60% |

Development stability ARI: 0.9888 ± 0.0025 across the predeclared subsamples.

## Intended use

Descriptive customer analysis, campaign hypothesis generation, service planning, and segment monitoring. Human review and controlled experiments are required before customer treatment decisions.

## Limitations

- Clusters are descriptive and are not causal effects, customer lifetime value, or guaranteed future behavior.
- The data describe one UK retailer in 2010–2011.
- Customers without identifiers are excluded from customer-level segmentation.
- Returns and cancellations are represented through diagnostics and a cancellation-rate feature; positive sales define monetary value.
- Cluster membership depends on the observation window, preparation choices, and distance/model assumptions.
- The hill-climb objective is a declared engineering heuristic, not scientific truth.
