# SegmentForge: Stable Customer Segmentation through Bounded Configuration Search

## 1. Introduction

Customer segmentation is useful when it compresses heterogeneous purchase behavior into groups that are stable enough to inspect and specific enough to support testable operational hypotheses. It becomes misleading when one attractive projection or one internal metric is treated as proof of true customer types. SegmentForge addresses that risk with CRISP-DM, a raw-data audit, RFM feature definitions, multiple clustering families, resampling stability, a locked audit population, and a fully logged configuration search.

The project adapts the professor's public Customer Intelligence Clustering prompt with attribution while using an independent implementation and independent results.

## 2. Data

The [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) contains transactions from a UK non-store retailer between December 2010 and December 2011. UCI distributes it under CC BY 4.0 with DOI 10.24432/C5BW33.

The executed raw audit measured 541,909 rows, 135,080 missing CustomerID values, 1,454 missing descriptions, 5,268 exact duplicates, 9,288 cancellation-invoice rows, 10,624 nonpositive quantities, and 2,517 nonpositive prices. Counts overlap and therefore are not summed.

## 3. Feature construction

Valid sales were defined as deduplicated, identified, non-cancellation rows with positive quantity and price. The fixed snapshot was one day after the latest valid purchase. Core features were:

\[
R_i = \text{snapshot date} - \max(\text{purchase date}_i)
\]

\[
F_i = \text{distinct completed invoices}_i
\]

\[
M_i = \sum_j \text{Quantity}_{ij}\text{UnitPrice}_{ij}
\]

RFM cells can discard information and should not automatically be interpreted as lifetime value; see [Fader, Hardie, and Lee](https://www.brucehardie.com/papers/rfm_clv_2005-02-16.pdf). SegmentForge therefore calls its clusters descriptive behavioral groups.

## 4. Methods

The candidate families were K-Means, Gaussian Mixture Models, and Ward agglomerative clustering. K-Means provides an interpretable centroid baseline but favors compact Euclidean groups and is sensitive to scale and outliers. The historical basis is [MacQueen's formulation](https://digicoll.lib.berkeley.edu/nanna/record/113015/files/math_s5_v1_article-17.pdf?registerDownload=1&version=1&withMetadata=0&withWatermark=0). Gaussian mixtures add elliptical density assumptions and soft memberships through expectation maximization; see [Dempster, Laird, and Rubin](https://academic.oup.com/jrsssb/article/39/1/1/7027539).

The search varied RFM, compact, and behavioral feature views; log1p and Yeo–Johnson transformations; standard and robust scaling; keeping versus 99th-percentile winsorization; three algorithms; and two through eight clusters.

The objective combined normalized silhouette, Calinski–Harabasz, Davies–Bouldin, adjusted-Rand stability, cluster balance, and a rule-based interpretability check. It penalized complexity and rejected tiny or dominant clusters. Silhouette follows [Rousseeuw](https://doi.org/10.1016/0377-0427(87)90125-7). Stability concepts have multiple interpretations; [von Luxburg](https://arxiv.org/pdf/1007.1075v1.pdf) motivates treating stability as one diagnostic rather than universal proof.

## 5. AutoResearch protocol

AutoResearch was implemented as bounded hill climbing rather than unrestricted code modification. Each immutable state contained feature view, transformation, scaler, algorithm, cluster count, and outlier policy. A mutation changed one component. The system logged the parent trial, mutation, configuration, metrics, stability, cluster shares, objective, runtime, status, and rejection reason.

The maximum was 42 trials with fixed seeds. The search never accessed the deterministic audit population. The composite weights and validity constraints were frozen in `configs/autoresearch.json` before execution.

## 6. Results

Thirty-four of 42 trials were valid and eight were rejected for cluster-size constraints. No model fit failed in the corrected run. The selected state was K-Means, four clusters, RFM, Yeo–Johnson, StandardScaler, and no outlier removal.

| Metric | Discovery | Locked audit |
|---|---:|---:|
| Silhouette | 0.3331 | 0.3383 |
| Calinski–Harabasz | 2872.26 | 741.30 |
| Davies–Bouldin | 1.0184 | 1.0219 |
| Minimum cluster share | 21.49% | 23.13% |
| Maximum cluster share | 29.33% | 27.60% |

Discovery subsample stability was ARI 0.9888 ± 0.0025. This result describes assignment agreement under the declared subsampling protocol. It does not establish causal or temporal permanence.

Cluster 0 contained 24.5% of customers and 72.3% of historical valid-purchase revenue. That concentration is descriptive and does not prove future value. The remaining personas were established mid-value customers, recent occasional customers, and long-recency customers.

## 7. Dashboard

The administration dashboard separates executive summary, data quality, segment exploration, masked customer lookup, experiment inspection, explanation, quality/drift context, and deployment metadata. Each page reads frozen aggregate artifacts. Raw transactions and credentials are not embedded.

## 8. Limitations

The dataset represents one retailer in 2010–2011. Missing identifiers remove many rows from customer segmentation. Cluster labels depend on the observation window and feature engineering. PCA is a two-dimensional diagnostic. The audit split tests customer generalization under the same observation window, not future production drift. Business actions remain hypotheses requiring controlled experiments.

## 9. Conclusion

The main contribution is not a claim that four natural customer types exist. It is a reproducible process for comparing segmentation choices while retaining failed and rejected evidence, isolating an audit population, and exposing methodological limitations in the dashboard.
