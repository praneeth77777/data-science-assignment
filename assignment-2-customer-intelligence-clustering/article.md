# From Transactions to Defensible Customer Segments

Customer clustering tutorials often end with a colorful scatterplot and memorable labels. SegmentForge began with a different question: how can I tell whether a segmentation is stable, operationally usable, and honest about uncertainty?

The project uses 541,909 transaction lines from the UCI Online Retail dataset. Before cleaning, I recorded missing customer identifiers, cancellations, nonpositive quantities and prices, and exact duplicates. That order matters because cleaning first would erase evidence about the source data.

I aggregated valid purchases into customer-level Recency, Frequency, and Monetary features. CustomerID was used only to group transactions and was prohibited from the clustering matrix. I also created richer behavioral views, but they had to earn their place through evaluation rather than being assumed superior.

The experiment compared K-Means, Gaussian Mixture, and Agglomerative clustering. It also varied cluster count, feature view, transformation, scaling, and outlier policy. Instead of running an uncontrolled grid and selecting the prettiest result, I created a bounded hill-climbing experiment manager. Each trial changed one decision, logged its parent, preserved errors and rejections, and respected a fixed 42-trial budget.

The selected configuration was K-Means with four clusters, RFM features, a Yeo–Johnson transformation, StandardScaler, and no removal of extreme customers. Its silhouette score was 0.3331 on discovery customers and 0.3383 on a locked customer audit group. Subsample stability reached adjusted Rand index 0.9888 ± 0.0025.

Those values are not classification accuracy. Clustering has no supplied correct customer label. I therefore displayed silhouette together with Calinski–Harabasz, Davies–Bouldin, cluster shares, stability, runtime, and interpretability checks.

The four profiles became high-value frequent, established mid-value, recent occasional, and long-recency customers. The highest-value segment accounted for 72.3% of historical valid-purchase revenue, but I did not call that lifetime value or expected campaign return. Those conclusions require future data and experiments.

The Streamlit dashboard makes the scientific process visible. Business users can explore aggregate segment profiles, while data scientists can inspect the complete experiment leaderboard, rejected trials, stability, data-quality evidence, and model metadata. Customer lookup uses masked identifiers, and the drift page explicitly says that production drift is not yet measured.

The main lesson was that customer segmentation is a model-governance problem as much as a clustering problem. A useful solution needs traceable data rules, several candidate algorithms, stability analysis, a locked audit, reproducible artifacts, and clear language about what the clusters cannot prove.
