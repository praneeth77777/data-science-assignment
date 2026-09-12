import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "artifacts" / "tables"
FIGURES = PROJECT_ROOT / "artifacts" / "figures"


class ArtifactContractTests(unittest.TestCase):
    def test_final_model_beats_seasonal_baseline(self):
        metrics = pd.read_csv(TABLES / "final_locked_test_metrics.csv").set_index("candidate")
        final = metrics.loc["rf_raw_depth18_leaf1"]
        seasonal = metrics.loc["seasonal_naive"]
        self.assertLess(final["rmsle"], seasonal["rmsle"])
        self.assertLess(final["mae"], seasonal["mae"])

    def test_audit_is_complete_and_passing(self):
        audit = pd.read_csv(TABLES / "final_leakage_reproducibility_audit.csv")
        self.assertEqual(len(audit), 6)
        self.assertTrue(audit["status"].eq("PASS").all())

    def test_dashboard_assets_exist(self):
        required = {
            "temporal_demand_patterns.png",
            "cluster_pca_projection.png",
            "model_family_comparison.png",
            "performance_slices.png",
            "permutation_importance.png",
        }
        self.assertTrue(required.issubset({path.name for path in FIGURES.glob("*.png")}))


if __name__ == "__main__":
    unittest.main()
