from __future__ import annotations

import pandas as pd
import pytest

from segmentforge.data import EXPECTED_COLUMNS, prepare_transactions, raw_quality_summary
from segmentforge.features import build_customer_features, deterministic_customer_split, feature_columns


def test_quality_audit_precedes_cleaning(raw_transactions):
    duplicate = pd.concat([raw_transactions, raw_transactions.iloc[[0]]], ignore_index=True)
    summary = raw_quality_summary(duplicate).set_index("check")["value"]
    assert int(summary["exact_duplicate_rows"]) == 1
    assert int(summary["rows"]) == len(raw_transactions) + 1


def test_declared_preparation_removes_duplicate(raw_transactions):
    duplicate = pd.concat([raw_transactions, raw_transactions.iloc[[0]]], ignore_index=True)
    identified, valid, exclusion = prepare_transactions(duplicate)
    assert len(identified) == len(raw_transactions)
    assert len(valid) == len(raw_transactions)
    assert int(exclusion.loc[exclusion.reason == "exact duplicate", "rows"].iloc[0]) == 1


def test_customer_features_are_nonnegative(raw_transactions):
    identified, valid, _ = prepare_transactions(raw_transactions)
    features = build_customer_features(identified, valid)
    columns = feature_columns("behavioral")
    assert len(features) == raw_transactions.CustomerID.nunique()
    assert (features[columns] >= 0).all().all()


def test_customer_identifier_is_never_a_feature():
    for view in ("rfm", "compact", "behavioral"):
        assert "CustomerID" not in feature_columns(view)
        assert "Country" not in feature_columns(view)


def test_snapshot_must_follow_observations(raw_transactions):
    identified, valid, _ = prepare_transactions(raw_transactions)
    with pytest.raises(ValueError, match="snapshot_date"):
        build_customer_features(identified, valid, snapshot_date=valid.InvoiceDate.max())


def test_deterministic_split_is_stable(raw_transactions):
    identified, valid, _ = prepare_transactions(raw_transactions)
    features = build_customer_features(identified, valid)
    first = deterministic_customer_split(features)
    second = deterministic_customer_split(features)
    assert first[0].CustomerID.tolist() == second[0].CustomerID.tolist()
    assert set(first[0].CustomerID).isdisjoint(set(first[1].CustomerID))
