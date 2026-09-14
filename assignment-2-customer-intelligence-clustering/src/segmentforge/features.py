"""Customer-level behavioral feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VIEWS = {
    "rfm": ["recency_days", "frequency", "monetary"],
    "compact": ["recency_days", "frequency", "monetary", "product_breadth", "cancellation_rate"],
    "behavioral": [
        "recency_days", "frequency", "monetary", "average_order_value",
        "product_breadth", "total_units", "average_basket_units",
        "tenure_days", "cancellation_rate",
    ],
}


def build_customer_features(
    identified: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = valid["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    if snapshot_date <= valid["InvoiceDate"].max():
        raise ValueError("snapshot_date must be after the latest included purchase")

    invoice = valid.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(
        order_value=("line_value", "sum"),
        order_units=("Quantity", "sum"),
        order_date=("InvoiceDate", "max"),
    )
    customer = valid.groupby("CustomerID", as_index=True).agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("line_value", "sum"),
        product_breadth=("StockCode", "nunique"),
        total_units=("Quantity", "sum"),
        primary_country=("Country", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    )
    order_stats = invoice.groupby("CustomerID").agg(
        average_order_value=("order_value", "mean"),
        average_basket_units=("order_units", "mean"),
    )
    cancellation = identified.groupby("CustomerID")["is_cancellation"].mean().rename("cancellation_rate")
    customer = customer.join(order_stats).join(cancellation, how="left")
    customer["recency_days"] = (snapshot_date - customer["last_purchase"].dt.normalize()).dt.days
    customer["tenure_days"] = (
        customer["last_purchase"].dt.normalize() - customer["first_purchase"].dt.normalize()
    ).dt.days
    customer["cancellation_rate"] = customer["cancellation_rate"].fillna(0.0)
    customer["snapshot_date"] = snapshot_date
    customer.index = customer.index.astype("int64")
    customer.index.name = "CustomerID"

    numeric = sorted({f for fields in FEATURE_VIEWS.values() for f in fields})
    if not np.isfinite(customer[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Customer features contain non-finite values")
    if (customer[numeric] < 0).any().any():
        raise ValueError("Customer features contain unexpected negative values")
    return customer.reset_index()


def deterministic_customer_split(
    customer: pd.DataFrame,
    *,
    audit_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < audit_fraction < 1:
        raise ValueError("audit_fraction must be between 0 and 1")
    hashed = pd.util.hash_pandas_object(
        customer["CustomerID"].astype(str) + f"-{seed}", index=False
    ).to_numpy(dtype="uint64")
    threshold = int(audit_fraction * 10_000)
    audit_mask = (hashed % 10_000) < threshold
    discovery = customer.loc[~audit_mask].sort_values("CustomerID").reset_index(drop=True)
    audit = customer.loc[audit_mask].sort_values("CustomerID").reset_index(drop=True)
    if discovery.empty or audit.empty:
        raise ValueError("Deterministic split produced an empty partition")
    return discovery, audit


def feature_columns(view: str) -> list[str]:
    if view not in FEATURE_VIEWS:
        raise ValueError(f"Unknown feature view: {view}")
    return list(FEATURE_VIEWS[view])
