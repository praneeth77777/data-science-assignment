"""Raw Online Retail loading, validation, and auditable preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = (
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Online Retail workbook not found: {path}. "
            "Follow data/raw/README.md to download it from UCI."
        )
    frame = pd.read_excel(path, engine="openpyxl")
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing={missing}; extra={extra}")
    frame = frame.loc[:, EXPECTED_COLUMNS].copy()
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="raise")
    if frame["InvoiceDate"].isna().any():
        raise ValueError("InvoiceDate contains missing values")
    return frame


def raw_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    invoice = frame["InvoiceNo"].astype(str)
    return pd.DataFrame([
        {"check": "rows", "value": len(frame)},
        {"check": "columns", "value": frame.shape[1]},
        {"check": "exact_duplicate_rows", "value": int(frame.duplicated().sum())},
        {"check": "missing_customer_id", "value": int(frame["CustomerID"].isna().sum())},
        {"check": "missing_description", "value": int(frame["Description"].isna().sum())},
        {"check": "cancellation_invoice_rows", "value": int(invoice.str.startswith("C").sum())},
        {"check": "nonpositive_quantity_rows", "value": int((frame["Quantity"] <= 0).sum())},
        {"check": "nonpositive_price_rows", "value": int((frame["UnitPrice"] <= 0).sum())},
        {"check": "unique_identified_customers", "value": int(frame["CustomerID"].nunique(dropna=True))},
        {"check": "minimum_invoice_date", "value": frame["InvoiceDate"].min().isoformat()},
        {"check": "maximum_invoice_date", "value": frame["InvoiceDate"].max().isoformat()},
    ])


def prepare_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return deduplicated identified rows, valid sales, and exclusion audit."""
    work = frame.copy()
    work["is_duplicate"] = work.duplicated(keep="first")
    work["is_cancellation"] = work["InvoiceNo"].astype(str).str.startswith("C")
    work["missing_customer"] = work["CustomerID"].isna()
    work["missing_description"] = work["Description"].isna()
    work["nonpositive_quantity"] = work["Quantity"] <= 0
    work["nonpositive_price"] = work["UnitPrice"] <= 0
    work["line_value"] = work["Quantity"] * work["UnitPrice"]

    reasons = {
        "exact duplicate": work["is_duplicate"],
        "missing CustomerID": work["missing_customer"],
        "cancellation invoice": work["is_cancellation"],
        "nonpositive quantity": work["nonpositive_quantity"],
        "nonpositive price": work["nonpositive_price"],
    }
    exclusion = pd.DataFrame([
        {"reason": name, "rows": int(mask.sum()), "share_of_raw": float(mask.mean())}
        for name, mask in reasons.items()
    ])

    identified = work.loc[~work["is_duplicate"] & ~work["missing_customer"]].copy()
    identified["CustomerID"] = identified["CustomerID"].astype("int64")
    valid = identified.loc[
        ~identified["is_cancellation"]
        & ~identified["nonpositive_quantity"]
        & ~identified["nonpositive_price"]
    ].copy()
    if valid.empty:
        raise ValueError("No valid purchase rows remain after declared rules")
    return identified, valid, exclusion
