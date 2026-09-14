from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    rows = []
    for customer in range(10001, 10121):
        group = (customer - 10001) // 30
        for order in range(1, group + 2):
            invoice = str(500000 + customer * 10 + order)
            rows.append({
                "InvoiceNo": invoice,
                "StockCode": f"SKU-{order % 7}",
                "Description": "Test product",
                "Quantity": 1 + group * 2,
                "InvoiceDate": pd.Timestamp("2011-01-01") + pd.Timedelta(days=group * 60 + order),
                "UnitPrice": 2.0 + group * 4,
                "CustomerID": float(customer),
                "Country": "United Kingdom",
            })
    return pd.DataFrame(rows)
