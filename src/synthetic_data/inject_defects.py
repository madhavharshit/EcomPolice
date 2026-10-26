"""Inject configurable data-quality defects into copies of clean synthetic tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synthetic_data.config import DefectRates


def _sample_indices(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> list[int]:
    count = max(0, int(round(len(df) * rate)))
    if count == 0 or df.empty:
        return []
    return list(rng.choice(df.index.to_numpy(), size=min(count, len(df)), replace=False))


def _record(
    manifest: list[dict[str, Any]],
    table: str,
    row_id: str,
    column: str,
    defect_type: str,
    original: Any,
    injected: Any,
) -> None:
    manifest.append(
        {
            "Table": table,
            "Row_ID": row_id,
            "Column": column,
            "Defect_Type": defect_type,
            "Original_Value": original,
            "Injected_Value": injected,
        }
    )


def inject_defects(
    tables: dict[str, pd.DataFrame],
    rates: DefectRates,
    rng: np.random.Generator,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return messy copies of clean tables plus a detailed defect manifest."""
    messy = {name: df.copy() for name, df in tables.items()}
    for column in ["Quantity", "Unit_Price_INR", "Discount_Pct"]:
        if column in messy["orders"].columns:
            messy["orders"][column] = messy["orders"][column].astype(object)
    manifest: list[dict[str, Any]] = []

    orders = messy["orders"]
    for idx in _sample_indices(orders, rates.missing_values, rng):
        col = str(rng.choice(["Payment_Method", "Shipping_Method", "Discount_Pct"]))
        original = orders.at[idx, col]
        orders.at[idx, col] = np.nan
        _record(manifest, "orders", orders.at[idx, "Order_ID"], col, "missing_value", original, np.nan)

    for idx in _sample_indices(orders, rates.inconsistent_categories, rng):
        col = str(rng.choice(["Payment_Method", "Shipping_Method"]))
        original = orders.at[idx, col]
        injected = str(original).lower().replace("-", " ")
        orders.at[idx, col] = injected
        _record(manifest, "orders", orders.at[idx, "Order_ID"], col, "inconsistent_category", original, injected)

    for idx in _sample_indices(orders, rates.mixed_date_formats, rng):
        original = orders.at[idx, "Order_Date"]
        injected = pd.Timestamp(original).strftime("%d/%m/%Y")
        orders.at[idx, "Order_Date"] = injected
        _record(manifest, "orders", orders.at[idx, "Order_ID"], "Order_Date", "mixed_date_format", original, injected)

    duplicate_count = int(round(len(orders) * rates.duplicate_rows))
    if duplicate_count:
        duplicates = orders.sample(n=duplicate_count, random_state=int(rng.integers(0, 1_000_000)))
        messy["orders"] = pd.concat([orders, duplicates], ignore_index=True)
        for _, row in duplicates.iterrows():
            _record(manifest, "orders", row["Order_ID"], "*", "duplicate_row", "<row>", "<duplicated row>")

    returns = messy["returns"]
    returned_mask = returns["Return_Date"].notna() & returns["Return_Date"].ne("")
    returned_indices = list(returns[returned_mask].index)
    invalid_count = min(len(returned_indices), int(round(len(returns) * rates.invalid_chronology)))
    for idx in list(rng.choice(returned_indices, size=invalid_count, replace=False)) if invalid_count else []:
        order_id = returns.at[idx, "Order_ID"]
        order_date = pd.Timestamp(messy["orders"].drop_duplicates("Order_ID").set_index("Order_ID").loc[order_id, "Order_Date"])
        original = returns.at[idx, "Return_Date"]
        injected = (order_date - pd.Timedelta(days=int(rng.integers(1, 15)))).date().isoformat()
        returns.at[idx, "Return_Date"] = injected
        _record(manifest, "returns", order_id, "Return_Date", "invalid_date_chronology", original, injected)

    for idx in _sample_indices(messy["orders"], rates.numeric_as_string, rng):
        col = str(rng.choice(["Quantity", "Unit_Price_INR", "Discount_Pct"]))
        original = messy["orders"].at[idx, col]
        injected = f"INR {original}" if col == "Unit_Price_INR" else str(original)
        messy["orders"].at[idx, col] = injected
        _record(manifest, "orders", messy["orders"].at[idx, "Order_ID"], col, "numeric_as_string", original, injected)

    for idx in _sample_indices(messy["orders"], rates.missing_identifiers, rng):
        col = str(rng.choice(["Customer_ID", "Product_ID"]))
        original = messy["orders"].at[idx, col]
        messy["orders"].at[idx, col] = ""
        _record(manifest, "orders", messy["orders"].at[idx, "Order_ID"], col, "missing_identifier", original, "")

    for idx in _sample_indices(messy["orders"], rates.impossible_negative_values, rng):
        col = str(rng.choice(["Quantity", "Unit_Price_INR"]))
        original = messy["orders"].at[idx, col]
        injected = -abs(float(original))
        messy["orders"].at[idx, col] = injected
        _record(manifest, "orders", messy["orders"].at[idx, "Order_ID"], col, "impossible_negative_value", original, injected)

    for idx in _sample_indices(messy["orders"], rates.implausible_outliers, rng):
        col = str(rng.choice(["Quantity", "Unit_Price_INR"]))
        original = messy["orders"].at[idx, col]
        injected = int(rng.integers(50, 200)) if col == "Quantity" else float(original) * 25
        messy["orders"].at[idx, col] = injected
        _record(manifest, "orders", messy["orders"].at[idx, "Order_ID"], col, "implausible_outlier", original, injected)

    return messy, pd.DataFrame(manifest)
