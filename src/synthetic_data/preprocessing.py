"""Preprocessing helpers for the synthetic EcomPolice notebook workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RETURNED_STATUSES = {"Returned", "Refunded", "Exchange Issued"}
ORDER_REQUIRED_COLUMNS = [
    "Order_ID",
    "Customer_ID",
    "Product_ID",
    "Order_Date",
    "Quantity",
    "Unit_Price_INR",
    "Discount_Pct",
    "Payment_Method",
    "Shipping_Method",
    "Order_Value_INR",
]


@dataclass(frozen=True)
class SyntheticTables:
    """Container for related synthetic e-commerce tables."""

    customers: pd.DataFrame
    products: pd.DataFrame
    orders: pd.DataFrame
    deliveries: pd.DataFrame
    returns: pd.DataFrame


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names without changing their meaning."""
    result = df.copy()
    result.columns = [column.strip() for column in result.columns]
    return result


def clean_category_text(value: object, mapping: dict[str, str]) -> object:
    """Normalize unambiguous category spelling and capitalization variants."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    key = text.lower().replace("-", " ")
    return mapping.get(key, text)


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse ISO and day-first date strings into pandas datetimes."""
    parsed = pd.to_datetime(series, errors="coerce")
    missing = parsed.isna() & series.notna() & series.astype(str).str.strip().ne("")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce", dayfirst=True)
    return parsed


def parse_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric values, including strings such as 'INR 499.0'."""
    cleaned = (
        series.astype(str)
        .str.replace("INR", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _blank_or_missing(series: pd.Series) -> pd.Series:
    """Identify null or whitespace-only values without changing source values."""
    return series.isna() | series.astype(str).str.strip().eq("")


def _choose_canonical_order_rows(orders: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep one defensible row per duplicated order ID and return discarded count.

    The generator's duplicate defect is intended to create repeated copies of an
    order. When a duplicate later receives another injected defect, the rows are
    no longer exactly equal. The record with fewer known quality problems is kept;
    ties keep the original input order. No values are merged or invented.
    """
    duplicate_mask = orders["Order_ID"].duplicated(keep=False) & orders["Order_ID"].notna()
    if not duplicate_mask.any():
        orders["Duplicate_Order_ID_Resolved_Flag"] = False
        return orders, 0

    result = orders.copy()
    result["_input_order"] = np.arange(len(result))
    result["_quality_score"] = (
        result["Invalid_Identifier_Flag"].astype(int)
        + result["Invalid_Numeric_Flag"].astype(int)
        + result["Missing_Required_Order_Field_Flag"].astype(int)
    )
    canonical_index = (
        result.loc[duplicate_mask]
        .sort_values(["Order_ID", "_quality_score", "_input_order"])
        .groupby("Order_ID", sort=False)
        .head(1)
        .index
    )
    keep_mask = ~duplicate_mask | result.index.isin(canonical_index)
    result["Duplicate_Order_ID_Resolved_Flag"] = duplicate_mask & result.index.isin(canonical_index)
    discarded = int((~keep_mask).sum())
    return result.loc[keep_mask].drop(columns=["_input_order", "_quality_score"]), discarded


def preprocess_messy_tables(tables: SyntheticTables) -> tuple[SyntheticTables, pd.DataFrame]:
    """Clean messy synthetic tables and return a quality-action log."""
    actions: list[dict[str, object]] = []
    customers = standardize_column_names(tables.customers)
    products = standardize_column_names(tables.products)
    orders = standardize_column_names(tables.orders)
    deliveries = standardize_column_names(tables.deliveries)
    returns = standardize_column_names(tables.returns)

    payment_mapping = {
        "upi": "UPI",
        "credit card": "Credit Card",
        "debit card": "Debit Card",
        "net banking": "Net Banking",
        "wallet": "Wallet",
        "cash on delivery": "Cash on Delivery",
    }
    shipping_mapping = {
        "standard": "Standard",
        "express": "Express",
        "same day": "Same-Day",
        "same-day": "Same-Day",
    }
    original_payment = orders["Payment_Method"].copy()
    original_shipping = orders["Shipping_Method"].copy()
    orders["Payment_Method"] = orders["Payment_Method"].apply(lambda value: clean_category_text(value, payment_mapping))
    orders["Shipping_Method"] = orders["Shipping_Method"].apply(lambda value: clean_category_text(value, shipping_mapping))
    category_corrections = int((orders["Payment_Method"] != original_payment).fillna(False).sum() + (orders["Shipping_Method"] != original_shipping).fillna(False).sum())
    actions.append({"step": "categorical_normalization", "action": "Corrected unambiguous payment and shipping spelling variants.", "affected_rows": category_corrections, "treatment": "corrected"})

    duplicate_count = int(orders.duplicated().sum())
    orders = orders.drop_duplicates().copy()
    actions.append({"step": "exact_duplicate_rows", "action": "Removed exact duplicate order rows only.", "affected_rows": duplicate_count, "treatment": "removed"})

    for column in ["Order_Date"]:
        parsed = parse_mixed_dates(orders[column])
        failed = int((parsed.isna() & orders[column].notna()).sum())
        orders[column] = parsed
        actions.append({"step": f"parse_{column}", "action": "Parsed ISO and day-first date formats.", "affected_rows": failed, "treatment": "flagged"})

    for column in ["Dispatch_Date", "Delivery_Date"]:
        parsed = parse_mixed_dates(deliveries[column])
        failed = int((parsed.isna() & deliveries[column].notna() & deliveries[column].astype(str).str.strip().ne("")).sum())
        deliveries[column] = parsed
        actions.append({"step": f"parse_{column}", "action": "Parsed delivery date values.", "affected_rows": failed, "treatment": "flagged"})

    returns["Return_Date"] = parse_mixed_dates(returns["Return_Date"])
    customers["Signup_Date"] = parse_mixed_dates(customers["Signup_Date"])
    for column in ["Order_ID", "Customer_ID", "Product_ID", "Payment_Method", "Shipping_Method"]:
        orders[column] = orders[column].replace(r"^\s*$", pd.NA, regex=True)
    # A blank return field for a non-returned order is structural missingness, not a zero value.
    returns["Return_Reason"] = returns["Return_Reason"].replace(r"^\s*$", pd.NA, regex=True)

    for column in ["Quantity", "Unit_Price_INR", "Discount_Pct", "Order_Value_INR"]:
        before_missing = orders[column].isna().sum()
        orders[column] = parse_numeric(orders[column])
        after_missing = orders[column].isna().sum()
        actions.append({"step": f"parse_{column}", "action": "Converted numeric-looking strings to numeric values.", "affected_rows": int(after_missing - before_missing), "treatment": "corrected"})
    for column in ["List_Price_INR"]:
        products[column] = parse_numeric(products[column])
    for column in ["Refund_Amount_INR"]:
        returns[column] = parse_numeric(returns[column])

    orders["Invalid_Identifier_Flag"] = _blank_or_missing(orders["Order_ID"]) | _blank_or_missing(orders["Customer_ID"]) | _blank_or_missing(orders["Product_ID"])
    orders["Missing_Required_Order_Field_Flag"] = pd.concat(
        [_blank_or_missing(orders[column]).rename(column) for column in ORDER_REQUIRED_COLUMNS], axis=1
    ).any(axis=1)
    orders["Invalid_Numeric_Flag"] = (
        orders["Quantity"].isna()
        | orders["Unit_Price_INR"].isna()
        | (orders["Quantity"] <= 0)
        | (orders["Unit_Price_INR"] <= 0)
        | (orders["Discount_Pct"] < 0)
        | (orders["Discount_Pct"] > 90)
    )
    orders["Suspicious_Outlier_Flag"] = (
        (orders["Quantity"] > 20)
        | (orders["Unit_Price_INR"] > products["List_Price_INR"].quantile(0.99) * 10)
    )
    orders, non_exact_duplicate_count = _choose_canonical_order_rows(orders)
    actions.extend(
        [
            {"step": "non_exact_duplicate_order_ids", "action": "Kept the least-problematic row for duplicate order IDs after exact duplicates were removed; no values were merged.", "affected_rows": non_exact_duplicate_count, "treatment": "removed"},
            {"step": "quality_flags", "action": "Retained invalid identifiers, missing required fields, invalid numeric values, and suspicious outliers with explicit flags.", "affected_rows": int((orders["Invalid_Identifier_Flag"] | orders["Invalid_Numeric_Flag"] | orders["Missing_Required_Order_Field_Flag"] | orders["Suspicious_Outlier_Flag"]).sum()), "treatment": "flagged"},
        ]
    )

    deliveries["Invalid_Delivery_Chronology_Flag"] = (
        deliveries["Dispatch_Date"].notna()
        & deliveries["Delivery_Date"].notna()
        & (deliveries["Delivery_Date"] < deliveries["Dispatch_Date"])
    )
    order_dates = orders.set_index("Order_ID")["Order_Date"]
    delivery_dates = deliveries.drop_duplicates("Order_ID").set_index("Order_ID")["Delivery_Date"]
    returns["Invalid_Return_Chronology_Flag"] = False
    known_return_orders = returns["Order_ID"].isin(order_dates.index)
    returns.loc[known_return_orders, "Invalid_Return_Chronology_Flag"] = (
        returns.loc[known_return_orders, "Return_Date"].notna()
        & (returns.loc[known_return_orders, "Return_Date"] < returns.loc[known_return_orders, "Order_ID"].map(delivery_dates))
    )
    actions.append({"step": "chronology_flags", "action": "Retained impossible delivery and return chronology with explicit flags.", "affected_rows": int(deliveries["Invalid_Delivery_Chronology_Flag"].sum() + returns["Invalid_Return_Chronology_Flag"].sum()), "treatment": "flagged"})

    cleaned = SyntheticTables(
        customers=customers,
        products=products,
        orders=orders,
        deliveries=deliveries,
        returns=returns,
    )
    return cleaned, pd.DataFrame(actions)


def build_clean_order_dataset(tables: SyntheticTables) -> pd.DataFrame:
    """Merge cleaned synthetic tables into one row per order."""
    orders = tables.orders.copy()
    deliveries = tables.deliveries.drop_duplicates("Order_ID").copy()
    returns = tables.returns.drop_duplicates("Order_ID").copy()
    if orders["Order_ID"].duplicated().any():
        raise ValueError("Orders must be deduplicated before building an order-level dataset.")
    if tables.customers["Customer_ID"].duplicated().any() or tables.products["Product_ID"].duplicated().any():
        raise ValueError("Customer and product reference keys must be unique for a many-to-one order join.")
    dataset = (
        orders.merge(tables.products, on="Product_ID", how="left", indicator="product_join_status", validate="m:1")
        .merge(tables.customers, on="Customer_ID", how="left", indicator="customer_join_status", validate="m:1")
        .merge(deliveries, on="Order_ID", how="left", indicator="delivery_join_status", validate="1:1")
        .merge(returns, on="Order_ID", how="left", indicator="return_join_status", validate="1:1")
    )
    dataset["Unmatched_Product_Flag"] = dataset["product_join_status"].ne("both")
    dataset["Unmatched_Customer_Flag"] = dataset["customer_join_status"].ne("both")
    dataset["Is_Returned"] = dataset["Return_Status"].isin(RETURNED_STATUSES).astype(int)
    dataset["Return_Duration_Days"] = (dataset["Return_Date"] - dataset["Delivery_Date"]).dt.days
    dataset["Order_Value_Recalculated_INR"] = (
        dataset["Quantity"] * dataset["Unit_Price_INR"] * (1 - dataset["Discount_Pct"] / 100)
    ).round(2)
    dataset["One_Row_Per_Order_Flag"] = ~dataset["Order_ID"].duplicated(keep=False)
    dataset["Any_Quality_Issue_Flag"] = (
        dataset["Invalid_Identifier_Flag"]
        | dataset["Missing_Required_Order_Field_Flag"]
        | dataset["Invalid_Numeric_Flag"]
        | dataset["Suspicious_Outlier_Flag"]
        | dataset["Unmatched_Product_Flag"]
        | dataset["Unmatched_Customer_Flag"]
        | dataset["Invalid_Delivery_Chronology_Flag"].fillna(False)
        | dataset["Invalid_Return_Chronology_Flag"].fillna(False)
    )
    return dataset


def quality_summary(dataset: pd.DataFrame) -> pd.Series:
    """Return compact quality metrics for reporting."""
    return pd.Series(
        {
            "rows": len(dataset),
            "unique_orders": dataset["Order_ID"].nunique(dropna=True),
            "duplicate_order_ids": int(dataset["Order_ID"].duplicated().sum()),
            "invalid_identifiers": int(dataset.get("Invalid_Identifier_Flag", pd.Series(False, index=dataset.index)).sum()),
            "missing_required_order_fields": int(dataset.get("Missing_Required_Order_Field_Flag", pd.Series(False, index=dataset.index)).sum()),
            "invalid_numeric_rows": int(dataset.get("Invalid_Numeric_Flag", pd.Series(False, index=dataset.index)).sum()),
            "suspicious_outlier_rows": int(dataset.get("Suspicious_Outlier_Flag", pd.Series(False, index=dataset.index)).sum()),
            "invalid_delivery_chronology": int(dataset.get("Invalid_Delivery_Chronology_Flag", pd.Series(False, index=dataset.index).fillna(False)).sum()),
            "invalid_return_chronology": int(dataset.get("Invalid_Return_Chronology_Flag", pd.Series(False, index=dataset.index)).sum()),
            "missing_customer_joins": int(dataset["customer_join_status"].ne("both").sum()) if "customer_join_status" in dataset else 0,
            "missing_product_joins": int(dataset["product_join_status"].ne("both").sum()) if "product_join_status" in dataset else 0,
        }
    )


def missing_value_audit(
    messy_orders: pd.DataFrame,
    cleaned_orders: pd.DataFrame,
    consolidated_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Compare missingness across source orders, cleaned orders, and final joins."""
    columns = [
        "Order_ID", "Customer_ID", "Product_ID", "Order_Date", "Quantity", "Unit_Price_INR",
        "Discount_Pct", "Payment_Method", "Shipping_Method", "Order_Value_INR", "Full_Name",
        "City", "State_Region", "Signup_Date", "Age_Band", "Gender", "Loyalty_Tier", "Category",
        "Brand", "Product_Name", "List_Price_INR",
    ]
    rows: list[dict[str, object]] = []
    for column in columns:
        source_available = column in messy_orders
        cleaned_available = column in cleaned_orders
        source = messy_orders[column] if source_available else pd.Series(index=messy_orders.index, dtype="object")
        cleaned = cleaned_orders[column] if cleaned_available else pd.Series(index=cleaned_orders.index, dtype="object")
        final = consolidated_orders[column] if column in consolidated_orders else pd.Series(index=consolidated_orders.index, dtype="object")
        source_missing = int(_blank_or_missing(source).sum())
        cleaned_missing = int(_blank_or_missing(cleaned).sum())
        final_missing = int(_blank_or_missing(final).sum())
        rows.append(
            {
                "column": column,
                "messy_missing_count": source_missing if source_available else pd.NA,
                "messy_missing_pct": round(source_missing / len(messy_orders) * 100, 3) if source_available else pd.NA,
                "cleaned_orders_missing_count": cleaned_missing if cleaned_available else pd.NA,
                "final_missing_count": final_missing,
                "final_missing_pct": round(final_missing / len(consolidated_orders) * 100, 3),
                "change_messy_to_final": final_missing - source_missing if source_available else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def build_model_targets(labels: pd.DataFrame) -> pd.DataFrame:
    """Create a separate target table aligned by stable order identifiers."""
    return labels[["Order_ID", "Customer_ID", "Scenario_Label", "Simulated_Abuse_Indicator"]].copy()
