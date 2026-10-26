"""Preprocessing helpers for the synthetic EcomPolice notebook workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RETURNED_STATUSES = {"Returned", "Refunded", "Exchange Issued"}


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
    orders["Payment_Method"] = orders["Payment_Method"].apply(lambda value: clean_category_text(value, payment_mapping))
    orders["Shipping_Method"] = orders["Shipping_Method"].apply(lambda value: clean_category_text(value, shipping_mapping))
    actions.append({"step": "categorical_normalization", "action": "Corrected unambiguous payment and shipping spelling variants."})

    duplicate_count = int(orders.duplicated().sum())
    orders = orders.drop_duplicates().copy()
    actions.append({"step": "duplicate_rows", "action": "Dropped exact duplicate order rows only.", "affected_rows": duplicate_count})

    for column in ["Order_Date"]:
        parsed = parse_mixed_dates(orders[column])
        failed = int((parsed.isna() & orders[column].notna()).sum())
        orders[column] = parsed
        actions.append({"step": f"parse_{column}", "action": "Parsed ISO and day-first date formats.", "affected_rows": failed})

    for column in ["Dispatch_Date", "Delivery_Date"]:
        parsed = parse_mixed_dates(deliveries[column])
        failed = int((parsed.isna() & deliveries[column].notna() & deliveries[column].astype(str).str.strip().ne("")).sum())
        deliveries[column] = parsed
        actions.append({"step": f"parse_{column}", "action": "Parsed delivery date values.", "affected_rows": failed})

    returns["Return_Date"] = parse_mixed_dates(returns["Return_Date"])
    customers["Signup_Date"] = parse_mixed_dates(customers["Signup_Date"])

    for column in ["Quantity", "Unit_Price_INR", "Discount_Pct", "Order_Value_INR"]:
        before_missing = orders[column].isna().sum()
        orders[column] = parse_numeric(orders[column])
        after_missing = orders[column].isna().sum()
        actions.append({"step": f"parse_{column}", "action": "Converted numeric-looking strings to numeric values.", "affected_rows": int(after_missing - before_missing)})
    for column in ["List_Price_INR"]:
        products[column] = parse_numeric(products[column])
    for column in ["Refund_Amount_INR"]:
        returns[column] = parse_numeric(returns[column])

    orders["Invalid_Identifier_Flag"] = (
        orders["Order_ID"].isna()
        | orders["Order_ID"].astype(str).str.strip().eq("")
        | orders["Customer_ID"].isna()
        | orders["Customer_ID"].astype(str).str.strip().eq("")
        | orders["Product_ID"].isna()
        | orders["Product_ID"].astype(str).str.strip().eq("")
    )
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
    actions.append({"step": "quality_flags", "action": "Flagged invalid identifiers, numeric defects, and suspicious outliers without blindly deleting them."})

    deliveries["Invalid_Delivery_Chronology_Flag"] = (
        deliveries["Dispatch_Date"].notna()
        & deliveries["Delivery_Date"].notna()
        & (deliveries["Delivery_Date"] < deliveries["Dispatch_Date"])
    )
    order_dates = orders.drop_duplicates("Order_ID").set_index("Order_ID")["Order_Date"]
    returns["Invalid_Return_Chronology_Flag"] = False
    known_return_orders = returns["Order_ID"].isin(order_dates.index)
    returns.loc[known_return_orders, "Invalid_Return_Chronology_Flag"] = (
        returns.loc[known_return_orders, "Return_Date"].notna()
        & (returns.loc[known_return_orders, "Return_Date"] < returns.loc[known_return_orders, "Order_ID"].map(order_dates))
    )
    actions.append({"step": "chronology_flags", "action": "Flagged impossible delivery and return chronology."})

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
    orders = tables.orders.drop_duplicates("Order_ID").copy()
    deliveries = tables.deliveries.drop_duplicates("Order_ID").copy()
    returns = tables.returns.drop_duplicates("Order_ID").copy()
    dataset = (
        orders.merge(tables.products, on="Product_ID", how="left", indicator="product_join_status")
        .merge(tables.customers, on="Customer_ID", how="left", indicator="customer_join_status")
        .merge(deliveries, on="Order_ID", how="left", indicator="delivery_join_status")
        .merge(returns, on="Order_ID", how="left", indicator="return_join_status")
    )
    dataset["Is_Returned"] = dataset["Return_Status"].isin(RETURNED_STATUSES).astype(int)
    dataset["Return_Duration_Days"] = (dataset["Return_Date"] - dataset["Delivery_Date"]).dt.days
    dataset["Order_Value_Recalculated_INR"] = (
        dataset["Quantity"] * dataset["Unit_Price_INR"] * (1 - dataset["Discount_Pct"] / 100)
    ).round(2)
    dataset["One_Row_Per_Order_Flag"] = ~dataset["Order_ID"].duplicated(keep=False)
    return dataset


def quality_summary(dataset: pd.DataFrame) -> pd.Series:
    """Return compact quality metrics for reporting."""
    return pd.Series(
        {
            "rows": len(dataset),
            "unique_orders": dataset["Order_ID"].nunique(dropna=True),
            "duplicate_order_ids": int(dataset["Order_ID"].duplicated().sum()),
            "invalid_identifiers": int(dataset.get("Invalid_Identifier_Flag", pd.Series(False, index=dataset.index)).sum()),
            "invalid_numeric_rows": int(dataset.get("Invalid_Numeric_Flag", pd.Series(False, index=dataset.index)).sum()),
            "suspicious_outlier_rows": int(dataset.get("Suspicious_Outlier_Flag", pd.Series(False, index=dataset.index)).sum()),
            "invalid_return_chronology": int(dataset.get("Invalid_Return_Chronology_Flag", pd.Series(False, index=dataset.index)).sum()),
            "missing_customer_joins": int(dataset["customer_join_status"].ne("both").sum()) if "customer_join_status" in dataset else 0,
            "missing_product_joins": int(dataset["product_join_status"].ne("both").sum()) if "product_join_status" in dataset else 0,
        }
    )


def build_model_targets(labels: pd.DataFrame) -> pd.DataFrame:
    """Create a separate target table aligned by stable order identifiers."""
    return labels[["Order_ID", "Customer_ID", "Scenario_Label", "Simulated_Abuse_Indicator"]].copy()
