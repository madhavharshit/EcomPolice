"""Validation checks for clean and feature synthetic datasets."""

from __future__ import annotations

import pandas as pd


def validate_clean_dataset(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    deliveries: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict[str, int | bool]:
    """Validate referential integrity and intended clean chronology."""
    order_dates = pd.to_datetime(orders["Order_Date"])
    signup_dates = pd.to_datetime(customers.set_index("Customer_ID").loc[orders["Customer_ID"], "Signup_Date"].values)
    dispatch_dates = pd.to_datetime(deliveries.set_index("Order_ID").loc[orders["Order_ID"], "Dispatch_Date"].values)
    delivery_dates = pd.to_datetime(deliveries.set_index("Order_ID").loc[orders["Order_ID"], "Delivery_Date"].replace("", pd.NA).values, errors="coerce")
    return_dates = pd.to_datetime(returns.set_index("Order_ID").loc[orders["Order_ID"], "Return_Date"].replace("", pd.NA).values, errors="coerce")
    user_order_counts = orders["Customer_ID"].value_counts()
    results = {
        "orders_have_valid_customers": bool(orders["Customer_ID"].isin(customers["Customer_ID"]).all()),
        "orders_have_valid_products": bool(orders["Product_ID"].isin(products["Product_ID"]).all()),
        "deliveries_have_valid_orders": bool(deliveries["Order_ID"].isin(orders["Order_ID"]).all()),
        "returns_have_valid_orders": bool(returns["Order_ID"].isin(orders["Order_ID"]).all()),
        "orders_before_signup_count": int((order_dates < signup_dates).sum()),
        "dispatch_before_order_count": int((dispatch_dates < order_dates).sum()),
        "delivery_before_dispatch_count": int((delivery_dates.notna() & (delivery_dates < dispatch_dates)).sum()),
        "return_before_delivery_count": int((return_dates.notna() & delivery_dates.notna() & (return_dates < delivery_dates)).sum()),
        "customers_with_multiple_orders": int((user_order_counts > 1).sum()),
        "max_orders_per_customer": int(user_order_counts.max()),
    }
    return results


def validate_feature_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, bool | int]:
    """Validate leakage-sensitive feature outputs."""
    label_columns = {"Scenario_Label", "Simulated_Abuse_Indicator"}
    post_return_columns = {"Return_Date", "Return_Reason", "Return_Status", "Refund_Amount_INR"}
    merged = features.merge(labels[["Order_ID", "Scenario_Label"]], on="Order_ID", how="left")
    chronological = features.sort_values(["Customer_ID", "Order_Date", "Order_ID"])
    return {
        "label_columns_excluded": bool(label_columns.isdisjoint(features.columns)),
        "post_return_columns_excluded": bool(post_return_columns.isdisjoint(features.columns)),
        "all_orders_have_labels": bool(merged["Scenario_Label"].notna().all()),
        "feature_rows": int(len(features)),
        "chronological_sort_check_rows": int(len(chronological)),
    }

