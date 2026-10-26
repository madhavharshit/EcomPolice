"""Leakage-safe feature construction for synthetic EcomPolice data."""

from __future__ import annotations

import pandas as pd


RETURNED_STATUSES = {"Returned", "Refunded", "Exchange Issued"}


def build_order_level_table(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    deliveries: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Join clean entity tables into an analytical order-level table."""
    table = (
        orders.merge(products, on="Product_ID", how="left")
        .merge(customers, on="Customer_ID", how="left")
        .merge(deliveries, on="Order_ID", how="left")
        .merge(returns, on="Order_ID", how="left")
    )
    table["Order_Date"] = pd.to_datetime(table["Order_Date"])
    table["Return_Date"] = pd.to_datetime(table["Return_Date"], errors="coerce")
    table["Delivery_Date"] = pd.to_datetime(table["Delivery_Date"], errors="coerce")
    table["Is_Returned"] = table["Return_Status"].isin(RETURNED_STATUSES).astype(int)
    table["Return_Duration_Days"] = (table["Return_Date"] - table["Delivery_Date"]).dt.days
    table["Is_High_Value_Order"] = table["Order_Value_INR"] >= table["Order_Value_INR"].quantile(0.80)
    table["Is_Defect_Claim"] = table["Return_Reason"].eq("Defective")
    return table.sort_values(["Customer_ID", "Order_Date", "Order_ID"]).reset_index(drop=True)


def build_pre_return_features(order_table: pd.DataFrame) -> pd.DataFrame:
    """Build features using only customer history before the current order."""
    df = order_table.sort_values(["Customer_ID", "Order_Date", "Order_ID"]).copy()
    rows = []
    for _, group in df.groupby("Customer_ID", sort=False):
        history = group.copy()
        prior_order_count = []
        prior_return_count = []
        prior_return_rate = []
        returns_30 = []
        returns_90 = []
        prior_high_value_returns = []
        prior_defect_claims = []
        prior_avg_return_duration = []
        for idx, row in history.iterrows():
            prior = history[history["Order_Date"] < row["Order_Date"]]
            if prior.empty:
                prior_order_count.append(0)
                prior_return_count.append(0)
                prior_return_rate.append(0.0)
                returns_30.append(0)
                returns_90.append(0)
                prior_high_value_returns.append(0)
                prior_defect_claims.append(0)
                prior_avg_return_duration.append(float("nan"))
                continue
            returned_prior = prior[prior["Is_Returned"].eq(1)]
            prior_order_count.append(len(prior))
            prior_return_count.append(int(returned_prior["Is_Returned"].sum()))
            prior_return_rate.append(float(returned_prior["Is_Returned"].sum() / len(prior)))
            returns_30.append(int(((returned_prior["Return_Date"] < row["Order_Date"]) & (returned_prior["Return_Date"] >= row["Order_Date"] - pd.Timedelta(days=30))).sum()))
            returns_90.append(int(((returned_prior["Return_Date"] < row["Order_Date"]) & (returned_prior["Return_Date"] >= row["Order_Date"] - pd.Timedelta(days=90))).sum()))
            prior_high_value_returns.append(int((returned_prior["Is_High_Value_Order"]).sum()))
            prior_defect_claims.append(int((returned_prior["Is_Defect_Claim"]).sum()))
            prior_avg_return_duration.append(float(returned_prior["Return_Duration_Days"].dropna().mean()) if not returned_prior["Return_Duration_Days"].dropna().empty else float("nan"))
        history["Customer_Order_Count_Before"] = prior_order_count
        history["Prior_Return_Count"] = prior_return_count
        history["Historical_Return_Rate"] = prior_return_rate
        history["Returns_Previous_30_Days"] = returns_30
        history["Returns_Previous_90_Days"] = returns_90
        history["Previous_High_Value_Return_Count"] = prior_high_value_returns
        history["Historical_Defective_Claim_Count"] = prior_defect_claims
        history["Historical_Avg_Return_Duration_Days"] = prior_avg_return_duration
        rows.append(history)

    features = pd.concat(rows, ignore_index=True).sort_values(["Order_Date", "Order_ID"])
    allowed_columns = [
        "Order_ID",
        "Customer_ID",
        "Product_ID",
        "Order_Date",
        "Quantity",
        "Unit_Price_INR",
        "Discount_Pct",
        "Order_Value_INR",
        "Payment_Method",
        "Shipping_Method",
        "Category",
        "Brand",
        "City",
        "State_Region",
        "Age_Band",
        "Gender",
        "Loyalty_Tier",
        "Customer_Order_Count_Before",
        "Prior_Return_Count",
        "Historical_Return_Rate",
        "Returns_Previous_30_Days",
        "Returns_Previous_90_Days",
        "Previous_High_Value_Return_Count",
        "Historical_Defective_Claim_Count",
        "Historical_Avg_Return_Duration_Days",
    ]
    return features[allowed_columns].reset_index(drop=True)

