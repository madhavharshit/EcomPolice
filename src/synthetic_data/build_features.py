"""Leakage-safe feature construction for synthetic EcomPolice data."""

from __future__ import annotations

import pandas as pd


RETURNED_STATUSES = {"Returned", "Refunded", "Exchange Issued"}
HIGH_VALUE_ORDER_THRESHOLD_INR = 10_000.0


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


def build_return_level_features(order_table: pd.DataFrame) -> pd.DataFrame:
    """Build one leakage-aware row per return request.

    The prediction point is ``Return_Date``: current return-request details are
    available, while histories use only earlier orders and earlier return requests.
    ``HIGH_VALUE_ORDER_THRESHOLD_INR`` is a fixed policy threshold, avoiding a
    dataset-wide quantile that could use future observations.
    """
    working = order_table.copy()
    working["Order_Date"] = pd.to_datetime(working["Order_Date"])
    working["Delivery_Date"] = pd.to_datetime(working["Delivery_Date"], errors="coerce")
    working["Return_Date"] = pd.to_datetime(working["Return_Date"], errors="coerce")
    working["Is_High_Value_Order"] = working["Order_Value_INR"].ge(HIGH_VALUE_ORDER_THRESHOLD_INR)
    working["Is_Defect_Claim"] = working["Return_Reason"].eq("Defective")
    return_requests = working.loc[
        working["Return_Status"].isin(RETURNED_STATUSES) & working["Return_Date"].notna()
    ].copy()

    rows: list[dict[str, object]] = []
    for customer_id, current_returns in return_requests.groupby("Customer_ID", sort=False):
        customer_orders = working.loc[working["Customer_ID"].eq(customer_id)].copy()
        customer_returns = current_returns.sort_values(["Return_Date", "Order_ID"])
        for _, current in customer_returns.iterrows():
            request_date = current["Return_Date"]
            prior_orders = customer_orders.loc[
                customer_orders["Order_Date"].lt(request_date) & customer_orders["Order_ID"].ne(current["Order_ID"])
            ]
            prior_returns = customer_returns.loc[customer_returns["Return_Date"].lt(request_date)]
            prior_return_count = len(prior_returns)
            prior_order_count = len(prior_orders)
            prior_defect_count = int(prior_returns["Is_Defect_Claim"].sum())
            rows.append(
                {
                    "Order_ID": current["Order_ID"],
                    "Customer_ID": customer_id,
                    "Return_Request_Date": request_date,
                    "Order_Value_INR": current["Order_Value_INR"],
                    "Product_Category": current["Category"],
                    "Payment_Method": current["Payment_Method"],
                    "Shipping_Method": current["Shipping_Method"],
                    "Return_Reason": current["Return_Reason"],
                    "Days_Since_Delivery": (request_date - current["Delivery_Date"]).days if pd.notna(current["Delivery_Date"]) else pd.NA,
                    "Prior_Order_Count": prior_order_count,
                    "Prior_Return_Count": prior_return_count,
                    "Historical_Return_Rate": prior_return_count / prior_order_count if prior_order_count else 0.0,
                    "Recent_Return_Count_30D": int(prior_returns["Return_Date"].ge(request_date - pd.Timedelta(days=30)).sum()),
                    "Recent_Return_Count_90D": int(prior_returns["Return_Date"].ge(request_date - pd.Timedelta(days=90)).sum()),
                    "High_Value_Return_Count": int(prior_returns["Is_High_Value_Order"].sum()),
                    "Repeated_Defect_Claim_Rate": prior_defect_count / prior_return_count if prior_return_count else 0.0,
                    "Average_Days_To_Return": float(prior_returns["Return_Duration_Days"].mean()) if prior_return_count else 0.0,
                    "Has_Prior_Order_History": int(prior_order_count > 0),
                    "Has_Prior_Return_History": int(prior_return_count > 0),
                }
            )
    return pd.DataFrame(rows).sort_values(["Return_Request_Date", "Order_ID"]).reset_index(drop=True)


def build_customer_level_features(
    order_table: pd.DataFrame,
    observation_date: pd.Timestamp | None = None,
    customer_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Create one customer-level analytical row at a fixed observation date."""
    working = order_table.copy()
    working["Order_Date"] = pd.to_datetime(working["Order_Date"])
    working["Return_Date"] = pd.to_datetime(working["Return_Date"], errors="coerce")
    working["Is_High_Value_Order"] = working["Order_Value_INR"].ge(HIGH_VALUE_ORDER_THRESHOLD_INR)
    working["Is_Defect_Claim"] = working["Return_Reason"].eq("Defective")
    if observation_date is None:
        observation_date = max(working["Order_Date"].max(), working["Return_Date"].max())
    observation_date = pd.Timestamp(observation_date)
    rows: list[dict[str, object]] = []
    for customer_id, group in working.groupby("Customer_ID", sort=False):
        prior_orders = group.loc[group["Order_Date"].le(observation_date)]
        prior_returns = prior_orders.loc[prior_orders["Return_Date"].notna() & prior_orders["Return_Date"].le(observation_date)]
        return_count = len(prior_returns)
        rows.append(
            {
                "Customer_ID": customer_id,
                "Observation_Date": observation_date,
                "Prior_Order_Count": len(prior_orders),
                "Prior_Return_Count": return_count,
                "Historical_Return_Rate": return_count / len(prior_orders) if len(prior_orders) else 0.0,
                "Recent_Return_Count_30D": int(prior_returns["Return_Date"].ge(observation_date - pd.Timedelta(days=30)).sum()),
                "Recent_Return_Count_90D": int(prior_returns["Return_Date"].ge(observation_date - pd.Timedelta(days=90)).sum()),
                "High_Value_Return_Count": int(prior_returns["Is_High_Value_Order"].sum()),
                "Repeated_Defect_Claim_Rate": float(prior_returns["Is_Defect_Claim"].mean()) if return_count else 0.0,
                "Average_Days_To_Return": float(prior_returns["Return_Duration_Days"].mean()) if return_count else 0.0,
                "Has_Prior_Return_History": int(return_count > 0),
            }
        )
    features = pd.DataFrame(rows)
    if customer_ids is not None:
        all_customers = pd.DataFrame({"Customer_ID": pd.Series(customer_ids).drop_duplicates()})
        features = all_customers.merge(features, on="Customer_ID", how="left", validate="1:1")
        features["Observation_Date"] = features["Observation_Date"].fillna(observation_date)
        numeric_columns = [column for column in features.columns if column not in {"Customer_ID", "Observation_Date"}]
        features[numeric_columns] = features[numeric_columns].fillna(0)
    return features.sort_values("Customer_ID").reset_index(drop=True)


def build_return_level_targets(order_labels: pd.DataFrame, return_features: pd.DataFrame) -> pd.DataFrame:
    """Align generator-provided order labels to return requests without using them as features."""
    labels = order_labels[["Order_ID", "Customer_ID", "Scenario_Label", "Simulated_Abuse_Indicator"]].copy()
    targets = return_features[["Order_ID", "Customer_ID", "Return_Request_Date"]].merge(
        labels, on=["Order_ID", "Customer_ID"], how="left", validate="1:1"
    )
    if targets["Scenario_Label"].isna().any():
        raise ValueError("Every return request must have a matching generator label.")
    targets = targets.rename(columns={"Simulated_Abuse_Indicator": "Simulated_Abuse_Target"})
    targets["Target_Construction"] = "Generator order label inherited from the customer scenario; selected only for returned orders."
    return targets


def build_customer_level_targets(customer_labels: pd.DataFrame, customer_features: pd.DataFrame) -> pd.DataFrame:
    """Align customer-level generator labels to customer-level analytical rows."""
    targets = customer_features[["Customer_ID", "Observation_Date"]].merge(
        customer_labels[["Customer_ID", "Scenario_Label", "Simulated_Abuse_Indicator"]],
        on="Customer_ID",
        how="left",
        validate="1:1",
    )
    if targets["Scenario_Label"].isna().any():
        raise ValueError("Every customer feature row must have a matching generator label.")
    targets = targets.rename(columns={"Simulated_Abuse_Indicator": "Simulated_Abuse_Target"})
    targets["Target_Construction"] = "Generator customer-level scenario label at the documented observation date."
    return targets


def validate_feature_outputs(
    order_table: pd.DataFrame,
    return_features: pd.DataFrame,
    return_targets: pd.DataFrame,
    customer_features: pd.DataFrame,
    customer_targets: pd.DataFrame,
) -> dict[str, int | bool]:
    """Validate feature shape, label separation, alignment, and history direction."""
    forbidden = {"Scenario_Label", "Simulated_Abuse_Indicator", "Simulated_Abuse_Target", "Refund_Amount_INR", "Return_Status"}
    return_ids = set(order_table.loc[order_table["Return_Status"].isin(RETURNED_STATUSES), "Order_ID"])
    chronological_violations = 0
    for _, row in return_features.iterrows():
        prior_returns = order_table.loc[
            order_table["Customer_ID"].eq(row["Customer_ID"])
            & pd.to_datetime(order_table["Return_Date"], errors="coerce").lt(row["Return_Request_Date"])
        ]
        if int(row["Prior_Return_Count"]) != len(prior_returns):
            chronological_violations += 1
    return {
        "return_features_one_row_per_return": bool(return_features["Order_ID"].is_unique),
        "all_return_requests_represented": set(return_features["Order_ID"]) == return_ids,
        "customer_features_one_row_per_customer": bool(customer_features["Customer_ID"].is_unique),
        "no_target_columns_in_return_features": bool(forbidden.isdisjoint(return_features.columns)),
        "return_feature_target_alignment": set(return_features["Order_ID"]) == set(return_targets["Order_ID"]),
        "customer_feature_target_alignment": set(customer_features["Customer_ID"]) == set(customer_targets["Customer_ID"]),
        "chronological_feature_violations": chronological_violations,
    }

