"""Generate order and delivery tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_data.common import PAYMENT_METHODS, SHIPPING_METHODS, parse_date
from synthetic_data.config import SyntheticConfig


def generate_orders(
    config: SyntheticConfig,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    labels: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate orders with repeated customers and scenario-aware product choices."""
    start = parse_date(config.period_start)
    end = start + pd.DateOffset(months=config.months) - pd.Timedelta(days=1)
    scenario_by_customer = labels.set_index("Customer_ID")["Scenario_Label"]
    signup_by_customer = pd.to_datetime(customers.set_index("Customer_ID")["Signup_Date"])

    activity_multiplier = labels["Scenario_Label"].map(
        {
            "NORMAL": 1.0,
            "FREQUENT_LEGITIMATE_RETURNS": 1.45,
            "REPEATED_DEFECT_CLAIMS": 1.25,
            "HIGH_VALUE_RETURN_PATTERN": 1.15,
            "RETURN_WINDOW_PATTERN": 1.20,
            "MIXED_SUSPICIOUS_PATTERN": 1.55,
        }
    )
    weights = activity_multiplier / activity_multiplier.sum()
    selected_customers = rng.choice(labels["Customer_ID"], size=config.n_orders, p=weights)

    high_value_products = products[products["List_Price_INR"] >= products["List_Price_INR"].quantile(0.75)]
    rows = []
    for i, customer_id in enumerate(selected_customers):
        scenario = str(scenario_by_customer.loc[customer_id])
        signup_date = signup_by_customer.loc[customer_id]
        earliest = max(start, signup_date)
        if scenario in {"HIGH_VALUE_RETURN_PATTERN", "MIXED_SUSPICIOUS_PATTERN"} and rng.random() < 0.50:
            product = high_value_products.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        else:
            product = products.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        order_span = max((end - earliest).days, 0)
        order_date = earliest + pd.Timedelta(days=int(rng.integers(0, order_span + 1)))
        quantity = int(rng.choice([1, 1, 1, 2, 2, 3, 4], p=[0.30, 0.22, 0.18, 0.13, 0.08, 0.06, 0.03]))
        discount_pct = float(np.round(np.clip(rng.normal(8, 7), 0, 35), 2))
        rows.append(
            {
                "Order_ID": f"ORD{i:07d}",
                "Customer_ID": customer_id,
                "Product_ID": product["Product_ID"],
                "Order_Date": order_date.date().isoformat(),
                "Quantity": quantity,
                "Unit_Price_INR": float(product["List_Price_INR"]),
                "Discount_Pct": discount_pct,
                "Payment_Method": str(rng.choice(PAYMENT_METHODS, p=[0.39, 0.18, 0.15, 0.10, 0.10, 0.08])),
                "Shipping_Method": str(rng.choice(SHIPPING_METHODS, p=[0.67, 0.26, 0.07])),
            }
        )
    orders = pd.DataFrame(rows)
    orders["Order_Value_INR"] = (
        orders["Quantity"] * orders["Unit_Price_INR"] * (1 - orders["Discount_Pct"] / 100)
    ).round(2)
    return orders.sort_values(["Order_Date", "Order_ID"]).reset_index(drop=True)


def generate_deliveries(orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate delivery dates and statuses from order dates."""
    rows = []
    for _, order in orders.iterrows():
        order_date = pd.Timestamp(order["Order_Date"])
        dispatch_days = int(rng.choice([0, 1, 2, 3], p=[0.20, 0.50, 0.25, 0.05]))
        base_delivery = {"Standard": 5, "Express": 3, "Same-Day": 0}[order["Shipping_Method"]]
        delay = int(rng.choice([0, 0, 0, 1, 2, 5], p=[0.45, 0.25, 0.12, 0.10, 0.06, 0.02]))
        status = str(rng.choice(["Delivered", "Delayed", "Lost"], p=[0.94, 0.055, 0.005]))
        dispatch_date = order_date + pd.Timedelta(days=dispatch_days)
        delivery_date = dispatch_date + pd.Timedelta(days=base_delivery + delay)
        rows.append(
            {
                "Order_ID": order["Order_ID"],
                "Dispatch_Date": dispatch_date.date().isoformat(),
                "Delivery_Date": delivery_date.date().isoformat() if status != "Lost" else "",
                "Delivery_Status": status,
            }
        )
    return pd.DataFrame(rows)

