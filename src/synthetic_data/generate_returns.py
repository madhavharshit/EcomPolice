"""Generate return records and ground-truth order labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_data.common import RETURN_REASONS
from synthetic_data.generate_scenarios import scenario_return_probability


def choose_return_reason(scenario: str, category: str, rng: np.random.Generator) -> str:
    """Choose a category-aware return reason."""
    if scenario in {"REPEATED_DEFECT_CLAIMS", "MIXED_SUSPICIOUS_PATTERN"} and rng.random() < 0.58:
        return "Defective"
    reasons = RETURN_REASONS.get(category, ["Changed mind", "Wrong item", "Damaged item"])
    return str(rng.choice(reasons))


def generate_returns(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    deliveries: pd.DataFrame,
    labels: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate return outcomes and separate order-level scenario labels."""
    product_category = products.set_index("Product_ID")["Category"]
    scenario_by_customer = labels.set_index("Customer_ID")["Scenario_Label"]
    abuse_by_customer = labels.set_index("Customer_ID")["Simulated_Abuse_Indicator"]
    delivery_by_order = deliveries.set_index("Order_ID")
    rows = []
    order_labels = []
    for _, order in orders.iterrows():
        scenario = str(scenario_by_customer.loc[order["Customer_ID"]])
        category = str(product_category.loc[order["Product_ID"]])
        probability = scenario_return_probability(scenario, category, float(order["Order_Value_INR"]))
        returned = bool(rng.random() < probability and delivery_by_order.loc[order["Order_ID"], "Delivery_Status"] != "Lost")
        return_date = ""
        reason = ""
        refund = np.nan
        status = "Not Returned"
        if returned:
            delivered_on = pd.Timestamp(delivery_by_order.loc[order["Order_ID"], "Delivery_Date"])
            if scenario in {"RETURN_WINDOW_PATTERN", "MIXED_SUSPICIOUS_PATTERN"}:
                days_after_delivery = int(rng.choice([24, 25, 26, 27, 28, 29, 30], p=[0.06, 0.08, 0.12, 0.18, 0.24, 0.22, 0.10]))
            else:
                days_after_delivery = int(rng.choice(range(2, 31)))
            return_date = (delivered_on + pd.Timedelta(days=days_after_delivery)).date().isoformat()
            reason = choose_return_reason(scenario, category, rng)
            refund_ratio = float(rng.uniform(0.80, 1.0))
            refund = round(float(order["Order_Value_INR"]) * refund_ratio, 2)
            status = str(rng.choice(["Returned", "Refunded", "Exchange Issued"], p=[0.55, 0.35, 0.10]))

        rows.append(
            {
                "Order_ID": order["Order_ID"],
                "Return_Date": return_date,
                "Return_Reason": reason,
                "Refund_Amount_INR": refund,
                "Return_Status": status,
            }
        )
        order_labels.append(
            {
                "Order_ID": order["Order_ID"],
                "Customer_ID": order["Customer_ID"],
                "Scenario_Label": scenario,
                "Simulated_Abuse_Indicator": int(abuse_by_customer.loc[order["Customer_ID"]]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(order_labels)

