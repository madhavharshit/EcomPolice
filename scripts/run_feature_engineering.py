"""Create leakage-aware, synthetic feature datasets for future experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthetic_data.build_features import (  # noqa: E402
    HIGH_VALUE_ORDER_THRESHOLD_INR,
    build_customer_level_features,
    build_customer_level_targets,
    build_order_level_table,
    build_return_level_features,
    build_return_level_targets,
    validate_feature_outputs,
)


def load_clean_tables() -> dict[str, pd.DataFrame]:
    """Load clean synthetic entity tables used for model-ready derivations."""
    clean_dir = PROJECT_ROOT / "data" / "synthetic" / "clean"
    return {name: pd.read_csv(clean_dir / f"{name}.csv") for name in ["customers", "products", "orders", "deliveries", "returns"]}


def build_data_dictionary(return_features: pd.DataFrame, customer_features: pd.DataFrame) -> pd.DataFrame:
    """Describe identifiers, features, timestamps, and deliberately excluded targets."""
    descriptions = {
        "Order_ID": "Stable return-request identifier; used for feature-target alignment, not a model input.",
        "Customer_ID": "Stable customer identifier; used for grouping and alignment, not a model input.",
        "Return_Request_Date": "Prediction timestamp; used for chronological validation, not a model input.",
        "Observation_Date": "Customer feature observation timestamp; used for documentation, not a model input.",
        "Order_Value_INR": "Value of the order associated with the current return request.",
        "Product_Category": "Category of the product on the current returned order.",
        "Payment_Method": "Payment method available from the original order.",
        "Shipping_Method": "Shipping method available from the original order.",
        "Return_Reason": "Reason supplied with the current return request.",
        "Days_Since_Delivery": "Days between recorded delivery and the return request.",
        "Prior_Order_Count": "Number of earlier customer orders available before the prediction timestamp.",
        "Prior_Return_Count": "Number of earlier return requests available before the prediction timestamp.",
        "Historical_Return_Rate": "Prior_Return_Count divided by Prior_Order_Count; zero when there is no prior order.",
        "Recent_Return_Count_30D": "Earlier return requests in the 30 days before the prediction timestamp.",
        "Recent_Return_Count_90D": "Earlier return requests in the 90 days before the prediction timestamp.",
        "High_Value_Return_Count": f"Earlier returned orders with value at least INR {HIGH_VALUE_ORDER_THRESHOLD_INR:,.0f}.",
        "Repeated_Defect_Claim_Rate": "Earlier defective claims divided by earlier return requests; zero when there is no prior return.",
        "Average_Days_To_Return": "Mean delivery-to-return duration for earlier return requests; zero when there is no prior return.",
        "Has_Prior_Order_History": "Indicator that at least one earlier order was available.",
        "Has_Prior_Return_History": "Indicator that at least one earlier return request was available.",
    }
    rows = []
    for dataset_name, frame in [("return_level_features", return_features), ("customer_level_features", customer_features)]:
        for column in frame.columns:
            if column in {"Order_ID", "Customer_ID"}:
                role = "stable identifier"
            elif column in {"Return_Request_Date", "Observation_Date"}:
                role = "prediction/observation timestamp"
            else:
                role = "model feature"
            rows.append({"dataset": dataset_name, "column": column, "role": role, "description": descriptions.get(column, "Documented feature column.")})
    rows.extend(
        [
            {"dataset": "targets_only", "column": "Scenario_Label", "role": "excluded target metadata", "description": "Generator-defined simulated scenario; never included in feature inputs."},
            {"dataset": "targets_only", "column": "Simulated_Abuse_Target", "role": "target", "description": "Synthetic target only; not verified real-world abuse."},
            {"dataset": "excluded_from_return_features", "column": "Refund_Amount_INR", "role": "post-request outcome", "description": "Refund completion information is unavailable at return request time."},
            {"dataset": "excluded_from_return_features", "column": "Return_Status", "role": "post-request outcome", "description": "Return outcome is excluded from model inputs."},
        ]
    )
    return pd.DataFrame(rows)


def write_reports(
    return_features: pd.DataFrame,
    customer_features: pd.DataFrame,
    return_targets: pd.DataFrame,
    customer_targets: pd.DataFrame,
    validation: dict[str, int | bool],
) -> None:
    """Write explanatory feature-engineering and leakage-audit reports."""
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    feature_report = [
        "# Feature Engineering Report",
        "## Prediction Task",
        "The primary task is a simulated return-level suspicious-behaviour assessment when a customer submits a return request. The prediction point is `Return_Request_Date`.",
        "## Source and Label Granularity",
        "Features are calculated from `data/synthetic/clean/` because the messy files intentionally contain controlled defects for the preprocessing exercise. `order_ground_truth.csv` contains one generator label per order, but each label is inherited from the customer's scenario. The return-level target file selects that existing order label only for returned orders and documents this construction explicitly. It is not a verified real-world return-level abuse label.",
        "## Datasets Created",
        f"- Return-level features: {len(return_features):,} rows, one per return request.\n- Customer-level features: {len(customer_features):,} rows, one per customer.\n- Return-level targets: {len(return_targets):,} rows aligned by `Order_ID` and `Customer_ID`.\n- Customer-level targets: {len(customer_targets):,} rows aligned by `Customer_ID`.",
        "## Historical Feature Formulae",
        "- `Historical_Return_Rate = Prior_Return_Count / Prior_Order_Count`, with zero when no earlier orders exist.\n- `Repeated_Defect_Claim_Rate = prior defective claims / Prior_Return_Count`, with zero when no earlier returns exist.\n- Recent windows count return requests with dates in `[prediction date - window, prediction date)`.\n- `High_Value_Return_Count` uses the fixed INR 10,000 threshold, not a dataset-wide future-informed quantile.",
        "## Availability Boundaries",
        "Order value, category, payment method, shipping method, current return reason, and days since delivery are available by the return request. Refund amount, return completion status, investigation outcome, future returns, scenario labels, and simulated targets are excluded from feature files.",
        "## Validation Results",
        "```text\n" + pd.Series(validation).to_string() + "\n```",
        "## Limitations",
        "The scenario labels and simulated abuse targets were generated for this project. They must not be interpreted as verified fraud labels or real-world performance evidence. No model is trained in this phase.",
    ]
    (reports / "feature_engineering_report.md").write_text("\n\n".join(feature_report) + "\n", encoding="utf-8")

    leakage_report = [
        "# Leakage Audit Report",
        "## Prediction Point",
        "Each return-level row is scored at `Return_Request_Date`. Historical fields use only orders and return requests strictly earlier than that timestamp.",
        "## Excluded From Return-Level Feature Inputs",
        "- `Scenario_Label`\n- `Simulated_Abuse_Target`\n- `Refund_Amount_INR`\n- `Return_Status`\n- future order and return outcomes\n- investigation results",
        "## Validation Results",
        "```text\n" + pd.Series(validation).to_string() + "\n```",
        "## Label Construction",
        "`return_level_targets.csv` retains `Scenario_Label` and `Simulated_Abuse_Target` outside the feature CSV. The target is an existing generator-provided order label inherited from the customer's scenario and is selected only for returned orders. This is a transparent simulation rule, not an assertion that each returned order is truly abusive.",
        "## What This Proves",
        "The audit verifies identifier alignment, one-row-per-return/customer grain, target separation, and the absence of future return requests in historical counts. It does not prove that a future model would perform well on real data.",
    ]
    (reports / "leakage_audit_report.md").write_text("\n\n".join(leakage_report) + "\n", encoding="utf-8")


def main() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    processed.mkdir(exist_ok=True)
    tables = load_clean_tables()
    order_labels = pd.read_csv(PROJECT_ROOT / "data" / "synthetic" / "labels" / "order_ground_truth.csv")
    customer_labels = pd.read_csv(PROJECT_ROOT / "data" / "synthetic" / "labels" / "customer_ground_truth.csv")
    order_table = build_order_level_table(**tables)
    return_features = build_return_level_features(order_table)
    customer_features = build_customer_level_features(order_table, customer_ids=tables["customers"]["Customer_ID"])
    return_targets = build_return_level_targets(order_labels, return_features)
    customer_targets = build_customer_level_targets(customer_labels, customer_features)
    validation = validate_feature_outputs(order_table, return_features, return_targets, customer_features, customer_targets)
    if not all(value is True or value == 0 for value in validation.values()):
        raise ValueError(f"Feature validation failed: {validation}")
    dictionary = build_data_dictionary(return_features, customer_features)
    return_features.to_csv(processed / "return_level_features.csv", index=False)
    customer_features.to_csv(processed / "customer_level_features.csv", index=False)
    return_targets.to_csv(processed / "return_level_targets.csv", index=False)
    customer_targets.to_csv(processed / "customer_level_targets.csv", index=False)
    dictionary.to_csv(processed / "feature_data_dictionary.csv", index=False)
    write_reports(return_features, customer_features, return_targets, customer_targets, validation)
    print(pd.Series(validation).to_string())
    print(f"Wrote {len(return_features):,} return-level and {len(customer_features):,} customer-level feature rows.")


if __name__ == "__main__":
    main()
