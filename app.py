"""Teacher-ready EcomPolice dashboard for the synthetic-data workflow.

Run locally with: streamlit run app.py
This application is an explainer and audit interface.  It deliberately does not
train, score, or deploy a fraud model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SYNTHETIC = PROJECT_ROOT / "data" / "synthetic"
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

st.set_page_config(page_title="EcomPolice | Synthetic Return Analytics", page_icon="🛍️", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame:
    """Load a project CSV with caching for a responsive local presentation."""
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metric_row(values: list[tuple[str, str, str]]) -> None:
    """Render a row of concise dashboard metrics."""
    columns = st.columns(len(values))
    for column, (label, value, help_text) in zip(columns, values):
        column.metric(label, value, help=help_text)


def missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return non-zero missingness counts and percentages."""
    result = pd.DataFrame({"Missing values": frame.isna().sum()})
    result["Missing %"] = (result["Missing values"] / len(frame) * 100).round(2)
    return result.loc[result["Missing values"].gt(0)].sort_values("Missing values", ascending=False)


def require_outputs() -> bool:
    """Explain how to create derived files rather than silently fabricating them."""
    required = [
        PROCESSED / "cleaned_orders.csv",
        PROCESSED / "return_level_features.csv",
        PROCESSED / "return_level_targets.csv",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("Derived files are missing: " + ", ".join(missing))
        st.code(
            "PYTHONPATH=src python scripts/run_preprocessing_audit.py\n"
            "PYTHONPATH=src python scripts/run_feature_engineering.py\n"
            "PYTHONPATH=src python scripts/run_ml_dataset_validation.py",
            language="bash",
        )
        return False
    return True


def overview_page() -> None:
    customers = load_csv("data/synthetic/clean/customers.csv")
    products = load_csv("data/synthetic/clean/products.csv")
    orders = load_csv("data/synthetic/clean/orders.csv")
    returns = load_csv("data/synthetic/clean/returns.csv")
    labels = load_csv("data/synthetic/labels/order_ground_truth.csv")

    st.title("EcomPolice")
    st.subheader("Synthetic e-commerce return analytics for an academic demonstration")
    st.info(
        "This project uses fictional, generator-created Indian e-commerce records. "
        "Scenario labels are simulated ground truth for classroom experimentation, not verified fraud findings."
    )
    metric_row([
        ("Customers", f"{len(customers):,}", "Fictional customer profiles"),
        ("Products", f"{len(products):,}", "Synthetic catalogue items"),
        ("Orders", f"{len(orders):,}", "Clean transaction records"),
        ("Return requests", f"{len(returns):,}", "Rows in the return table"),
    ])
    st.markdown("### What this application demonstrates")
    st.markdown(
        "1. Synthetic data generation and controlled defect injection.\n"
        "2. Data-quality investigation before cleaning.\n"
        "3. Conservative preprocessing with quality flags and join checks.\n"
        "4. Leakage-aware feature engineering and an ML experiment design, with no model trained yet."
    )
    st.markdown("### Entity relationship")
    st.code("Customers 1 ── * Orders * ── 1 Products\nOrders 1 ── 1 Deliveries\nOrders 1 ── 0..1 Returns\nOrders 1 ── 1 Generator-defined labels (stored separately)", language="text")
    st.markdown("### Important limitation")
    st.warning(
        "A return is not proof of abuse. The application does not make fraud predictions, and its simulated labels must not be treated as real-world evidence."
    )
    with st.expander("Inspect source-table schemas"):
        for name, frame in {"customers": customers, "products": products, "orders": orders, "returns": returns, "order labels": labels}.items():
            st.markdown(f"**{name}**: {frame.shape[0]:,} rows × {frame.shape[1]} columns")
            st.dataframe(pd.DataFrame({"Column": frame.columns, "Data type": frame.dtypes.astype(str)}), use_container_width=True, hide_index=True)


def quality_page() -> None:
    if not require_outputs():
        return
    messy = load_csv("data/synthetic/messy/orders.csv")
    cleaned = load_csv("data/processed/cleaned_orders.csv")
    manifest = load_csv("data/synthetic/manifests/defect_manifest.csv")
    missing_audit = load_csv("data/processed/preprocessing_missing_value_audit.csv")
    duplicate_audit = load_csv("data/processed/preprocessing_duplicate_audit.csv")
    join_audit = load_csv("data/processed/preprocessing_join_audit.csv")

    st.title("Data Quality and Preprocessing Evidence")
    st.caption("The messy tables are preserved. Cleaning produces a new derived output; it never overwrites source data.")
    metric_row([
        ("Messy order rows", f"{len(messy):,}", "Input to the preprocessing exercise"),
        ("Cleaned order rows", f"{len(cleaned):,}", "One row per final order"),
        ("Injected defects", f"{len(manifest):,}", "Manifest entries recorded by the generator"),
        ("Unique output orders", f"{cleaned['Order_ID'].nunique():,}", "Expected analytical grain"),
    ])
    left, right = st.columns(2)
    with left:
        st.markdown("### Missing values before preprocessing")
        before = missing_table(messy)
        st.dataframe(before, use_container_width=True)
    with right:
        st.markdown("### Missing values after preprocessing")
        after = missing_table(cleaned)
        st.dataframe(after, use_container_width=True)
    st.markdown("### Calculated before/after missing-value audit")
    st.dataframe(missing_audit, use_container_width=True, hide_index=True)
    st.markdown("### Duplicate and join integrity checks")
    left, right = st.columns(2)
    left.dataframe(duplicate_audit, use_container_width=True, hide_index=True)
    right.dataframe(join_audit, use_container_width=True, hide_index=True)
    st.markdown("### How to explain remaining blanks")
    st.write(
        "Blank return fields for an order that was never returned are expected structural missingness. "
        "Missing customer or product attributes can also remain when the original messy order lacks a recoverable identifier. "
        "Those records are retained and flagged rather than assigned invented IDs or guessed attributes."
    )
    with st.expander("View generator defect manifest"):
        st.dataframe(manifest, use_container_width=True, hide_index=True)


def exploration_page() -> None:
    orders = load_csv("data/synthetic/clean/orders.csv")
    products = load_csv("data/synthetic/clean/products.csv")
    returns = load_csv("data/synthetic/clean/returns.csv")
    orders["Order_Date"] = pd.to_datetime(orders["Order_Date"], errors="coerce")
    st.title("Exploratory Data Analysis")
    st.caption("Patterns shown below are generator-defined assumptions, not findings about real customers or real fraud.")
    merged = orders.merge(products[["Product_ID", "Category"]], on="Product_ID", how="left", validate="m:1")
    monthly = merged.groupby(merged["Order_Date"].dt.to_period("M")).size().rename("Orders").reset_index()
    monthly["Order_Date"] = monthly["Order_Date"].astype(str)
    st.markdown("### Order volume over time")
    st.line_chart(monthly.set_index("Order_Date"))
    left, right = st.columns(2)
    with left:
        st.markdown("### Product category distribution")
        st.bar_chart(merged["Category"].value_counts())
    with right:
        st.markdown("### Payment-method distribution")
        st.bar_chart(orders["Payment_Method"].value_counts())
    returned_ids = set(returns.loc[returns["Return_Status"].eq("Returned"), "Order_ID"])
    merged["Was_Returned"] = merged["Order_ID"].isin(returned_ids)
    category_rates = merged.groupby("Category")["Was_Returned"].mean().sort_values(ascending=False).mul(100).round(2)
    st.markdown("### Return rate by product category")
    st.bar_chart(category_rates)
    st.write("A higher generated return rate is a descriptive pattern only. It does not prove suspicious behaviour.")
    returned = returns.loc[returns["Return_Status"].eq("Returned")]
    st.markdown("### Return reasons")
    st.bar_chart(returned["Return_Reason"].value_counts())


def features_page() -> None:
    if not require_outputs():
        return
    features = load_csv("data/processed/return_level_features.csv")
    targets = load_csv("data/processed/return_level_targets.csv")
    customers = load_csv("data/processed/customer_level_features.csv")
    dictionary = load_csv("data/processed/feature_data_dictionary.csv")
    target_counts = targets["Simulated_Abuse_Target"].value_counts().sort_index()
    st.title("Feature Engineering and ML Readiness")
    st.info(
        "The future task is an assessment at the moment a return request is submitted. "
        "Historical fields use only earlier records for that customer. No classifier is trained in this project."
    )
    metric_row([
        ("Return-level rows", f"{len(features):,}", "One row per return request"),
        ("Model-safe features", str(features.shape[1] - 3), "Excludes IDs and timestamp"),
        ("Customer-level rows", f"{len(customers):,}", "One row per customer at observation date"),
        ("Target positive rate", f"{target_counts.get(1, 0) / len(targets):.1%}", "Synthetic inherited target only"),
    ])
    st.markdown("### Feature availability at the prediction point")
    st.write(
        "Included: order value, category, payment and shipping method, current return reason, days since delivery, and strictly historical customer behaviour. "
        "Excluded: scenario label, simulated target, refund amount, return completion status, investigation outcomes, and future returns."
    )
    st.markdown("### Synthetic target distribution")
    st.bar_chart(target_counts.rename({0: "Synthetic normal", 1: "Synthetic suspicious"}))
    with st.expander("Feature dictionary"):
        st.dataframe(dictionary, use_container_width=True, hide_index=True)
    with st.expander("Sample feature rows and separately stored targets"):
        left, right = st.columns(2)
        left.dataframe(features.head(10), use_container_width=True, hide_index=True)
        right.dataframe(targets.head(10), use_container_width=True, hide_index=True)


def experiment_page() -> None:
    if not require_outputs():
        return
    assignments = load_csv("data/processed/return_level_split_assignments.csv")
    st.title("Machine-Learning Experiment Design")
    st.warning("Design only: no algorithm has been trained, evaluated, or used to make predictions.")
    summary = assignments.groupby("Split").agg(Return_requests=("Order_ID", "size"), Customers=("Customer_ID", "nunique"), Synthetic_positive_rate=("Simulated_Abuse_Target", "mean")).reindex(["train", "validation", "test"])
    summary["Synthetic_positive_rate"] = summary["Synthetic_positive_rate"].map("{:.2%}".format)
    st.markdown("### Customer-grouped split")
    st.dataframe(summary, use_container_width=True)
    st.write(
        "Every customer is assigned to exactly one partition. This stops the same customer’s history appearing in both training and evaluation data."
    )
    st.markdown("### Planned preprocessing inside a future scikit-learn pipeline")
    st.markdown(
        "- Numeric fields: median imputation followed by scaling.\n"
        "- Categorical fields: most-frequent imputation followed by one-hot encoding.\n"
        "- All transformers: fit only on the training partition.\n"
        "- Class imbalance: evaluate class weights first; any resampling would happen only inside training folds."
    )
    st.markdown("### Why this is leakage-safe")
    st.write(
        "Identifiers, target labels, scenario labels, refund outcomes, and future return outcomes are excluded from model inputs. "
        "The split is grouped by customer, and historical features are calculated before each return request date.")


PAGES = {
    "Project overview": overview_page,
    "Data quality and preprocessing": quality_page,
    "Exploratory analysis": exploration_page,
    "Features and targets": features_page,
    "ML experiment design": experiment_page,
}

with st.sidebar:
    st.header("EcomPolice")
    selected = st.radio("Presentation section", list(PAGES))
    st.divider()
    st.caption("Synthetic-only academic project. No Kaggle data, no deployed model, no fraud decisions.")

PAGES[selected]()
