"""Run the EcomPolice preprocessing and feature-engineering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    CATEGORICAL_COLUMNS,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from feature_engineering import (
    add_order_features,
    customer_history_summary,
    historical_customer_features,
    make_pre_return_features,
    numeric_bin_return_rates,
    return_rate_by,
    return_reason_distribution,
)
from load_data import discover_raw_csv, load_raw_data
from preprocess import preprocess_orders
from validate_data import (
    categorical_distributions,
    duplicate_summary,
    missingness_table,
    numeric_validation,
    return_duration_summary,
    return_info_consistency,
    unexpected_status_values,
    validate_schema,
    write_markdown_report,
)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a compact markdown table."""
    table = df if max_rows is None else df.head(max_rows)
    if table.empty:
        return "_No rows._"
    safe = table.copy().astype(object).where(pd.notna(table), "")
    headers = [str(column) for column in safe.columns]
    rows = [[str(value) for value in row] for row in safe.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def save_outputs(
    cleaned: pd.DataFrame,
    order_features: pd.DataFrame,
    pre_return_features: pd.DataFrame,
    analysis_tables: dict[str, pd.DataFrame],
) -> None:
    """Save cleaned and engineered CSV outputs."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(INTERIM_DATA_DIR / "cleaned_orders.csv", index=False)
    order_features.to_csv(PROCESSED_DATA_DIR / "order_features.csv", index=False)
    pre_return_features.to_csv(PROCESSED_DATA_DIR / "pre_return_features.csv", index=False)
    analysis_tables["category"].to_csv(PROCESSED_DATA_DIR / "category_return_analysis.csv", index=False)
    analysis_tables["payment"].to_csv(PROCESSED_DATA_DIR / "payment_return_analysis.csv", index=False)
    analysis_tables["shipping"].to_csv(PROCESSED_DATA_DIR / "shipping_return_analysis.csv", index=False)
    analysis_tables["return_reason"].to_csv(
        PROCESSED_DATA_DIR / "return_reason_analysis.csv", index=False
    )


def build_data_quality_report(
    csv_path: Path,
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    invalid_date_counts: dict[str, int],
) -> str:
    """Build the data-quality markdown report."""
    schema = validate_schema(raw)
    duplicates = duplicate_summary(raw)
    missing = missingness_table(raw)
    numeric_checks = numeric_validation(raw)
    return_info = return_info_consistency(raw)
    duration = return_duration_summary(cleaned)
    status_distribution = raw["Return_Status"].value_counts(dropna=False).reset_index()
    status_distribution.columns = ["Return_Status", "count"]
    status_distribution["percent"] = (
        status_distribution["count"] / len(raw) * 100
    ).round(2)
    categorical = categorical_distributions(raw, CATEGORICAL_COLUMNS)

    sections = [
        "# EcomPolice Data Quality Report",
        f"Raw file used: `{csv_path.relative_to(Path.cwd())}`",
        f"Rows: {len(raw):,}\n\nColumns: {raw.shape[1]:,}",
        "## Columns\n\n" + "\n".join(f"- `{column}`: `{dtype}`" for column, dtype in raw.dtypes.items()),
        "## Schema Validation\n\n"
        f"- Matches expected column order: {schema['matches_expected_order']}\n"
        f"- Missing expected columns: {schema['missing_columns']}\n"
        f"- Extra columns: {schema['extra_columns']}",
        "## Duplicate Checks\n\n"
        + "\n".join(f"- {key}: {value:,}" for key, value in duplicates.items()),
        "## Missing Values\n\n" + markdown_table(missing),
        "## Return Status Distribution\n\n" + markdown_table(status_distribution),
        "## Date Parsing\n\n"
        + "\n".join(f"- Invalid or unparseable `{key}` values: {value:,}" for key, value in invalid_date_counts.items()),
        "## Return Information Consistency\n\n"
        + "\n".join(f"- {key}: {value:,}" for key, value in return_info.items()),
        "## Return Duration Quality\n\n"
        + "\n".join(f"- {key}: {value:,}" for key, value in duration.items()),
        "## Numeric Validation\n\n"
        + "\n".join(f"- {key}: {value:,.2f}" if isinstance(value, float) else f"- {key}: {value:,}" for key, value in numeric_checks.items()),
        "## Unexpected Categories\n\n"
        f"- Unexpected `Return_Status` values: {unexpected_status_values(raw)}",
        "## Numeric Summaries\n\n" + markdown_table(raw.describe(include="number").reset_index()),
    ]

    for column, table in categorical.items():
        sections.append(f"## `{column}` Distribution\n\n" + markdown_table(table, max_rows=30))

    report = "\n\n".join(sections)
    write_markdown_report(REPORTS_DIR / "data_quality_report.md", [report])
    return report


def build_feature_report(
    order_features: pd.DataFrame,
    analysis_tables: dict[str, pd.DataFrame],
    customer_summary: dict[str, int],
    numeric_bins: dict[str, pd.DataFrame],
) -> str:
    """Build the feature-engineering markdown report."""
    returned = int(order_features["Return_Status_Binary"].sum())
    total = len(order_features)
    return_rate = returned / total if total else 0
    duration_summary = return_duration_summary(order_features)

    sections = [
        "# EcomPolice Feature Engineering Report",
        f"Rows in order feature table: {total:,}\n\nColumns: {order_features.shape[1]:,}",
        f"Overall return rate: {return_rate:.2%} ({returned:,} of {total:,} orders)",
        "## Feature Definitions\n\n"
        "- `Order_Value`: `Product_Price * Order_Quantity`.\n"
        "- `Order_Month`, `Order_Quarter`, `Order_DayOfWeek`, `Order_Day`, `Order_IsWeekend`: calendar features from parsed order date.\n"
        "- `Calculated_Days_To_Return`: date-derived return duration for audit.\n"
        "- `Invalid_Return_Chronology`: returned order with return date before order date.\n"
        "- `Validated_Days_To_Return`: date-derived duration retained only when returned and chronology is valid.\n"
        "- `Return_Status_Binary`: 1 for returned, 0 for not returned.\n"
        "- Pre-return candidate features exclude return outcome, return date, return reason, stored duration, calculated durations, and other post-return information.",
        "## Return Duration Counts\n\n"
        + "\n".join(f"- {key}: {value:,}" for key, value in duration_summary.items()),
        "## Customer-History Limitation\n\n"
        f"- Unique users: {customer_summary['unique_users']:,}\n"
        f"- Users with repeated orders: {customer_summary['users_with_repeated_orders']:,}\n"
        f"- Maximum orders for one user: {customer_summary['max_orders_per_user']:,}\n\n"
        "If repeated users exist in a future dataset, prior customer counts and rates should be calculated by sorting each user's orders chronologically and using only earlier orders. In this file, every user appears once, so meaningful customer-level behavioral history cannot be derived from this dataset alone.",
        "## Fraud-Related Limitation\n\n"
        "No verified fraud label was found. `Return_Status` only states whether an order was returned. This phase does not create fraud labels, train a model, run anomaly detection, or claim supervised fraud-detection readiness.",
        "## Category Return Analysis\n\n" + markdown_table(analysis_tables["category"]),
        "## Payment Return Analysis\n\n" + markdown_table(analysis_tables["payment"]),
        "## Shipping Return Analysis\n\n" + markdown_table(analysis_tables["shipping"]),
        "## Return Reason Analysis\n\n" + markdown_table(analysis_tables["return_reason"]),
    ]

    for name, table in numeric_bins.items():
        sections.append(f"## {name} Return Rate Summary\n\n" + markdown_table(table))

    report = "\n\n".join(sections)
    write_markdown_report(REPORTS_DIR / "feature_engineering_report.md", [report])
    return report


def run(input_csv: str | Path | None = None) -> None:
    """Execute the full preprocessing and feature-engineering pipeline."""
    csv_path = discover_raw_csv(input_path=input_csv)
    raw = load_raw_data(csv_path)

    print(f"Raw file used: {csv_path}")
    print(f"Rows: {raw.shape[0]}")
    print(f"Columns: {raw.shape[1]}")
    print("Column names:", list(raw.columns))
    print("Data types:")
    print(raw.dtypes)
    print("Duplicate summary:", duplicate_summary(raw))
    print("Missing values:")
    print(missingness_table(raw).to_string(index=False))
    print("Numeric summary:")
    print(raw.describe(include='number').to_string())

    schema = validate_schema(raw)
    if schema["missing_columns"] or schema["extra_columns"]:
        print("Schema discrepancy detected. Actual file remains the source of truth.")
        print(schema)

    cleaned, invalid_date_counts = preprocess_orders(raw)
    order_features = add_order_features(cleaned)
    customer_features = historical_customer_features(order_features)
    pre_return_features = make_pre_return_features(order_features)

    analysis_tables = {
        "category": return_rate_by(order_features, "Product_Category"),
        "payment": return_rate_by(order_features, "Payment_Method"),
        "shipping": return_rate_by(order_features, "Shipping_Method"),
        "return_reason": return_reason_distribution(order_features),
    }

    save_outputs(cleaned, customer_features, pre_return_features, analysis_tables)
    build_data_quality_report(csv_path, raw, cleaned, invalid_date_counts)
    build_feature_report(
        customer_features,
        analysis_tables,
        customer_history_summary(order_features),
        numeric_bin_return_rates(order_features),
    )

    print("Saved outputs under data/interim, data/processed, and reports.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EcomPolice preprocessing only.")
    parser.add_argument("--input-csv", help="Optional path to a raw CSV file.")
    args = parser.parse_args()
    run(args.input_csv)


if __name__ == "__main__":
    main()
