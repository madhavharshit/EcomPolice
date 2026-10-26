"""Validation and reporting utilities for EcomPolice data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import EXPECTED_COLUMNS, EXPECTED_RETURN_STATUSES, ID_COLUMNS, RETURN_INFO_COLUMNS


def validate_schema(df: pd.DataFrame) -> dict[str, list[str] | bool]:
    """Compare actual columns against the expected Kaggle dataset schema."""
    actual = list(df.columns)
    return {
        "matches_expected_order": actual == EXPECTED_COLUMNS,
        "missing_columns": [column for column in EXPECTED_COLUMNS if column not in actual],
        "extra_columns": [column for column in actual if column not in EXPECTED_COLUMNS],
    }


def duplicate_summary(df: pd.DataFrame) -> dict[str, int]:
    """Count exact duplicate rows and duplicated key IDs."""
    summary = {"exact_duplicate_rows": int(df.duplicated().sum())}
    for column in ID_COLUMNS:
        if column in df.columns:
            summary[f"duplicate_{column}"] = int(df[column].duplicated().sum())
            summary[f"unique_{column}"] = int(df[column].nunique(dropna=False))
    return summary


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages by column."""
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.values,
            "missing_percent": (missing.values / len(df) * 100).round(2),
        }
    )


def categorical_distributions(df: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    """Return value-count tables for selected categorical columns."""
    distributions = {}
    for column in columns:
        if column in df.columns:
            counts = df[column].value_counts(dropna=False).reset_index()
            counts.columns = [column, "count"]
            counts["percent"] = (counts["count"] / len(df) * 100).round(2)
            distributions[column] = counts
    return distributions


def numeric_validation(df: pd.DataFrame) -> dict[str, int | float]:
    """Identify invalid numeric values without removing rows."""
    order_value = df["Product_Price"] * df["Order_Quantity"]
    return {
        "nonpositive_product_prices": int((df["Product_Price"] <= 0).sum()),
        "nonpositive_order_quantities": int((df["Order_Quantity"] <= 0).sum()),
        "negative_discounts": int((df["Discount_Applied"] < 0).sum()),
        "discounts_over_100": int((df["Discount_Applied"] > 100).sum()),
        "negative_user_age": int((df["User_Age"] < 0).sum()),
        "negative_order_values": int((order_value < 0).sum()),
        "max_order_value": float(order_value.max()),
        "p99_order_value": float(order_value.quantile(0.99)),
    }


def return_info_consistency(df: pd.DataFrame) -> dict[str, int]:
    """Check expected structural missingness and returned-order inconsistencies."""
    returned = df["Return_Status"].eq("Returned")
    not_returned = df["Return_Status"].eq("Not Returned")
    summary = {}
    for column in RETURN_INFO_COLUMNS:
        summary[f"not_returned_missing_{column}"] = int((not_returned & df[column].isna()).sum())
        summary[f"not_returned_present_{column}"] = int((not_returned & df[column].notna()).sum())
        summary[f"returned_missing_{column}"] = int((returned & df[column].isna()).sum())
    return summary


def return_duration_summary(df: pd.DataFrame) -> dict[str, int]:
    """Summarize calculated return duration quality."""
    returned = df["Return_Status"].eq("Returned")
    return {
        "returned_orders": int(returned.sum()),
        "invalid_return_chronology": int(df["Invalid_Return_Chronology"].sum()),
        "valid_return_durations": int((returned & df["Validated_Days_To_Return"].notna()).sum()),
        "missing_validated_return_durations": int((returned & df["Validated_Days_To_Return"].isna()).sum()),
        "stored_duration_mismatches": int(df["Stored_Duration_Mismatch"].sum()),
    }


def unexpected_status_values(df: pd.DataFrame) -> list[str]:
    """Return status values outside the documented allowed set."""
    observed = set(df["Return_Status"].dropna().unique())
    return sorted(observed - EXPECTED_RETURN_STATUSES)


def write_markdown_report(path: Path, sections: list[str]) -> None:
    """Write a markdown report from preformatted sections."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")

