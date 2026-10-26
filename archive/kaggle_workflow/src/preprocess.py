
import pandas as pd

from config import CATEGORICAL_COLUMNS, EXPECTED_RETURN_STATUSES


def normalize_categorical_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim accidental whitespace in categorical string columns."""
    result = df.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in result.columns:
            result[column] = result[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    return result


def parse_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse order and return dates, preserving original text fields."""
    result = df.copy()
    invalid_counts: dict[str, int] = {}
    for column in ["Order_Date", "Return_Date"]:
        parsed_column = f"Parsed_{column}"
        original_not_missing = result[column].notna()
        result[parsed_column] = pd.to_datetime(result[column], errors="coerce")
        invalid_counts[column] = int((original_not_missing & result[parsed_column].isna()).sum())
    return result, invalid_counts


def add_return_quality_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration, chronology, and status fields without overwriting raw columns."""
    result = df.copy()
    if not EXPECTED_RETURN_STATUSES.issuperset(set(result["Return_Status"].dropna().unique())):
        unexpected = sorted(set(result["Return_Status"].dropna().unique()) - EXPECTED_RETURN_STATUSES)
        raise ValueError(f"Unexpected Return_Status values: {unexpected}")

    returned = result["Return_Status"].eq("Returned")
    result["Calculated_Days_To_Return"] = (
        result["Parsed_Return_Date"] - result["Parsed_Order_Date"]
    ).dt.days
    result["Invalid_Return_Chronology"] = (
        returned
        & result["Parsed_Order_Date"].notna()
        & result["Parsed_Return_Date"].notna()
        & (result["Calculated_Days_To_Return"] < 0)
    )
    result["Validated_Days_To_Return"] = result["Calculated_Days_To_Return"].where(
        returned & ~result["Invalid_Return_Chronology"]
    )
    result["Return_Status_Binary"] = result["Return_Status"].map(
        {"Returned": 1, "Not Returned": 0}
    )
    result["Stored_Duration_Mismatch"] = (
        returned
        & result["Days_to_Return"].notna()
        & result["Calculated_Days_To_Return"].notna()
        & (result["Days_to_Return"] != result["Calculated_Days_To_Return"])
    )
    return result


def preprocess_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize strings, parse dates, and add data-quality fields."""
    normalized = normalize_categorical_strings(df)
    parsed, invalid_date_counts = parse_dates(normalized)
    cleaned = add_return_quality_fields(parsed)
    return cleaned, invalid_date_counts

