import pandas as pd


def add_order_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create order-level analysis features while preserving original columns."""
    features = cleaned.copy()
    features["Order_Value"] = features["Product_Price"] * features["Order_Quantity"]
    features["Order_Month"] = features["Parsed_Order_Date"].dt.month
    features["Order_Quarter"] = features["Parsed_Order_Date"].dt.quarter
    features["Order_DayOfWeek"] = features["Parsed_Order_Date"].dt.day_name()
    features["Order_Day"] = features["Parsed_Order_Date"].dt.day
    features["Order_IsWeekend"] = features["Parsed_Order_Date"].dt.dayofweek.isin([5, 6])
    return features


def make_pre_return_features(order_features: pd.DataFrame) -> pd.DataFrame:
    """Return candidate pre-return features available before return outcome is known."""
    columns = [
        "Order_ID",
        "Product_ID",
        "User_ID",
        "Order_Date",
        "Product_Category",
        "Product_Price",
        "Order_Quantity",
        "User_Age",
        "User_Gender",
        "User_Location",
        "Payment_Method",
        "Shipping_Method",
        "Discount_Applied",
        "Order_Value",
        "Order_Month",
        "Order_Quarter",
        "Order_DayOfWeek",
        "Order_Day",
        "Order_IsWeekend",
    ]
    return order_features[columns].copy()


def return_rate_by(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Calculate return counts and rates by a categorical field."""
    grouped = (
        df.groupby(group_column, dropna=False)
        .agg(
            order_count=("Return_Status_Binary", "size"),
            return_count=("Return_Status_Binary", "sum"),
        )
        .reset_index()
    )
    grouped["not_returned_count"] = grouped["order_count"] - grouped["return_count"]
    grouped["return_rate"] = grouped["return_count"] / grouped["order_count"]
    return grouped.sort_values(["return_rate", "order_count"], ascending=[False, False])


def return_reason_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return reason distribution among returned orders only."""
    returned = df[df["Return_Status"].eq("Returned")]
    counts = returned["Return_Reason"].value_counts(dropna=False).rename_axis("Return_Reason")
    result = counts.reset_index(name="return_count")
    result["share_of_returned_orders"] = result["return_count"] / len(returned) if len(returned) else 0
    return result


def numeric_bin_return_rates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create return-rate summaries for selected numeric bins."""
    summaries: dict[str, pd.DataFrame] = {}
    bins = {
        "Product_Price_Bin": pd.qcut(df["Product_Price"], q=4, duplicates="drop"),
        "Order_Value_Bin": pd.qcut(df["Order_Value"], q=4, duplicates="drop"),
        "Discount_Bin": pd.qcut(df["Discount_Applied"], q=4, duplicates="drop"),
        "User_Age_Bin": pd.cut(df["User_Age"], bins=[0, 24, 34, 44, 54, 64, 120]),
    }
    for name, series in bins.items():
        temp = df.assign(**{name: series.astype(str)})
        summaries[name] = return_rate_by(temp, name)
    return summaries


def customer_history_summary(df: pd.DataFrame) -> dict[str, int]:
    """Summarize whether the dataset contains repeated users."""
    user_counts = df["User_ID"].value_counts(dropna=False)
    return {
        "unique_users": int(user_counts.size),
        "users_with_repeated_orders": int((user_counts > 1).sum()),
        "max_orders_per_user": int(user_counts.max()) if not user_counts.empty else 0,
    }


def historical_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe prior customer features when repeated users exist."""
    ordered = df.sort_values(["User_ID", "Parsed_Order_Date", "Order_ID"]).copy()
    prior_orders = ordered.groupby("User_ID").cumcount()
    prior_returns = ordered.groupby("User_ID")["Return_Status_Binary"].cumsum() - ordered[
        "Return_Status_Binary"
    ]
    ordered["Prior_User_Order_Count"] = prior_orders
    ordered["Prior_User_Return_Count"] = prior_returns
    ordered["Prior_User_Return_Rate"] = prior_returns.where(prior_orders > 0) / prior_orders.where(
        prior_orders > 0
    )
    return ordered.sort_index()

