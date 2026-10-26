"""Project paths and expected dataset schema."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXPECTED_COLUMNS = [
    "Order_ID",
    "Product_ID",
    "User_ID",
    "Order_Date",
    "Return_Date",
    "Product_Category",
    "Product_Price",
    "Order_Quantity",
    "Return_Reason",
    "Return_Status",
    "Days_to_Return",
    "User_Age",
    "User_Gender",
    "User_Location",
    "Payment_Method",
    "Shipping_Method",
    "Discount_Applied",
]

ID_COLUMNS = ["Order_ID", "Product_ID", "User_ID"]
DATE_COLUMNS = ["Order_Date", "Return_Date"]
RETURN_INFO_COLUMNS = ["Return_Date", "Return_Reason", "Days_to_Return"]
CATEGORICAL_COLUMNS = [
    "Product_Category",
    "Return_Reason",
    "Return_Status",
    "User_Gender",
    "User_Location",
    "Payment_Method",
    "Shipping_Method",
]
NUMERIC_COLUMNS = [
    "Product_Price",
    "Order_Quantity",
    "Days_to_Return",
    "User_Age",
    "Discount_Applied",
]
EXPECTED_RETURN_STATUSES = {"Returned", "Not Returned"}

