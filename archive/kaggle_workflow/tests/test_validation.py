import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from validate_data import duplicate_summary, missingness_table, validate_schema


class ValidationTests(unittest.TestCase):
    def test_validate_schema_detects_expected_columns(self) -> None:
        df = pd.DataFrame(columns=[
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
        ])

        result = validate_schema(df)

        self.assertTrue(result["matches_expected_order"])
        self.assertEqual(result["missing_columns"], [])
        self.assertEqual(result["extra_columns"], [])


    def test_duplicate_and_missing_summaries(self) -> None:
        df = pd.DataFrame(
            {
                "Order_ID": ["O1", "O1"],
                "Product_ID": ["P1", "P2"],
                "User_ID": ["U1", "U2"],
                "Value": [None, 3],
            }
        )

        duplicates = duplicate_summary(df)
        missing = missingness_table(df)

        self.assertEqual(duplicates["duplicate_Order_ID"], 1)
        self.assertEqual(duplicates["unique_User_ID"], 2)
        self.assertEqual(missing.loc[missing["column"].eq("Value"), "missing_count"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
