import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_engineering import add_order_features, make_pre_return_features, return_rate_by
from preprocess import preprocess_orders


def sample_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order_ID": ["O1", "O2", "O3"],
            "Product_ID": ["P1", "P2", "P3"],
            "User_ID": ["U1", "U1", "U2"],
            "Order_Date": ["2024-01-05", "2024-01-10", "2024-01-06"],
            "Return_Date": ["2024-01-08", "2024-01-09", None],
            "Product_Category": ["Books ", "Books", "Toys"],
            "Product_Price": [10.0, 20.0, 5.0],
            "Order_Quantity": [2, 1, 3],
            "Return_Reason": ["Wrong item", "Changed mind", None],
            "Return_Status": ["Returned", "Returned", "Not Returned"],
            "Days_to_Return": [3.0, -1.0, None],
            "User_Age": [30, 30, 25],
            "User_Gender": ["Female", "Female", "Male"],
            "User_Location": ["City1", "City1", "City2"],
            "Payment_Method": ["Credit Card", "Credit Card", "Debit Card"],
            "Shipping_Method": ["Express", "Express", "Standard"],
            "Discount_Applied": [1.0, 0.0, 2.0],
        }
    )


class FeatureEngineeringTests(unittest.TestCase):
    def test_invalid_chronology_and_validated_duration(self) -> None:
        cleaned, invalid_dates = preprocess_orders(sample_orders())

        self.assertEqual(invalid_dates, {"Order_Date": 0, "Return_Date": 0})
        self.assertEqual(cleaned.loc[0, "Validated_Days_To_Return"], 3)
        self.assertTrue(cleaned.loc[1, "Invalid_Return_Chronology"])
        self.assertTrue(pd.isna(cleaned.loc[1, "Validated_Days_To_Return"]))
        self.assertTrue(pd.isna(cleaned.loc[2, "Validated_Days_To_Return"]))


    def test_order_value_and_calendar_features(self) -> None:
        cleaned, _ = preprocess_orders(sample_orders())
        features = add_order_features(cleaned)

        self.assertEqual(features.loc[0, "Product_Category"], "Books")
        self.assertEqual(features.loc[0, "Order_Value"], 20.0)
        self.assertEqual(features.loc[0, "Order_Month"], 1)
        self.assertTrue(features.loc[2, "Order_IsWeekend"])


    def test_return_rate_by_category(self) -> None:
        cleaned, _ = preprocess_orders(sample_orders())
        features = add_order_features(cleaned)
        result = return_rate_by(features, "Product_Category")
        books = result[result["Product_Category"].eq("Books")].iloc[0]

        self.assertEqual(books["order_count"], 2)
        self.assertEqual(books["return_count"], 2)
        self.assertEqual(books["return_rate"], 1.0)


    def test_pre_return_feature_availability_excludes_post_return_fields(self) -> None:
        cleaned, _ = preprocess_orders(sample_orders())
        features = add_order_features(cleaned)
        pre_return = make_pre_return_features(features)

        excluded = {
            "Return_Status",
            "Return_Status_Binary",
            "Return_Date",
            "Return_Reason",
            "Days_to_Return",
            "Calculated_Days_To_Return",
            "Validated_Days_To_Return",
            "Invalid_Return_Chronology",
        }
        self.assertTrue(excluded.isdisjoint(pre_return.columns))
        self.assertIn("Order_Value", pre_return.columns)


if __name__ == "__main__":
    unittest.main()
