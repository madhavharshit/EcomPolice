import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthetic_data.build_features import (
    build_customer_level_features,
    build_customer_level_targets,
    build_order_level_table,
    build_pre_return_features,
    build_return_level_features,
    build_return_level_targets,
    validate_feature_outputs,
)
from synthetic_data.config import DefectRates, SyntheticConfig
from synthetic_data.generate_customers import generate_customers
from synthetic_data.generate_orders import generate_deliveries, generate_orders
from synthetic_data.generate_products import generate_products
from synthetic_data.generate_returns import generate_returns
from synthetic_data.generate_scenarios import assign_customer_scenarios
from synthetic_data.inject_defects import inject_defects
from synthetic_data.preprocessing import (
    SyntheticTables,
    build_clean_order_dataset,
    build_model_targets,
    missing_value_audit,
    preprocess_messy_tables,
)
from synthetic_data.run import generate
from synthetic_data.validate import validate_clean_dataset, validate_feature_dataset

import numpy as np

from scripts.run_ml_dataset_validation import build_grouped_split, validate_grouped_split


def make_small_dataset(seed: int = 123):
    config = SyntheticConfig(n_customers=80, n_products=35, n_orders=360, seed=seed)
    rng = np.random.default_rng(config.seed)
    labels = assign_customer_scenarios(config, rng)
    customers = generate_customers(config, labels, rng)
    products = generate_products(config, rng)
    orders = generate_orders(config, customers, products, labels, rng)
    deliveries = generate_deliveries(orders, rng)
    returns, order_labels = generate_returns(orders, products, deliveries, labels, rng)
    order_table = build_order_level_table(customers, products, orders, deliveries, returns)
    features = build_pre_return_features(order_table)
    return config, customers, products, orders, deliveries, returns, labels, order_labels, features


class SyntheticDataTests(unittest.TestCase):
    def test_clean_ids_and_foreign_keys_are_valid(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        results = validate_clean_dataset(customers, products, orders, deliveries, returns)

        self.assertTrue(results["orders_have_valid_customers"])
        self.assertTrue(results["orders_have_valid_products"])
        self.assertTrue(results["deliveries_have_valid_orders"])
        self.assertTrue(results["returns_have_valid_orders"])

    def test_clean_chronology_rules_hold(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        results = validate_clean_dataset(customers, products, orders, deliveries, returns)

        self.assertEqual(results["orders_before_signup_count"], 0)
        self.assertEqual(results["dispatch_before_order_count"], 0)
        self.assertEqual(results["delivery_before_dispatch_count"], 0)
        self.assertEqual(results["return_before_delivery_count"], 0)

    def test_customers_have_multiple_orders_overall(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        results = validate_clean_dataset(customers, products, orders, deliveries, returns)

        self.assertGreater(results["customers_with_multiple_orders"], 0)
        self.assertGreater(results["max_orders_per_customer"], 1)

    def test_historical_features_do_not_use_future_data(self) -> None:
        _, _, _, orders, _, _, _, _, features = make_small_dataset()
        sample_customer = orders["Customer_ID"].value_counts().idxmax()
        customer_features = features[features["Customer_ID"].eq(sample_customer)].sort_values(["Order_Date", "Order_ID"])

        for _, row in customer_features.iterrows():
            prior = customer_features[pd.to_datetime(customer_features["Order_Date"]) < pd.Timestamp(row["Order_Date"])]
            self.assertEqual(row["Customer_Order_Count_Before"], len(prior))

    def test_labels_and_post_return_fields_are_excluded_from_features(self) -> None:
        _, _, _, _, _, _, _, order_labels, features = make_small_dataset()
        results = validate_feature_dataset(features, order_labels)

        self.assertTrue(results["label_columns_excluded"])
        self.assertTrue(results["post_return_columns_excluded"])
        self.assertTrue(results["all_orders_have_labels"])

    def test_defect_manifest_reports_injected_defects(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        rates = DefectRates(
            missing_values=0.02,
            inconsistent_categories=0.02,
            mixed_date_formats=0.02,
            duplicate_rows=0.02,
            invalid_chronology=0.02,
            numeric_as_string=0.02,
            missing_identifiers=0.02,
            impossible_negative_values=0.02,
            implausible_outliers=0.02,
        )
        messy, manifest = inject_defects(
            {
                "customers": customers,
                "products": products,
                "orders": orders,
                "deliveries": deliveries,
                "returns": returns,
            },
            rates,
            np.random.default_rng(999),
        )

        self.assertFalse(manifest.empty)
        self.assertIn("Defect_Type", manifest.columns)
        self.assertGreater(len(messy["orders"]), len(orders))
        self.assertEqual(len(orders), 360)

    def test_random_seed_reproducibility(self) -> None:
        first = make_small_dataset(seed=321)[3]
        second = make_small_dataset(seed=321)[3]

        pd.testing.assert_frame_equal(first.head(50).reset_index(drop=True), second.head(50).reset_index(drop=True))

    def test_clean_and_messy_outputs_remain_separate(self) -> None:
        config = replace(
            SyntheticConfig(),
            n_customers=40,
            n_products=20,
            n_orders=120,
            seed=777,
            defect_rates=DefectRates(duplicate_rows=0.05),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("synthetic_data.run.CLEAN_DIR", root / "clean"),
                patch("synthetic_data.run.MESSY_DIR", root / "messy"),
                patch("synthetic_data.run.LABELS_DIR", root / "labels"),
                patch("synthetic_data.run.MANIFESTS_DIR", root / "manifests"),
                patch("synthetic_data.run.REPORTS_DIR", root / "reports"),
            ):
                generate(config)

            self.assertTrue((root / "clean" / "orders.csv").exists())
            self.assertTrue((root / "messy" / "orders.csv").exists())
            self.assertTrue((root / "manifests" / "defect_manifest.csv").exists())

    def test_preprocessing_keeps_one_row_per_order_after_duplicates(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        rates = DefectRates(duplicate_rows=0.05, numeric_as_string=0.05, inconsistent_categories=0.05)
        messy, _ = inject_defects(
            {
                "customers": customers,
                "products": products,
                "orders": orders,
                "deliveries": deliveries,
                "returns": returns,
            },
            rates,
            np.random.default_rng(444),
        )

        cleaned, actions = preprocess_messy_tables(SyntheticTables(**messy))
        order_dataset = build_clean_order_dataset(cleaned)

        self.assertFalse(actions.empty)
        self.assertEqual(order_dataset["Order_ID"].nunique(), len(orders))
        self.assertTrue(order_dataset["One_Row_Per_Order_Flag"].all())

    def test_model_targets_are_separate_from_features(self) -> None:
        _, _, _, _, _, _, _, order_labels, features = make_small_dataset()
        targets = build_model_targets(order_labels)

        self.assertEqual(len(targets), len(features))
        self.assertTrue(set(targets["Order_ID"]).issubset(set(features["Order_ID"])))
        self.assertFalse({"Scenario_Label", "Simulated_Abuse_Indicator"}.intersection(features.columns))

    def test_missing_identifiers_remain_flagged_and_create_unmatched_joins(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        orders.loc[0, "Customer_ID"] = ""
        orders.loc[1, "Product_ID"] = ""

        cleaned, _ = preprocess_messy_tables(SyntheticTables(customers, products, orders, deliveries, returns))
        final = build_clean_order_dataset(cleaned)

        self.assertEqual(int(final["Invalid_Identifier_Flag"].sum()), 2)
        self.assertEqual(int(final["Unmatched_Customer_Flag"].sum()), 1)
        self.assertEqual(int(final["Unmatched_Product_Flag"].sum()), 1)
        self.assertTrue(final.loc[final["Unmatched_Customer_Flag"], "Full_Name"].isna().all())
        self.assertTrue(final.loc[final["Unmatched_Product_Flag"], "Category"].isna().all())

    def test_join_rejects_duplicate_reference_keys(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        duplicate_products = pd.concat([products, products.iloc[[0]]], ignore_index=True)
        cleaned, _ = preprocess_messy_tables(SyntheticTables(customers, duplicate_products, orders, deliveries, returns))

        with self.assertRaisesRegex(ValueError, "reference keys must be unique"):
            build_clean_order_dataset(cleaned)

    def test_return_blanks_for_not_returned_orders_are_structural(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        cleaned, _ = preprocess_messy_tables(SyntheticTables(customers, products, orders, deliveries, returns))
        final = build_clean_order_dataset(cleaned)
        not_returned = final[final["Return_Status"].eq("Not Returned")]

        self.assertTrue(not_returned["Return_Date"].isna().all())
        self.assertTrue(not_returned["Return_Reason"].isna().all())
        self.assertTrue(not_returned["Refund_Amount_INR"].isna().all())

    def test_missing_value_audit_matches_final_dataset(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        orders.loc[0, "Customer_ID"] = ""
        cleaned, _ = preprocess_messy_tables(SyntheticTables(customers, products, orders, deliveries, returns))
        final = build_clean_order_dataset(cleaned)
        audit = missing_value_audit(orders, cleaned.orders, final).set_index("column")

        self.assertEqual(audit.loc["Customer_ID", "final_missing_count"], int(final["Customer_ID"].isna().sum()))
        self.assertTrue(pd.isna(audit.loc["Full_Name", "cleaned_orders_missing_count"]))

    def test_return_level_features_are_chronological_and_separate_from_targets(self) -> None:
        _, customers, products, orders, deliveries, returns, customer_labels, order_labels, _ = make_small_dataset()
        order_table = build_order_level_table(customers, products, orders, deliveries, returns)
        return_features = build_return_level_features(order_table)
        customer_features = build_customer_level_features(order_table, customer_ids=customers["Customer_ID"])
        return_targets = build_return_level_targets(order_labels, return_features)
        customer_targets = build_customer_level_targets(customer_labels, customer_features)
        validation = validate_feature_outputs(
            order_table, return_features, return_targets, customer_features, customer_targets
        )

        self.assertTrue(validation["return_features_one_row_per_return"])
        self.assertTrue(validation["customer_features_one_row_per_customer"])
        self.assertTrue(validation["no_target_columns_in_return_features"])
        self.assertTrue(validation["return_feature_target_alignment"])
        self.assertTrue(validation["customer_feature_target_alignment"])
        self.assertEqual(validation["chronological_feature_violations"], 0)

    def test_customer_with_no_orders_has_zero_history(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset()
        order_table = build_order_level_table(customers, products, orders, deliveries, returns)
        customer_ids = pd.concat([customers["Customer_ID"], pd.Series(["CUST_NO_ORDERS"])], ignore_index=True)
        features = build_customer_level_features(order_table, customer_ids=customer_ids)
        no_order_row = features.loc[features["Customer_ID"].eq("CUST_NO_ORDERS")].iloc[0]

        self.assertEqual(no_order_row["Prior_Order_Count"], 0)
        self.assertEqual(no_order_row["Prior_Return_Count"], 0)
        self.assertEqual(no_order_row["Historical_Return_Rate"], 0)
        self.assertEqual(no_order_row["Has_Prior_Return_History"], 0)

    def test_return_level_feature_build_is_reproducible(self) -> None:
        _, customers, products, orders, deliveries, returns, _, _, _ = make_small_dataset(seed=987)
        order_table = build_order_level_table(customers, products, orders, deliveries, returns)

        first = build_return_level_features(order_table)
        second = build_return_level_features(order_table)

        pd.testing.assert_frame_equal(first, second)

    def test_grouped_split_keeps_customers_in_one_partition(self) -> None:
        rows = []
        for customer_number in range(10):
            target = customer_number % 2
            for return_number in range(2):
                rows.append(
                    {
                        "Order_ID": f"ORD{customer_number:02d}{return_number}",
                        "Customer_ID": f"CUST{customer_number:02d}",
                        "Simulated_Abuse_Target": target,
                    }
                )
        dataset = pd.DataFrame(rows)
        assignments, _ = build_grouped_split(dataset)
        validation = validate_grouped_split(assignments)

        self.assertTrue(validation["all_customers_in_one_partition"])
        self.assertEqual(validation["customers_shared_across_partitions"], 0)
        self.assertTrue(validation["both_target_classes_in_each_partition"])


if __name__ == "__main__":
    unittest.main()
