import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthetic_data.build_features import build_order_level_table, build_pre_return_features
from synthetic_data.config import DefectRates, SyntheticConfig
from synthetic_data.generate_customers import generate_customers
from synthetic_data.generate_orders import generate_deliveries, generate_orders
from synthetic_data.generate_products import generate_products
from synthetic_data.generate_returns import generate_returns
from synthetic_data.generate_scenarios import assign_customer_scenarios
from synthetic_data.inject_defects import inject_defects
from synthetic_data.preprocessing import SyntheticTables, build_clean_order_dataset, build_model_targets, preprocess_messy_tables
from synthetic_data.run import generate
from synthetic_data.validate import validate_clean_dataset, validate_feature_dataset

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
