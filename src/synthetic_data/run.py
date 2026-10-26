"""CLI entry point for generating synthetic EcomPolice datasets."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_data.build_features import build_order_level_table, build_pre_return_features
from synthetic_data.config import CLEAN_DIR, LABELS_DIR, MANIFESTS_DIR, MESSY_DIR, REPORTS_DIR, SyntheticConfig
from synthetic_data.generate_customers import generate_customers
from synthetic_data.generate_orders import generate_deliveries, generate_orders
from synthetic_data.generate_products import generate_products
from synthetic_data.generate_returns import generate_returns
from synthetic_data.generate_scenarios import SCENARIO_RULES, assign_customer_scenarios
from synthetic_data.inject_defects import inject_defects
from synthetic_data.validate import validate_clean_dataset, validate_feature_dataset


LOGGER = logging.getLogger(__name__)


def ensure_dirs() -> None:
    """Create synthetic output directories."""
    for path in [CLEAN_DIR, MESSY_DIR, LABELS_DIR, MANIFESTS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_csvs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Write named tables to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def write_data_dictionary(path: Path) -> None:
    """Write a concise data dictionary for generated CSVs."""
    content = """# Synthetic Data Dictionary

## Clean entity tables

- `customers.csv`: fictional customer profiles with city, signup date, demographic bands, and loyalty tier.
- `products.csv`: synthetic product catalog with category, brand, product name, and rupee list price.
- `orders.csv`: order events with customer/product foreign keys, order date, quantity, price, discount, payment method, shipping method, and order value.
- `deliveries.csv`: dispatch and delivery dates plus delivery status.
- `returns.csv`: return outcome fields. Non-returned orders have structurally empty return fields.

## Labels

- `customer_ground_truth.csv`: customer-level latent scenario and simulated abuse indicator.
- `order_ground_truth.csv`: order-level scenario inherited from the customer. Keep this separate from model features.

## Features

- `pre_return_features.csv`: leakage-safe candidate features available before the current return outcome.
- Historical features are calculated chronologically using only prior orders for the same customer.

## Defect manifest

- `defect_manifest.csv`: table, row identifier, column, defect type, original value, and injected value for the messy data copy.
"""
    path.write_text(content, encoding="utf-8")


def build_report(
    config: SyntheticConfig,
    clean_validation: dict[str, int | bool],
    feature_validation: dict[str, int | bool],
    labels: pd.DataFrame,
    order_labels: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    """Write the synthetic-data report."""
    scenario_counts = labels["Scenario_Label"].value_counts().rename_axis("scenario").reset_index(name="customers")
    order_scenario_counts = order_labels["Scenario_Label"].value_counts().rename_axis("scenario").reset_index(name="orders")
    defect_counts = manifest["Defect_Type"].value_counts().rename_axis("defect_type").reset_index(name="count") if not manifest.empty else pd.DataFrame(columns=["defect_type", "count"])
    sections = [
        "# Synthetic EcomPolice Dataset Report",
        f"Customers: {config.n_customers:,}\n\nProducts: {config.n_products:,}\n\nOrders: {config.n_orders:,}\n\nSimulated period: {config.months} months from {config.period_start}\n\nRandom seed: {config.seed}",
        "## Generation Assumptions\n\n"
        + "\n".join(f"- `{name}`: {rule}" for name, rule in SCENARIO_RULES.items())
        + "\n- Indian locations and names are fictional/synthetic.\n- Returns are scenario-influenced but high return rate alone is not treated as proof of abuse.",
        "## Customer Scenario Distribution\n\n" + scenario_counts.to_string(index=False),
        "## Order Scenario Distribution\n\n" + order_scenario_counts.to_string(index=False),
        "## Defects Injected\n\n" + (defect_counts.to_string(index=False) if not defect_counts.empty else "No defects injected."),
        "## Clean Validation Results\n\n" + "\n".join(f"- {key}: {value}" for key, value in clean_validation.items()),
        "## Feature Validation Results\n\n" + "\n".join(f"- {key}: {value}" for key, value in feature_validation.items()),
        "## Known Limitations\n\n"
        "- This is simulated data, not evidence of real fraud or real customer behavior.\n"
        "- Scenario labels are artificial ground truth for experiments and must not be mixed into input features.\n"
        "- The generator approximates plausible dependencies but cannot capture all operational details of Indian e-commerce.\n"
        "- Messy defects are injected for preprocessing experiments and are documented separately in the manifest.",
    ]
    (REPORTS_DIR / "synthetic_data_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def generate(config: SyntheticConfig) -> dict[str, pd.DataFrame]:
    """Generate clean, messy, label, manifest, and feature outputs."""
    config.validate()
    ensure_dirs()
    rng = np.random.default_rng(config.seed)
    LOGGER.info("Generating synthetic EcomPolice data with seed %s", config.seed)

    customer_labels = assign_customer_scenarios(config, rng)
    customers = generate_customers(config, customer_labels, rng)
    products = generate_products(config, rng)
    orders = generate_orders(config, customers, products, customer_labels, rng)
    deliveries = generate_deliveries(orders, rng)
    returns, order_labels = generate_returns(orders, products, deliveries, customer_labels, rng)
    order_table = build_order_level_table(customers, products, orders, deliveries, returns)
    features = build_pre_return_features(order_table)

    clean_tables = {
        "customers": customers,
        "products": products,
        "orders": orders,
        "deliveries": deliveries,
        "returns": returns,
    }
    labels = {
        "customer_ground_truth": customer_labels,
        "order_ground_truth": order_labels,
    }
    feature_tables = {"pre_return_features": features}

    messy_tables, manifest = inject_defects(clean_tables, config.defect_rates, rng)
    clean_validation = validate_clean_dataset(customers, products, orders, deliveries, returns)
    feature_validation = validate_feature_dataset(features, order_labels)

    write_csvs(CLEAN_DIR, clean_tables | feature_tables)
    write_csvs(MESSY_DIR, messy_tables)
    write_csvs(LABELS_DIR, labels)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFESTS_DIR / "defect_manifest.csv", index=False)
    write_data_dictionary(MANIFESTS_DIR / "data_dictionary.md")
    (MANIFESTS_DIR / "generation_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    build_report(config, clean_validation, feature_validation, customer_labels, order_labels, manifest)

    return clean_tables | feature_tables | labels | {"defect_manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EcomPolice data.")
    parser.add_argument("--customers", type=int, default=3_000)
    parser.add_argument("--products", type=int, default=500)
    parser.add_argument("--orders", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(message)s")
    config = replace(
        SyntheticConfig(),
        n_customers=args.customers,
        n_products=args.products,
        n_orders=args.orders,
        seed=args.seed,
        months=args.months,
    )
    outputs = generate(config)
    print("Synthetic dataset generated.")
    for name, df in outputs.items():
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()

