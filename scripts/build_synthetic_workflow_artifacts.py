"""Build reports and notebooks for the synthetic-only EcomPolice workflow."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks" / "synthetic_pipeline"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


COMMON_SETUP = r"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "src").exists() and (path / "data").exists():
            return path
    raise RuntimeError("Run this notebook from inside the EcomPolice project.")

PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CLEAN_DIR = PROJECT_ROOT / "data" / "synthetic" / "clean"
MESSY_DIR = PROJECT_ROOT / "data" / "synthetic" / "messy"
LABELS_DIR = PROJECT_ROOT / "data" / "synthetic" / "labels"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "synthetic" / "manifests"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_tables(base_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(base_dir / "customers.csv"),
        "products": pd.read_csv(base_dir / "products.csv"),
        "orders": pd.read_csv(base_dir / "orders.csv"),
        "deliveries": pd.read_csv(base_dir / "deliveries.csv"),
        "returns": pd.read_csv(base_dir / "returns.csv"),
    }
"""


def build_reports() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_files = sorted((SYNTHETIC_DIR / "clean").glob("*.csv"))
    messy_files = sorted((SYNTHETIC_DIR / "messy").glob("*.csv"))
    label_files = sorted((SYNTHETIC_DIR / "labels").glob("*.csv"))
    manifest_files = sorted((SYNTHETIC_DIR / "manifests").glob("*"))
    archived = sorted((PROJECT_ROOT / "archive" / "kaggle_workflow").glob("**/*"))
    archived_files = [p for p in archived if p.is_file()]

    def shape_line(path: Path) -> str:
        if path.suffix.lower() != ".csv":
            return f"- `{path.relative_to(PROJECT_ROOT)}`"
        df = pd.read_csv(path)
        return f"- `{path.relative_to(PROJECT_ROOT)}`: {df.shape[0]:,} rows x {df.shape[1]:,} columns"

    migration = [
        "# Migration Report",
        "## What Was Found",
        "The active project contained a complete synthetic-data generator under `src/synthetic_data/` and generated synthetic outputs under `data/synthetic/`. The active `README.md` still described the old Kaggle CSV workflow, while Kaggle-specific notebooks, scripts, tests, reports, and processed CSVs were already located under `archive/kaggle_workflow/`.",
        "## Kaggle Workflow Archived",
        "\n".join(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in archived_files) or "- No archived Kaggle files found.",
        "## Synthetic Files Reused",
        "\n".join(shape_line(p) for p in clean_files + messy_files + label_files) + "\n" + "\n".join(shape_line(p) for p in manifest_files),
        "## What Needed Generation Or Fixing",
        "- No regeneration was required because the expected clean, messy, label, and manifest files were present at the standard 20,000-order scale.\n- Active documentation needed to be rewritten for the synthetic-only workflow.\n- Teacher-ready notebooks and active reports needed to be created.\n- Tests needed a safety fix so temporary generation does not overwrite the main synthetic outputs.",
    ]
    (REPORTS_DIR / "migration_report.md").write_text("\n\n".join(migration) + "\n", encoding="utf-8")

    customers = pd.read_csv(SYNTHETIC_DIR / "clean" / "customers.csv")
    products = pd.read_csv(SYNTHETIC_DIR / "clean" / "products.csv")
    orders = pd.read_csv(SYNTHETIC_DIR / "clean" / "orders.csv")
    deliveries = pd.read_csv(SYNTHETIC_DIR / "clean" / "deliveries.csv")
    returns = pd.read_csv(SYNTHETIC_DIR / "clean" / "returns.csv")
    labels = pd.read_csv(SYNTHETIC_DIR / "labels" / "order_ground_truth.csv")
    manifest = pd.read_csv(SYNTHETIC_DIR / "manifests" / "defect_manifest.csv")

    order_dates = pd.to_datetime(orders["Order_Date"])
    dispatch_dates = pd.to_datetime(deliveries.set_index("Order_ID").loc[orders["Order_ID"], "Dispatch_Date"].to_numpy())
    delivery_dates = pd.to_datetime(deliveries.set_index("Order_ID").loc[orders["Order_ID"], "Delivery_Date"].to_numpy(), errors="coerce")
    return_dates = pd.to_datetime(returns.set_index("Order_ID").loc[orders["Order_ID"], "Return_Date"].to_numpy(), errors="coerce")
    validation_lines = [
        "# Synthetic Data Validation Report",
        "## Schema And Volume",
        "\n".join(shape_line(p) for p in clean_files + label_files + [SYNTHETIC_DIR / "manifests" / "defect_manifest.csv"]),
        "## Integrity Checks",
        f"- Orders with valid customers: {orders['Customer_ID'].isin(customers['Customer_ID']).all()}",
        f"- Orders with valid products: {orders['Product_ID'].isin(products['Product_ID']).all()}",
        f"- Deliveries with valid orders: {deliveries['Order_ID'].isin(orders['Order_ID']).all()}",
        f"- Returns with valid orders: {returns['Order_ID'].isin(orders['Order_ID']).all()}",
        f"- Duplicate order IDs in clean orders: {orders['Order_ID'].duplicated().sum()}",
        "## Chronology Checks",
        f"- Dispatch before order rows: {(dispatch_dates < order_dates).sum()}",
        f"- Delivery before dispatch rows: {(delivery_dates < dispatch_dates).sum()}",
        f"- Return before delivery rows: {(return_dates.notna() & delivery_dates.notna() & (return_dates < delivery_dates)).sum()}",
        "## Label Separation",
        f"- Order labels exist for all orders: {orders['Order_ID'].isin(labels['Order_ID']).all()}",
        "- Scenario labels and simulated abuse indicators are stored in `data/synthetic/labels/`, not in active feature inputs.",
        "## Messy Dataset",
        f"- Defect manifest rows: {len(manifest):,}",
        "- The messy dataset is a controlled copy for quality analysis and preprocessing practice.",
    ]
    (REPORTS_DIR / "synthetic_data_validation_report.md").write_text("\n\n".join(validation_lines) + "\n", encoding="utf-8")


def build_notebooks() -> None:
    overview_cells = [
        markdown_cell("# 01 Synthetic Data Overview\n\nThis notebook introduces the synthetic EcomPolice dataset and verifies that the active workflow uses generated data only."),
        code_cell(COMMON_SETUP + "\nclean_tables = load_tables(CLEAN_DIR)\ncustomer_labels = pd.read_csv(LABELS_DIR / 'customer_ground_truth.csv')\norder_labels = pd.read_csv(LABELS_DIR / 'order_ground_truth.csv')\nprint(PROJECT_ROOT)"),
        markdown_cell("## Why Synthetic Data?\n\nThe project studies return behaviour without using real customer records or the old Kaggle workflow. Scenario labels are generated assumptions for academic experimentation; they are not verified real-world fraud labels."),
        code_cell("for name, df in clean_tables.items():\n    display(pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str).values}))\n    print(f'{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns')\n    display(df.head())"),
        markdown_cell("## Data Dictionary\n\nThese fields are the main columns used in the following notebooks."),
        code_cell("dictionary = pd.DataFrame([\n    ('Customer_ID', 'Stable synthetic customer identifier'),\n    ('Product_ID', 'Stable synthetic product identifier'),\n    ('Order_ID', 'Stable synthetic order identifier and analytical grain'),\n    ('Return_Status', 'Observed return outcome such as Not Returned, Returned, Refunded, or Exchange Issued'),\n    ('Scenario_Label', 'Generator-defined behavioural scenario stored separately from features'),\n    ('Simulated_Abuse_Indicator', 'Artificial label for future supervised experiments, not a real fraud finding'),\n], columns=['Column', 'Meaning'])\ndisplay(dictionary)"),
        code_cell("summary = []\nfor name, df in clean_tables.items():\n    id_cols = [col for col in df.columns if col.endswith('_ID')]\n    summary.append({'table': name, 'rows': len(df), 'columns': df.shape[1], 'id_columns': ', '.join(id_cols), 'missing_cells': int(df.isna().sum().sum())})\ndisplay(pd.DataFrame(summary))"),
        code_cell("customers, products, orders, deliveries, returns = [clean_tables[k] for k in ['customers','products','orders','deliveries','returns']]\nchecks = {\n    'orders_have_valid_customers': orders['Customer_ID'].isin(customers['Customer_ID']).all(),\n    'orders_have_valid_products': orders['Product_ID'].isin(products['Product_ID']).all(),\n    'deliveries_have_valid_orders': deliveries['Order_ID'].isin(orders['Order_ID']).all(),\n    'returns_have_valid_orders': returns['Order_ID'].isin(orders['Order_ID']).all(),\n    'order_labels_for_all_orders': orders['Order_ID'].isin(order_labels['Order_ID']).all(),\n}\ndisplay(pd.Series(checks, name='passed'))"),
        code_cell("return_summary = returns['Return_Status'].value_counts(dropna=False).rename_axis('Return_Status').reset_index(name='orders')\nscenario_summary = order_labels['Scenario_Label'].value_counts().rename_axis('Scenario_Label').reset_index(name='orders')\ndisplay(return_summary)\ndisplay(scenario_summary)"),
        markdown_cell("## Return Status vs Scenario Labels\n\n`Return_Status` is an event outcome in the return table. Scenario labels are generator-defined ground truth kept separately for future modelling experiments. A returned order is not automatically suspicious."),
        markdown_cell("## Conclusion\n\nThe active data source is the synthetic entity set. The clean tables have stable identifiers, separate labels, and clear joins for later analysis."),
    ]
    write_notebook(NOTEBOOK_DIR / "01_synthetic_data_overview.ipynb", overview_cells)

    dq_cells = [
        markdown_cell("# 02 Data Quality Analysis\n\nThis notebook inspects the messy synthetic copy and the defect manifest before any correction is applied."),
        code_cell(COMMON_SETUP + "\nmessy_tables = load_tables(MESSY_DIR)\nmanifest = pd.read_csv(MANIFESTS_DIR / 'defect_manifest.csv')"),
        code_cell("for name, df in messy_tables.items():\n    missing = df.isna().sum().rename('missing_count').to_frame()\n    missing['missing_pct'] = (missing['missing_count'] / len(df) * 100).round(2)\n    print(f'\\n{name}')\n    display(missing[missing['missing_count'] > 0].sort_values('missing_count', ascending=False))"),
        code_cell("duplicate_summary = pd.DataFrame([{'table': name, 'exact_duplicate_rows': int(df.duplicated().sum())} for name, df in messy_tables.items()])\ndisplay(duplicate_summary)"),
        code_cell("orders = messy_tables['orders'].copy()\nfor col in ['Order_Date']:\n    parsed = pd.to_datetime(orders[col], errors='coerce')\n    dayfirst = pd.to_datetime(orders.loc[parsed.isna(), col], errors='coerce', dayfirst=True)\n    print(col, 'initial failures:', int(parsed.isna().sum()), 'recoverable day-first:', int(dayfirst.notna().sum()))"),
        code_cell("numeric_cols = ['Quantity', 'Unit_Price_INR', 'Discount_Pct', 'Order_Value_INR']\nconverted = orders[numeric_cols].apply(lambda s: pd.to_numeric(s.astype(str).str.replace('INR', '', regex=False).str.strip(), errors='coerce'))\ndisplay(converted.describe())\ndisplay(converted[(converted[['Quantity','Unit_Price_INR']] <= 0).any(axis=1)].head())"),
        code_cell("category_variants = {\n    'Payment_Method': sorted(orders['Payment_Method'].dropna().astype(str).unique()),\n    'Shipping_Method': sorted(orders['Shipping_Method'].dropna().astype(str).unique()),\n}\ndisplay(category_variants)"),
        code_cell("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\nmanifest['Defect_Type'].value_counts().plot(kind='bar', ax=axes[0], title='Injected Defect Types')\naxes[0].set_xlabel('Defect type')\naxes[0].set_ylabel('Count')\nmanifest['Table'].value_counts().plot(kind='bar', ax=axes[1], title='Defects by Table')\naxes[1].set_xlabel('Table')\naxes[1].set_ylabel('Count')\nplt.tight_layout()\nplt.show()"),
        code_cell("clean_orders = pd.read_csv(CLEAN_DIR / 'orders.csv')\nbroken_customers = ~orders['Customer_ID'].isin(pd.read_csv(CLEAN_DIR / 'customers.csv')['Customer_ID'])\nbroken_products = ~orders['Product_ID'].isin(pd.read_csv(CLEAN_DIR / 'products.csv')['Product_ID'])\nquality_findings = pd.Series({\n    'messy_order_rows': len(orders),\n    'clean_order_rows': len(clean_orders),\n    'broken_customer_references': int(broken_customers.sum()),\n    'broken_product_references': int(broken_products.sum()),\n    'manifest_rows': len(manifest),\n})\ndisplay(quality_findings)"),
        markdown_cell("## Assessment\n\nExact duplicate rows can be removed. Unambiguous spelling/case variations can be standardized. Missing return dates for non-returned orders are structural and should not be treated as defects. Invalid identifiers, impossible negative values, chronology failures, and extreme outliers should be flagged or excluded from model-ready tables unless the original value can be recovered from a trusted source."),
    ]
    write_notebook(NOTEBOOK_DIR / "02_data_quality_analysis.ipynb", dq_cells)

    eda_cells = [
        markdown_cell("# 03 Exploratory Data Analysis\n\nThis notebook studies generated customer and order behaviour before modelling. The results describe synthetic assumptions, not real-world fraud patterns."),
        code_cell(COMMON_SETUP + "\nclean_tables = load_tables(CLEAN_DIR)\nlabels = pd.read_csv(LABELS_DIR / 'order_ground_truth.csv')\norders = clean_tables['orders']; products = clean_tables['products']; returns = clean_tables['returns']; deliveries = clean_tables['deliveries']; customers = clean_tables['customers']\norder_level = orders.merge(products, on='Product_ID').merge(returns, on='Order_ID').merge(deliveries, on='Order_ID').merge(labels, on=['Order_ID','Customer_ID'])\norder_level['Order_Date'] = pd.to_datetime(order_level['Order_Date'])\norder_level['Delivery_Date'] = pd.to_datetime(order_level['Delivery_Date'])\norder_level['Return_Date'] = pd.to_datetime(order_level['Return_Date'], errors='coerce')\norder_level['Is_Returned'] = order_level['Return_Status'].ne('Not Returned')"),
        code_cell("monthly = order_level.set_index('Order_Date').resample('ME')['Order_ID'].count()\nax = monthly.plot(figsize=(10,4), marker='o', title='Orders Over Time')\nax.set_xlabel('Month')\nax.set_ylabel('Order count')\nplt.show()\ndisplay(monthly.to_frame('orders').tail())"),
        code_cell("fig, axes = plt.subplots(1, 2, figsize=(13, 4))\nsns.countplot(data=order_level, y='Category', order=order_level['Category'].value_counts().index, ax=axes[0])\naxes[0].set_title('Product Category Distribution'); axes[0].set_xlabel('Orders'); axes[0].set_ylabel('Category')\nsns.histplot(order_level['Order_Value_INR'], bins=40, ax=axes[1])\naxes[1].set_title('Order Value Distribution'); axes[1].set_xlabel('Order value INR'); axes[1].set_ylabel('Orders')\nplt.tight_layout(); plt.show()"),
        code_cell("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\nsns.countplot(data=order_level, y='Payment_Method', order=order_level['Payment_Method'].value_counts().index, ax=axes[0])\naxes[0].set_title('Payment Methods'); axes[0].set_xlabel('Orders'); axes[0].set_ylabel('Payment method')\nsns.countplot(data=order_level, y='Shipping_Method', order=order_level['Shipping_Method'].value_counts().index, ax=axes[1])\naxes[1].set_title('Shipping Methods'); axes[1].set_xlabel('Orders'); axes[1].set_ylabel('Shipping method')\nplt.tight_layout(); plt.show()"),
        code_cell("category_return = order_level.groupby('Category')['Is_Returned'].mean().sort_values(ascending=False).mul(100).rename('return_rate_pct')\nax = category_return.plot(kind='bar', figsize=(10,4), title='Return Rate by Product Category')\nax.set_xlabel('Category'); ax.set_ylabel('Return rate (%)')\nplt.show()\ndisplay(category_return.round(2))"),
        code_cell("returned = order_level[order_level['Is_Returned']].copy()\nfig, axes = plt.subplots(1, 2, figsize=(13, 4))\nsns.countplot(data=returned, y='Return_Reason', order=returned['Return_Reason'].value_counts().index, ax=axes[0])\naxes[0].set_title('Return Reasons'); axes[0].set_xlabel('Returned orders'); axes[0].set_ylabel('Reason')\nreturned['Delivery_Delay_Days'] = (returned['Delivery_Date'] - pd.to_datetime(returned['Dispatch_Date'])).dt.days\nreturned['Return_Duration_Days'] = (returned['Return_Date'] - returned['Delivery_Date']).dt.days\nsns.histplot(returned['Return_Duration_Days'], bins=30, ax=axes[1])\naxes[1].set_title('Days from Delivery to Return'); axes[1].set_xlabel('Days'); axes[1].set_ylabel('Returned orders')\nplt.tight_layout(); plt.show()"),
        code_cell("customer_behaviour = order_level.groupby('Customer_ID').agg(order_count=('Order_ID','count'), return_count=('Is_Returned','sum'), scenario=('Scenario_Label','first'))\ncustomer_behaviour['return_rate'] = customer_behaviour['return_count'] / customer_behaviour['order_count']\ndisplay(customer_behaviour.describe())\ndisplay(customer_behaviour.groupby('scenario')[['order_count','return_count','return_rate']].mean().round(3))"),
        code_cell("scenario_rates = order_level.groupby('Scenario_Label')['Is_Returned'].mean().sort_values(ascending=False).mul(100)\nax = scenario_rates.plot(kind='bar', figsize=(11,4), title='Return Rate by Generator-Defined Scenario')\nax.set_xlabel('Scenario label'); ax.set_ylabel('Return rate (%)')\nplt.xticks(rotation=35, ha='right'); plt.tight_layout(); plt.show()"),
        markdown_cell("## Interpretation\n\nScenario groups differ because the generator intentionally created behavioural patterns. These differences are useful for demonstrating analysis and leakage-aware preprocessing, but they are not evidence that similar real customers are abusive. High return rate alone should be treated as a signal for review, not proof."),
    ]
    write_notebook(NOTEBOOK_DIR / "03_exploratory_data_analysis.ipynb", eda_cells)

    prep_cells = [
        markdown_cell("# 04 Data Preprocessing\n\nThis notebook cleans the messy synthetic copy while preserving the original input files. The final grain is one row per order."),
        code_cell(COMMON_SETUP + "\nfrom synthetic_data.preprocessing import SyntheticTables, build_clean_order_dataset, preprocess_messy_tables, quality_summary\nmessy = load_tables(MESSY_DIR)\nmanifest = pd.read_csv(MANIFESTS_DIR / 'defect_manifest.csv')\nraw_tables = SyntheticTables(**messy)"),
        code_cell("before_orders = messy['orders'].copy()\nprint('Original messy order rows:', len(before_orders))\ndisplay(manifest['Defect_Type'].value_counts().rename_axis('defect_type').reset_index(name='count'))"),
        code_cell("cleaned_tables, action_log = preprocess_messy_tables(raw_tables)\ncleaned_orders = build_clean_order_dataset(cleaned_tables)\ndisplay(action_log)\ndisplay(quality_summary(cleaned_orders).to_frame('after_cleaning'))"),
        code_cell("before_summary = pd.Series({\n    'rows': len(before_orders),\n    'unique_orders': before_orders['Order_ID'].nunique(dropna=True),\n    'duplicate_order_ids': int(before_orders['Order_ID'].duplicated().sum()),\n    'missing_customer_ids': int(before_orders['Customer_ID'].isna().sum() + before_orders['Customer_ID'].astype(str).str.strip().eq('').sum()),\n    'missing_product_ids': int(before_orders['Product_ID'].isna().sum() + before_orders['Product_ID'].astype(str).str.strip().eq('').sum()),\n})\nafter_summary = quality_summary(cleaned_orders)\ncomparison = pd.concat([before_summary.rename('before_preprocessing'), after_summary.rename('after_preprocessing')], axis=1)\ndisplay(comparison)"),
        code_cell("invalid_rows = cleaned_orders[\n    cleaned_orders.get('Invalid_Identifier_Flag', False)\n    | cleaned_orders.get('Invalid_Numeric_Flag', False)\n    | cleaned_orders.get('Invalid_Return_Chronology_Flag', False)\n    | cleaned_orders.get('Suspicious_Outlier_Flag', False)\n]\ndisplay(invalid_rows[['Order_ID','Customer_ID','Product_ID','Invalid_Identifier_Flag','Invalid_Numeric_Flag','Invalid_Return_Chronology_Flag','Suspicious_Outlier_Flag']].head(10))"),
        code_cell("cleaned_orders.to_csv(PROCESSED_DIR / 'cleaned_orders.csv', index=False)\nreport = [\n    '# Preprocessing Report',\n    '## Strategy',\n    '- Original messy files were loaded from `data/synthetic/messy/` and were not modified in place.',\n    '- Column names were standardized, mixed dates and numeric-looking strings were parsed, and unambiguous category variants were normalized.',\n    '- Exact duplicate order rows were removed, while invalid identifiers, invalid numeric values, chronology issues, and outliers were retained with quality flags where possible.',\n    '- The consolidated dataset uses one row per order and avoids accidental row multiplication from joins.',\n    '## Quality Metrics',\n    '```text\\n' + comparison.to_string() + '\\n```',\n    '## Known Limits',\n    '- When an injected defect destroys the original value, this notebook flags the issue instead of claiming perfect recovery.',\n]\n(REPORTS_DIR / 'preprocessing_report.md').write_text('\\n\\n'.join(report), encoding='utf-8')\nprint('Saved:', PROCESSED_DIR / 'cleaned_orders.csv')"),
        markdown_cell("## Summary\n\nThe cleaned analytical dataset preserves one row per order. Quality flags make questionable records visible for future decisions instead of silently deleting all unusual values."),
    ]
    write_notebook(NOTEBOOK_DIR / "04_data_preprocessing.ipynb", prep_cells)

    feature_cells = [
        markdown_cell("# 05 Feature Engineering and Model-Ready Data\n\nThis notebook prepares leakage-aware features for a future return-level simulated abuse detection phase. It does not train or evaluate any model."),
        code_cell(COMMON_SETUP + "\nfrom synthetic_data.build_features import build_order_level_table, build_pre_return_features\nfrom synthetic_data.preprocessing import build_model_targets\nclean = load_tables(CLEAN_DIR)\nlabels = pd.read_csv(LABELS_DIR / 'order_ground_truth.csv')"),
        markdown_cell("## Prediction Point\n\nThe main future task is return-level simulated abuse detection. Pre-return features are calculated as of the current order date and use only older customer history. Post-return investigation fields such as return date, return reason, refund amount, and scenario labels are excluded from `X`."),
        code_cell("order_level = build_order_level_table(clean['customers'], clean['products'], clean['orders'], clean['deliveries'], clean['returns'])\nfeatures = build_pre_return_features(order_level)\ntargets = build_model_targets(labels)\nfeatures['Order_Date'] = pd.to_datetime(features['Order_Date'])\ndisplay(features.head())\ndisplay(targets.head())"),
        code_cell("forbidden = {'Scenario_Label','Simulated_Abuse_Indicator','Return_Date','Return_Reason','Return_Status','Refund_Amount_INR'}\nleakage_check = pd.Series({\n    'forbidden_columns_in_features': sorted(forbidden.intersection(features.columns)),\n    'feature_rows': len(features),\n    'target_rows': len(targets),\n    'order_id_alignment': set(features['Order_ID']) == set(targets['Order_ID']),\n})\ndisplay(leakage_check)"),
        code_cell("violations = []\nfor customer_id, group in features.sort_values(['Customer_ID','Order_Date','Order_ID']).groupby('Customer_ID'):\n    for _, row in group.iterrows():\n        prior = group[group['Order_Date'] < row['Order_Date']]\n        if row['Customer_Order_Count_Before'] != len(prior):\n            violations.append(row['Order_ID'])\n            break\nprint('Chronological feature violations:', len(violations))"),
        code_cell("pre_return_feature_columns = [col for col in features.columns if col not in ['Order_ID', 'Customer_ID', 'Product_ID', 'Order_Date']]\ndata_dictionary = pd.DataFrame({'column': features.columns})\ndata_dictionary['role'] = np.where(data_dictionary['column'].isin(['Order_ID','Customer_ID','Product_ID']), 'stable identifier', 'pre-return feature')\ndata_dictionary.loc[data_dictionary['column'].eq('Order_Date'), 'role'] = 'prediction timestamp'\ndisplay(data_dictionary)"),
        code_cell("features.to_csv(PROCESSED_DIR / 'pre_return_features.csv', index=False)\ntargets.to_csv(PROCESSED_DIR / 'model_targets.csv', index=False)\ndata_dictionary.to_csv(PROCESSED_DIR / 'feature_data_dictionary.csv', index=False)\nreport = [\n    '# Feature Engineering Report',\n    '## Prediction Point',\n    'Features are available at the current order date before the current return outcome is known.',\n    '## Leakage Controls',\n    f'- Forbidden columns in features: {sorted(forbidden.intersection(features.columns))}',\n    f'- Feature-target order alignment: {set(features[\"Order_ID\"]) == set(targets[\"Order_ID\"])}',\n    f'- Chronological feature violations found: {len(violations)}',\n    '## Outputs',\n    '- `data/processed/pre_return_features.csv`',\n    '- `data/processed/model_targets.csv`',\n    '- `data/processed/feature_data_dictionary.csv`',\n    '## Limitations',\n    '- Labels are generator-defined simulated scenarios, not real verified fraud labels.',\n    '- No model training is performed in this phase.',\n]\n(REPORTS_DIR / 'feature_engineering_report.md').write_text('\\n\\n'.join(report), encoding='utf-8')\nprint('Saved feature outputs and report.')"),
        markdown_cell("## Summary\n\nThe model-ready files keep `X` and `y` separate. Historical customer features use only earlier transactions, and post-return information remains outside the pre-return feature set."),
    ]
    write_notebook(NOTEBOOK_DIR / "05_feature_engineering_and_model_ready_data.ipynb", feature_cells)


def main() -> None:
    build_reports()
    build_notebooks()


if __name__ == "__main__":
    main()
