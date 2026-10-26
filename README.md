# EcomPolice

EcomPolice is an academic data-science project for studying synthetic e-commerce return behaviour. The active workflow now uses only generated synthetic data and intentionally avoids the previous Kaggle dataset.

This phase covers data understanding, data-quality analysis, exploratory data analysis, preprocessing, and leakage-aware feature preparation. It does not train models, score fraud, build a dashboard, create an API, or treat returned orders as fraudulent by default.

## Active Synthetic Workflow

Run the notebooks in this order:

1. `notebooks/synthetic_pipeline/01_synthetic_data_overview.ipynb`
2. `notebooks/synthetic_pipeline/02_data_quality_analysis.ipynb`
3. `notebooks/synthetic_pipeline/03_exploratory_data_analysis.ipynb`
4. `notebooks/synthetic_pipeline/04_data_preprocessing.ipynb`
5. `notebooks/synthetic_pipeline/05_feature_engineering_and_model_ready_data.ipynb`

The old Kaggle workflow is preserved for reference under `archive/kaggle_workflow/` and is not part of the active analysis path.

## Project Structure

```text
data/
  synthetic/
    clean/       Valid generated entity tables.
    messy/       Generated copy with controlled data-quality defects.
    labels/      Separate generator-defined scenario labels.
    manifests/   Defect manifest, data dictionary, and generation config.
  processed/     Outputs produced by preprocessing and feature notebooks.
notebooks/
  synthetic_pipeline/  Teacher-ready notebook series.
reports/        Migration, validation, preprocessing, and feature reports.
src/
  synthetic_data/      Synthetic generator, validation, preprocessing, and feature helpers.
archive/
  kaggle_workflow/     Retired Kaggle-specific workflow artifacts.
tests/          Unit tests for generation, validation, preprocessing, and leakage checks.
```

## Synthetic Data

The synthetic dataset contains fictional Indian customers, products, orders, deliveries, returns, separate scenario labels, and a defect manifest. Scenario labels are simulated ground truth from the generator and must not be used as input features.

Expected files:

- `data/synthetic/clean/customers.csv`
- `data/synthetic/clean/products.csv`
- `data/synthetic/clean/orders.csv`
- `data/synthetic/clean/deliveries.csv`
- `data/synthetic/clean/returns.csv`
- `data/synthetic/messy/*.csv`
- `data/synthetic/labels/customer_ground_truth.csv`
- `data/synthetic/labels/order_ground_truth.csv`
- `data/synthetic/manifests/defect_manifest.csv`

Regenerate only when these files are missing or intentionally stale:

```bash
PYTHONPATH=src python -m synthetic_data.run
```

Optional size and seed overrides:

```bash
PYTHONPATH=src python -m synthetic_data.run --customers 3000 --products 500 --orders 20000 --seed 42 --months 12
```

## VS Code Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, install or enable the Python and Jupyter extensions. Open the first notebook, choose **Select Kernel**, and pick the `.venv` Python interpreter from this project.

## Outputs

The preprocessing and feature notebooks write:

- `data/processed/cleaned_orders.csv`
- `data/processed/pre_return_features.csv`
- `data/processed/model_targets.csv`
- `data/processed/feature_data_dictionary.csv`
- `reports/preprocessing_report.md`
- `reports/feature_engineering_report.md`

Project-level reports:

- `reports/migration_report.md`
- `reports/synthetic_data_validation_report.md`
- `reports/synthetic_data_report.md`

## Tests

Run tests from the project root:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

The tests cover clean synthetic data validation, chronological constraints, defect injection, preprocessing behaviour, and leakage prevention for feature construction.
