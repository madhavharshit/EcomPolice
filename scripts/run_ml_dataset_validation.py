"""Audit the synthetic ML dataset and design a grouped evaluation experiment without training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
TARGET_COLUMN = "Simulated_Abuse_Target"
IDENTIFIER_COLUMNS = ["Order_ID", "Customer_ID", "Return_Request_Date"]


def load_return_level_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load separate feature and target tables and verify their stable-ID alignment."""
    processed = PROJECT_ROOT / "data" / "processed"
    features = pd.read_csv(processed / "return_level_features.csv", parse_dates=["Return_Request_Date"])
    targets = pd.read_csv(processed / "return_level_targets.csv", parse_dates=["Return_Request_Date"])
    if features["Order_ID"].duplicated().any() or targets["Order_ID"].duplicated().any():
        raise ValueError("Return-level IDs must be unique before experiment design.")
    merged = features.merge(
        targets[["Order_ID", "Customer_ID", "Return_Request_Date", "Scenario_Label", TARGET_COLUMN]],
        on=["Order_ID", "Customer_ID", "Return_Request_Date"],
        how="left",
        validate="1:1",
    )
    if merged[TARGET_COLUMN].isna().any():
        raise ValueError("Every feature row must align with a target row.")
    return features, merged


def feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return model-safe numeric, categorical, and excluded feature columns."""
    excluded = IDENTIFIER_COLUMNS.copy()
    model_features = features.drop(columns=excluded)
    numeric = model_features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = model_features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    unknown = sorted(set(model_features.columns) - set(numeric) - set(categorical))
    if unknown:
        raise ValueError(f"Unexpected feature dtypes require an explicit plan: {unknown}")
    return numeric, categorical, excluded


def build_grouped_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible 60/20/20 customer-grouped stratified assignments."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    assignments = pd.Series(index=dataset.index, dtype="object")
    folds = list(splitter.split(dataset, dataset[TARGET_COLUMN], groups=dataset["Customer_ID"]))
    for fold_number, (_, fold_indices) in enumerate(folds):
        partition = "test" if fold_number == 0 else "validation" if fold_number == 1 else "train"
        assignments.loc[fold_indices] = partition
    split_data = dataset[["Order_ID", "Customer_ID", TARGET_COLUMN]].copy()
    split_data["Split"] = assignments
    summary = (
        split_data.groupby("Split", observed=True)
        .agg(
            records=("Order_ID", "size"),
            customers=("Customer_ID", "nunique"),
            positive_records=(TARGET_COLUMN, "sum"),
            positive_rate=(TARGET_COLUMN, "mean"),
        )
        .reindex(["train", "validation", "test"])
        .reset_index()
    )
    summary["positive_rate"] = summary["positive_rate"].round(4)
    return split_data, summary


def build_audit_tables(features: pd.DataFrame, dataset: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Calculate data, target, redundancy, and generation-mechanism diagnostics."""
    numeric, categorical, excluded = feature_columns(features)
    model_input = features.drop(columns=excluded)
    missing = pd.DataFrame(
        {
            "feature": model_input.columns,
            "missing_count": model_input.isna().sum().values,
            "missing_pct": (model_input.isna().mean().mul(100).round(3)).values,
            "dtype": model_input.dtypes.astype(str).values,
        }
    )
    uniqueness = pd.DataFrame(
        {
            "feature": model_input.columns,
            "unique_values": model_input.nunique(dropna=False).values,
            "dominant_value_share": [model_input[column].value_counts(dropna=False, normalize=True).iloc[0] for column in model_input.columns],
        }
    )
    constants = uniqueness.loc[uniqueness["unique_values"].eq(1)].copy()
    near_constants = uniqueness.loc[uniqueness["dominant_value_share"].ge(0.99)].copy()

    corr = model_input[numeric].astype(float).corr(method="spearman")
    pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(numeric):
        for right in numeric[left_index + 1 :]:
            value = corr.loc[left, right]
            if abs(value) >= 0.80:
                pairs.append({"feature_a": left, "feature_b": right, "spearman_correlation": round(float(value), 4)})
    high_corr = pd.DataFrame(pairs).sort_values("spearman_correlation", key=lambda values: values.abs(), ascending=False) if pairs else pd.DataFrame(columns=["feature_a", "feature_b", "spearman_correlation"])

    target_distribution = dataset[TARGET_COLUMN].value_counts().sort_index().rename_axis(TARGET_COLUMN).reset_index(name="return_requests")
    target_distribution["percentage"] = (target_distribution["return_requests"] / len(dataset) * 100).round(2)
    customer_distribution = (
        dataset.groupby(TARGET_COLUMN)["Customer_ID"].nunique().rename("unique_customers").reset_index()
    )
    labels_per_customer = dataset.groupby("Customer_ID")[TARGET_COLUMN].nunique()
    mixed_label_customers = int(labels_per_customer.gt(1).sum())

    numeric_association = pd.DataFrame(
        [
            {"feature": column, "point_biserial_like_pearson": round(float(pd.to_numeric(model_input[column]).corr(dataset[TARGET_COLUMN])), 4)}
            for column in numeric
        ]
    ).assign(abs_association=lambda frame: frame["point_biserial_like_pearson"].abs()).sort_values("abs_association", ascending=False)
    categorical_association_rows = []
    for column in categorical:
        rates = dataset.groupby(features[column], dropna=False)[TARGET_COLUMN].agg(["size", "mean"])
        supported = rates.loc[rates["size"].ge(10)]
        categorical_association_rows.append(
            {
                "feature": column,
                "levels": int(rates.shape[0]),
                "supported_target_rate_min": round(float(supported["mean"].min()), 4),
                "supported_target_rate_max": round(float(supported["mean"].max()), 4),
                "supported_target_rate_range": round(float(supported["mean"].max() - supported["mean"].min()), 4),
            }
        )
    categorical_association = pd.DataFrame(categorical_association_rows).sort_values("supported_target_rate_range", ascending=False)
    overview = pd.Series(
        {
            "records": len(features),
            "model_input_features": model_input.shape[1],
            "numeric_features": len(numeric),
            "categorical_features": len(categorical),
            "duplicate_feature_ids": int(features["Order_ID"].duplicated().sum()),
            "feature_target_id_alignment": bool(set(features["Order_ID"]) == set(dataset["Order_ID"])),
            "target_or_scenario_columns_in_features": sorted(set([TARGET_COLUMN, "Scenario_Label", "Simulated_Abuse_Indicator"]).intersection(features.columns)),
            "customers_with_mixed_inherited_labels": mixed_label_customers,
        }
    )
    return {
        "overview": overview,
        "missing": missing,
        "constants": constants,
        "near_constants": near_constants,
        "high_corr": high_corr,
        "target_distribution": target_distribution,
        "customer_distribution": customer_distribution,
        "numeric_association": numeric_association,
        "categorical_association": categorical_association,
        "numeric_columns": pd.DataFrame({"numeric_feature": numeric}),
        "categorical_columns": pd.DataFrame({"categorical_feature": categorical}),
    }


def validate_grouped_split(assignments: pd.DataFrame) -> dict[str, int | bool]:
    """Confirm customer isolation, non-empty partitions, and two target classes per split."""
    split_per_customer = assignments.groupby("Customer_ID")["Split"].nunique()
    class_counts = assignments.groupby("Split")[TARGET_COLUMN].nunique()
    return {
        "all_customers_in_one_partition": bool(split_per_customer.eq(1).all()),
        "customers_shared_across_partitions": int(split_per_customer.gt(1).sum()),
        "all_partitions_non_empty": bool(assignments["Split"].value_counts().reindex(["train", "validation", "test"], fill_value=0).gt(0).all()),
        "both_target_classes_in_each_partition": bool(class_counts.reindex(["train", "validation", "test"], fill_value=0).eq(2).all()),
    }


def write_reports(audit: dict[str, pd.DataFrame | pd.Series], split_summary: pd.DataFrame, split_validation: dict[str, int | bool]) -> None:
    """Write college-readable audit and experiment design Markdown reports."""
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    overview = audit["overview"]
    validation_report = [
        "# ML Dataset Validation Report",
        "## Dataset Overview",
        "```text\n" + overview.to_string() + "\n```",
        "## Target Distribution",
        "```text\n" + audit["target_distribution"].to_string(index=False) + "\n```",
        "## Customers by Target Class",
        "```text\n" + audit["customer_distribution"].to_string(index=False) + "\n```",
        "## Missing Values by Model Feature",
        "```text\n" + audit["missing"].to_string(index=False) + "\n```",
        "## Constant and Near-Constant Features",
        "Constant features:\n```text\n" + audit["constants"].to_string(index=False) + "\n```\n\nNear-constant features (dominant value at least 99%):\n```text\n" + audit["near_constants"].to_string(index=False) + "\n```",
        "## Highly Correlated Numerical Features",
        "```text\n" + audit["high_corr"].to_string(index=False) + "\n```",
        "## Potential Generation-Mechanism Signals",
        "The following are association diagnostics, not model results. High association can indicate a synthetic generator pattern rather than a useful real-world signal. In particular, current `Return_Reason`, `Days_Since_Delivery`, historical defect-claim behaviour, and high-value-return history are scenario-influenced and require careful ablation in a later modelling phase.",
        "Numeric associations with the simulated target:\n```text\n" + audit["numeric_association"].to_string(index=False) + "\n```\n\nCategorical target-rate ranges for levels with at least 10 records:\n```text\n" + audit["categorical_association"].to_string(index=False) + "\n```",
        "## Target Limitation",
        "Every return of a customer inherits the same generator scenario label, so customers do not have differing inherited labels across returns. This makes customer grouping mandatory and means some features may reflect the generation mechanism. The target is not verified fraud or abuse.",
    ]
    (reports / "ml_dataset_validation_report.md").write_text("\n\n".join(validation_report) + "\n", encoding="utf-8")

    design_report = [
        "# ML Experiment Design",
        "## Evaluation Split",
        f"Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state={RANDOM_SEED})`. Fold 0 is test, fold 1 is validation, and folds 2-4 are training, producing an approximately 60/20/20 split. Group is `Customer_ID`; every customer is confined to one partition.",
        "```text\n" + split_summary.to_string(index=False) + "\n```",
        "Split validation:\n```text\n" + pd.Series(split_validation).to_string() + "\n```",
        "## Planned Model Inputs",
        "Exclude `Order_ID`, `Customer_ID`, and `Return_Request_Date` from the model matrix while retaining them externally for grouping, traceability, and temporal audit. Never include `Scenario_Label`, `Simulated_Abuse_Target`, `Refund_Amount_INR`, or `Return_Status` as features.",
        "## Planned scikit-learn Preprocessing",
        "- Numeric columns: `SimpleImputer(strategy='median')` followed by `StandardScaler()` for a linear baseline.\n- Categorical columns: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.\n- Combine with `ColumnTransformer`; fit it on training data only.\n- A future estimator must be appended inside a `Pipeline` only after this design is approved.",
        "## Class Imbalance Plan",
        "Do not oversample before splitting. Begin with grouped splits and report class-sensitive metrics later (precision, recall, PR-AUC, balanced accuracy). Consider estimator class weights first. Do not use SMOTE yet; if ever used, it must be applied only within training folds through an imbalanced-learn pipeline.",
        "## Recommended Next Step",
        "Review the scenario-mechanism signals and agree on a small feature-ablation plan before training a baseline. Then fit preprocessing and the first classifier only on the grouped training partition.",
    ]
    (reports / "ml_experiment_design.md").write_text("\n\n".join(design_report) + "\n", encoding="utf-8")


def main() -> None:
    features, dataset = load_return_level_data()
    audit = build_audit_tables(features, dataset)
    assignments, split_summary = build_grouped_split(dataset)
    split_validation = validate_grouped_split(assignments)
    if not all(value is True or value == 0 for value in split_validation.values()):
        raise ValueError(f"Grouped split validation failed: {split_validation}")
    output = PROJECT_ROOT / "data" / "processed"
    output.mkdir(exist_ok=True)
    assignments.to_csv(output / "return_level_split_assignments.csv", index=False)
    write_reports(audit, split_summary, split_validation)
    print(audit["overview"].to_string())
    print(split_summary.to_string(index=False))
    print(pd.Series(split_validation).to_string())


if __name__ == "__main__":
    main()
