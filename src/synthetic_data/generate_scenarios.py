"""Customer scenario assignment and scenario behavior rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_data.config import SCENARIOS, SyntheticConfig


ABUSE_SCENARIOS = {
    "REPEATED_DEFECT_CLAIMS",
    "HIGH_VALUE_RETURN_PATTERN",
    "RETURN_WINDOW_PATTERN",
    "MIXED_SUSPICIOUS_PATTERN",
}


SCENARIO_RULES = {
    "NORMAL": "Low-to-moderate returns with varied reasons and ordinary order values.",
    "FREQUENT_LEGITIMATE_RETURNS": "Higher return rate concentrated in apparel and footwear, often size-related.",
    "REPEATED_DEFECT_CLAIMS": "Elevated repeated defective-claim returns across categories.",
    "HIGH_VALUE_RETURN_PATTERN": "More expensive products and higher-value returns than baseline customers.",
    "RETURN_WINDOW_PATTERN": "Returns tend to occur close to the policy window end.",
    "MIXED_SUSPICIOUS_PATTERN": "Combination of high return propensity, defect claims, high-value items, and late-window returns.",
}


def assign_customer_scenarios(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Assign every customer a latent behavior scenario."""
    scenarios = list(config.scenario_shares)
    shares = [config.scenario_shares[name] for name in scenarios]
    labels = rng.choice(scenarios, size=config.n_customers, p=shares)
    return pd.DataFrame(
        {
            "Customer_ID": [f"CUST{i:06d}" for i in range(config.n_customers)],
            "Scenario_Label": labels,
            "Simulated_Abuse_Indicator": [int(label in ABUSE_SCENARIOS) for label in labels],
        }
    )


def scenario_return_probability(scenario: str, category: str, order_value: float) -> float:
    """Return a scenario-aware probability that an order is returned."""
    base_by_category = {
        "Clothing": 0.20,
        "Footwear": 0.22,
        "Electronics": 0.12,
        "Mobile Accessories": 0.10,
        "Home & Kitchen": 0.11,
        "Beauty": 0.09,
        "Books": 0.07,
        "Sports": 0.13,
    }
    probability = base_by_category.get(category, 0.10)
    if scenario == "FREQUENT_LEGITIMATE_RETURNS":
        probability += 0.18 if category in {"Clothing", "Footwear"} else 0.08
    elif scenario == "REPEATED_DEFECT_CLAIMS":
        probability += 0.20
    elif scenario == "HIGH_VALUE_RETURN_PATTERN":
        probability += 0.18 if order_value >= 20_000 else 0.08
    elif scenario == "RETURN_WINDOW_PATTERN":
        probability += 0.12
    elif scenario == "MIXED_SUSPICIOUS_PATTERN":
        probability += 0.26
    return min(max(probability, 0.01), 0.85)

