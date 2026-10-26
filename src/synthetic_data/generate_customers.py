"""Generate fictional Indian e-commerce customers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

from synthetic_data.common import GENDERS, INDIAN_CITIES, parse_date, random_dates
from synthetic_data.config import SyntheticConfig


def generate_customers(config: SyntheticConfig, labels: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic customer profiles with fictional names."""
    fake = Faker("en_IN")
    Faker.seed(config.seed)
    period_start = parse_date(config.period_start)
    signup_start = period_start - pd.Timedelta(days=540)
    signup_end = period_start + pd.Timedelta(days=60)
    signup_dates = random_dates(rng, signup_start, signup_end, config.n_customers)

    customers = labels[["Customer_ID"]].copy()
    customers["Full_Name"] = [fake.name() for _ in range(config.n_customers)]
    customers["City"] = rng.choice(INDIAN_CITIES, size=config.n_customers)
    customers["State_Region"] = customers["City"].map(
        {
            "Mumbai": "Maharashtra",
            "Delhi": "Delhi",
            "Bengaluru": "Karnataka",
            "Hyderabad": "Telangana",
            "Chennai": "Tamil Nadu",
            "Kolkata": "West Bengal",
            "Pune": "Maharashtra",
            "Ahmedabad": "Gujarat",
            "Jaipur": "Rajasthan",
            "Lucknow": "Uttar Pradesh",
            "Surat": "Gujarat",
            "Indore": "Madhya Pradesh",
            "Kochi": "Kerala",
            "Chandigarh": "Chandigarh",
            "Guwahati": "Assam",
        }
    )
    customers["Signup_Date"] = signup_dates.dt.date.astype(str)
    customers["Age_Band"] = rng.choice(
        ["18-24", "25-34", "35-44", "45-54", "55-64"],
        size=config.n_customers,
        p=[0.16, 0.34, 0.25, 0.17, 0.08],
    )
    customers["Gender"] = rng.choice(GENDERS, size=config.n_customers, p=[0.47, 0.48, 0.02, 0.03])
    customers["Loyalty_Tier"] = rng.choice(
        ["Bronze", "Silver", "Gold", "Platinum"],
        size=config.n_customers,
        p=[0.50, 0.30, 0.15, 0.05],
    )
    return customers

