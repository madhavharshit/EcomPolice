"""Generate synthetic product catalog data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_data.common import PRODUCT_CATEGORIES
from synthetic_data.config import SyntheticConfig


BRANDS = ["Aarika", "BharatBuy", "UrbanKart", "Nila", "TaraTech", "Zentra", "Kosha", "MetroNest"]


def generate_products(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate products with category-dependent rupee prices."""
    categories = list(PRODUCT_CATEGORIES)
    rows = []
    for i in range(config.n_products):
        category = str(rng.choice(categories, p=[0.18, 0.12, 0.14, 0.14, 0.14, 0.10, 0.09, 0.09]))
        low, high = PRODUCT_CATEGORIES[category]
        price = float(np.round(rng.lognormal(mean=np.log((low + high) / 2), sigma=0.45), 2))
        price = float(np.clip(price, low, high))
        brand = str(rng.choice(BRANDS))
        rows.append(
            {
                "Product_ID": f"PROD{i:05d}",
                "Category": category,
                "Brand": brand,
                "Product_Name": f"{brand} {category} Item {i + 1}",
                "List_Price_INR": round(price, 2),
            }
        )
    return pd.DataFrame(rows)

