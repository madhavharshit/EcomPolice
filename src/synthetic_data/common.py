"""Shared constants and helpers for synthetic data generation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


INDIAN_CITIES = [
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Ahmedabad",
    "Jaipur",
    "Lucknow",
    "Surat",
    "Indore",
    "Kochi",
    "Chandigarh",
    "Guwahati",
]

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet", "Cash on Delivery"]
SHIPPING_METHODS = ["Standard", "Express", "Same-Day"]
GENDERS = ["Female", "Male", "Non-binary", "Prefer not to say"]

PRODUCT_CATEGORIES = {
    "Clothing": (399, 4999),
    "Footwear": (499, 6999),
    "Electronics": (3999, 89999),
    "Mobile Accessories": (149, 2999),
    "Home & Kitchen": (299, 24999),
    "Beauty": (99, 4999),
    "Books": (99, 1999),
    "Sports": (299, 14999),
}

RETURN_REASONS = {
    "Clothing": ["Size issue", "Changed mind", "Not as described", "Damaged item"],
    "Footwear": ["Size issue", "Changed mind", "Damaged item", "Not as described"],
    "Electronics": ["Defective", "Missing accessory", "Not as described", "Damaged item"],
    "Mobile Accessories": ["Defective", "Wrong item", "Changed mind", "Damaged item"],
    "Home & Kitchen": ["Damaged item", "Not as described", "Wrong item", "Changed mind"],
    "Beauty": ["Damaged item", "Wrong item", "Changed mind", "Allergic reaction"],
    "Books": ["Damaged item", "Wrong item", "Not as described", "Changed mind"],
    "Sports": ["Size issue", "Damaged item", "Not as described", "Changed mind"],
}


def rng_choice(rng: np.random.Generator, values: list[str], p: list[float] | None = None) -> str:
    """Return one random string choice."""
    return str(rng.choice(values, p=p))


def random_dates(
    rng: np.random.Generator,
    start: pd.Timestamp,
    end: pd.Timestamp,
    count: int,
) -> pd.Series:
    """Generate random dates between inclusive start and end."""
    if end < start:
        return pd.Series([start] * count)
    span = (end - start).days
    offsets = rng.integers(0, span + 1, size=count)
    return pd.Series(start + pd.to_timedelta(offsets, unit="D"))


def parse_date(value: str) -> pd.Timestamp:
    """Parse a config date string."""
    return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))

