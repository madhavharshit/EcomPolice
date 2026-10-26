"""Configuration for synthetic EcomPolice data generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
CLEAN_DIR = SYNTHETIC_DATA_DIR / "clean"
MESSY_DIR = SYNTHETIC_DATA_DIR / "messy"
LABELS_DIR = SYNTHETIC_DATA_DIR / "labels"
MANIFESTS_DIR = SYNTHETIC_DATA_DIR / "manifests"
REPORTS_DIR = PROJECT_ROOT / "reports"


SCENARIOS = [
    "NORMAL",
    "FREQUENT_LEGITIMATE_RETURNS",
    "REPEATED_DEFECT_CLAIMS",
    "HIGH_VALUE_RETURN_PATTERN",
    "RETURN_WINDOW_PATTERN",
    "MIXED_SUSPICIOUS_PATTERN",
]


@dataclass(frozen=True)
class DefectRates:
    """Configurable low default rates for messy-data injection."""

    missing_values: float = 0.004
    inconsistent_categories: float = 0.006
    mixed_date_formats: float = 0.015
    duplicate_rows: float = 0.003
    invalid_chronology: float = 0.003
    numeric_as_string: float = 0.006
    missing_identifiers: float = 0.002
    impossible_negative_values: float = 0.002
    implausible_outliers: float = 0.002


@dataclass(frozen=True)
class SyntheticConfig:
    """Top-level generator configuration."""

    n_customers: int = 3_000
    n_products: int = 500
    n_orders: int = 20_000
    seed: int = 42
    period_start: str = "2024-01-01"
    months: int = 12
    defect_rates: DefectRates = field(default_factory=DefectRates)
    scenario_shares: dict[str, float] = field(
        default_factory=lambda: {
            "NORMAL": 0.78,
            "FREQUENT_LEGITIMATE_RETURNS": 0.08,
            "REPEATED_DEFECT_CLAIMS": 0.04,
            "HIGH_VALUE_RETURN_PATTERN": 0.035,
            "RETURN_WINDOW_PATTERN": 0.035,
            "MIXED_SUSPICIOUS_PATTERN": 0.03,
        }
    )

    def validate(self) -> None:
        """Validate configuration values before generation."""
        if self.n_customers <= 0:
            raise ValueError("n_customers must be positive.")
        if self.n_products <= 0:
            raise ValueError("n_products must be positive.")
        if self.n_orders <= 0:
            raise ValueError("n_orders must be positive.")
        if self.months <= 0:
            raise ValueError("months must be positive.")
        missing = set(SCENARIOS) - set(self.scenario_shares)
        if missing:
            raise ValueError(f"Missing scenario shares for: {sorted(missing)}")
        total = sum(self.scenario_shares.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scenario_shares must sum to 1.0, got {total:.6f}.")
        for name, value in self.defect_rates.__dict__.items():
            if not 0 <= value <= 0.25:
                raise ValueError(f"Defect rate {name} must be between 0 and 0.25.")

