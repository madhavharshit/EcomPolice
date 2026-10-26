"""Data loading helpers for the EcomPolice preprocessing pipeline."""

from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR


def discover_raw_csv(raw_dir: Path = RAW_DATA_DIR, input_path: str | Path | None = None) -> Path:
    """Return the configured CSV path or discover a single CSV under data/raw."""
    if input_path is not None:
        path = Path(input_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Configured raw CSV was not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Configured raw data file is not a CSV: {path}")
        return path

    csv_files = sorted(path for path in raw_dir.glob("*.csv") if path.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    if len(csv_files) > 1:
        names = ", ".join(str(path) for path in csv_files)
        raise ValueError(
            "Multiple CSV files found in data/raw. Pass --input-csv to choose one: "
            f"{names}"
        )
    return csv_files[0]


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load a raw CSV without modifying source values."""
    return pd.read_csv(csv_path)

