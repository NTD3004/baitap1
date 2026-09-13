"""Load raw real estate features without fitting preprocessing steps."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "Real estate valuation data set.xlsx"
FEATURES = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]
TARGET = "Y house price of unit area"


def load_features(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return original data and validated, unscaled numeric features."""
    data = pd.read_excel(input_path, engine="openpyxl")
    data.columns = data.columns.str.strip()
    missing = sorted(set(FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    features = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Input features contain missing or infinite values.")
    if (features.nunique() < 2).any():
        raise ValueError("All six features must have nonzero variance.")
    return data, features
