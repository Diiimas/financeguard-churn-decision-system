"""Dataset loading, schema checks and leakage-safe splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Exited"
ID_COLUMNS = ("RowNumber", "CustomerId", "Surname")
FEATURE_COLUMNS = (
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
)


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the canonical churn dataset."""
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS) | {TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    target_values = set(frame[TARGET].unique())
    if not target_values <= {0, 1}:
        raise ValueError(f"Target must be binary 0/1; found {target_values}")
    return frame.loc[:, FEATURE_COLUMNS].copy(), frame[TARGET].astype(int).copy()


def development_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create one untouched test split; all selection happens on development data."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
