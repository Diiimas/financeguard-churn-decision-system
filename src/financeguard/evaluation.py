"""Metrics, lift and business-aware threshold selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BusinessAssumptions:
    contact_cost: float = 10.0
    retained_customer_value: float = 250.0
    campaign_success_rate: float = 0.30


def threshold_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    assumptions: BusinessAssumptions,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate campaign economics for candidate decision thresholds."""
    y = np.asarray(y_true, dtype=int)
    thresholds = np.linspace(0.05, 0.95, 181) if thresholds is None else thresholds
    rows = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        contacts = int(predicted.sum())
        true_positives = int(((y == 1) & predicted).sum())
        expected_saved = true_positives * assumptions.campaign_success_rate
        expected_benefit = (
            expected_saved * assumptions.retained_customer_value
            - contacts * assumptions.contact_cost
        )
        rows.append(
            {
                "threshold": float(threshold),
                "contacts": contacts,
                "contact_rate": float(contacts / len(y)),
                "true_positives": true_positives,
                "expected_customers_saved": float(expected_saved),
                "expected_net_benefit": float(expected_benefit),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    table: pd.DataFrame,
    max_contact_rate: float | None = None,
) -> float:
    candidates = table

    if max_contact_rate is not None:
        if not 0 < max_contact_rate <= 1:
            raise ValueError("max_contact_rate must be in (0, 1]")

        candidates = table.loc[
            table["contact_rate"] <= max_contact_rate
        ]

        if candidates.empty:
            raise ValueError(
                "No threshold satisfies the contact-rate constraint"
            )

    best = candidates.sort_values(
        ["expected_net_benefit", "threshold"],
        ascending=[False, False],
    ).iloc[0]

    return float(best["threshold"])


def lift_at_fraction(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray, fraction: float = 0.10
) -> float:
    y = np.asarray(y_true, dtype=int)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(np.ceil(len(y) * fraction)))
    top = y[np.argsort(probabilities)[::-1][:count]]
    baseline = y.mean()
    return float(top.mean() / baseline) if baseline else 0.0


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "lift_at_10_percent": lift_at_fraction(y, probabilities, 0.10),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def assumptions_to_dict(assumptions: BusinessAssumptions) -> dict:
    return asdict(assumptions)
