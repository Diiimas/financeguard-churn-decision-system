"""Leakage-safe pipelines and model selection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = (
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
)
CATEGORICAL_FEATURES = ("Geography", "Gender", "HasCrCard", "IsActiveMember")


@dataclass(frozen=True)
class CandidateResult:
    name: str
    search: GridSearchCV
    cv_roc_auc: float
    cv_pr_auc: float


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore",
    drop="first",
    sparse_output=False,)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, list(NUMERIC_FEATURES)),
         ("categorical", categorical, list(CATEGORICAL_FEATURES))],
        verbose_feature_names_out=False,
    )


def candidate_spaces(random_state: int = 42) -> dict[str, tuple[BaseEstimator, dict]]:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=random_state),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=random_state),
            {
                "model__learning_rate": [0.05, 0.10],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
    }


def select_model(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[CandidateResult, list[CandidateResult]]:
    """Tune candidates using development data only; choose by mean CV PR-AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results: list[CandidateResult] = []
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    for name, (estimator, grid) in candidate_spaces(random_state).items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(
            pipeline,
            grid,
            scoring=scoring,
            refit="pr_auc",
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        search.fit(features, target)
        index = search.best_index_
        results.append(
            CandidateResult(
                name=name,
                search=search,
                cv_roc_auc=float(search.cv_results_["mean_test_roc_auc"][index]),
                cv_pr_auc=float(search.cv_results_["mean_test_pr_auc"][index]),
            )
        )
    winner = max(results, key=lambda item: item.cv_pr_auc)
    return winner, results
