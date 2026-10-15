"""Command-line training workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from financeguard.data import development_test_split, load_dataset
from financeguard.evaluation import (
    BusinessAssumptions,
    assumptions_to_dict,
    classification_metrics,
    select_threshold,
    threshold_table,
)
from financeguard.modeling import select_model


def run_training(
    data_path: Path,
    output_dir: Path,
    model_dir: Path,
    assumptions: BusinessAssumptions,
    max_contact_rate: float | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict:
    features, target = load_dataset(data_path)
    x_dev, x_test, y_dev, y_test = development_test_split(
        features, target, random_state=random_state
    )

    winner, candidates = select_model(x_dev, y_dev, random_state=random_state, n_jobs=n_jobs)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probabilities = cross_val_predict(
        clone(winner.search.best_estimator_),
        x_dev,
        y_dev,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]
    business_curve = threshold_table(y_dev, oof_probabilities, assumptions)
    threshold = select_threshold(
        business_curve,
        max_contact_rate=max_contact_rate,
    )

    final_model = clone(winner.search.best_estimator_).fit(x_dev, y_dev)
    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_probabilities, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    business_curve.to_csv(output_dir / "threshold_business_curve.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": item.name,
                "cv_roc_auc": item.cv_roc_auc,
                "cv_pr_auc": item.cv_pr_auc,
                "best_parameters": json.dumps(item.search.best_params_, sort_keys=True),
            }
            for item in candidates
        ]
    ).to_csv(output_dir / "model_comparison.csv", index=False)

    report = {
        "data": {
            "rows": int(len(features)),
            "development_rows": int(len(x_dev)),
            "test_rows": int(len(x_test)),
            "positive_rate": float(target.mean()),
            "random_state": random_state,
        },
        "selection": {
            "winner": winner.name,
            "best_parameters": winner.search.best_params_,
            "cv_roc_auc": winner.cv_roc_auc,
            "cv_pr_auc": winner.cv_pr_auc,
            "criterion": "highest mean 5-fold development PR-AUC",
        },
        "business_assumptions": assumptions_to_dict(assumptions),
        "decision_policy": {
            "max_contact_rate": max_contact_rate,
            "selected_threshold": threshold,
        },
        "untouched_test_metrics": test_metrics,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(
        {"model": final_model, "threshold": threshold, "metadata": report},
        model_dir / "churn_model.joblib",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--contact-cost", type=float, default=10.0)
    parser.add_argument("--retained-customer-value", type=float, default=250.0)
    parser.add_argument("--campaign-success-rate", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--max-contact-rate",
        type=float,
        default=None,
        help="Maximum proportion of customers that can be contacted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assumptions = BusinessAssumptions(
        contact_cost=args.contact_cost,
        retained_customer_value=args.retained_customer_value,
        campaign_success_rate=args.campaign_success_rate,
    )
    report = run_training(
        args.data,
        args.output_dir,
        args.model_dir,
        assumptions,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        max_contact_rate=args.max_contact_rate,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
