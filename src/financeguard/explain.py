"""Generate model-agnostic permutation importance."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from financeguard.data import development_test_split, load_dataset


def generate_permutation_importance(
    data_path: Path,
    model_path: Path,
    output_path: Path,
    random_state: int = 42,
    n_repeats: int = 20,
    n_jobs: int = -1,
) -> pd.DataFrame:
    features, target = load_dataset(data_path)

    _, x_test, _, y_test = development_test_split(
        features,
        target,
        random_state=random_state,
    )

    payload = joblib.load(model_path)
    model = payload["model"]

    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)

    return importance

def save_importance_plot(
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = importance.sort_values(
        "importance_mean",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))

    axis.barh(
        ordered["feature"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color="#2563EB",
        alpha=0.85,
        capsize=3,
    )

    axis.axvline(0, color="#374151", linewidth=0.8)
    axis.set_title("Importancia por permutación del modelo")
    axis.set_xlabel("Caída media de PR-AUC al permutar la variable")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.20)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate permutation importance using PR-AUC."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/churn_model.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics/permutation_importance.csv"),
    )
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
    "--figure",
    type=Path,
    default=Path("reports/figures/permutation_importance.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    importance = generate_permutation_importance(
        data_path=args.data,
        model_path=args.model,
        output_path=args.output,
        random_state=args.random_state,
        n_repeats=args.n_repeats,
        n_jobs=args.n_jobs,
    )
    save_importance_plot(
        importance,
        args.figure,
    )

    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()