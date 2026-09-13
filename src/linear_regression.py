"""Compare linear regression with and without PCA on the same held-out data."""

import argparse
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(actual, predicted):
        return np.sqrt(mean_squared_error(actual, predicted))
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .data import DEFAULT_INPUT, PROJECT_ROOT, TARGET, load_features
    from .regression_plots import save_regression_plot
else:
    from data import DEFAULT_INPUT, PROJECT_ROOT, TARGET, load_features
    from regression_plots import save_regression_plot


def build_pipelines(pca_components: int = 3) -> dict[str, Pipeline]:
    """Create independent, unfitted pipelines accepting raw feature values."""
    return {
        "with_pca": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, svd_solver="full")),
            ("model", LinearRegression()),
        ]),
        "without_pca": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
    }


def regression_metrics(actual, predicted) -> dict[str, float]:
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": mean_absolute_error(actual, predicted),
        "mse": mse,
        "rmse": root_mean_squared_error(actual, predicted),
        "r2": r2_score(actual, predicted),
    }


def run_regression(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = PROJECT_ROOT / "results" / "regression",
    test_size: float = 0.2,
    random_state: int = 42,
    show_plot: bool = False,
    pca_components: int = 3,
    cv_folds: int = 5,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Run shared CV on training data, then refit and evaluate on test data.

    Returned pipelines predict directly from raw features in training column order.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if (not isinstance(pca_components, Integral) or isinstance(pca_components, bool)
            or not 1 <= pca_components <= 6):
        raise ValueError("pca-components must be an integer between 1 and 6.")
    if (not isinstance(cv_folds, Integral) or isinstance(cv_folds, bool) or cv_folds < 2):
        raise ValueError("cv-folds must be an integer of at least 2.")
    data, features = load_features(input_path)
    if TARGET not in data.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    target = pd.to_numeric(data[TARGET], errors="raise")
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Target contains missing or infinite values.")
    n_test = int(np.ceil(len(features) * test_size))
    n_train = len(features) - n_test
    if n_test < 2 or n_train // cv_folds < 2:
        raise ValueError("Need at least two rows in each CV validation fold and the test set for R².")
    if pca_components > n_train - int(np.ceil(n_train / cv_folds)):
        raise ValueError("pca-components cannot exceed the smallest CV training fold.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state,
    )
    pipelines = build_pipelines(pca_components)
    # One shared fold partition ensures a paired comparison of both pipelines.
    folds = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_metrics = []
    fold_assignments = pd.Series(index=x_train.index, dtype="int64", name="validation_fold")
    for fold, (fit_indices, validation_indices) in enumerate(folds.split(x_train), start=1):
        x_fit, x_validation = x_train.iloc[fit_indices], x_train.iloc[validation_indices]
        y_fit, y_validation = y_train.iloc[fit_indices], y_train.iloc[validation_indices]
        fold_assignments.loc[x_validation.index] = fold
        for name, pipeline in pipelines.items():
            # A fresh full pipeline fits preprocessing on this fold's training rows.
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(x_fit, y_fit)
            cv_metrics.append({
                "pipeline": name,
                "fold": fold,
                "train_rows": len(x_fit),
                "validation_rows": len(x_validation),
                **regression_metrics(y_validation, fold_pipeline.predict(x_validation)),
            })

    cv_results = pd.DataFrame(cv_metrics)
    cv_summary = cv_results.groupby("pipeline", sort=False)[["mae", "mse", "rmse", "r2"]].agg(["mean", "std"])
    cv_summary.columns = [f"{metric}_{stat}" for metric, stat in cv_summary.columns]

    # Refit both pipelines on the entire 80% training set before touching the test set.
    predictions = pd.DataFrame({"row_index": x_test.index})
    if "No" in data.columns:
        predictions["No"] = data.loc[x_test.index, "No"].to_numpy()
    predictions["actual"] = y_test.to_numpy()
    metrics = []
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        predictions[name] = predicted
        metrics.append({
            "pipeline": name,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "test_size": test_size,
            "random_state": random_state,
            "pca_components": pca_components,
            "cv_folds": cv_folds,
            **regression_metrics(y_test, predicted),
        })

    summary = pd.DataFrame(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.loc[x_train.index].to_csv(output_dir / "train.csv", index_label="row_index")
    data.loc[x_test.index].to_csv(output_dir / "test.csv", index_label="row_index")
    fold_assignments.astype(int).to_csv(output_dir / "cv_folds.csv", index_label="row_index")
    cv_results.to_csv(output_dir / "cv_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "cv_summary.csv")
    summary.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_regression_plot(predictions, output_dir, cv_summary, show=show_plot,
                         pca_components=pca_components, cv_folds=cv_folds)
    print(f"Split: {len(x_train)} training rows, {len(x_test)} test rows.")
    print(f"{cv_folds}-fold cross-validation on training data (mean and sample standard deviation):")
    print(cv_summary.to_string(float_format=lambda value: f"{value:.6f}"))
    return pipelines, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "regression")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    try:
        _, summary = run_regression(args.input, args.output_dir, args.test_size, args.random_state,
                                    show_plot=True, pca_components=args.pca_components,
                                    cv_folds=args.cv_folds)
    except ValueError as error:
        parser.error(str(error))
    print("\nFinal held-out test metrics:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nResults saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
