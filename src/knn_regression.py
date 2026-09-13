"""Select k-NN neighbours by development-set CV, then evaluate the final test set."""

import argparse
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .data import DEFAULT_INPUT, PROJECT_ROOT, TARGET, FEATURES, load_features
    from .linear_regression import regression_metrics
    from .regression_plots import save_figure, save_regression_plot
else:
    from data import DEFAULT_INPUT, PROJECT_ROOT, TARGET, FEATURES, load_features
    from linear_regression import regression_metrics
    from regression_plots import save_figure, save_regression_plot


K_VALUES = [3, 5, 7, 9, 11, 15, 20]
TEST_SIZE = 0.2
RANDOM_STATE = 42
REPRESENTATIONS = {"without_pca": "Original", "with_pca": "PCA"}


def build_pipeline(representation: str, k: int, pca_components: int = 3) -> Pipeline:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    steps = [("scaler", StandardScaler())]
    if representation == "with_pca":
        steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
    steps.append(("model", KNeighborsRegressor(n_neighbors=k, weights="uniform", metric="euclidean")))
    return Pipeline(steps)


def positive_integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def select_candidates(cv_summary: pd.DataFrame) -> pd.DataFrame:
    """Choose minimum mean validation RMSE per representation; ties prefer smaller k."""
    return (cv_summary.sort_values(["rmse_mean", "k"], kind="stable")
            .drop_duplicates("pipeline").set_index("pipeline")
            .loc[list(REPRESENTATIONS)])


def save_cv_curve(cv_summary: pd.DataFrame, selected: pd.DataFrame,
                  output_dir: Path, cv_folds: int, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for name, color in [("without_pca", "#2563eb"), ("with_pca", "#ea580c")]:
        rows = cv_summary[cv_summary["pipeline"] == name].sort_values("k")
        ax.errorbar(rows["k"], rows["rmse_mean"], yerr=rows["rmse_std"],
                    fmt="o-", capsize=4, color=color, label=REPRESENTATIONS[name])
        best = selected.loc[name]
        ax.scatter([best["k"]], [best["rmse_mean"]], s=160, facecolors="none",
                   edgecolors=color, linewidths=2, zorder=4)
    ax.set(title=f"k-NN: development {cv_folds}-fold CV\nMean RMSE ± sample SD; circles mark selected k",
           xlabel="Number of neighbours (k)", ylabel="Validation RMSE",
           xticks=sorted(cv_summary["k"].unique()))
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, output_dir, "cv_rmse_vs_k", show)


def run_knn(input_path: Path = DEFAULT_INPUT,
            output_dir: Path = PROJECT_ROOT / "results" / "knn",
            pca_components: int = 3, k_values=None, cv_folds: int = 5,
            show_plot: bool = False) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Select each representation's k using CV only; return final fitted models/results."""
    candidates = list(K_VALUES if k_values is None else k_values)
    if not candidates or any(not positive_integer(k) for k in candidates):
        raise ValueError("k-values must contain positive integers.")
    candidates = sorted(set(candidates))
    if not positive_integer(cv_folds) or cv_folds < 2:
        raise ValueError("cv-folds must be an integer of at least 2.")
    if not positive_integer(pca_components) or pca_components > len(FEATURES):
        raise ValueError("pca-components must be an integer between 1 and 6.")
    data, features = load_features(input_path)
    if TARGET not in data:
        raise ValueError(f"Missing target column: {TARGET}")
    target = pd.to_numeric(data[TARGET], errors="raise")
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Target contains missing or infinite values.")
    x_dev, x_test, y_dev, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    if len(x_test) < 2 or len(x_dev) // cv_folds < 2:
        raise ValueError("Need at least two rows in the test set and in each CV validation fold for R².")
    folds = list(KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE).split(x_dev))
    minimum_training = min(len(train) for train, _ in folds)
    if candidates[-1] > minimum_training:
        raise ValueError(f"k-values cannot exceed {minimum_training}, the smallest CV training fold.")
    if pca_components > minimum_training:
        raise ValueError(f"pca-components cannot exceed the smallest CV training fold ({minimum_training}).")

    cv_rows = []
    assignments = pd.Series(0, index=x_dev.index, name="validation_fold")
    for fold, (fit, validate) in enumerate(folds, start=1):
        assignments.loc[x_dev.iloc[validate].index] = fold
        for name in REPRESENTATIONS:
            for k in candidates:
                pipeline = build_pipeline(name, k, pca_components)
                pipeline.fit(x_dev.iloc[fit], y_dev.iloc[fit])
                predicted = pipeline.predict(x_dev.iloc[validate])
                cv_rows.append({"pipeline": name, "k": k, "fold": fold,
                                "train_rows": len(fit), "validation_rows": len(validate),
                                **regression_metrics(y_dev.iloc[validate], predicted)})
    cv_metrics = pd.DataFrame(cv_rows)
    cv_summary = cv_metrics.groupby(["pipeline", "k"])[["mae", "rmse", "r2"]].agg(["mean", "std"])
    cv_summary.columns = [f"{metric}_{stat}" for metric, stat in cv_summary.columns]
    cv_summary = cv_summary.reset_index()
    selected = select_candidates(cv_summary)

    # Both choices are fixed before any final-test prediction is computed.
    models, results = {}, []
    predictions = pd.DataFrame({"row_index": x_test.index, "actual": y_test.to_numpy()})
    for name, label in REPRESENTATIONS.items():
        k = int(selected.loc[name, "k"])
        model = build_pipeline(name, k, pca_components)
        model.fit(x_dev, y_dev)
        predicted = model.predict(x_test)
        models[name] = model
        predictions[name] = predicted
        results.append({"representation": label, "pipeline": name, "k": k,
                        **{f"cv_{key}": value for key, value in selected.loc[name].items() if key != "k"},
                        **{f"test_{key}": value for key, value in regression_metrics(y_test, predicted).items()},
                        "development_rows": len(x_dev), "test_rows": len(x_test),
                        "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                        "cv_folds": cv_folds, "pca_components": pca_components})
    results = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.loc[x_dev.index].to_csv(output_dir / "development.csv", index_label="row_index")
    data.loc[x_test.index].to_csv(output_dir / "test.csv", index_label="row_index")
    assignments.to_csv(output_dir / "cv_folds.csv", index_label="row_index")
    cv_metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    results.to_csv(output_dir / "results.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_cv_curve(cv_summary, selected, output_dir, cv_folds, show_plot)
    save_regression_plot(predictions, output_dir, selected, show=show_plot,
                         model_name="k-NN regression", pca_components=pca_components,
                         cv_folds=cv_folds, selected_k=selected["k"].astype(int).to_dict())
    print(f"Split: {len(x_dev)} development / {len(x_test)} final-test rows; random_state={RANDOM_STATE}.")
    print(f"Selected minimum mean {cv_folds}-fold validation RMSE among k={candidates}; ties prefer smaller k.")
    table = results[["representation", "k"]].copy()
    for metric, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]:
        table[f"CV {label} mean ± sample SD"] = [
            f"{row[f'cv_{metric}_mean']:.3f} ± {row[f'cv_{metric}_std']:.3f}"
            for _, row in results.iterrows()]
    for metric, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]:
        table[f"Test {label}"] = results[f"test_{metric}"]
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Results saved to {output_dir.resolve()}")
    return models, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "knn")
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    try:
        run_knn(args.input, args.output_dir, args.pca_components, args.k_values, args.cv_folds,
                show_plot=True)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
