"""Run regression-tree selection and evaluation with and without PCA."""

import argparse
from itertools import product
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

if __package__:
    from .data import DEFAULT_INPUT, PROJECT_ROOT, FEATURES, TARGET, load_features
    from .linear_regression import regression_metrics
    from .regression_plots import save_figure, save_regression_plot
else:
    from data import DEFAULT_INPUT, PROJECT_ROOT, FEATURES, TARGET, load_features
    from linear_regression import regression_metrics
    from regression_plots import save_figure, save_regression_plot


MAX_DEPTH_VALUES = [2, 3, 4, 5, None]
MIN_SAMPLES_LEAF_VALUES = [1, 5, 10, 20]
REPRESENTATIONS = {"without_pca": "Original", "with_pca": "PCA"}
SPLIT_RANDOM_STATE = 42


def parse_depth(value: str):
    if value.lower() == "none":
        return None
    try:
        result = int(value)
        if result > 0:
            return result
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("max depth must be a positive integer or none")


def positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def configurations(depths=None, leaves=None):
    depths = list(MAX_DEPTH_VALUES if depths is None else depths)
    leaves = list(MIN_SAMPLES_LEAF_VALUES if leaves is None else leaves)
    if not depths or any(d is not None and not positive_integer(d) for d in depths):
        raise ValueError("max-depth-values must contain positive integers or None.")
    if not leaves or any(not positive_integer(n) for n in leaves):
        raise ValueError("min-samples-leaf-values must contain positive integers.")
    depths = sorted(set(depths), key=lambda d: float("inf") if d is None else d)
    return list(product(depths, sorted(set(leaves))))


def depth_label(depth):
    return "none" if depth is None else str(int(depth))


def build_tree_pipeline(name, depth, leaf, pca_components=3, random_state=42):
    if name not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {name}")
    steps = []
    if name == "with_pca":
        steps = [("scaler", StandardScaler()),
                 ("pca", PCA(n_components=pca_components, svd_solver="full"))]
    # A regression node R stores mean(t_R) = sum(t_n in R) / N_R.
    # Its squared-error impurity is I(R) = sum((t_n - mean(t_R))**2) / N_R.
    # Each split minimizes (N_L/N)*I(R_L) + (N_R/N)*I(R_R), equivalently
    # maximizing the reduction from parent impurity. Entropy is not used.
    # This builds a tree; development validation RMSE selects its hyperparameters.
    steps.append(("model", DecisionTreeRegressor(
        criterion="squared_error", min_samples_split=2, random_state=random_state,
        max_depth=depth, min_samples_leaf=leaf,
    )))
    return Pipeline(steps)


def select_configurations(summary):
    """Independent minimum mean validation RMSE; ties prefer shallow trees/larger leaves."""
    ranked = summary.assign(depth_order=summary["max_depth"].map(
        lambda d: float("inf") if d == "none" else int(d)))
    return (ranked.sort_values(["rmse_mean", "depth_order", "min_samples_leaf"],
                              ascending=[True, True, False], kind="stable")
            .drop_duplicates("pipeline").drop(columns="depth_order")
            .set_index("pipeline").loc[list(REPRESENTATIONS)])


def save_tree_cv_plots(summary, selected, output_dir, show=False):
    depths = sorted(summary["max_depth"].unique(),
                    key=lambda d: float("inf") if d == "none" else int(d))
    leaves = sorted(summary["min_samples_leaf"].unique())
    vmin, vmax = summary["rmse_mean"].min(), summary["rmse_mean"].max()
    for name, label in REPRESENTATIONS.items():
        rows = summary[summary["pipeline"] == name]
        grid = rows.pivot(index="min_samples_leaf", columns="max_depth", values="rmse_mean").loc[leaves, depths]
        best = selected.loc[name]
        best_x, best_y = depths.index(best["max_depth"]), leaves.index(best["min_samples_leaf"])
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        for leaf in leaves:
            ax.plot(range(len(depths)), grid.loc[leaf], "o-", label=f"min_samples_leaf={leaf}")
        ax.scatter([best_x], [best["rmse_mean"]], marker="*", s=260, color="black",
                   zorder=4, label=f"Selected: depth={best['max_depth']}, leaf={best['min_samples_leaf']}")
        ax.set(title=f"{label}: development CV", xlabel="max_depth (none = unlimited)",
               ylabel="Mean validation RMSE", xticks=range(len(depths)), xticklabels=depths)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        save_figure(fig, output_dir, f"cv_rmse_vs_depth_{name}", show)

        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        im = ax.imshow(grid.to_numpy(), cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label="Mean validation RMSE")
        for y in range(len(leaves)):
            for x in range(len(depths)):
                value = grid.iloc[y, x]
                ax.text(x, y, f"{value:.3f}", ha="center", va="center",
                        color="white" if value < (vmin + vmax) / 2 else "black")
        ax.add_patch(Rectangle((best_x - .48, best_y - .48), .96, .96,
                               fill=False, edgecolor="red", linewidth=3))
        ax.set(title=f"{label}: mean CV RMSE (red border = selected)",
               xlabel="max_depth (none = unlimited)", ylabel="min_samples_leaf",
               xticks=range(len(depths)), xticklabels=depths,
               yticks=range(len(leaves)), yticklabels=leaves)
        save_figure(fig, output_dir, f"cv_heatmap_{name}", show)


def save_selected_tree(pipeline, name, output_dir, pca_components, show=False):
    tree = pipeline["model"]
    names = FEATURES if name == "without_pca" else [f"PC{i}" for i in range(1, pca_components + 1)]
    # Large trees get a readable overview and a complete, zoomable SVG.
    large = tree.get_n_leaves() > 32
    width = max(16, min(tree.get_n_leaves(), 16) * 2.5)
    fig, ax = plt.subplots(figsize=(width, max(8, min(tree.get_depth(), 5) * 2)), layout="constrained")
    plot_tree(tree, feature_names=names, filled=True, rounded=True, precision=3,
              max_depth=3 if large else None, fontsize=9, ax=ax)
    ax.set_title(f"{REPRESENTATIONS[name]} selected tree · depth={tree.get_depth()} · leaves={tree.get_n_leaves()}"
                 + (" · overview through depth 3; full tree in SVG" if large else ""))
    save_figure(fig, output_dir, f"tree_{name}", show)
    if large:
        fig, ax = plt.subplots(figsize=(tree.get_n_leaves() * 2.5, max(8, tree.get_depth() * 2)))
        plot_tree(tree, feature_names=names, filled=True, rounded=True, precision=3, fontsize=9, ax=ax)
        fig.savefig(output_dir / f"tree_{name}_full.svg", bbox_inches="tight")
        plt.close(fig)
    (output_dir / f"tree_{name}.txt").write_text(
        export_text(tree, feature_names=names, max_depth=tree.get_depth() + 1, decimals=4), encoding="utf-8")


def run_decision_tree(input_path=DEFAULT_INPUT, output_dir=PROJECT_ROOT / "results" / "decision_tree",
                      pca_components=3, cv_folds=5, max_depth_values=None,
                      min_samples_leaf_values=None, random_state=42, show_plot=False):
    candidates = configurations(max_depth_values, min_samples_leaf_values)
    if not positive_integer(pca_components) or pca_components > len(FEATURES):
        raise ValueError("pca-components must be an integer between 1 and 6.")
    if not positive_integer(cv_folds) or cv_folds < 2:
        raise ValueError("cv-folds must be an integer of at least 2.")
    if isinstance(random_state, bool) or not isinstance(random_state, Integral) or not 0 <= random_state < 2**32:
        raise ValueError("random-state must be an integer in [0, 2**32).")
    data, features = load_features(input_path)
    if TARGET not in data:
        raise ValueError(f"Missing target column: {TARGET}")
    target = pd.to_numeric(data[TARGET], errors="raise")
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Target contains missing or infinite values.")
    # Preserve the project's final-test membership even when the CV/tree seed changes.
    x_dev, x_test, y_dev, y_test = train_test_split(
        features, target, test_size=.20, random_state=SPLIT_RANDOM_STATE)
    if len(x_test) < 2 or len(x_dev) // cv_folds < 2:
        raise ValueError("Need at least two test rows and two validation rows per CV fold for R².")
    folds = list(KFold(n_splits=cv_folds, shuffle=True, random_state=random_state).split(x_dev))
    if pca_components > min(len(fit) for fit, _ in folds):
        raise ValueError("pca-components exceeds the smallest CV training fold.")
    assignments = pd.Series(0, index=x_dev.index, name="validation_fold")
    scores = []
    for fold, (fit, validate) in enumerate(folds, 1):
        assignments.loc[x_dev.iloc[validate].index] = fold
        for name in REPRESENTATIONS:
            for depth, leaf in candidates:
                pipeline = build_tree_pipeline(name, depth, leaf, pca_components, random_state)
                pipeline.fit(x_dev.iloc[fit], y_dev.iloc[fit])
                predicted = pipeline.predict(x_dev.iloc[validate])
                scores.append({"pipeline": name, "max_depth": depth_label(depth),
                               "min_samples_leaf": leaf, "fold": fold,
                               "train_rows": len(fit), "validation_rows": len(validate),
                               **regression_metrics(y_dev.iloc[validate], predicted)})
    del pipeline  # Temporary CV estimators are never reused for the final models.
    cv_metrics = pd.DataFrame(scores)
    keys = ["pipeline", "max_depth", "min_samples_leaf"]
    grouped = cv_metrics.groupby(keys)[["mae", "rmse", "r2"]]
    # Explicit ddof=1: sample SD across validation folds.
    cv_summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std")).reset_index()
    selected = select_configurations(cv_summary)
    models, rows = {}, []
    predictions = pd.DataFrame({"row_index": x_test.index, "actual": y_test.to_numpy()})
    # Both selections are complete before the final test set is predicted once per model.
    for name, label in REPRESENTATIONS.items():
        best = selected.loc[name]
        depth, leaf = parse_depth(best["max_depth"]), int(best["min_samples_leaf"])
        pipeline = build_tree_pipeline(name, depth, leaf, pca_components, random_state)
        pipeline.fit(x_dev, y_dev)
        predicted = pipeline.predict(x_test)
        models[name] = pipeline
        predictions[name] = predicted
        predictions[f"residual_{name}"] = y_test.to_numpy() - predicted
        tree = pipeline["model"]
        names = FEATURES if name == "without_pca" else [f"PC{i}" for i in range(1, pca_components + 1)]
        root = tree.tree_.feature[0]
        rows.append({"representation": label, "pipeline": name, "max_depth": depth_label(depth),
                     "min_samples_leaf": leaf, "tree_depth": tree.get_depth(),
                     "number_of_leaves": tree.get_n_leaves(),
                     **{f"cv_{key}": best[key] for key in best.index if key.endswith(("_mean", "_std"))},
                     **{f"test_{key}": val for key, val in regression_metrics(y_test, predicted).items()},
                     "root_feature": names[root] if root >= 0 else "leaf (no split)",
                     "root_threshold": tree.tree_.threshold[0] if root >= 0 else np.nan,
                     "development_rows": len(x_dev), "test_rows": len(x_test), "test_size": .20,
                     "split_random_state": SPLIT_RANDOM_STATE, "random_state": random_state,
                     "cv_folds": cv_folds, "pca_components": pca_components})
    results = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, filename in [(cv_metrics, "cv_metrics"), (cv_summary, "cv_summary"),
                            (predictions, "predictions"), (results, "summary")]:
        frame.to_csv(output_dir / f"{filename}.csv", index=False)
    data.loc[x_dev.index].to_csv(output_dir / "development.csv", index_label="row_index")
    data.loc[x_test.index].to_csv(output_dir / "test.csv", index_label="row_index")
    assignments.to_csv(output_dir / "cv_folds.csv", index_label="row_index")
    save_tree_cv_plots(cv_summary, selected, output_dir, show_plot)
    save_regression_plot(predictions, output_dir, selected, show=show_plot,
                         model_name="Decision-tree regression", pca_components=pca_components,
                         cv_folds=cv_folds, original_label="Original features (6 unscaled)")
    for name, pipeline in models.items():
        save_selected_tree(pipeline, name, output_dir, pca_components, show_plot)
    table = results[["representation", "max_depth", "min_samples_leaf", "tree_depth", "number_of_leaves"]].copy()
    for metric, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]:
        table[f"CV {label} mean ± SD"] = [f"{r[f'cv_{metric}_mean']:.3f} ± {r[f'cv_{metric}_std']:.3f}"
                                          for _, r in results.iterrows()]
    for col in ["test_mae", "test_rmse", "test_r2", "root_feature", "root_threshold"]:
        table[col] = results[col]
    print(f"Split: {len(x_dev)} development / {len(x_test)} final-test rows; split seed={SPLIT_RANDOM_STATE}.")
    print(f"Selected by mean {cv_folds}-fold validation RMSE among {len(candidates)} supplied configurations per representation.")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Results saved to {output_dir.resolve()}")
    return models, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "decision_tree")
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-depth-values", type=parse_depth, nargs="+", default=MAX_DEPTH_VALUES)
    parser.add_argument("--min-samples-leaf-values", type=int, nargs="+", default=MIN_SAMPLES_LEAF_VALUES)
    parser.add_argument("--random-state", type=int, default=42,
                        help="Tree and CV seed; the final-test split remains fixed at 42.")
    args = parser.parse_args()
    try:
        run_decision_tree(
            args.input, args.output_dir, pca_components=args.pca_components,
            cv_folds=args.cv_folds, max_depth_values=args.max_depth_values,
            min_samples_leaf_values=args.min_samples_leaf_values,
            random_state=args.random_state, show_plot=True,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
