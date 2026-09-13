"""Visual comparison of both regression pipelines on held-out test rows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_regression_figure(predictions: pd.DataFrame, cv_summary: pd.DataFrame,
                            model_name: str = "Linear regression", pca_components: int = 3,
                            cv_folds: int = 5, selected_k: dict | None = None,
                            original_label: str = "Original features (6 standardized)"):
    """Build test plots with test metrics and training CV mean/sample SD."""
    actual = predictions["actual"].to_numpy()
    models = [
        ("without_pca", original_label, "#2563eb"),
        ("with_pca", f"PCA ({pca_components} components)", "#ea580c"),
    ]
    values = predictions[["actual", "without_pca", "with_pca"]].to_numpy()
    padding = max(float(np.ptp(values)) * 0.06, 1.0)
    price_limits = (float(values.min()) - padding, float(values.max()) + padding)
    residual_extent = max(
        float(np.abs(actual[:, None] - values[:, 1:]).max()) * 1.12, 1.0,
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.26, top=0.87,
                        hspace=0.36, wspace=0.27)
    fig.suptitle(f"{model_name}: effect of PCA", fontsize=19, fontweight="bold", y=0.97)
    fig.text(0.5, 0.935, f"Held-out test set · {len(actual)} properties · identical rows for both models",
             ha="center", fontsize=11, color="#475569")

    for column, (name, title, color) in enumerate(models):
        if selected_k is not None:
            title += f" · k={selected_k[name]}"
        predicted = predictions[name].to_numpy()
        residuals = actual - predicted
        top, bottom = axes[:, column]
        top.set_title(title, fontsize=13, fontweight="bold", pad=12)
        top.scatter(actual, predicted, color=color, alpha=0.7, s=33,
                    edgecolors="white", linewidths=0.4, zorder=3)
        top.plot(price_limits, price_limits, "--", color="#334155", linewidth=1.4,
                 label=r"Perfect prediction: $\widehat{t}=t$")
        top.set(xlabel=r"Actual price $t_n$", ylabel=r"Predicted price $\widehat{t}_n$",
                xlim=price_limits, ylim=price_limits)
        top.set_aspect("equal", adjustable="box")
        top.legend(loc="upper left", fontsize=9, frameon=False)

        bottom.scatter(predicted, residuals, color=color, alpha=0.7, s=33,
                       edgecolors="white", linewidths=0.4, zorder=3)
        bottom.axhline(0, linestyle="--", color="#334155", linewidth=1.4, label=r"$e=0$")
        bottom.set(xlabel=r"Predicted price $\widehat{t}_n$",
                   ylabel=r"Residual $e_n=t_n-\widehat{t}_n$",
                   xlim=price_limits, ylim=(-residual_extent, residual_extent))
        bottom.set_title("Residual vs predicted", fontsize=11, pad=10)
        bottom.legend(loc="upper left", fontsize=9, frameon=False)

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        position = bottom.get_position()
        center = (position.x0 + position.x1) / 2
        fig.text(center, 0.20,
                 f"Test MAE: {mae:.3f}    RMSE: {rmse:.3f}    R²: {r2:.3f}",
                 ha="center", fontsize=11, color=color, fontweight="bold")
        cv = cv_summary.loc[name]
        fig.text(center, 0.17, f"Training {cv_folds}-fold CV (mean ± sample SD)",
                 ha="center", fontsize=11, color=color, fontweight="bold")
        for y, metric, label in [(0.147, "mae", "MAE"), (0.127, "rmse", "RMSE"),
                                 (0.107, "r2", "R²")]:
            fig.text(center, y, f"{label}: {cv[f'{metric}_mean']:.3f} ± {cv[f'{metric}_std']:.3f}",
                     ha="center", fontsize=11, color=color)

    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.075, "Prices and errors are in the workbook's house-price-per-unit-area units.",
             ha="center", fontsize=10, color="#475569")
    fig.text(0.5, 0.047, "Above diagonal: overprediction · Below diagonal: underprediction",
             ha="center", fontsize=10, color="#475569")
    fig.text(0.5, 0.022, "Positive residual: underprediction · Negative residual: overprediction",
             ha="center", fontsize=10, color="#475569")
    return fig


def save_regression_plot(predictions: pd.DataFrame, output_dir: Path, cv_summary: pd.DataFrame,
                         show: bool = False, **figure_options) -> Path:
    """Save PNG/SVG versions and optionally display inline in Jupyter or a GUI."""
    fig = build_regression_figure(predictions, cv_summary, **figure_options)
    return save_figure(fig, output_dir, "regression_diagnostics", show)


def save_figure(fig, output_dir: Path, stem: str, show: bool = False) -> Path:
    """Save a figure in both formats and close it after optional display."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{stem}.png"
    try:
        fig.savefig(image_path, dpi=200)
        fig.savefig(output_dir / f"{stem}.svg")
        if show:
            try:
                from IPython import get_ipython
                shell = get_ipython()
            except ImportError:
                shell = None
            if shell is not None and getattr(shell, "kernel", None) is not None:
                from IPython.display import display
                display(fig)
            elif plt.get_backend().lower() not in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
                plt.show()
            else:
                print(f"Chart saved: {image_path.resolve()}")
    finally:
        plt.close(fig)
    return image_path
