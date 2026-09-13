"""Run PCA for M=1,...,6 on the real estate workbook.

Run from any directory: python /path/to/project/src/pca_analysis.py
Outputs default to project/results/pca; override with --output-dir.
The scree chart is saved and displayed in Jupyter or a desktop plot window.
In Jupyter, run: %run src/pca_analysis.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

if __package__:
    from .data import DEFAULT_INPUT, FEATURES, PROJECT_ROOT, load_features
else:
    from data import DEFAULT_INPUT, FEATURES, PROJECT_ROOT, load_features


ELBOW_REGION = (2, 3)
CEV_TARGET = 0.95


def select_m_binary_search(cev: np.ndarray, lower: int, upper: int,
                           target: float) -> tuple[int, bool]:
    """Find the smallest integer M meeting target in an inclusive region.

    CEV must be nondecreasing and indexed by M-1. If no feasible M exists,
    return the upper boundary as an explicit best-variance fallback.
    """
    if not 1 <= lower <= upper <= len(cev):
        raise ValueError("Invalid component interval.")
    if not 0 < target <= 1 or not np.isfinite(cev).all():
        raise ValueError("A finite CEV curve and target in (0, 1] are required.")
    if np.any(np.diff(cev) < -1e-12):
        raise ValueError("CEV must be nondecreasing for binary search.")
    left, right = lower, upper
    match = None
    while left <= right:
        midpoint = (left + right) // 2
        if cev[midpoint - 1] >= target - 1e-12:
            match = midpoint
            right = midpoint - 1
        else:
            left = midpoint + 1
    return (match, True) if match is not None else (upper, False)


def calculate_cev(summary: pd.DataFrame) -> None:
    """Calculate CEV and select an integer M in the user-specified region."""
    counts = summary["M"].to_numpy()
    cev = summary["cumulative_variance_ratio"].to_numpy()
    lower, upper = ELBOW_REGION
    selected, reached = select_m_binary_search(cev, lower, upper, CEV_TARGET)
    summary["cev_percent"] = cev * 100
    summary["in_elbow_region"] = (counts >= lower) & (counts <= upper)
    summary["selected_m"] = counts == selected
    summary["meets_cev_target"] = cev >= CEV_TARGET - 1e-12
    print(f"Elbow region: PC{lower}-PC{upper}; binary-search CEV target: {CEV_TARGET:.0%}.")
    if not reached:
        print("No M in the region meets the target; using its upper boundary "
              "to retain the most variance within the region.")
    print(f"Selected M={selected}: CEV={cev[selected - 1]:.2%}.")


def display_plot(fig, image_path: Path) -> None:
    """Display a chart inline or in a GUI, with a saved-file fallback."""
    import matplotlib.pyplot as plt
    # %run executes inside the notebook kernel, allowing a real inline image.
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is not None and getattr(shell, "kernel", None) is not None:
        from IPython.display import display
        display(fig)
    elif plt.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
        print(f"Chart saved: {image_path.resolve()}")
        print("To see the chart inline in Jupyter, run: %run src/pca_analysis.py")
    else:
        plt.show()
    plt.close(fig)


def save_scree_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot each component's individual (not cumulative) explained variance."""
    import matplotlib.pyplot as plt

    counts = summary["M"].to_numpy()
    percentages = summary["explained_variance_ratio"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    fig.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.fill_between(counts, percentages, color="#2563eb", alpha=0.07)
    ax.plot(counts, percentages, "o-", color="#2563eb", linewidth=2.8,
            markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    lower, upper = ELBOW_REGION
    ax.axvspan(lower, upper, color="#ea580c", alpha=0.12, zorder=0)
    for boundary in (lower, upper):
        ax.axvline(boundary, color="#ea580c", linestyle="--", linewidth=1.2, alpha=0.65)
    ax.text((lower + upper) / 2, 0.94, "Elbow region", ha="center",
            transform=ax.get_xaxis_transform(), color="#c2410c", fontsize=11)
    selected_index = int(np.flatnonzero(summary["selected_m"].to_numpy())[0])
    ax.scatter([counts[selected_index]], [percentages[selected_index]], s=180,
               facecolors="none", edgecolors="#ea580c", linewidths=2, zorder=3)
    for m, value in zip(counts, percentages):
        ax.annotate(f"{value:.2f}%", (m, value), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=11,
                    color="#334155")
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)",
           xticks=counts, xticklabels=[f"PC{m}" for m in counts],
           xlim=(0.8, 6.2), ylim=(0, percentages.max() * 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", length=0, pad=8)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "scree.png", dpi=200)
    fig.savefig(output_dir / "scree.svg")
    display_plot(fig, output_dir / "scree.png")


def run_pca(input_path: Path, output_dir: Path) -> pd.DataFrame:
    """Save scores, component directions, and variance for each PCA model."""
    data, features = load_features(input_path)
    if len(features) < 6:
        raise ValueError("At least six observations are required for M=6.")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for m in range(1, 7):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=m, svd_solver="full")),
        ])
        scores = pipeline.fit_transform(features)
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["pca"]
        standardized = scaler.transform(features)
        if m == 1:
            pd.DataFrame(
                {"feature": FEATURES, "mean": scaler.mean_, "scale": scaler.scale_}
            ).to_csv(output_dir / "scaling.csv", index=False)
        components = [f"PC{i}" for i in range(1, m + 1)]
        # Keep identifiers and targets as metadata, never as PCA inputs.
        metadata = data[[c for c in data.columns if c not in FEATURES]]
        pd.concat(
            [metadata, pd.DataFrame(scores, columns=components, index=data.index)],
            axis=1,
        ).to_csv(output_dir / f"scores_m{m}.csv", index=False)
        pd.DataFrame(
            model.components_.T, index=FEATURES, columns=components
        ).to_csv(output_dir / f"components_m{m}.csv", index_label="feature")
        pd.DataFrame(
            {
                "component": components,
                "explained_variance": model.explained_variance_,
                "explained_variance_ratio": model.explained_variance_ratio_,
                "cumulative_variance_ratio": np.cumsum(model.explained_variance_ratio_),
            }
        ).to_csv(output_dir / f"variance_m{m}.csv", index=False)
        reconstruction = model.inverse_transform(scores)
        summary.append(
            {
                "M": m,
                "explained_variance_ratio": model.explained_variance_ratio_[-1],
                "cumulative_variance_ratio": model.explained_variance_ratio_.sum(),
                "reconstruction_mse_standardized": np.mean(
                    (standardized - reconstruction) ** 2
                ),
            }
        )

    result = pd.DataFrame(summary)
    calculate_cev(result)
    result.to_csv(output_dir / "summary.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "pca")
    args = parser.parse_args()
    summary = run_pca(args.input, args.output_dir)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    save_scree_plot(summary, args.output_dir)
    print(f"\nResults saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
