"""Integration checks for preprocessing isolation and analysis outputs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from src.data import DEFAULT_INPUT, FEATURES, TARGET, PROJECT_ROOT, load_features
from src.linear_regression import run_regression
from src.pca_analysis import run_pca
from src.regression_plots import build_regression_figure


class AnalysisTests(unittest.TestCase):
    def test_configurable_linear_components_and_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.linear_regression.save_regression_plot") as plot:
                models, metrics = run_regression(output_dir=root, pca_components=2, cv_folds=4)
            self.assertEqual(models["with_pca"]["pca"].n_components_, 2)
            self.assertNotIn("pca", models["without_pca"].named_steps)
            cv = pd.read_csv(root / "cv_metrics.csv")
            self.assertEqual(len(cv), 8)
            self.assertEqual(set(cv["fold"]), {1, 2, 3, 4})
            self.assertTrue((metrics["pca_components"] == 2).all())
            self.assertTrue((metrics["cv_folds"] == 4).all())
            self.assertEqual(plot.call_args.kwargs["pca_components"], 2)
            self.assertEqual(plot.call_args.kwargs["cv_folds"], 4)
            for options in [{"pca_components": 0}, {"pca_components": 7},
                            {"pca_components": 2.5}, {"cv_folds": 1},
                            {"cv_folds": 166}, {"cv_folds": True}]:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run_regression(output_dir=root, **options)

    def test_diagnostic_plot_axes_residual_sign_and_metrics(self):
        predictions = pd.DataFrame({
            "actual": [10., 20., 30.],
            "without_pca": [12., 18., 30.],
            "with_pca": [8., 25., 29.],
        })
        cv_summary = pd.DataFrame({
            "pipeline": ["with_pca", "without_pca"],
            "mae_mean": [2.1, 1.1], "mae_std": [0.2, 0.1],
            "rmse_mean": [3.1, 2.1], "rmse_std": [0.4, 0.3],
            "r2_mean": [0.7, 0.8], "r2_std": [0.06, 0.05],
        }).set_index("pipeline")
        fig = build_regression_figure(predictions, cv_summary)
        try:
            self.assertEqual(len(fig.axes), 4)
            for column, name in enumerate(["without_pca", "with_pca"]):
                top, bottom = fig.axes[column], fig.axes[column + 2]
                np.testing.assert_allclose(top.collections[0].get_offsets(),
                                           predictions[["actual", name]].to_numpy())
                np.testing.assert_allclose(bottom.collections[0].get_offsets(),
                                           np.column_stack((predictions[name],
                                                            predictions["actual"] - predictions[name])))
                np.testing.assert_allclose(top.lines[0].get_xdata(), top.lines[0].get_ydata())
                np.testing.assert_allclose(bottom.lines[0].get_ydata(), 0)
                self.assertEqual(top.get_xlim(), top.get_ylim())
            self.assertEqual(fig.axes[0].get_xlim(), fig.axes[1].get_xlim())
            self.assertEqual(fig.axes[2].get_ylim(), fig.axes[3].get_ylim())
            labels = "\n".join(text.get_text() for text in fig.texts)
            self.assertIn("MAE: 1.333    RMSE: 1.633    R²: 0.960", labels)
            self.assertIn("MAE: 2.667    RMSE: 3.162    R²: 0.850", labels)
            for name, x in [("without_pca", 0.09), ("with_pca", 0.5)]:
                cv_labels = [text for text in fig.texts
                             if x < text.get_position()[0] < x + 0.4
                             and 0.10 < text.get_position()[1] < 0.18]
                self.assertEqual(len(cv_labels), 4)
                self.assertIn("Training 5-fold CV (mean ± sample SD)",
                              [text.get_text() for text in cv_labels])
                for metric, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]:
                    cv = cv_summary.loc[name]
                    self.assertIn(f"{label}: {cv[f'{metric}_mean']:.3f} ± {cv[f'{metric}_std']:.3f}",
                                  [text.get_text() for text in cv_labels])
        finally:
            plt.close(fig)

    def test_shared_five_folds_exclude_test_rows_and_fit_fresh_pipelines(self):
        _, features = load_features(DEFAULT_INPUT)
        train, test = train_test_split(features, test_size=0.2, random_state=42)
        fitted = []
        original_fit = Pipeline.fit

        def record_fit(pipeline, x, y, **kwargs):
            result = original_fit(pipeline, x, y, **kwargs)
            fitted.append((pipeline, x.index.copy()))
            np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, x.mean())
            if "pca" in pipeline.named_steps:
                self.assertEqual(pipeline.named_steps["pca"].n_samples_, len(x))
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(Pipeline, "fit", new=record_fit):
                run_regression(output_dir=root)
            assignments = pd.read_csv(root / "cv_folds.csv").set_index("row_index")
            cv = pd.read_csv(root / "cv_metrics.csv")
            self.assertEqual(len(cv), 10)
            self.assertEqual(set(assignments.index), set(train.index))
            self.assertTrue(set(assignments.index).isdisjoint(test.index))
            self.assertEqual(set(assignments["validation_fold"]), set(range(1, 6)))
            self.assertEqual(len(fitted), 12)
            self.assertEqual(len({id(p) for p, _ in fitted}), 12)
            for fold in range(1, 6):
                expected = set(assignments.index[assignments["validation_fold"] != fold])
                for _, indices in fitted[2 * (fold - 1):2 * fold]:
                    self.assertEqual(set(indices), expected)
                rows = cv[cv["fold"] == fold]
                self.assertEqual(set(rows["pipeline"]), {"with_pca", "without_pca"})
                self.assertTrue((rows["train_rows"] == len(expected)).all())
                self.assertTrue((rows["validation_rows"] == len(train) - len(expected)).all())
            for _, indices in fitted[-2:]:
                self.assertEqual(set(indices), set(train.index))
            for name, expected in [("train", train), ("test", test)]:
                saved = pd.read_csv(root / f"{name}.csv")
                np.testing.assert_array_equal(saved["row_index"], expected.index)
            summary = pd.read_csv(root / "cv_summary.csv").set_index("pipeline")
            for metric in ["mae", "mse", "rmse", "r2"]:
                grouped = cv.groupby("pipeline")[metric]
                np.testing.assert_allclose(summary[f"{metric}_mean"].sort_index(), grouped.mean())
                np.testing.assert_allclose(summary[f"{metric}_std"].sort_index(), grouped.std())

    def test_regression_uses_training_statistics_and_aligned_predictions(self):
        data, features = load_features(DEFAULT_INPUT)
        train, test = train_test_split(features, test_size=0.2, random_state=42)
        with tempfile.TemporaryDirectory() as directory:
            pipelines, metrics = run_regression(output_dir=Path(directory))
            predictions = pd.read_csv(Path(directory) / "predictions.csv")
            for extension in ["png", "svg"]:
                self.assertGreater((Path(directory) / f"regression_diagnostics.{extension}").stat().st_size, 0)
            np.testing.assert_array_equal(predictions["row_index"], test.index)
            np.testing.assert_allclose(predictions["actual"], data.loc[test.index, TARGET])
            self.assertTrue(np.isfinite(metrics[["mae", "mse", "rmse", "r2"]]).all().all())
            for name, pipeline in pipelines.items():
                scaler = pipeline.named_steps["scaler"]
                self.assertEqual(scaler.n_samples_seen_, len(train))
                self.assertEqual(list(scaler.feature_names_in_), FEATURES)
                np.testing.assert_allclose(scaler.mean_, train.mean())
                np.testing.assert_allclose(predictions[name], pipeline.predict(test))
            self.assertNotIn("pca", pipelines["without_pca"].named_steps)
            self.assertEqual(pipelines["with_pca"].named_steps["pca"].n_components_, 3)
            self.assertIsNot(pipelines["with_pca"].named_steps["scaler"],
                             pipelines["without_pca"].named_steps["scaler"])

    def test_held_out_changes_do_not_affect_fitted_preprocessing(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(size=(30, 6)), columns=FEATURES)
        data[TARGET] = data[FEATURES[0]] * 2 + 3
        _, test = train_test_split(data.index, test_size=0.2, random_state=42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "input.xlsx"
            data.to_excel(workbook, index=False)
            before, _ = run_regression(workbook, root / "before")
            data.loc[test, FEATURES] += 10000
            data.loc[test, TARGET] += 50000
            data.to_excel(workbook, index=False)
            after, _ = run_regression(workbook, root / "after")
            for filename in ["cv_metrics.csv", "cv_summary.csv", "cv_folds.csv"]:
                pd.testing.assert_frame_equal(pd.read_csv(root / "before" / filename),
                                              pd.read_csv(root / "after" / filename))
            for name in before:
                np.testing.assert_allclose(before[name].named_steps["scaler"].mean_,
                                           after[name].named_steps["scaler"].mean_)
                np.testing.assert_allclose(before[name].named_steps["model"].coef_,
                                           after[name].named_steps["model"].coef_)
            np.testing.assert_allclose(before["with_pca"].named_steps["pca"].components_,
                                       after["with_pca"].named_steps["pca"].components_)

    def test_invalid_target_and_split(self):
        data, _ = load_features(DEFAULT_INPUT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "input.xlsx"
            for value in [np.nan, np.inf, "invalid"]:
                invalid = data.copy()
                invalid[TARGET] = invalid[TARGET].astype(object)
                invalid.loc[0, TARGET] = value
                invalid.to_excel(workbook, index=False)
                with self.assertRaises(ValueError):
                    run_regression(workbook, root / "output")
            data.drop(columns=TARGET).to_excel(workbook, index=False)
            with self.assertRaisesRegex(ValueError, "Missing target"):
                run_regression(workbook, root / "output")
            for size in [0, 1, 0.001, 0.98, 0.999]:
                with self.assertRaises(ValueError):
                    run_regression(output_dir=root / "output", test_size=size)

    def test_pca_summary_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_pca(DEFAULT_INPUT, Path(directory))
            expected = pd.read_csv(PROJECT_ROOT / "results" / "pca" / "summary.csv")
            pd.testing.assert_frame_equal(summary, expected, check_exact=False, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
