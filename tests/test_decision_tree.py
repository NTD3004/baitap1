import argparse
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import DEFAULT_INPUT, FEATURES, TARGET, load_features
from src.decision_tree import configurations, parse_depth, run_decision_tree, select_configurations


class DecisionTreeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(redirect_stdout(StringIO()))

    def suppress_plots(self):
        for function in ["save_tree_cv_plots", "save_regression_plot", "save_selected_tree"]:
            self.stack.enter_context(patch(f"src.decision_tree.{function}"))

    def test_grid_none_and_independent_selection(self):
        self.assertEqual(len(configurations()), 20)
        self.assertEqual(len(set(configurations())), 20)
        self.assertIn((None, 20), configurations())
        self.assertIsNone(parse_depth("none"))
        self.assertIsNone(parse_depth("None"))
        self.assertEqual(parse_depth("3"), 3)
        for value in ["0", "-1", "2.5", "bad"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_depth(value)
        summary = pd.DataFrame({"pipeline": ["without_pca"] * 2 + ["with_pca"] * 2,
                                "max_depth": ["none", "2", "none", "2"],
                                "min_samples_leaf": [1, 5, 1, 5],
                                "rmse_mean": [2., 1., 1., 2.]})
        selected = select_configurations(summary)
        self.assertEqual(selected.loc["without_pca", "max_depth"], "2")
        self.assertEqual(selected.loc["with_pca", "max_depth"], "none")
        summary["rmse_mean"] = 1.
        self.assertTrue((select_configurations(summary)["max_depth"] == "2").all())

    def test_fold_isolation_metrics_and_final_refit(self):
        self.suppress_plots()
        data, features = load_features(DEFAULT_INPUT)
        dev, test = train_test_split(features, test_size=.2, random_state=42)
        fits, predictions = [], []
        original_fit, original_predict = Pipeline.fit, Pipeline.predict

        def fit(model, x, y, **kwargs):
            result = original_fit(model, x, y, **kwargs)
            self.assertTrue(set(x.index).isdisjoint(test.index))
            self.assertEqual(list(x.columns), FEATURES)
            self.assertEqual(model["model"].criterion, "squared_error")
            self.assertEqual(model["model"].min_samples_split, 2)
            self.assertEqual(model["model"].random_state, 42)
            if "pca" in model.named_steps:
                np.testing.assert_allclose(model["scaler"].mean_, x.mean())
                self.assertEqual(model["pca"].n_samples_, len(x))
            else:
                self.assertEqual(list(model.named_steps), ["model"])
            fits.append((model, set(x.index)))
            return result

        def predict(model, x, **kwargs):
            if set(x.index) & set(test.index):
                self.assertEqual(set(x.index), set(test.index))
                self.assertGreaterEqual(len(fits), 201)
            predictions.append(set(x.index))
            return original_predict(model, x, **kwargs)

        with patch.object(Pipeline, "fit", new=fit), patch.object(Pipeline, "predict", new=predict):
            models, results = run_decision_tree(output_dir=self.root)
        self.assertEqual(len(fits), 202)
        self.assertEqual(len({id(model) for model, _ in fits}), 202)
        self.assertEqual(sum(bool(rows & set(test.index)) for rows in predictions), 2)
        folds = pd.read_csv(self.root / "cv_folds.csv").set_index("row_index")
        self.assertEqual(set(folds.index), set(dev.index))
        for fold in range(1, 6):
            validation = set(folds.index[folds["validation_fold"] == fold])
            for i in range((fold - 1) * 40, fold * 40):
                self.assertEqual(fits[i][1], set(dev.index) - validation)
                self.assertEqual(predictions[i], validation)
        for _, rows in fits[-2:]:
            self.assertEqual(rows, set(dev.index))
        cv = pd.read_csv(self.root / "cv_metrics.csv", keep_default_na=False)
        summary = pd.read_csv(self.root / "cv_summary.csv", keep_default_na=False)
        self.assertEqual(len(cv), 200)
        self.assertEqual(len(summary), 40)
        keys = ["pipeline", "max_depth", "min_samples_leaf"]
        grouped = cv.groupby(keys)
        ordered = summary.set_index(keys).sort_index()
        for metric in ["mae", "rmse", "r2"]:
            np.testing.assert_allclose(ordered[f"{metric}_mean"], grouped[metric].mean())
            np.testing.assert_allclose(ordered[f"{metric}_std"], grouped[metric].std(ddof=1))
        saved = pd.read_csv(self.root / "predictions.csv")
        np.testing.assert_array_equal(saved["row_index"], test.index)
        np.testing.assert_allclose(saved["actual"], data.loc[test.index, TARGET])
        for name, model in models.items():
            np.testing.assert_allclose(saved[f"residual_{name}"], saved["actual"] - saved[name])
            np.testing.assert_allclose(saved[name], model.predict(test))
            row = results.set_index("pipeline").loc[name]
            self.assertAlmostEqual(row["cv_rmse_mean"], summary.loc[summary["pipeline"] == name, "rmse_mean"].min())
            self.assertEqual(row["tree_depth"], model["model"].get_depth())
            self.assertEqual(row["number_of_leaves"], model["model"].get_n_leaves())
            names = FEATURES if name == "without_pca" else ["PC1", "PC2", "PC3"]
            self.assertEqual(row["root_feature"], names[model["model"].tree_.feature[0]])

    def test_test_changes_do_not_affect_cv_or_selection(self):
        self.suppress_plots()
        rng = np.random.default_rng(4)
        data = pd.DataFrame(rng.normal(size=(40, 6)), columns=FEATURES)
        data[TARGET] = data[FEATURES[0]] ** 2 + rng.normal(size=40)
        _, test = train_test_split(data.index, test_size=.2, random_state=42)
        workbook = self.root / "input.xlsx"
        data.to_excel(workbook, index=False)
        options = dict(max_depth_values=[2, None], min_samples_leaf_values=[1, 5])
        _, before = run_decision_tree(workbook, self.root / "before", **options)
        data.loc[test, FEATURES] += 1000
        data.loc[test, TARGET] += 10000
        data.to_excel(workbook, index=False)
        _, after = run_decision_tree(workbook, self.root / "after", **options)
        for filename in ["cv_metrics.csv", "cv_summary.csv", "cv_folds.csv"]:
            pd.testing.assert_frame_equal(pd.read_csv(self.root / "before" / filename),
                                          pd.read_csv(self.root / "after" / filename))
        pd.testing.assert_frame_equal(before[["max_depth", "min_samples_leaf"]],
                                      after[["max_depth", "min_samples_leaf"]])

    def test_outputs_and_seed_preserves_split(self):
        run_decision_tree(output_dir=self.root, max_depth_values=[2], min_samples_leaf_values=[5],
                          random_state=7, pca_components=2, cv_folds=4)
        for stem in ["cv_metrics", "cv_summary", "predictions", "summary", "development", "test", "cv_folds"]:
            self.assertTrue((self.root / f"{stem}.csv").is_file())
        for stem in ["regression_diagnostics", "cv_rmse_vs_depth_without_pca", "cv_rmse_vs_depth_with_pca",
                     "cv_heatmap_without_pca", "cv_heatmap_with_pca", "tree_without_pca", "tree_with_pca"]:
            for extension in ["png", "svg"]:
                self.assertGreater((self.root / f"{stem}.{extension}").stat().st_size, 0)
        _, features = load_features(DEFAULT_INPUT)
        _, test = train_test_split(features, test_size=.2, random_state=42)
        np.testing.assert_array_equal(pd.read_csv(self.root / "test.csv")["row_index"], test.index)
        for options in [dict(max_depth_values=[]), dict(min_samples_leaf_values=[0]),
                        dict(cv_folds=1), dict(cv_folds=166), dict(pca_components=7), dict(random_state=-1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_decision_tree(output_dir=self.root, **options)


if __name__ == "__main__":
    unittest.main()
