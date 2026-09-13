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
from src.knn_regression import K_VALUES, run_knn, select_candidates


class KNNTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch("src.knn_regression.save_cv_curve"))
        self.stack.enter_context(patch("src.knn_regression.save_regression_plot"))
        self.stack.enter_context(redirect_stdout(StringIO()))

    def test_shared_folds_selection_and_final_test(self):
        data, features = load_features(DEFAULT_INPUT)
        dev, test = train_test_split(features, test_size=0.2, random_state=42)
        fits, test_predictions = [], []
        original_fit, original_predict = Pipeline.fit, Pipeline.predict

        def fit(pipeline, x, y, **kwargs):
            result = original_fit(pipeline, x, y, **kwargs)
            self.assertTrue(set(x.index).isdisjoint(test.index))
            np.testing.assert_allclose(pipeline["scaler"].mean_, x.mean())
            self.assertEqual(list(pipeline["scaler"].feature_names_in_), FEATURES)
            self.assertEqual(pipeline["model"].weights, "uniform")
            self.assertEqual(pipeline["model"].metric, "euclidean")
            self.assertLessEqual(pipeline["model"].n_neighbors, len(x))
            if "pca" in pipeline.named_steps:
                self.assertEqual(pipeline["pca"].n_samples_, len(x))
            fits.append((pipeline, set(x.index)))
            return result

        def predict(pipeline, x, **kwargs):
            if set(x.index) & set(test.index):
                self.assertEqual(set(x.index), set(test.index))
                self.assertGreaterEqual(len(fits), 71)
                test_predictions.append(pipeline)
            return original_predict(pipeline, x, **kwargs)

        with patch.object(Pipeline, "fit", new=fit), patch.object(Pipeline, "predict", new=predict):
            models, results = run_knn(output_dir=self.root)
        self.assertEqual(len(fits), 72)
        self.assertEqual(len({id(model) for model, _ in fits}), 72)
        self.assertEqual(len(test_predictions), 2)
        folds = pd.read_csv(self.root / "cv_folds.csv").set_index("row_index")
        self.assertEqual(set(folds.index), set(dev.index))
        for fold in range(1, 6):
            expected = set(folds.index[folds["validation_fold"] != fold])
            for _, indices in fits[(fold - 1) * 14:fold * 14]:
                self.assertEqual(indices, expected)
        cv = pd.read_csv(self.root / "cv_metrics.csv")
        self.assertEqual(len(cv), 70)
        summary = pd.read_csv(self.root / "cv_summary.csv").set_index(["pipeline", "k"])
        grouped = cv.groupby(["pipeline", "k"])
        for metric in ["mae", "rmse", "r2"]:
            np.testing.assert_allclose(summary[f"{metric}_mean"], grouped[metric].mean())
            np.testing.assert_allclose(summary[f"{metric}_std"], grouped[metric].std(ddof=1))
        predictions = pd.read_csv(self.root / "predictions.csv")
        np.testing.assert_array_equal(predictions["row_index"], test.index)
        np.testing.assert_allclose(predictions["actual"], data.loc[test.index, TARGET])
        for name, model in models.items():
            expected_k = summary.loc[name].sort_values("rmse_mean", kind="stable").index[0]
            self.assertEqual(model["model"].n_neighbors, expected_k)
            self.assertEqual(model["scaler"].n_samples_seen_, len(dev))
            np.testing.assert_allclose(predictions[name], model.predict(test))
            self.assertEqual(int(results.set_index("pipeline").loc[name, "k"]), expected_k)

    def test_final_test_changes_cannot_change_selection_or_cv(self):
        rng = np.random.default_rng(13)
        data = pd.DataFrame(rng.normal(size=(40, 6)), columns=FEATURES)
        data[TARGET] = 4 * data[FEATURES[0]] + rng.normal(size=40)
        _, test = train_test_split(data.index, test_size=0.2, random_state=42)
        workbook = self.root / "input.xlsx"
        data.to_excel(workbook, index=False)
        _, before = run_knn(workbook, self.root / "before", k_values=[3, 5], cv_folds=4)
        data.loc[test, FEATURES] += 1000
        data.loc[test, TARGET] -= 10000
        data.to_excel(workbook, index=False)
        _, after = run_knn(workbook, self.root / "after", k_values=[3, 5], cv_folds=4)
        np.testing.assert_array_equal(before["k"], after["k"])
        for filename in ["cv_metrics.csv", "cv_summary.csv", "cv_folds.csv"]:
            pd.testing.assert_frame_equal(pd.read_csv(self.root / "before" / filename),
                                          pd.read_csv(self.root / "after" / filename))

    def test_invalid_candidates_and_fold_sizes(self):
        for options in [dict(k_values=[]), dict(k_values=[0]), dict(k_values=[-1]),
                        dict(k_values=[3.5]), dict(k_values=[True]), dict(k_values=[265]),
                        dict(cv_folds=1), dict(cv_folds=166), dict(pca_components=7),
                        dict(pca_components=0)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_knn(output_dir=self.root, **options)
        # 331 development rows: smallest training fold has 264 observations.
        models, _ = run_knn(output_dir=self.root, k_values=[264])
        self.assertEqual(models["without_pca"]["model"].n_neighbors, 264)

    def test_selection_is_independent_and_ties_choose_smaller_k(self):
        summary = pd.DataFrame({"pipeline": ["without_pca"] * 2 + ["with_pca"] * 2,
                                "k": [5, 3, 3, 5], "rmse_mean": [1., 1., 2., 1.]})
        selected = select_candidates(summary)
        self.assertEqual(selected.loc["without_pca", "k"], 3)
        self.assertEqual(selected.loc["with_pca", "k"], 5)


if __name__ == "__main__":
    unittest.main()
