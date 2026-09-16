"""Regression checks for preprocessing, fold isolation, and deployment policy."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.config import get_project_root
from src.evaluation import create_cv_splits
from src.experiment import (
    MODEL_NAMES, PRIMARY, ExperimentConfig, add_extra_features,
    apply_calibration, bootstrap_summary, calibrate, make_submission, select_members,
)
from src.features import build_features
from src.imputation import make_preprocessor

EMPTY_DICTIONARY = pd.DataFrame(columns=["Field", "Type", "Values"])


class ProjectTests(unittest.TestCase):
    def test_missing_measurements_ignore_season(self):
        raw = pd.DataFrame({"id": ["a", "b", "c"], "FGC-Season": ["Spring", "Summer", None],
                            "FGC-FGC_CU": [np.nan, 2, np.nan]})
        prepared = build_features(raw, EMPTY_DICTIONARY)
        self.assertEqual(prepared.FGC_measurements_missing.tolist(), [1, 0, 1])
        self.assertNotIn("FGC_not_administered", prepared)

    def test_features_are_row_local_and_remove_leakage(self):
        raw = pd.DataFrame({"id": ["a", "b"], "sii": [0, 3], "PCIAT-PCIAT_Total": [10, 90],
                            "CGAS-Season": ["Spring", "Winter"], "CGAS-CGAS_Score": [999, 80]})
        full = build_features(raw, EMPTY_DICTIONARY)
        individual = pd.concat([build_features(raw.iloc[[i]], EMPTY_DICTIONARY) for i in range(len(raw))])
        pd.testing.assert_frame_equal(full, individual)
        self.assertFalse(any(c.startswith("PCIAT-") for c in full))
        self.assertEqual(full.sii.tolist(), [0, 3])
        self.assertTrue(pd.isna(full.loc[0, "CGAS-CGAS_Score"]))
        self.assertEqual(full.CGAS_measurements_missing.tolist(), [1, 0])

    def test_default_imputation_is_median_and_training_only(self):
        train = pd.DataFrame({"value": [1.0, np.nan, 3.0], "season": ["Spring"] * 3})
        pipeline = Pipeline([("preprocessor", make_preprocessor(train)), ("model", Ridge())])
        pipeline.fit(train, [0, 1, 2])
        imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
        self.assertEqual(imputer.strategy, "median")
        np.testing.assert_allclose(imputer.statistics_, [2.0])
        pipeline.predict(pd.DataFrame({"value": [1e9, np.nan], "season": ["NewSeason", "Spring"]}))
        np.testing.assert_allclose(imputer.statistics_, [2.0])

    def test_folds_are_id_stable_disjoint_and_cover_once(self):
        ids = pd.Series([f"p{i:03}" for i in range(80)])
        X, y = pd.DataFrame({"a": np.arange(80)}), pd.Series(np.tile(np.arange(4), 20))
        splits = create_cv_splits(X, y, ids)
        self.assertEqual(sorted(np.concatenate([v for _, v in splits]).tolist()), list(range(80)))
        for train, val in splits:
            self.assertEqual(len(np.intersect1d(train, val)), 0)
        permutation = np.random.default_rng(4).permutation(80)
        reordered = create_cv_splits(X.iloc[permutation], y.iloc[permutation], ids.iloc[permutation])
        for (_, a), (_, b) in zip(splits, reordered):
            self.assertEqual(set(ids.iloc[a]), set(ids.iloc[permutation].iloc[b]))

    def test_clipping_bounds_are_training_only(self):
        train = pd.DataFrame({"value": np.arange(100, dtype=float)})
        preprocessor = make_preprocessor(train, scale=False, winsorize=0.01)
        preprocessor.fit(train)
        clipper = preprocessor.named_transformers_["numeric"].named_steps["clipper"]
        np.testing.assert_allclose(clipper.lower_, [0.99])
        np.testing.assert_allclose(clipper.upper_, [98.01])
        transformed = preprocessor.transform(pd.DataFrame({"value": [-1e9, 1e9]}))
        np.testing.assert_allclose(transformed.ravel(), [0.99, 98.01])
        np.testing.assert_allclose(clipper.lower_, [0.99])
        np.testing.assert_allclose(clipper.upper_, [98.01])

    def test_candidate_training_cannot_see_its_validation_rows(self):
        # Use a tiny fake estimator to audit the data access of every candidate.
        X = pd.DataFrame({"row": np.arange(48)})
        y = pd.Series(np.tile(np.arange(4), 12))
        ids = pd.Series([f"p{i:03}" for i in range(48)])
        calls = []

        class FakeModel:
            def __init__(self, fit_rows):
                self.fit_rows = set(fit_rows)

            def predict(self, validation):
                val_rows = set(validation["row"])
                if self.fit_rows & val_rows:
                    raise AssertionError("A validation participant was included in training")
                calls.append((self.fit_rows, val_rows))
                return validation["row"].to_numpy() % 4

        def fake_fit(name, frame, labels, params, seed, config):
            self.assertEqual(frame.index.tolist(), labels.index.tolist())
            return FakeModel(frame["row"]), None

        grids = {name: [{"candidate": 0}, {"candidate": 1}] for name in MODEL_NAMES}
        with patch("src.experiment.fit_member", side_effect=fake_fit), \
             patch("src.experiment.model_frame", side_effect=lambda name, base: base), \
             contextlib.redirect_stdout(io.StringIO()):
            baseline, tuned, _, _ = select_members(X, y, ids, 42, ExperimentConfig(), grids)
        self.assertEqual(len(calls), 4 * 2 * 3)
        np.testing.assert_allclose(baseline, tuned)

    def test_primary_is_exactly_equal_weight_policy(self):
        y = np.tile(np.arange(4), 15)
        inner = np.column_stack([y * 0.3 + offset for offset in [0.1, 0.2, 0.3, 0.4]])
        calibration = calibrate(inner, inner, y, ExperimentConfig())
        predictions = apply_calibration(inner, inner, calibration, 0)
        from src.ensemble import to_classes
        np.testing.assert_array_equal(predictions[PRIMARY], to_classes(inner.mean(axis=1), calibration["equal_thresholds"]))

    def test_submission_id_alignment_and_validation(self):
        sample = pd.DataFrame({"id": ["b", "a"], "sii": [0, 0]})
        submission = make_submission(np.array([1, 3]), pd.Series(["a", "b"]), sample)
        self.assertEqual(submission.sii.tolist(), [3, 1])
        with self.assertRaises(ValueError):
            make_submission(np.array([1, 3]), pd.Series(["a", "a"]), sample)
        with self.assertRaises(ValueError):
            make_submission(np.array([4, 3]), pd.Series(["a", "b"]), sample)

    def test_bootstrap_clusters_repeats_by_participant(self):
        y = np.tile(np.arange(4), 10)
        from src.experiment import REFERENCE
        predictions = {PRIMARY: y.copy(), REFERENCE: y.copy(), "Baseline equal ensemble": y.copy(), "Tuned xgboost": y.copy()}
        summary, deltas = bootstrap_summary(y, {42: predictions, 2026: predictions}, samples=20)
        np.testing.assert_allclose(summary.mean_oof_qwk, 1)
        np.testing.assert_allclose(deltas[["delta_qwk", "ci_low", "ci_high"]], 0)

    def test_invalid_cv_configuration_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(outer_seeds=(42, 42))
        with self.assertRaises(ValueError):
            ExperimentConfig(inner_splits=1)

    def test_extra_feature_preserves_missing_block(self):
        columns = ["Fitness_Endurance-Time_Mins", "Fitness_Endurance-Time_Sec", "Physical-Waist_Circumference",
                   "Physical-Height", "Physical-Systolic_BP", "Physical-Diastolic_BP", "Physical-BMI", "Basic_Demos-Age",
                   "FGC-FGC_CU", "FGC-FGC_GSND", "FGC-FGC_GSD", "FGC-FGC_PU", "FGC-FGC_SRL", "FGC-FGC_SRR", "FGC-FGC_TL"]
        result = add_extra_features(pd.DataFrame({c: [np.nan] for c in columns}))
        self.assertTrue(pd.isna(result.FGC_total_score.iloc[0]))


if __name__ == "__main__":
    unittest.main()
