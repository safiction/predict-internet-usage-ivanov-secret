"""Small deterministic checks; no competition data or boosting dependencies."""

import unittest

import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.ensemble import _qwk, fit_blend, optimize_thresholds, simplex_weights, to_classes


class EnsembleTests(unittest.TestCase):
    def test_qwk_matches_sklearn(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 4, 200)
        pred = rng.integers(0, 4, 200)
        self.assertAlmostEqual(_qwk(y, pred), cohen_kappa_score(y, pred, weights="quadratic"))

    def test_qwk_preserves_distances_when_a_class_is_absent(self):
        y = np.array([0, 0, 1, 1, 3, 3])
        pred = np.array([0, 1, 1, 3, 1, 3])
        self.assertAlmostEqual(
            _qwk(y, pred),
            cohen_kappa_score(y, pred, labels=[0, 1, 2, 3], weights="quadratic"),
        )

    def test_thresholds_improve_compressed_predictions(self):
        y = np.tile(np.arange(4), 20)
        predictions = 0.2 + y * 0.3
        thresholds = optimize_thresholds(predictions, y)
        np.testing.assert_array_equal(to_classes(predictions, thresholds), y)
        self.assertTrue(np.all(np.diff(thresholds) > 0))

    def test_duplicates_and_constant_predictions(self):
        y = np.tile(np.arange(4), 10)
        thresholds = optimize_thresholds(np.ones(len(y)), y)
        self.assertTrue(np.all(np.diff(thresholds) > 0))
        self.assertEqual(len(np.unique(to_classes(np.ones(len(y)), thresholds))), 1)

    def test_search_never_worse_than_midpoints(self):
        rng = np.random.default_rng(12)
        y = rng.integers(0, 4, 300)
        predictions = y * 0.35 + rng.normal(0, 0.4, len(y))
        fitted = optimize_thresholds(predictions, y)
        self.assertGreaterEqual(_qwk(y, to_classes(predictions, fitted)), _qwk(y, to_classes(predictions)))

    def test_simplex_includes_vertices_and_equal_blend(self):
        weights = simplex_weights(4)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertTrue((weights >= 0).all())
        for vertex in np.eye(4):
            self.assertTrue(any(np.array_equal(vertex, w) for w in weights))
        self.assertTrue(any(np.allclose(w, 0.25) for w in weights))

    def test_blend_can_select_best_single_model(self):
        y = np.tile(np.arange(4), 20)
        matrix = np.column_stack([y * 0.3, np.ones(len(y))])
        fitted = fit_blend(matrix, y)
        self.assertAlmostEqual(fitted["calibration_qwk"], 1)
        np.testing.assert_array_equal(to_classes(matrix @ fitted["weights"], fitted["thresholds"]), y)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            to_classes([0, np.nan])
        with self.assertRaises(ValueError):
            to_classes([0], [1, 1, 2])
        with self.assertRaises(ValueError):
            optimize_thresholds([0.1, 0.2], [0, 4])
        with self.assertRaises(ValueError):
            fit_blend([[0, np.inf]], [0])


if __name__ == "__main__":
    unittest.main()
