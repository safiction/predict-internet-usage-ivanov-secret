"""Ordinal blending utilities for notebook 09.

Fit blend weights and thresholds on inner OOF predictions, never on the outer
validation labels. Full-OOF calibration is reserved for deployment.
"""

from itertools import product

import numpy as np


DEFAULT_THRESHOLDS = np.array([0.5, 1.5, 2.5])


def _validate_labels(y, n):
    labels = np.asarray(y)
    if labels.shape != (n,) or not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError("y must be a one-dimensional array of sii classes 0..3")
    return labels.astype(int)


def to_classes(predictions, thresholds=DEFAULT_THRESHOLDS):
    predictions = np.asarray(predictions, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ValueError("predictions must be a finite one-dimensional array")
    if thresholds.shape != (3,) or not np.isfinite(thresholds).all():
        raise ValueError("three finite thresholds are required")
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly increasing")
    return np.digitize(predictions, thresholds).astype(int)


def _qwk(y, predictions):
    """QWK with the fixed ordinal scale 0..3, using squared distances."""
    observed = np.square(y - predictions).sum()
    expected = (
        np.square(y).sum() + np.square(predictions).sum()
        - 2 * y.mean() * predictions.sum()
    )
    return float(1 - observed / expected) if expected > 0 else 0.0


def optimize_thresholds(predictions, y, max_rounds=8):
    """Multi-start coordinate search over all distinct prediction cut points.

    QWK is piecewise constant, so gradient-based optimizers can stop on a
    plateau. Each coordinate is instead optimized exactly using cumulative
    label sums. This is a local search, not a guarantee of a global optimum.
    A class may receive no predictions, important for the rare severe class.
    """
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("predictions must be a nonempty finite vector")
    labels = _validate_labels(y, len(values))
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_labels = values[order], labels[order]
    unique = np.unique(sorted_values)
    cuts = np.concatenate([
        [np.nextafter(unique[0], -np.inf)],
        unique[:-1] + np.diff(unique) / 2,
        [np.nextafter(unique[-1], np.inf)],
    ])
    counts = np.searchsorted(sorted_values, cuts, side="left")
    label_sums = np.concatenate([[0], np.cumsum(sorted_labels)])
    quantiles = np.cumsum(np.bincount(labels, minlength=4))[:3] / len(labels)
    starts = [DEFAULT_THRESHOLDS, np.array([0.5, 0.9, 1.5]), np.quantile(values, quantiles)]
    best_score, best_thresholds = -np.inf, None

    for start in starts:
        thresholds = np.asarray(start, dtype=float).copy()
        for j in range(1, 3):
            thresholds[j] = max(thresholds[j], np.nextafter(thresholds[j - 1], np.inf))
        for _ in range(max_rounds):
            changed = False
            for boundary in range(3):
                low = thresholds[boundary - 1] if boundary else -np.inf
                high = thresholds[boundary + 1] if boundary < 2 else np.inf
                feasible = (cuts > low) & (cuts < high)
                if not feasible.any():
                    continue
                candidates = cuts[feasible]
                positions = counts[feasible]
                current_position = np.searchsorted(sorted_values, thresholds[boundary], side="left")
                classes = to_classes(values, thresholds)
                observed = np.square(labels - classes).sum()
                expected = (
                    np.square(labels).sum() + np.square(classes).sum()
                    - 2 * labels.mean() * classes.sum()
                )
                moved_count = positions - current_position
                moved_sum = label_sums[positions] - label_sums[current_position]
                candidate_observed = observed + 2 * moved_sum - (2 * boundary + 1) * moved_count
                candidate_expected = expected + (2 * labels.mean() - 2 * boundary - 1) * moved_count
                scores = np.zeros(len(candidates))
                np.divide(candidate_observed, candidate_expected, out=scores, where=candidate_expected > 0)
                scores = np.where(candidate_expected > 0, 1 - scores, 0)
                optimum = np.max(scores)
                if optimum > _qwk(labels, classes) + 1e-12:
                    ties = np.flatnonzero(np.isclose(scores, optimum, rtol=0, atol=1e-12))
                    chosen = ties[np.argmin(np.abs(candidates[ties] - thresholds[boundary]))]
                    thresholds[boundary] = candidates[chosen]
                    changed = True
            if not changed:
                break
        score = _qwk(labels, to_classes(values, thresholds))
        if score > best_score:
            best_score, best_thresholds = score, thresholds.copy()
    return best_thresholds


def simplex_weights(n_models, resolution=4):
    """Nonnegative blend weights summing to one; include every single model."""
    if n_models < 1 or resolution < 1:
        raise ValueError("n_models and resolution must be positive integers")
    candidates = [
        np.asarray(weights, dtype=float) / resolution
        for weights in product(range(resolution + 1), repeat=n_models)
        if sum(weights) == resolution
    ]
    equal = np.full(n_models, 1 / n_models)
    if not any(np.allclose(equal, weights) for weights in candidates):
        candidates.append(equal)
    return np.asarray(candidates)


def fit_blend(prediction_matrix, y, resolution=4):
    """Select weights and rounding thresholds on a calibration set only."""
    matrix = np.asarray(prediction_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("prediction_matrix must be a finite (rows, models) array")
    labels = _validate_labels(y, len(matrix))
    # Prefer genuine blends on ties; no claimed improvement if a vertex wins.
    candidates = simplex_weights(matrix.shape[1], resolution)
    candidates = sorted(candidates, key=lambda weights: np.square(weights).sum())
    best = {"calibration_qwk": -np.inf}
    for weights in candidates:
        continuous = matrix @ weights
        thresholds = optimize_thresholds(continuous, labels)
        score = _qwk(labels, to_classes(continuous, thresholds))
        if score > best["calibration_qwk"] + 1e-12:
            best = {"weights": weights.copy(), "thresholds": thresholds, "calibration_qwk": score}
    return best
