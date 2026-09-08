"""Layer B: learned imputation and scaling, delivered as pipeline components.

These transformers estimate values from the data distribution and
must be fit inside the cross-validation pipeline, once per fold, and refit on
the full training data before predicting on the test set. Never fit them on the
whole dataset and never materialize their output.

Usage:

    from sklearn.pipeline import Pipeline
    from src.imputation import make_preprocessor

    model = Pipeline([
        ("preprocessor", make_preprocessor(X, strategy="median", winsorize=0.01)),
        ("model", SomeEstimator()),
    ])
    # then evaluate: src.evaluation.evaluate_model(model, X, y, cv_splits)

The preprocessor is a ColumnTransformer, so ``clone`` inside evaluate_model gives every fold a fresh, unfitted copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE

# Physiologically linked numeric blocks.
_MULTIVARIATE_BLOCK_PREFIXES = ("BIA-", "Physical-")


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to a lower/upper percentile learned in ``fit``.

    Fold-safe outlier handling for scale-sensitive linear models: the clip
    bounds are estimated only from the training fold. Percentile-based clipping is data-dependent.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_, self.upper_)


def _split_numeric_columns(numeric_columns: list[str]) -> tuple[list[str], list[str]]:
    """Separate block-imputed numeric columns from the rest."""
    block = [c for c in numeric_columns if c.startswith(_MULTIVARIATE_BLOCK_PREFIXES)]
    other = [c for c in numeric_columns if c not in block]
    return block, other


def _numeric_pipeline(imputer, scale: bool, winsorize: float | None) -> Pipeline:
    steps = [("imputer", imputer)]
    if winsorize is not None:
        # Clip extreme values before scaling.
        steps.append(("clipper", PercentileClipper(lower=winsorize, upper=1 - winsorize)))
    if scale:
        # Linear models (e.g. Ridge) need scaled inputs; harmless for others.
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def make_preprocessor(
    feature_frame: pd.DataFrame,
    strategy: str = "iterative",
    scale: bool = True,
    winsorize: float | None = None,
    nominal_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Build a fold-safe preprocessor for a Layer A feature frame.

    Parameters
    ----------
    feature_frame:
        Layer A output frame without ``id``/``sii``
    strategy:
        ``"median"`` imputes every numeric column with its median.
        ``"iterative"`` imputes the linked BIA/Physical blocks jointly with
        IterativeImputer and falls back to the median elsewhere.
    scale:
        Whether to standardize numeric features after imputation.
    winsorize:
        If set, clip numeric features to the ``[winsorize, 1 - winsorize]`` percentiles,
        fit per fold. ``None`` disables clipping.
    nominal_columns:
        Numeric-dtyped columns to treat as categorical instead of
        numeric (e.g. a nominal code stored as an int). Defaults to none.
    """
    nominal = set(nominal_columns or [])

    categorical_columns = feature_frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = feature_frame.select_dtypes(include="number").columns.tolist()

    # Move any explicitly-nominal numeric columns into the categorical branch.
    categorical_columns += [c for c in numeric_columns if c in nominal]
    numeric_columns = [c for c in numeric_columns if c not in nominal]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # A validation fold may hold a category unseen in its training fold.
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple] = []

    if strategy == "median":
        transformers.append(
            (
                "numeric",
                _numeric_pipeline(SimpleImputer(strategy="median"), scale, winsorize),
                numeric_columns,
            )
        )
    elif strategy == "iterative":
        block_columns, other_numeric = _split_numeric_columns(numeric_columns)
        if block_columns:
            transformers.append(
                (
                    "numeric_block",
                    _numeric_pipeline(
                        IterativeImputer(random_state=RANDOM_STATE), scale, winsorize
                    ),
                    block_columns,
                )
            )
        if other_numeric:
            transformers.append(
                (
                    "numeric_other",
                    _numeric_pipeline(SimpleImputer(strategy="median"), scale, winsorize),
                    other_numeric,
                )
            )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")
