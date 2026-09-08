"""Layer B: learned imputation, delivered as pipeline components.

Transformations estimate fill values from the data distribution (medians,
most-frequent categories, iterative regression). They MUST be fit inside the
cross-validation pipeline, once per fold, and refit on the full training data
before predicting on the test set. Never fit them on the whole dataset: doing so leaks
test information into training.

Usage:

    from sklearn.pipeline import Pipeline
    from src.imputation import make_preprocessor

    model = Pipeline([
        ("preprocessor", make_preprocessor(X, strategy="iterative")),
        ("model", SomeEstimator()),
    ])
    # then evaluate with src.evaluation.evaluate_model(model, X, y, cv_splits)

The preprocessor is a ColumnTransformer, so ``clone`` inside evaluate_model gives every fold an unfitted copy.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE

# Columns inside each are derived from one another, so imputing them jointly
# with IterativeImputer keeps the filled combinations internally consistent.
_MULTIVARIATE_BLOCK_PREFIXES = ("BIA-", "Physical-")


def _split_numeric_columns(numeric_columns: list[str]) -> tuple[list[str], list[str]]:
    """Separate block-imputed numeric columns from the rest."""
    block = [
        c
        for c in numeric_columns
        if c.startswith(_MULTIVARIATE_BLOCK_PREFIXES)
    ]
    other = [c for c in numeric_columns if c not in block]
    return block, other


def _numeric_pipeline(imputer, scale: bool) -> Pipeline:
    steps = [("imputer", imputer)]
    if scale:
        # Linear models (e.g. Ridge) need scaled inputs; harmless for others.
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def make_preprocessor(
    feature_frame: pd.DataFrame,
    strategy: str = "iterative",
    scale: bool = True,
) -> ColumnTransformer:
    """Build a fold-safe preprocessor for a Layer A feature frame.

    Parameters
    ----------
    feature_frame:
        Layer A output frame without ``id``/``sii``.
    strategy:
        ``"median"`` imputes every numeric column with its median.
        ``"iterative"`` imputes the linked BIA/Physical blocks jointly with
        IterativeImputer and falls back to the median elsewhere.
    scale:
        Whether to standardize numeric features after imputation.
    """
    categorical_columns = feature_frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = feature_frame.select_dtypes(include="number").columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple] = []

    if strategy == "median":
        transformers.append(
            (
                "numeric",
                _numeric_pipeline(SimpleImputer(strategy="median"), scale),
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
                        IterativeImputer(random_state=RANDOM_STATE),
                        scale,
                    ),
                    block_columns,
                )
            )
        if other_numeric:
            transformers.append(
                (
                    "numeric_other",
                    _numeric_pipeline(SimpleImputer(strategy="median"), scale),
                    other_numeric,
                )
            )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
