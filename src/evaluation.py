"""Shared cross-validation and evaluation utilities"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from src.config import N_SPLITS, RANDOM_STATE


def create_cv_splits(X, y, ids):
    """Create reproducible stratified splits independent of row order"""
    if not (len(X) == len(y) == len(ids)):
        raise ValueError("X, y, and ids must contain the same number of rows")
    if ids.isna().any():
        raise ValueError("IDs must not contain missing values")
    if ids.duplicated().any():
        raise ValueError("IDs must be unique")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,  # Randomize row order before assigning folds
        random_state=RANDOM_STATE,  # Reproduce the same shuffled folds
    )

    # Sort unique IDs so row reordering cannot change fold membership
    sorted_positions = np.argsort(ids.astype(str).to_numpy())
    sorted_splits = cv.split(
        X.iloc[sorted_positions],
        y.iloc[sorted_positions],
    )

    # Convert sorted positions back to positions in the provided X and y
    return [
        (
            sorted_positions[train_indices],
            sorted_positions[validation_indices],
        )
        for train_indices, validation_indices in sorted_splits
    ]


def quadratic_weighted_kappa(y_true, y_pred):
    """Calculate QWK for ordered target classes"""
    return cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic",  # Give larger penalties to more distant errors
    )


def regression_to_classes(predictions):
    """Convert continuous predictions to sii classes"""
    return np.digitize(
        predictions,
        bins=[0.5, 1.5, 2.5],  # Use midpoints between adjacent sii classes
    )


def evaluate_model(
    model,
    X,
    y,
    cv_splits,
    prediction_transform=None,
):
    """Evaluate a model on fixed cross-validation splits"""
    fold_scores = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cv_splits,
        start=1,
    ):
        # Create an unfitted copy so model state is not shared between folds
        fold_model = clone(model)

        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        y_train = y.iloc[train_indices]
        y_validation = y.iloc[validation_indices]

        fold_model.fit(X_train, y_train)
        predictions = fold_model.predict(X_validation)

        # Convert continuous regression outputs before calculating QWK
        if prediction_transform is not None:
            predictions = prediction_transform(predictions)

        score = quadratic_weighted_kappa(y_validation, predictions)
        fold_scores.append(score)

        print(f"Fold {fold_number} QWK: {score:.4f}")

    print(f"Mean QWK: {np.mean(fold_scores):.4f}")
    print(f"QWK standard deviation: {np.std(fold_scores):.4f}")

    return fold_scores
