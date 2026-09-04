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
    training_scores = []
    validation_scores = []
    oof_predictions = np.empty(len(y), dtype=int)

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
        training_predictions = fold_model.predict(X_train)
        validation_predictions = fold_model.predict(X_validation)

        # Convert continuous regression outputs before calculating QWK
        if prediction_transform is not None:
            training_predictions = prediction_transform(training_predictions)
            validation_predictions = prediction_transform(validation_predictions)

        training_score = quadratic_weighted_kappa(
            y_train,
            training_predictions,
        )
        validation_score = quadratic_weighted_kappa(
            y_validation,
            validation_predictions,
        )

        training_scores.append(training_score)
        validation_scores.append(validation_score)
        oof_predictions[validation_indices] = validation_predictions

        print(
            f"Fold {fold_number}: "
            f"training QWK={training_score:.4f}, "
            f"validation QWK={validation_score:.4f}"
        )

    print(f"Mean training QWK: {np.mean(training_scores):.4f}")
    print(f"Mean validation QWK: {np.mean(validation_scores):.4f}")
    print(
        "Validation QWK standard deviation: "
        f"{np.std(validation_scores):.4f}"
    )

    return {
        "training_scores": training_scores,
        "validation_scores": validation_scores,
        "oof_predictions": oof_predictions,
    }
