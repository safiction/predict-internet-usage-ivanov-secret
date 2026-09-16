"""Independently recheck saved CV numbers, folds, provenance, and deployment."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score

from src.config import ID_COLUMN, PROJECT_ROOT, TARGET
from src.evaluation import create_cv_splits
from src.experiment import ExperimentConfig, MODEL_NAMES, PRIMARY, make_submission, predict_bundle, prepare_data, results_match


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_artifacts(root=PROJECT_ROOT):
    root = Path(root)
    results = root / "results"
    metadata = json.loads((results / "nested_metadata.json").read_text())
    values = metadata["config"].copy()
    values["outer_seeds"] = tuple(values["outer_seeds"])
    config = ExperimentConfig(**values)
    data = prepare_data(root / "data", save_processed=False)
    require(results_match(config, data, metadata, results), "Data, code, environment, configuration, or result digests differ from the saved experiment; rerun training")
    oof = pd.read_csv(results / "nested_oof_predictions.csv")
    summary = pd.read_csv(results / "nested_cv_results.csv")
    repeats = pd.read_csv(results / "nested_repeat_results.csv")
    folds = pd.read_csv(results / "nested_fold_results.csv")
    deltas = pd.read_csv(results / "nested_paired_deltas.csv")
    classes = pd.read_csv(results / "nested_class_metrics.csv")
    require(not oof.duplicated(["seed", ID_COLUMN]).any(), "OOF participant/repeat rows are duplicated")
    require(set(oof.seed) == set(config.outer_seeds), "OOF seed set differs from the configuration")
    setups = summary.setup.tolist()
    actual_repeat_scores = {setup: [] for setup in setups}
    for seed in config.outer_seeds:
        rows = oof.loc[oof.seed.eq(seed)].sort_values(ID_COLUMN).reset_index(drop=True)
        require(rows[ID_COLUMN].tolist() == data["ids"].tolist(), "OOF IDs differ from labeled participants")
        require(np.array_equal(rows[TARGET], data["y"]), "OOF labels differ from raw-data labels")
        expected_splits = create_cv_splits(data["X"], data["y"], data["ids"], config.outer_splits, seed)
        for fold, (_, val) in enumerate(expected_splits, start=1):
            require(set(rows.loc[rows.fold.eq(fold), ID_COLUMN]) == set(data["ids"].iloc[val]), "OOF fold assignment differs from ID-stable CV")
        for setup in setups:
            require(rows[setup].isin([0, 1, 2, 3]).all(), f"Invalid OOF classes: {setup}")
            score = cohen_kappa_score(rows[TARGET], rows[setup], labels=[0, 1, 2, 3], weights="quadratic")
            saved = repeats.loc[repeats.seed.eq(seed) & repeats.setup.eq(setup)]
            require(len(saved) == 1 and np.isclose(score, saved.oof_qwk.iloc[0], atol=1e-12, rtol=0), "Saved repeat QWK does not match sklearn recomputation")
            actual_repeat_scores[setup].append(score)
            fold_scores = []
            for fold in range(1, config.outer_splits + 1):
                mask = rows.fold.eq(fold)
                fold_score = cohen_kappa_score(rows.loc[mask, TARGET], rows.loc[mask, setup], labels=[0, 1, 2, 3], weights="quadratic")
                saved_fold = folds.loc[folds.seed.eq(seed) & folds.fold.eq(fold) & folds.setup.eq(setup)]
                require(len(saved_fold) == 1 and np.isclose(fold_score, saved_fold.validation_qwk.iloc[0], atol=1e-12, rtol=0), "Saved fold QWK differs")
                fold_scores.append(fold_score)
            require(np.isclose(np.mean(fold_scores), saved.mean_fold_qwk.iloc[0]), "Saved mean fold QWK differs")
            require(np.isclose(np.std(fold_scores), saved.std_fold_qwk.iloc[0]), "Saved fold deviation differs")
            actual_classes = classification_report(rows[TARGET], rows[setup], labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
            for label in range(4):
                saved_class = classes.loc[classes.seed.eq(seed) & classes.setup.eq(setup) & classes["class"].eq(label)]
                require(len(saved_class) == 1, "Missing/duplicated class metric row")
                for metric in ("precision", "recall", "f1-score", "support"):
                    require(np.isclose(actual_classes[str(label)][metric], saved_class[metric].iloc[0]), "Saved class metric differs")
                true_positives = int((rows[TARGET].eq(label) & rows[setup].eq(label)).sum())
                require(true_positives == saved_class.true_positives.iloc[0], "Saved true-positive count differs")
    for row in summary.itertuples():
        scores = actual_repeat_scores[row.setup]
        require(np.isclose(np.mean(scores), row.mean_oof_qwk), "Saved mean whole-OOF QWK differs")
        require(np.isclose(np.std(scores), row.repeat_std_qwk), "Saved repeat deviation differs")
        require(row.ci_low <= row.ci_high and row.repeats == len(config.outer_seeds), "Invalid uncertainty metadata")
    for row in deltas.itertuples():
        require(row.primary == PRIMARY, "Paired comparison uses a different primary policy")
        expected_delta = np.mean(actual_repeat_scores[PRIMARY]) - np.mean(actual_repeat_scores[row.reference])
        require(np.isclose(expected_delta, row.delta_qwk, atol=1e-12, rtol=0), "Saved paired QWK change differs")
        require(row.ci_low <= row.ci_high, "Invalid paired interval")
        require(bool(row.interval_excludes_zero) == bool(row.ci_low > 0 or row.ci_high < 0), "Paired significance flag differs from its interval")
    bundle = joblib.load(root / "data" / "models" / "equal_ensemble.joblib")
    require(bundle["policy"] == PRIMARY == metadata["deployment"]["policy"], "Deployment policy differs from the evaluated primary policy")
    require(list(bundle["models"]) == list(MODEL_NAMES), "Deployment members differ")
    require(np.allclose(bundle["weights"], [0.25] * 4), "Primary deployment is not equal weight")
    require(bundle["weights"] == metadata["deployment"]["weights"], "Saved deployment weights differ")
    require(bundle["thresholds"] == metadata["deployment"]["thresholds"], "Saved deployment thresholds differ")
    predictions = predict_bundle(bundle, pd.read_csv(root / "data" / "test.csv"))
    expected = make_submission(predictions, data["test_ids"], data["sample"])
    actual = pd.read_csv(results / "submission_nested.csv")
    pd.testing.assert_frame_equal(expected, actual)
    print(f"Verified {len(setups)} models x {len(config.outer_seeds)} repeats: raw-data labels, all folds/QWK/class metrics, provenance, and reloaded deployment submission")
    return metadata


if __name__ == "__main__":
    verify_artifacts()
