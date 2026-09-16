"""Reproducible fully nested model-selection experiment for notebook 09.

Candidate grids are specified here, not read from historical winning trials.
Outer validation labels never enter tuning, early stopping, or calibration.
The primary deployment algorithm is always the tuned equal-weight ensemble.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from src.config import DATA_DIR, ID_COLUMN, LEAKAGE_PREFIX, PROCESSED_DIR, RESULTS_DIR, TARGET
from src.ensemble import _qwk, fit_blend, optimize_thresholds, to_classes
from src.evaluation import create_cv_splits
from src.features import build_features, feature_type_map, load_data_dictionary
from src.imputation import make_preprocessor

MODEL_NAMES = ("catboost", "xgboost", "extra_trees", "ridge")
PRIMARY = "Tuned equal ensemble"
REFERENCE = "Inner-selected single model"
PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "catboost", "xgboost", "optuna", "pyarrow", "joblib", "threadpoolctl")
SOURCE_FILES = ("experiment.py", "features.py", "imputation.py", "ensemble.py", "evaluation.py", "config.py")
# Small, generic capacity/regularization grids. Candidate 0 is the untuned
# reference. No numerical values from historical best Optuna trials are used.
CANDIDATES = {
    "catboost": [
        dict(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=3.0),
        dict(iterations=900, depth=4, learning_rate=0.03, l2_leaf_reg=10.0, bagging_temperature=1.0),
        dict(iterations=1200, depth=5, learning_rate=0.02, l2_leaf_reg=20.0, bagging_temperature=1.0),
    ],
    "xgboost": [
        dict(n_estimators=1200, max_depth=6, learning_rate=0.03, min_child_weight=1, reg_lambda=1.0),
        dict(n_estimators=1200, max_depth=3, learning_rate=0.03, min_child_weight=5, reg_lambda=10.0,
             subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1),
        dict(n_estimators=1600, max_depth=4, learning_rate=0.02, min_child_weight=10, reg_lambda=20.0,
             subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, gamma=0.1),
    ],
    "extra_trees": [
        dict(n_estimators=250, max_depth=None, min_samples_leaf=4, max_features=1.0),
        dict(n_estimators=250, max_depth=12, min_samples_leaf=8, max_features=0.8),
        dict(n_estimators=250, max_depth=8, min_samples_leaf=12, max_features=0.8),
    ],
    "ridge": [dict(alpha=1.0), dict(alpha=10.0, winsorize=0.01), dict(alpha=100.0, winsorize=0.01)],
}


@dataclass(frozen=True)
class ExperimentConfig:
    outer_seeds: tuple = (42, 2026)
    outer_splits: int = 5
    inner_splits: int = 3
    threads: int = 4
    bootstrap_samples: int = 2000
    stopping_rounds: int = 60
    weight_resolution: int = 4

    def __post_init__(self):
        if not self.outer_seeds or len(set(self.outer_seeds)) != len(self.outer_seeds):
            raise ValueError("Provide at least one distinct outer seed")
        if min(self.outer_splits, self.inner_splits) < 2:
            raise ValueError("Both CV levels need at least two splits")
        if min(self.threads, self.bootstrap_samples, self.stopping_rounds, self.weight_resolution) < 1:
            raise ValueError("Resource and search settings must be positive")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def results_match(config, data, metadata, results_dir=RESULTS_DIR):
    """Accept notebook cache only for identical data, code, environment, and config."""
    if metadata.get("historical_best_params_reused") is not False:
        return False
    expected_config = json.loads(json.dumps(asdict(config)))
    if metadata.get("config") != expected_config or metadata.get("data", {}).get("data_sha256") != data["provenance"]["data_sha256"]:
        return False
    expected_versions = {"python": platform.python_version(), **{name: version(name) for name in PACKAGES}}
    if metadata.get("versions") != expected_versions:
        return False
    sources = {name: file_digest(Path(__file__).parent / name) for name in SOURCE_FILES}
    if metadata.get("source_sha256") != sources or not metadata.get("artifact_sha256"):
        return False
    for name, digest in metadata["artifact_sha256"].items():
        path = Path(results_dir) / name
        if not path.exists() or file_digest(path) != digest:
            return False
    return True


def clean_model_frame(frame):
    """Only row-local string/missing normalization; no learned statistics."""
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object", "string", "category"]).columns:
        frame[column] = frame[column].astype("object").where(frame[column].notna(), "__missing__").astype(str)
    numeric = frame.select_dtypes(include="number").columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    return frame


def add_extra_features(frame):
    """Experimental row-local interactions, not validated clinical scores."""
    extra = frame.copy()
    extra["Fitness_Endurance_total_seconds"] = extra["Fitness_Endurance-Time_Mins"] * 60 + extra["Fitness_Endurance-Time_Sec"]
    extra["Physical-Waist_to_Height"] = extra["Physical-Waist_Circumference"] / extra["Physical-Height"].replace(0, np.nan)
    extra["Physical-Pulse_Pressure"] = extra["Physical-Systolic_BP"] - extra["Physical-Diastolic_BP"]
    extra["FGC_total_score"] = extra[[
        "FGC-FGC_CU", "FGC-FGC_GSND", "FGC-FGC_GSD", "FGC-FGC_PU",
        "FGC-FGC_SRL", "FGC-FGC_SRR", "FGC-FGC_TL",
    ]].sum(axis=1, min_count=1)  # All-missing results stay NaN, not a false zero.
    extra["BMI_per_Age"] = extra["Physical-BMI"] / extra["Basic_Demos-Age"].replace(0, np.nan)
    return extra.replace([np.inf, -np.inf], np.nan)


def model_frame(name, base):
    return add_extra_features(base) if name == "xgboost" else base


def prepare_data(data_dir=DATA_DIR, save_processed=True):
    data_dir = Path(data_dir)
    filenames = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
    for name in filenames:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Missing data/{name}; follow README data-access instructions")
    dictionary = pd.read_csv(data_dir / "data_dictionary.csv")
    raw_train, raw_test = (pd.read_csv(data_dir / f"{split}.csv") for split in ("train", "test"))
    for raw in (raw_train, raw_test):
        if raw[ID_COLUMN].isna().any() or not raw[ID_COLUMN].is_unique:
            raise ValueError("Participant IDs must be present and unique")
    train, cleaning = build_features(raw_train, dictionary, return_report=True)
    test = build_features(raw_test, dictionary)
    labeled = train.loc[train[TARGET].notna()].sort_values(ID_COLUMN).reset_index(drop=True)
    y = labeled[TARGET].astype(int)
    if not np.isin(labeled[TARGET], [0, 1, 2, 3]).all():
        raise ValueError("Known sii values must be integer classes 0..3")
    columns = [c for c in labeled if c not in {ID_COLUMN, TARGET} and not c.startswith(LEAKAGE_PREFIX)]
    if set(columns) - set(test.columns):
        raise ValueError("Train and test feature contracts do not match")
    if save_processed:
        processed = data_dir / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        train.to_parquet(processed / "train_features.parquet", index=False)
        test.to_parquet(processed / "test_features.parquet", index=False)
    return {
        "X": clean_model_frame(labeled[columns]), "y": y, "ids": labeled[ID_COLUMN],
        "X_test": clean_model_frame(test[columns]), "test_ids": test[ID_COLUMN],
        "sample": pd.read_csv(data_dir / "sample_submission.csv"), "feature_columns": columns,
        "provenance": {
            "data_sha256": {name: file_digest(data_dir / name) for name in filenames},
            "raw_train_rows": len(train), "labeled_rows": len(y), "unlabeled_rows": len(train) - len(y),
            "test_rows": len(test), "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "cleaning": cleaning,
        },
    }


def build_booster(name, frame, params, seed, config, early_stopping=False):
    params = params.copy()
    if name == "catboost":
        categories = frame.select_dtypes(include=["object", "string", "category"]).columns
        params.update(
            loss_function="RMSE", random_seed=seed, thread_count=config.threads,
            verbose=False, allow_writing_files=False,
            cat_features=tuple(frame.columns.get_loc(c) for c in categories),
        )
        if early_stopping:
            params["early_stopping_rounds"] = config.stopping_rounds
        return CatBoostRegressor(**params)
    params.update(objective="reg:squarederror", eval_metric="rmse", random_state=seed,
                  n_jobs=config.threads, tree_method="hist")
    if early_stopping:
        params["early_stopping_rounds"] = config.stopping_rounds
    return XGBRegressor(**params)


def fit_member(name, frame, labels, params, seed, config):
    """Fit only supplied training rows; this function cannot see outer labels."""
    if name in {"ridge", "extra_trees"}:
        params = params.copy()
        winsorize = params.pop("winsorize", None)
        estimator = Ridge(**params) if name == "ridge" else ExtraTreesRegressor(
            random_state=seed, n_jobs=config.threads, **params,
        )
        model = Pipeline([
            ("preprocessor", make_preprocessor(frame, strategy="median", scale=name == "ridge", winsorize=winsorize)),
            ("model", estimator),
        ])
        with threadpool_limits(limits=config.threads):
            model.fit(frame, labels)
        return model, None
    fit_pos, stop_pos = train_test_split(
        np.arange(len(labels)), test_size=0.15, random_state=seed, stratify=labels,
    )
    provisional = build_booster(name, frame, params, seed, config, early_stopping=True)
    final_params = params.copy()
    if name == "catboost":
        provisional.fit(frame.iloc[fit_pos], labels.iloc[fit_pos],
                        eval_set=(frame.iloc[stop_pos], labels.iloc[stop_pos]), use_best_model=True)
        tree_count = max(1, provisional.tree_count_)
        final_params["iterations"] = tree_count
        final = build_booster(name, frame, final_params, seed, config)
    else:
        preprocessor = make_preprocessor(frame.iloc[fit_pos], strategy="median", scale=False)
        fit_values = preprocessor.fit_transform(frame.iloc[fit_pos])
        stop_values = preprocessor.transform(frame.iloc[stop_pos])
        provisional.fit(fit_values, labels.iloc[fit_pos],
                        eval_set=[(stop_values, labels.iloc[stop_pos])], verbose=False)
        tree_count = provisional.best_iteration + 1
        final_params["n_estimators"] = tree_count
        final = Pipeline([
            ("preprocessor", make_preprocessor(frame, strategy="median", scale=False)),
            ("model", build_booster(name, frame, final_params, seed, config)),
        ])
    # New model/preprocessor: the stopping holdout joins the final training set.
    final.fit(frame, labels)
    return final, int(tree_count)


def select_members(base, y, ids, seed, config, grids=None):
    """Choose candidate capacity by inner OOF RMSE, without outer validation."""
    grids = CANDIDATES if grids is None else grids
    splits = create_cv_splits(base, y, ids, n_splits=config.inner_splits, random_state=seed)
    if y.value_counts().min() < config.inner_splits:
        raise ValueError("Too few rare-class examples for the requested inner splits")
    baseline = np.full((len(y), len(MODEL_NAMES)), np.nan)
    tuned = np.full_like(baseline, np.nan)
    selections, candidate_rows = {}, []
    for column, name in enumerate(MODEL_NAMES):
        frame = model_frame(name, base)
        candidate_predictions = np.full((len(y), len(grids[name])), np.nan)
        for candidate_id, params in enumerate(grids[name]):
            for fit_idx, val_idx in splits:
                if np.intersect1d(fit_idx, val_idx).size:
                    raise RuntimeError("Inner training and validation overlap")
                model, _ = fit_member(name, frame.iloc[fit_idx], y.iloc[fit_idx], params, seed, config)
                candidate_predictions[val_idx, candidate_id] = np.asarray(model.predict(frame.iloc[val_idx])).ravel()
            if not np.isfinite(candidate_predictions[:, candidate_id]).all():
                raise RuntimeError("An inner OOF row is missing")
            rmse = float(np.sqrt(mean_squared_error(y, candidate_predictions[:, candidate_id])))
            candidate_rows.append({"model": name, "candidate": candidate_id, "inner_rmse": rmse})
            print(f"    {name} candidate {candidate_id}: inner RMSE={rmse:.4f}", flush=True)
        errors = [mean_squared_error(y, candidate_predictions[:, i]) for i in range(len(grids[name]))]
        selected = int(np.argmin(errors))
        selections[name] = {"candidate": selected, "params": grids[name][selected].copy(),
                            "inner_rmse": float(np.sqrt(errors[selected]))}
        baseline[:, column] = candidate_predictions[:, 0]
        tuned[:, column] = candidate_predictions[:, selected]
    return baseline, tuned, selections, candidate_rows


def calibrate(inner_baseline, inner_tuned, labels, config):
    baseline_thresholds = [optimize_thresholds(inner_baseline[:, i], labels) for i in range(len(MODEL_NAMES))]
    tuned_thresholds = [optimize_thresholds(inner_tuned[:, i], labels) for i in range(len(MODEL_NAMES))]
    single_scores = [_qwk(np.asarray(labels), to_classes(inner_tuned[:, i], tuned_thresholds[i])) for i in range(len(MODEL_NAMES))]
    return {
        "baseline_thresholds": baseline_thresholds, "tuned_thresholds": tuned_thresholds,
        "baseline_equal_thresholds": optimize_thresholds(inner_baseline.mean(axis=1), labels),
        "equal_thresholds": optimize_thresholds(inner_tuned.mean(axis=1), labels),
        "weighted": fit_blend(inner_tuned, labels, config.weight_resolution),
        "selected_single": int(np.argmax(single_scores)),
    }


def apply_calibration(baseline, tuned, calibration, majority_class):
    """No labels required: all choices were made on the training partition."""
    predictions = {"Dummy majority": np.full(len(tuned), majority_class, dtype=int),
                   "Ridge fixed midpoints": to_classes(baseline[:, MODEL_NAMES.index("ridge")])}
    for column, name in enumerate(MODEL_NAMES):
        predictions[f"Baseline {name}"] = to_classes(baseline[:, column], calibration["baseline_thresholds"][column])
        predictions[f"Tuned {name}"] = to_classes(tuned[:, column], calibration["tuned_thresholds"][column])
    predictions["Baseline equal ensemble"] = to_classes(baseline.mean(axis=1), calibration["baseline_equal_thresholds"])
    predictions[PRIMARY] = to_classes(tuned.mean(axis=1), calibration["equal_thresholds"])
    predictions["Tuned weighted ensemble"] = to_classes(tuned @ calibration["weighted"]["weights"], calibration["weighted"]["thresholds"])
    selected = calibration["selected_single"]
    predictions[REFERENCE] = to_classes(tuned[:, selected], calibration["tuned_thresholds"][selected])
    return predictions


def make_submission(classes, test_ids, sample):
    if test_ids.isna().any() or not test_ids.is_unique or not sample[ID_COLUMN].is_unique:
        raise ValueError("Submission and test IDs must be unique and present")
    if sample[ID_COLUMN].isna().any() or set(test_ids) != set(sample[ID_COLUMN]):
        raise ValueError("Sample submission and test ID sets differ")
    if len(classes) != len(test_ids) or not np.isin(classes, [0, 1, 2, 3]).all():
        raise ValueError("Predictions must contain one sii class per test participant")
    result = sample.copy()
    result[TARGET] = result[ID_COLUMN].map(pd.Series(classes, index=test_ids.to_numpy())).astype(int)
    return result


def bootstrap_summary(y, predictions_by_seed, samples=2000, seed=31415):
    """Participant-paired, repeat-clustered bootstrap of fixed OOF predictions.

    The same participant indices are resampled in every CV repeat. These CIs
    are conditional diagnostics, not full retraining or post-selection CIs.
    """
    seeds = sorted(predictions_by_seed)
    setups = list(predictions_by_seed[seeds[0]])
    labels = np.asarray(y, dtype=int)
    point = {setup: np.mean([_qwk(labels, predictions_by_seed[s][setup]) for s in seeds]) for setup in setups}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((samples, len(setups)))
    for b in range(samples):
        positions = rng.integers(0, len(labels), len(labels))
        for j, setup in enumerate(setups):
            bootstrap[b, j] = np.mean([_qwk(labels[positions], predictions_by_seed[s][setup][positions]) for s in seeds])
    rows = []
    for j, setup in enumerate(setups):
        low, high = np.quantile(bootstrap[:, j], [0.025, 0.975])
        repeat_scores = [_qwk(labels, predictions_by_seed[s][setup]) for s in seeds]
        rows.append({"setup": setup, "mean_oof_qwk": point[setup], "ci_low": float(low), "ci_high": float(high),
                     "repeat_std_qwk": float(np.std(repeat_scores)), "repeats": len(seeds)})
    deltas = []
    for reference in (REFERENCE, "Baseline equal ensemble", "Tuned xgboost"):
        differences = bootstrap[:, setups.index(PRIMARY)] - bootstrap[:, setups.index(reference)]
        low, high = np.quantile(differences, [0.025, 0.975])
        deltas.append({"primary": PRIMARY, "reference": reference, "delta_qwk": point[PRIMARY] - point[reference],
                       "ci_low": float(low), "ci_high": float(high), "interval_excludes_zero": bool(low > 0 or high < 0)})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def predict_bundle(bundle, raw_test):
    """Apply the exact primary deployment policy to a fresh raw test frame."""
    base = clean_model_frame(build_features(raw_test, bundle["dictionary"])[bundle["feature_columns"]])
    continuous = np.column_stack([
        np.asarray(bundle["models"][name].predict(model_frame(name, base))).ravel() for name in MODEL_NAMES
    ]) @ np.asarray(bundle["weights"])
    return to_classes(continuous, bundle["thresholds"])


def run_experiment(config=None, data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    config = ExperimentConfig() if config is None else config
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(data_dir)
    X, y, ids = data["X"], data["y"], data["ids"]
    if y.value_counts().min() < config.outer_splits:
        raise ValueError("Too few rare-class participants for outer CV")
    fold_rows, repeat_rows, class_rows, tuning_rows, audit = [], [], [], [], []
    predictions_by_seed, oof_tables = {}, []
    print(f"Nested CV: {len(y)} labeled rows; {len(config.outer_seeds)} repeats x {config.outer_splits} outer x {config.inner_splits} inner", flush=True)
    for seed in config.outer_seeds:
        oof_predictions = {}
        fold_assignments = np.full(len(y), -1, dtype=int)
        for fold, (train_idx, val_idx) in enumerate(create_cv_splits(X, y, ids, config.outer_splits, seed), start=1):
            print(f"Repeat seed {seed}, outer fold {fold}/{config.outer_splits}", flush=True)
            if np.intersect1d(train_idx, val_idx).size:
                raise RuntimeError("Outer training and validation overlap")
            training, training_y = X.iloc[train_idx], y.iloc[train_idx]
            inner_base, inner_tuned, selections, candidates = select_members(training, training_y, ids.iloc[train_idx], seed, config)
            calibration = calibrate(inner_base, inner_tuned, training_y, config)
            baseline = np.full((len(val_idx), len(MODEL_NAMES)), np.nan)
            tuned = np.full_like(baseline, np.nan)
            tree_counts = {}
            for column, name in enumerate(MODEL_NAMES):
                frame = model_frame(name, X)
                base_model, base_trees = fit_member(name, frame.iloc[train_idx], training_y, CANDIDATES[name][0], seed, config)
                baseline[:, column] = np.asarray(base_model.predict(frame.iloc[val_idx])).ravel()
                if selections[name]["candidate"] == 0:
                    tuned[:, column] = baseline[:, column]
                    tree_counts[name] = base_trees
                else:
                    model, trees = fit_member(name, frame.iloc[train_idx], training_y, selections[name]["params"], seed, config)
                    tuned[:, column] = np.asarray(model.predict(frame.iloc[val_idx])).ravel()
                    tree_counts[name] = trees
            predictions = apply_calibration(baseline, tuned, calibration, int(training_y.mode().iloc[0]))
            fold_assignments[val_idx] = fold
            for setup, classes in predictions.items():
                if setup not in oof_predictions:
                    oof_predictions[setup] = np.full(len(y), -1, dtype=int)
                oof_predictions[setup][val_idx] = classes
                fold_rows.append({"seed": seed, "fold": fold, "setup": setup, "validation_qwk": _qwk(y.iloc[val_idx].to_numpy(), classes)})
            tuning_rows.extend({"seed": seed, "fold": fold, **candidate} for candidate in candidates)
            audit.append({
                "seed": seed, "fold": fold, "training_rows": len(train_idx), "validation_rows": len(val_idx),
                "selections": selections, "tree_counts": tree_counts,
                "member_thresholds": {name: calibration["tuned_thresholds"][i].tolist() for i, name in enumerate(MODEL_NAMES)},
                "equal_thresholds": calibration["equal_thresholds"].tolist(),
                "selected_single": MODEL_NAMES[calibration["selected_single"]],
                "weighted_weights": calibration["weighted"]["weights"].tolist(),
                "weighted_thresholds": calibration["weighted"]["thresholds"].tolist(),
            })
            # Aggregate checkpoints aid inspection; participant predictions stay local.
            pd.DataFrame(fold_rows).to_csv(results_dir / "nested_fold_results.csv", index=False)
            print(f"  Primary outer QWK={_qwk(y.iloc[val_idx].to_numpy(), predictions[PRIMARY]):.4f}", flush=True)
        if (fold_assignments < 1).any() or not all(np.isin(p, [0, 1, 2, 3]).all() for p in oof_predictions.values()):
            raise RuntimeError("Incomplete outer OOF coverage")
        predictions_by_seed[seed] = oof_predictions
        oof = pd.DataFrame({ID_COLUMN: ids, TARGET: y, "seed": seed, "fold": fold_assignments})
        for setup, classes in oof_predictions.items():
            oof[setup] = classes
            scores = [r["validation_qwk"] for r in fold_rows if r["seed"] == seed and r["setup"] == setup]
            repeat_rows.append({"seed": seed, "setup": setup, "oof_qwk": _qwk(y.to_numpy(), classes),
                                "mean_fold_qwk": float(np.mean(scores)), "std_fold_qwk": float(np.std(scores))})
            report = classification_report(y, classes, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
            for label in range(4):
                class_rows.append({"seed": seed, "setup": setup, "class": label, **report[str(label)],
                                   "true_positives": int(((y.to_numpy() == label) & (classes == label)).sum())})
        oof_tables.append(oof)
        pd.concat(oof_tables, ignore_index=True).to_csv(results_dir / "nested_oof_predictions.csv", index=False)
    summary, deltas = bootstrap_summary(y, predictions_by_seed, config.bootstrap_samples)
    summary.sort_values("mean_oof_qwk", ascending=False).to_csv(results_dir / "nested_cv_results.csv", index=False)
    deltas.to_csv(results_dir / "nested_paired_deltas.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(results_dir / "nested_repeat_results.csv", index=False)
    pd.DataFrame(class_rows).to_csv(results_dir / "nested_class_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(results_dir / "nested_tuning_results.csv", index=False)
    print("Final refit: repeat inner selection on ALL labeled training rows; equal weights are fixed", flush=True)
    deploy_seed = config.outer_seeds[0]
    inner_base, inner_tuned, selections, _ = select_members(X, y, ids, deploy_seed, config)
    calibration = calibrate(inner_base, inner_tuned, y, config)
    models, trees = {}, {}
    for name in MODEL_NAMES:
        models[name], trees[name] = fit_member(name, model_frame(name, X), y, selections[name]["params"], deploy_seed, config)
    bundle = {"models": models, "dictionary": pd.read_csv(Path(data_dir) / "data_dictionary.csv"),
              "feature_columns": data["feature_columns"], "weights": [0.25] * 4,
              "thresholds": calibration["equal_thresholds"].tolist(), "policy": PRIMARY}
    classes = predict_bundle(bundle, pd.read_csv(Path(data_dir) / "test.csv"))
    submission = make_submission(classes, data["test_ids"], data["sample"])
    submission.to_csv(results_dir / "submission_nested.csv", index=False)
    model_dir = Path(data_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "equal_ensemble.joblib", compress=3)
    restored = joblib.load(model_dir / "equal_ensemble.joblib")
    np.testing.assert_array_equal(classes, predict_bundle(restored, pd.read_csv(Path(data_dir) / "test.csv")))
    metadata = {
        "protocol": "Outer-held-out evaluation of inner hyperparameter selection, early stopping, and calibration",
        "primary_policy": PRIMARY, "historical_best_params_reused": False,
        "design_caveat": "The algorithm was redesigned after exploratory analysis of this dataset; these are resampling estimates, not a new untouched external cohort.",
        "config": asdict(config), "candidate_grids": CANDIDATES, "feature_columns": data["feature_columns"],
        "data": data["provenance"], "versions": {"python": platform.python_version(), **{name: version(name) for name in PACKAGES}},
        "source_sha256": {name: file_digest(Path(__file__).parent / name) for name in SOURCE_FILES},
        "artifact_sha256": {name: file_digest(results_dir / name) for name in (
            "nested_cv_results.csv", "nested_fold_results.csv", "nested_paired_deltas.csv", "nested_repeat_results.csv",
            "nested_class_metrics.csv", "nested_tuning_results.csv",
        )},
        "uncertainty": "95% percentile participant-paired bootstrap; same sampled IDs in every repeat; fixed OOF predictions, no retraining or post-selection adjustment",
        "fold_audit": audit,
        "deployment": {"policy": PRIMARY, "seed": deploy_seed, "selection": selections, "tree_counts": trees,
                       "weights": bundle["weights"], "thresholds": bundle["thresholds"],
                       "calibration_source": "inner OOF on all labeled train, following the exact outer-fold training algorithm",
                       "prediction_rows": len(submission), "serialization_round_trip_verified": True},
    }
    (results_dir / "nested_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary[["setup", "mean_oof_qwk", "ci_low", "ci_high"]].round(4).to_string(index=False), flush=True)
    print("Paired changes (do not claim significance when the interval includes zero):", flush=True)
    print(deltas.round(4).to_string(index=False), flush=True)
    return summary, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026])
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    run_experiment(ExperimentConfig(tuple(args.seeds), args.outer_splits, args.inner_splits,
                                   args.threads, args.bootstrap_samples), results_dir=args.results_dir)


if __name__ == "__main__":
    main()
