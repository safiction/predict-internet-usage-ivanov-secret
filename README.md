# Problematic Internet Use — reproducible ordinal ensemble

Predict `sii` (0–3) from tabular health, fitness, activity, and internet-use data
from the [Child Mind Institute competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use).
This is prediction, not a causal estimate of internet use on physical activity.
The model is not a clinical diagnostic or screening system.

## Complete reference workflow

Reference platform: **CPython 3.13.5, Linux x86_64**. The reference run uses a new
isolated environment; `requirements.lock.txt` pins direct and transitive packages.
Other platforms/Python versions are not claimed to reproduce identical numbers.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.download_data
.venv/bin/python -m src.experiment
.venv/bin/python -m src.verify
.venv/bin/python -m src.report
.venv/bin/python -m src.package
```

Before downloading, accept the [competition rules](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/rules)
and provide **your own** Kaggle credentials in `~/.kaggle/kaggle.json`, or set
`KAGGLE_USERNAME` and `KAGGLE_KEY`. Never put credentials in the repository or
submission. If the four CSVs are already available, skip the download command.
The access-controlled raw files, participant-level predictions, and trained
model bundle remain under Git-ignored paths.

The experiment prepares features directly from raw CSVs, regenerates the
processed parquet handoff, and runs **two repeats × five outer × three inner
folds**. CPU/BLAS pools are limited to four threads. Runtime depends on hardware;
allow time for hundreds of small fits. All settings are visible in
`src/experiment.py`; lower-budget custom runs must not be presented as the
reference experiment.

## What is evaluated and exported

The primary model is a trained **equal-weight CatBoost + XGBoost + ExtraTrees +
Ridge ensemble**, followed by three QWK-calibrated class cut points.

For each outer training partition:

1. Test three generic candidate configurations per member on three inner folds.
2. Choose member hyperparameters by inner-OOF RMSE.
3. Learn QWK thresholds on inner OOF only.
4. Refit selected members on outer training and predict the untouched outer fold.

Boosting uses a separate 85/15 training-only stopping holdout, then fresh
full-partition refits at the selected tree count. Imputation, clipping,
one-hot vocabularies, and scaling are always learned on the corresponding
training partition. **No best Optuna parameters from notebooks 07/08 are reused.**

The deployed model repeats exactly this training/calibration algorithm on all
labeled rows and keeps weights `[0.25, 0.25, 0.25, 0.25]`. A secondary weighted
ensemble is evaluated separately; it never silently replaces the primary export.
The saved bundle is reloaded and its predictions checked against the submission.

Results use whole-OOF QWK per repeat. Confidence intervals resample paired
participants with the same indices across CV repeats; repeated predictions are
not treated as independent people. These are conditional fixed-prediction
bootstrap diagnostics, not full retraining/post-selection intervals. Project
design followed earlier exploration of the same dataset, so even the revised
nested experiment is not an untouched external cohort. No statistically reliable
improvement is claimed when the paired interval includes zero.

## Features and scope

`src/features.py` provides row-local Layer A preparation:

- dictionary enum validation and fixed **heuristic**, not clinically validated, bounds;
- imperial-unit BMI consistency checks;
- removal of every `PCIAT-*` leakage feature;
- merged child/adolescent PAQ scores and missingness/version information;
- `<instrument>_measurements_missing` flags, excluding season metadata.

Flags describe missing **results**, not whether an assessment was administered.
Learned Layer B preprocessing lives in `src/imputation.py`; **median is the
actual function default**. Ridge tuning can add fold-learned 1%/99% clipping.
The XGBoost member also receives five row-local experimental interactions; a
fully absent fitness composite stays NaN. Actigraphy is outside this tabular
experiment. Unlabeled participants are excluded from supervised evaluation;
their nonrandom missingness limits population generalization.

## Repository map

```text
notebooks/01–05                   Raw data, target, missingness, and instrument EDA
notebooks/06_imputation_strategy.ipynb  Processed-feature handoff and preprocessing exploration
notebooks/baseline.ipynb          Historical classifier/linear baselines
notebooks/07_adaboost.ipynb       Historical AdaBoost exploration
notebooks/07_catboost.ipynb       Historical CatBoost exploration
notebooks/08_xgboost.ipynb        Historical XGBoost exploration
notebooks/09_ensemble.ipynb      Current fully nested workflow and result inspection
src/experiment.py               Authoritative training/evaluation/deployment algorithm
src/features.py                 Deterministic preparation
src/imputation.py               Fold-learned preprocessing
src/evaluation.py               ID-stable configurable splits and fixed-scale QWK
src/ensemble.py                 Coordinate threshold optimization and blending
src/verify.py                   Independent artifact/provenance/fold/score checks
src/inference.py                Predictions from a trusted locally trained bundle
src/report.py                   Data-driven anonymous project.pdf generation
src/package.py                  Allowlisted anonymous project.zip export
tests/                          Metric, preprocessing, fold isolation, deployment, export checks
docs/project.md                 Seven-section report template and source attribution
project.pdf                     Generated technical report
results/nested_*                 Current aggregate metrics and provenance
dist/project.zip                Anonymous source + report archive for manual submission
```

Earlier CSVs without the `nested_` prefix are historical exploratory artifacts,
not current validation scores. Their preprocessing, environment, and calibration
protocols may differ; do not compare them as a single experiment table. Cached
outputs/runtime metadata have been removed from historical notebooks for anonymity.

## Outputs and inspection

`results/nested_cv_results.csv` contains repeat-mean OOF QWK and conditional
95% intervals. `nested_paired_deltas.csv` distinguishes observed changes from
uncertain gains. Fold/repeat tables, class precision/recall/F1/true positives,
inner candidate RMSE, and the complete audit are saved separately.
`nested_metadata.json` records grids, choices, seeds, raw-data/source/result
SHA256 digests, and actual runtime versions. Severe-class limitations are
reported in the generated PDF rather than hidden by aggregate QWK.

Participant-level `nested_oof_predictions.csv`, `submission_nested.csv`, and
`data/models/equal_ensemble.joblib` are local artifacts. The example test file
is not the hidden Kaggle evaluation set.

To inspect notebook 09, choose this environment's Python kernel. It either runs
training or loads results after checking identical data/code/environment/config
and aggregate artifact digests. Set `FORCE_RETRAIN = True` to retrain.
For an optional local Jupyter kernel:

```bash
.venv/bin/python -m ipykernel install --user --name pml-repro --display-name "Python (PML reproducible)"
```

Command-line training does not require Jupyter or manual notebook execution.

For inference after training:

```bash
.venv/bin/python -m src.inference --input data/test.csv --output results/predictions.csv
```

Only load trusted locally trained joblib bundles; pickle deserialization is not
safe for untrusted files.

## Anonymous submission and attribution

`python -m src.report` creates all seven required report sections from verified
current results and regenerates plots. `python -m src.package` creates
`dist/project.zip` using an explicit allowlist, sanitized notebook copies, and
a checked `MANIFEST.json`. It excludes raw data, credentials, participant-level
records, models, environments, local paths, and Git history. It never rewrites or
deletes local Git history. Perform a final human inspection before **manually**
uploading to Moodle; automated checks and this code do not guarantee a grade.

Library algorithms are attributed in `docs/project.md` and `project.pdf`.
The ensemble architecture, integration, tests, and drafting used AI assistance;
all numbers come from executed, independently checked artifacts. No clinical
validity, causal effect, external-test score, or novel boosting algorithm is
claimed.
