# Problematic Internet Use — ML Baseline

Baseline and evaluation pipeline for the Kaggle competition
[Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use)

The project currently uses the tabular competition data and focuses on reproducible CPU-friendly baselines evaluated with Quadratic Weighted Kappa

## Reproducing the pipeline

Steps 1-2 are one-time manual prerequisites
(Kaggle data is access-controlled and cannot be redistributed); everything after is a single command.

1. **Accept the competition rules** once:
   <https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/rules>
2. **Add Kaggle credentials:** Kaggle → Settings → API → *Create New Token*
   (downloads `kaggle.json`), then move it to `~/.kaggle/kaggle.json`
   (`C:\Users\<you>\.kaggle\kaggle.json` on Windows). Alternatively set the
   `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.
3. **Install dependencies:** `python -m pip install -r requirements.txt`
4. **Download the data:** `python -m src.download_data`
   (fetches only the tabular CSVs into `data/` and verifies row counts).
5. **Build the processed features:** run `notebooks/06_imputation_strategy.ipynb`
   to produce `data/processed/*.parquet` (the modeling input).
6. **Run the baseline:** run `notebooks/baseline.ipynb`.

Details for each step are in the sections below.

## Repository structure

```text
notebooks/01_data_overview.ipynb      Initial data audit
notebooks/02_missing_values.ipynb     Structure of missingness (per feature/participant/instrument)
notebooks/03_target_analysis.ipynb    Target analysis
notebooks/04_tabular_feature_eda.ipynb Feature EDA
notebooks/05_instrument_analysis.ipynb Instrument-level analysis
notebooks/06_imputation_strategy.ipynb Data-processing handoff (Layer A + imputation reference)
notebooks/baseline.ipynb              Data audit and baseline experiments
src/config.py                         Shared project settings
src/evaluation.py                     Shared CV and QWK utilities
src/features.py                       Layer A: deterministic feature engineering
src/imputation.py                     Layer B: fold-safe imputation components
data/processed/                       Layer A output for modeling
results/baseline_cv_results.csv       Saved fold-level baseline results
results/imputation_cv_results.csv     Reference CV for preprocessing choices
```

Raw competition files belong under `data/` and are excluded from Git

## Data processing (imputation) stage

Missingness is **not neutral** (see `notebooks/02_missing_values.ipynb`):
the target `sii` is derived from `PCIAT-PCIAT_Total` and rows with a missing
target are systematically less complete; some blocks are **structurally** absent
by age (the child `PAQ_C` vs adolescent `PAQ_A` questionnaires, the treadmill);
and missingness is **block-structured** (a whole assessment is present or absent
together):

- **Layer A `src/features.py` (`build_features`).** Deterministic, row-local
  transforms that never read statistics from other rows: drop the `PCIAT-*`
  leakage columns, merge the age-gated `PAQ_C`/`PAQ_A` into one `PAQ_total`
  (+ `PAQ_version`, `PAQ_missing`, `PAQ_season`), and add one
  `<instrument>_not_administered` flag per assessment block. Contains NaN in
  some gaps **by design**.
- **Layer B `src/imputation.py` (`make_preprocessor`).** Learned imputation
  (median, or iterative for the linked BIA/Physical blocks) plus scaling and
  one-hot encoding, delivered as a scikit-learn `ColumnTransformer`. It is fit
  **inside each CV fold** and never materialized.

**One rule:** call `make_preprocessor(...)` only inside a `Pipeline`/CV; never
fit it on the full dataset and never save its output. For the final test
prediction, fit on all of train and apply to test exactly once. Fitting
imputation on the whole dataset leaks validation/test information into training
and gives optimistic, non-reproducible CV.

**Scope decision: actigraphy not used.** We use only the tabular CSVs.
The accelerometer time series (`series_*.parquet`) are a separate modality with
heavy missingness, absent for many participants, and cannot be used to fill
tabular gaps; they are out of scope for this stage.

**Reference CV** (`results/imputation_cv_results.csv`, on the shared stratified
folds): Ridge with Layer A + median imputation matches the raw baseline mean QWK
(0.369) while halving fold-to-fold variance (std ~0.026 vs ~0.039);
iterative block imputation does not beat the median on the linear model, so the
median is the default; an untuned NaN-native `HistGradientBoosting` overfits
(train QWK ~1.0) and is left as a direction to tune.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current environment was tested with Python 3.14.4. Compatibility with optional boosting libraries will be checked before adding them

## Data location

The CSV files live directly under `data/`:

```text
data/train.csv
data/test.csv
data/data_dictionary.csv
data/sample_submission.csv
data/processed/            # generated by notebooks/06 (Layer A output)
```

To fetch them, run once per machine:

```bash
python -m src.download_data
```

This requires Kaggle credentials (`~/.kaggle/kaggle.json` or the
`KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables) and accepting the
competition rules on Kaggle once.

Paths are defined once in `src/config.py` (`TRAIN_PATH`, `TEST_PATH`,
`DATA_DICT_PATH`, `DATA_DIR`, `PROCESSED_DIR`, `RESULTS_DIR`); notebooks and
scripts import them.

The baseline excludes `id`, the target `sii`, and all `PCIAT-*` columns from model features because the PCIAT score directly determines the target

## Running the baseline

Open `notebooks/baseline.ipynb`, select the `.venv` kernel, and run all cells in order

All models use the same five stratified folds with shuffling and random state 42. Fold assignment is made stable by participant ID
