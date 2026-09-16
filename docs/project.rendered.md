# Predicting problematic internet use from tabular health and activity data

Anonymous technical report. All numeric results and plots below are generated
from the current experiment artifacts, not entered by hand.

## 1. Problem statement

The task is to predict the ordinal Severity Impairment Index (sii) from the
tabular Child Mind Institute - Problematic Internet Use competition data [1].
The features describe demographics, physical measurements, physical activity,
fitness, body composition, sleep, and reported computer/internet use. The label
has four ordered classes: 0 (none), 1 (mild), 2 (moderate), and 3 (severe).
This is a predictive task in the internet-use/physical-activity domain, not an
estimate of a causal effect of internet use on physical activity.

The raw training table contains 3960 participants: 2736 labeled and 1224 unlabeled. Class counts are 0: 1594, 1: 730, 2: 378, 3: 34. The example test contains 20 rows.

Only labeled participants are used for supervised fitting and validation.
Missing labels are not assigned a class or pseudo-labeled. PCIAT questionnaire
items, total score, and season are excluded because the label is derived from
the questionnaire total. Participant ID is only used for alignment and stable
fold assignment. The public example test file has no usable evaluation labels;
producing a submission is not evidence of hidden-test or leaderboard accuracy.

The primary metric is quadratic weighted kappa (QWK), using the fixed class
scale 0, 1, 2, 3. With true classes y and predicted classes c, it is
1 - observed / expected, where observed = sum((y-c)^2), and expected =
sum(y^2) + sum(c^2) - 2*mean(y)*sum(c). Larger ordinal errors receive larger
penalties. Class precision, recall, F1, prediction counts, and confusion
matrices supplement QWK; aggregate agreement alone can hide rare-class errors.

## 2. Baselines

We test the standard implementations provided by scikit-learn, CatBoost,
and XGBoost [2-5]; these libraries and algorithms are not claimed as our own
inventions. The experiment includes a majority-class predictor, Ridge with
fixed midpoint rounding, and single CatBoost, XGBoost, ExtraTrees, and Ridge
regressors with training-only QWK calibration. Each untuned member uses
candidate 0 from the explicit grid in src/experiment.py.

The reference equal ensemble averages the four untuned continuous predictions
before calibration. An additional strong reference chooses a tuned single
member using inner-OOF QWK, never the outer-fold score. This avoids choosing
the best reference retrospectively on validation labels.

All current comparisons use the same participants, feature preparation, outer
folds, and QWK definition. Historical notebook 07/08/early-09 winning scores are
exploratory and are not used to choose current numerical hyperparameters or
assert a current improvement. The earlier global tuning and calibration
protocol differs from the current nested procedure.

## 3. Proposed model and training procedure

The proposed architecture is an equal-weight ensemble of four regressors:
CatBoost (categorical boosting [3]), XGBoost (histogram tree boosting [4]),
ExtraTrees (randomized trees [5]), and Ridge (regularized linear regression [2]).
Its continuous score is 0.25 times the sum of member scores. The three strictly
ordered cut points map that score to classes 0-3. Equal averaging provides
different inductive biases without adding learned blend weights to the primary
architecture. The contribution is this trained ordinal ensemble and its
evaluation/calibration pipeline, not a novel boosting algorithm.

Layer A preprocessing is deterministic and row-local. It checks enumerated
codes against the competition dictionary, applies fixed heuristic plausibility
bounds, recomputes BMI using the dictionary's inch/pound units, merges child
and adolescent PAQ totals, and flags blocks with all measurement results
missing. Season metadata does not count as a measurement. These flags do not
infer whether a questionnaire was actually administered. The plausibility
bounds and BMI discrepancy tolerances are project heuristics, not clinically
validated diagnostic rules; they must not be used to make clinical decisions.

CatBoost retains numeric NaNs and categorical strings; categorical missing
values become an explicit level. Other members use training-partition median
imputation and one-hot encoding; Ridge additionally uses standard scaling.
The XGBoost member also receives five deterministic interactions: treadmill
duration, waist/height ratio, pulse pressure, a raw fitness composite, and
BMI/age. The fitness composite combines heterogeneous measurements and is not
a clinical score. An entirely missing fitness block remains NaN, not zero.

The measured run uses 2 repeated ID-stable stratified outer CV runs, with seeds [42, 2026], 5 outer folds, and 3 inner folds per outer training partition. CPU pools are limited to 4 threads. Early stopping has patience 60.

Within each outer training partition, each member's generic candidate grid is
evaluated on three inner stratified folds. Candidate capacity/regularization is
chosen by inner-OOF RMSE. QWK cut points are then calibrated on the selected
inner-OOF scores. The coordinate search evaluates all distinct score cut points
for each boundary and uses multiple starting points; it is a local optimizer,
not a guarantee of a global optimum. Hyperparameter selection and calibration
may overfit the inner data; only the untouched outer score evaluates their
combined effect [6]. No previously winning Optuna parameters are reused.

For boosting, each individual training partition is split 85/15 for early
stopping. XGBoost preprocessing is learned on the 85% only. A fresh model and
fresh preprocessing are refit on the entire available partition with the
selected tree count. Outer validation labels never decide the tree count,
medians, vocabularies, scaling, capacity, weights, or class thresholds.

For deployment, the same inner-selection/calibration algorithm is run on all
labeled training participants, followed by final fits on all labeled rows.
Weights remain exactly 0.25 for each member. The saved model bundle reproduces
the submission after serialization and reload. The primary policy is specified
in code, not switched to a retrospectively higher full-OOF blend score.

## 4. Model improvements and controlled comparisons

The improvement over the untuned proposed ensemble is inner-fold selection of
tree capacity, regularization, sampling, Ridge alpha, and training-fold
percentile clipping for Ridge. Clipping bounds are learned on training data
only; they protect a scale-sensitive linear member against extreme input values.
Both ensemble versions
receive the same row-local features and inner-only rounding procedure; the
comparison isolates the adaptive member hyperparameter selection procedure.
The search has three candidates per member and a fixed, reproducible budget.
The full grids, selected candidates, inner RMSE, and stopping tree counts are
saved in results/nested_metadata.json and results/nested_tuning_results.csv.

A secondary weighted ensemble searches nonnegative weights in steps of 0.25
and calibrates them on inner OOF only. Its outer scores are reported but it is
not the primary exported policy. We do not assume that the most complex
ensemble or hyperparameter tuning must improve QWK.

| reference | delta_qwk | ci_low | ci_high |
| --- | --- | --- | --- |
| Inner-selected single model | 0.0091 | -0.0031 | 0.0213 |
| Baseline equal ensemble | 0.0107 | -0.0009 | 0.0227 |
| Tuned xgboost | 0.0068 | -0.0061 | 0.0202 |

Against Inner-selected single model, the observed change is +0.0091; its interval includes zero; a reliable gain is not established. Against Baseline equal ensemble, the observed change is +0.0107; its interval includes zero; a reliable gain is not established. Against Tuned xgboost, the observed change is +0.0068; its interval includes zero; a reliable gain is not established.

## 5. Results and error analysis

The following table reports the mean whole-OOF QWK across repeated nested CV
runs, not an average obtained by concatenating repeat rows as independent
participants. Fold-level and repeat-level results are saved separately.

| setup | mean_oof_qwk | ci_low | ci_high |
| --- | --- | --- | --- |
| Tuned equal ensemble | 0.4669 | 0.4340 | 0.4973 |
| Tuned weighted ensemble | 0.4627 | 0.4293 | 0.4926 |
| Tuned xgboost | 0.4601 | 0.4270 | 0.4904 |
| Inner-selected single model | 0.4579 | 0.4255 | 0.4890 |
| Baseline equal ensemble | 0.4562 | 0.4219 | 0.4873 |
| Tuned extra_trees | 0.4555 | 0.4208 | 0.4880 |
| Tuned catboost | 0.4553 | 0.4226 | 0.4852 |
| Baseline catboost | 0.4544 | 0.4208 | 0.4840 |
| Baseline extra_trees | 0.4504 | 0.4173 | 0.4825 |
| Tuned ridge | 0.4382 | 0.4045 | 0.4698 |
| Baseline ridge | 0.4362 | 0.4031 | 0.4679 |
| Baseline xgboost | 0.4268 | 0.3934 | 0.4578 |
| Ridge fixed midpoints | 0.3609 | 0.3330 | 0.3857 |
| Dummy majority | 0.0000 | 0.0000 | 0.0000 |

![Nested CV comparison](results/figures/nested_qwk.png)

The intervals are percentile bootstrap diagnostics conditional on the fixed
OOF predictions. The same sampled participant indices are used in every repeat
and in both sides of every comparison. This accounts for participant pairing
and avoids treating repeated predictions as independent observations. It does
not repeat model fitting or all historical design decisions; it is not a full
retraining or post-selection confidence interval. Two random seeds provide a
limited sensitivity check, not exhaustive uncertainty quantification.

| class | precision | recall | f1-score | true_positives |
| --- | --- | --- | --- | --- |
| 0 | 0.7657 | 0.6888 | 0.7252 | 1098.0 |
| 1 | 0.3548 | 0.3863 | 0.3698 | 282.0 |
| 2 | 0.3400 | 0.4325 | 0.3807 | 163.5 |
| 3 | 0.2692 | 0.2059 | 0.2333 | 7.0 |

Across repeats, severe-class recall is 0.206, with a mean 7.0 true positives out of 34 severe participants per repeat. The table averages class metrics across repeats; fractional true-positive counts are repeat means, not fractional participants. Confusion-matrix counts sum across repeats and are not counts of unique children.

![Primary ensemble confusion matrix](results/figures/nested_confusion.png)

Errors and rare-class performance are reported even when they weaken the
headline result. The model is not presented as ready for diagnosis or clinical
screening. The small severe-class sample limits conclusions and motivates more
labeled examples and external validation rather than unsupported accuracy
claims. Excluding unlabeled participants can induce selection bias because
their feature missingness differs; the evaluated population is the labeled
cohort, not every child or every competition participant.

## 6. Reproducibility

Reference runtime: python 3.13.5, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.15.3, catboost 1.2.10, xgboost 3.4.1, optuna 5.0.0, pyarrow 24.0.0, joblib 1.6.0, threadpoolctl 3.7.0.

The reference environment was installed in a new virtual environment, checked
with pip check, and tested before training. requirements.txt pins direct
dependencies; requirements.lock.txt also pins all resolved transitive packages
for CPython 3.13.5 on Linux x86_64. Other Python/platform combinations are not
claimed to reproduce the exact numerical run. Source and raw-data SHA256
digests, grids, seeds, fold selections, runtime versions, and result artifact
digests are recorded in the metadata. Notebook cache is accepted only when
the configuration, data, source, versions, and aggregate artifacts match.

Reproduce from the extracted anonymous source directory:

```text
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

Before downloading, the reviewer must accept the competition rules and provide
their own Kaggle credentials [1], as detailed in README. Access-controlled CSVs,
participant-level OOF records, and model binaries are excluded from the source
archive; the download/training commands recreate them. No private pretrained
weights are required. The experiment builds processed features from raw CSVs
and does not rely on a stale local parquet. The exported notebook is optional;
the command-line workflow requires no manually reconstructed notebook state.

The technical report and plots are generated from verified aggregate artifacts.
The anonymous archive excludes Git history, local environments, credentials,
personal paths, and notebook outputs/metadata. Packaging uses an explicit
allowlist and an automated path/identity-pattern scan. A final human anonymity
inspection is still recommended; an automated scan is not proof that every
possible identifying clue has been excluded.

## 7. Conclusion and limitations

The prespecified tuned equal ensemble achieves mean nested OOF QWK 0.4669, with a conditional bootstrap interval [0.4340, 0.4973]. Against Inner-selected single model, the observed change is +0.0091; its interval includes zero; a reliable gain is not established. Against Baseline equal ensemble, the observed change is +0.0107; its interval includes zero; a reliable gain is not established. Against Tuned xgboost, the observed change is +0.0068; its interval includes zero; a reliable gain is not established. The project implements several baselines, a trained proposed ensemble, and explicit inner-only hyperparameter modifications with measured outcomes; these are not claims of universal superiority.

The current procedure nests algorithmic hyperparameter selection and
calibration inside outer training folds. However, project design followed
exploratory analysis of this same dataset. We therefore do not describe these
resampling estimates as a completely new untouched external cohort. Future
work should use additional labeled data or an independently held-out cohort,
especially for severe cases, and assess transportability across age and sex.
Actigraphy is not used: it is a separate modality with incomplete coverage and
was excluded to keep this tabular experiment reproducible on CPU. This scope
choice may limit attainable performance and is not claimed to be optimal.

## References and attribution

[1] Child Mind Institute - Problematic Internet Use. Competition data,
evaluation task, rules, and data dictionary. https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

[2] scikit-learn documentation. Ridge regression, preprocessing, stratified
cross-validation, and metric implementations. https://scikit-learn.org/1.6/

[3] Dorogush, Ershov, and Gulin. CatBoost: gradient boosting with categorical
features support. https://arxiv.org/abs/1810.11363

[4] Chen and Guestrin. XGBoost: A Scalable Tree Boosting System.
https://arxiv.org/abs/1603.02754

[5] scikit-learn. ExtraTreesRegressor implementation and references.
https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

[6] scikit-learn. Nested versus non-nested cross-validation.
https://scikit-learn.org/1.6/auto_examples/model_selection/plot_nested_cross_validation_iris.html

All baseline/member algorithms are used through the attributed packages,
not copied and claimed as new implementations. Figures are generated from this
project's computed results. The composite architecture, glue code, tests, and
report were developed with AI assistance; all numeric claims are checked against
executed artifacts. This disclosure is not a claim of algorithmic novelty.
