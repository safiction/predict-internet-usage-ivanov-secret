"""Integration checks for the reviewed CatBoost/AdaBoost/XGBoost PRs."""

import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from src.features import add_measurement_missing_features, build_features
from src.package import export_files

ROOT = Path(__file__).resolve().parents[1]
EMPTY_DICTIONARY = pd.DataFrame(columns=["Field", "Type", "Values"])


class UpstreamTests(unittest.TestCase):
    def test_measurement_counts_ignore_season_and_are_row_local(self):
        frame = pd.DataFrame({"FGC-Season": ["Spring", "Winter", None],
                              "FGC-FGC_CU": [np.nan, 1., 2.], "FGC-FGC_PU": [np.nan, np.nan, 3.]})
        result = add_measurement_missing_features(frame, add_flags=True, add_counts=True)
        self.assertEqual(result.FGC_measurements_missing.tolist(), [1, 0, 0])
        self.assertEqual(result.FGC_observed_count.tolist(), [0, 1, 2])
        row_local = pd.concat([add_measurement_missing_features(frame.iloc[[i]], add_flags=True, add_counts=True)
                               for i in range(len(frame))])
        pd.testing.assert_frame_equal(result, row_local)
        self.assertNotIn("FGC_observed_count", frame)

    def test_missing_helper_is_idempotent_on_current_features(self):
        raw = pd.DataFrame({"id": ["a", "b"], "FGC-Season": ["Spring", "Winter"],
                            "FGC-FGC_CU": [np.nan, 1.]})
        prepared = build_features(raw, EMPTY_DICTIONARY)
        pd.testing.assert_frame_equal(prepared, add_measurement_missing_features(prepared, add_flags=True))
        pd.testing.assert_frame_equal(prepared, add_measurement_missing_features(prepared))
        once = add_measurement_missing_features(prepared, add_flags=True, add_counts=True)
        pd.testing.assert_frame_equal(once, add_measurement_missing_features(once, add_flags=True, add_counts=True))

    def test_season_only_block_has_no_synthetic_measurement_flag(self):
        frame = pd.DataFrame({"FGC-Season": ["Spring"]})
        pd.testing.assert_frame_equal(frame, add_measurement_missing_features(frame, add_flags=True, add_counts=True))

    def test_anonymous_allowlist_excludes_incoming_participant_artifacts(self):
        exported = [path.relative_to(ROOT).as_posix() for path in export_files(ROOT)]
        self.assertFalse(any(name.startswith(("data/", "results/catboost")) for name in exported))
        self.assertFalse(any("oof_predictions" in name or name.endswith((".cbm", ".joblib")) for name in exported))

    def test_notebook06_keeps_both_pr_changes(self):
        notebook = nbformat.read(ROOT / "notebooks" / "06_imputation_strategy.ipynb", as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        self.assertIn("add_measurement_missing_features", source)
        self.assertIn("random_state=RANDOM_STATE", source)
        self.assertNotIn("random_state=42", source)
        self.assertTrue((ROOT / "notebooks" / "09_catboost.ipynb").exists())


if __name__ == "__main__":
    unittest.main()
