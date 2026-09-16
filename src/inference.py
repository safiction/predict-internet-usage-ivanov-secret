"""Predict from the primary bundle recreated by python -m src.experiment."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import DATA_DIR, ID_COLUMN, RESULTS_DIR, TARGET
from src.experiment import predict_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument("--model", type=Path, default=DATA_DIR / "models" / "equal_ensemble.joblib")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "predictions.csv")
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    if raw[ID_COLUMN].isna().any() or not raw[ID_COLUMN].is_unique:
        raise ValueError("Input participant IDs must be present and unique")
    # Only load locally trained/trusted joblib bundles; pickle is not a sandbox.
    predictions = predict_bundle(joblib.load(args.model), raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COLUMN: raw[ID_COLUMN], TARGET: predictions}).to_csv(args.output, index=False)
    print(f"Predicted {len(predictions)} participants")


if __name__ == "__main__":
    main()
