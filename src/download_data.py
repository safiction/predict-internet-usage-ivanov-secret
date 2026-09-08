"""Fetch the competition CSVs into the flat ``data/`` layout, reproducibly.

Run once per machine:

    python -m src.download_data            # fetch missing files
    python -m src.download_data --force    # re-download and overwrite

Prerequisites must be done before downloading the data:

1. Accept the competition rules:
   https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/rules
2. Provide Kaggle credentials, either
   - ``~/.kaggle/kaggle.json`` (Kaggle -> Settings -> API -> Create New Token), or
   - the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.

Only the tabular CSVs are fetched; the actigraphy series are intentionally skipped.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import kagglehub

from src.config import DATA_DIR

COMPETITION = "child-mind-institute-problematic-internet-use"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"

# Tabular files needed by the project. 
CSV_FILES = [
    "train.csv",
    "test.csv",
    "data_dictionary.csv",
    "sample_submission.csv",
]

# Expected row counts, used only as a post-download sanity check.
EXPECTED_ROWS = {"train.csv": 3960, "test.csv": 20}


def _credentials_available() -> bool:
    """True if kagglehub can find credentials (env vars or kaggle.json)."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _print_credentials_help() -> None:
    print(
        "\n".join(
            [
                "Kaggle credentials not found.",
                "",
                "Fix:",
                "  1. Accept the competition rules:",
                f"     {RULES_URL}",
                "  2. Kaggle -> Settings -> API -> Create New Token "
                "(downloads kaggle.json).",
                f"  3. Move it to: {Path.home() / '.kaggle' / 'kaggle.json'}",
                "     (or set the KAGGLE_USERNAME / KAGGLE_KEY environment variables).",
                "",
                "Re-run: python -m src.download_data",
            ]
        ),
        file=sys.stderr,
    )


def _explain_http_error(status: int) -> None:
    if status == 401:
        message = [
            "Kaggle returned 401 (unauthenticated).",
            "The credentials are being rejected",
            "Most likely API token is stale: creating a new token expires the",
            "old one, so an old kaggle.json stops working.",
            "Fix: Kaggle -> Settings -> API -> Expire Token, then Create New Token,",
            f"and replace {Path.home() / '.kaggle' / 'kaggle.json'} with the fresh file.",
        ]
    elif status == 403:
        message = [
            "Kaggle returned 403 (forbidden).",
            "The credentials work, but this account has not accepted the",
            f"competition rules yet. Accept here:\n  {RULES_URL}",
        ]
    else:
        message = [f"Kaggle returned HTTP {status}."]
    print("\n".join(message), file=sys.stderr)


def download(force: bool = False) -> None:
    """Download each CSV into ``DATA_DIR`` unless it is already present."""
    if not _credentials_available():
        _print_credentials_help()
        raise SystemExit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in CSV_FILES:
        destination = DATA_DIR / filename
        if destination.exists() and not force:
            print(f"skip (already present): {filename}")
            continue

        # path=filename fetches one file, avoiding parquets.
        cached_path = Path(
            kagglehub.competition_download(
                COMPETITION,
                path=filename,
                force_download=force,
            )
        )
        shutil.copyfile(cached_path, destination)
        print(f"saved: {destination.relative_to(DATA_DIR.parent)}")


def verify() -> bool:
    """Check that the expected files exist and have the expected row counts."""
    import pandas as pd

    ok = True
    for filename in CSV_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"MISSING: {filename}", file=sys.stderr)
            ok = False
            continue
        expected = EXPECTED_ROWS.get(filename)
        if expected is not None:
            actual = len(pd.read_csv(path))
            status = "ok" if actual == expected else f"WARNING (expected {expected})"
            print(f"{filename}: {actual} rows [{status}]")
            if actual != expected:
                ok = False
        else:
            print(f"{filename}: present")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the competition CSVs into data/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite files that already exist in data/.",
    )
    args = parser.parse_args()

    try:
        download(force=args.force)
    except Exception as error:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            match = re.search(r"\b(401|403)\b", str(error))
            status_code = int(match.group(1)) if match else None
        if status_code in (401, 403):
            _explain_http_error(status_code)
            raise SystemExit(1)
        raise

    print("\nVerifying downloaded files...")
    if verify():
        print("\nData is ready in", DATA_DIR)
    else:
        print("\nSome files look wrong, see warnings", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
