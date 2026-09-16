"""Shared project configuration"""

from pathlib import Path

TARGET = "sii"
ID_COLUMN = "id"
LEAKAGE_PREFIX = "PCIAT-"

N_SPLITS = 5
RANDOM_STATE = 42


def get_project_root() -> Path:
    """Resolve paths from the source location, not the caller's working directory."""
    return Path(__file__).resolve().parents[1]


# Locate data in a flat ``data/``:
#   data/train.csv, data/test.csv, data/data_dictionary.csv, data/processed/
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
DATA_DICT_PATH = DATA_DIR / "data_dictionary.csv"
