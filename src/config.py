"""Shared project configuration"""

from pathlib import Path

TARGET = "sii"
ID_COLUMN = "id"
LEAKAGE_PREFIX = "PCIAT-"

N_SPLITS = 5
RANDOM_STATE = 42


def get_project_root() -> Path:
    """Resolve the repo root whether the cwd is the root or ``notebooks/``."""
    root = Path.cwd()
    return root.parent if root.name == "notebooks" else root


# Locate data in a flat ``data/``:
#   data/train.csv, data/test.csv, data/data_dictionary.csv, data/processed/
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
DATA_DICT_PATH = DATA_DIR / "data_dictionary.csv"
