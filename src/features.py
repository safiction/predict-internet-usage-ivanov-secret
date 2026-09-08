"""Layer A: deterministic, row-local data preparation.

Every transform operates on a single row and never reads statistics from other
rows. Output is saved to ``data/processed``.

Pipeline order (see ``build_features``):
1. codebook validation   - values outside the data-dictionary's allowed set -> NaN
2. domain-range validity - physiologically impossible values -> NaN
3. BMI cross-consistency - flag (and null gross) stated-vs-recomputed BMI
4. merge age-gated PAQ    - PAQ_C / PAQ_A -> one score + flags
5. instrument indicators  - one "assessment not administered" flag per block
6. drop leakage           - PCIAT-* columns encode the target
"""

from __future__ import annotations

import pandas as pd
from src.config import DATA_DICT_PATH, ID_COLUMN, LEAKAGE_PREFIX

# Assessment blocks for which "the whole assessment is missing" is
# informative. Demographics and Internet Use are excluded.
INDICATOR_INSTRUMENTS = [
    "CGAS",
    "Physical",
    "Fitness_Endurance",
    "FGC",
    "BIA",
    "SDS",
]

# Fixed clinical/plausibility bounds [low, high] (inclusive). Values outside are
# physiologically impossible for the cohort (children/adolescents) and are set  to NaN and imputer handles them.
DOMAIN_BOUNDS: dict[str, tuple[float, float]] = {
    "Basic_Demos-Age": (1, 100),
    "Physical-BMI": (5, 100),
    "Physical-Height": (20, 90),
    "Physical-Weight": (10, 600),
    "Physical-Waist_Circumference": (5, 100),
    "Physical-Diastolic_BP": (20, 200),
    "Physical-Systolic_BP": (40, 300),
    "Physical-HeartRate": (20, 250),
    "CGAS-CGAS_Score": (1, 100),
    "Fitness_Endurance-Time_Sec": (0, 59),
    "Fitness_Endurance-Time_Mins": (0, 120),
    "Fitness_Endurance-Max_Stage": (0, 40),
}

# BMI cross-consistency thresholds (relative difference vs recomputed BMI).
_BMI_FLAG_TOLERANCE = 0.15   # flag as inconsistent above this
_BMI_NULL_TOLERANCE = 0.50   # additionally null the stated BMI above this
_LBS_IN_BMI_CONSTANT = 703.0  # BMI = 703 * lb / in^2

# PAQ columns consumed for merge
_PAQ_CHILD_TOTAL = "PAQ_C-PAQ_C_Total"
_PAQ_ADOLESCENT_TOTAL = "PAQ_A-PAQ_A_Total"
_PAQ_CHILD_SEASON = "PAQ_C-Season"
_PAQ_ADOLESCENT_SEASON = "PAQ_A-Season"


# 1. Codebook validation
def load_data_dictionary() -> pd.DataFrame:
    """Load the competition data dictionary"""
    return pd.read_csv(DATA_DICT_PATH)


def _parse_allowed_values(type_text: str, values_text: str):
    """Parse the dictionary's ``Values`` cell into a set of allowed values."""
    if not isinstance(values_text, str) or not values_text.strip():
        return None
    tokens = [t.strip() for t in values_text.split(",") if t.strip()]
    if isinstance(type_text, str) and "int" in type_text:
        # categorical int -> compare against integer codes.
        try:
            return {int(t) for t in tokens}
        except ValueError:
            return None
    # str enumerations
    return set(tokens)


def build_codebook(data_dict: pd.DataFrame) -> dict[str, set]:
    """Map each enumerable feature to its allowed value set"""
    codebook: dict[str, set] = {}
    for _, row in data_dict.iterrows():
        field = row.get("Field")
        if not isinstance(field, str) or field.startswith(LEAKAGE_PREFIX):
            continue
        allowed = _parse_allowed_values(row.get("Type"), row.get("Values"))
        if allowed:
            codebook[field] = allowed
    return codebook


def _apply_codebook(df: pd.DataFrame, codebook: dict[str, set]) -> dict[str, int]:
    """Set values outside the allowed set to NaN. Return per-column counts."""
    counts: dict[str, int] = {}
    for column, allowed in codebook.items():
        if column not in df.columns:
            continue
        invalid = df[column].notna() & ~df[column].isin(allowed)
        n = int(invalid.sum())
        if n:
            df.loc[invalid, column] = pd.NA
            counts[column] = n
    return counts


# 2. Domain-range validity
def _apply_domain_bounds(df: pd.DataFrame) -> dict[str, int]:
    """Set physiologically impossible values to NaN. Returns per-column counts."""
    counts: dict[str, int] = {}
    for column, (low, high) in DOMAIN_BOUNDS.items():
        if column not in df.columns:
            continue
        out_of_range = df[column].notna() & ((df[column] < low) | (df[column] > high))
        n = int(out_of_range.sum())
        if n:
            df.loc[out_of_range, column] = pd.NA
            counts[column] = n
    return counts

# 3. BMI cross-consistency (compare stated and recomputed)
def _add_bmi_consistency(df: pd.DataFrame) -> dict[str, int]:
    """Flag (and null gross) disagreements between stated and recomputed BMI.

    Recomputes BMI = 703 * weight_lb / height_in^2 and compares to Physical-BMI.
    Adds a ``bmi_inconsistent`` flag. Returns counts of flagged and nulled rows.
    """
    needed = {"Physical-Height", "Physical-Weight", "Physical-BMI"}
    if not needed.issubset(df.columns):
        return {}

    height = df["Physical-Height"]
    weight = df["Physical-Weight"]
    stated = df["Physical-BMI"]

    recomputed = _LBS_IN_BMI_CONSTANT * weight / (height**2)
    comparable = recomputed.notna() & stated.notna() & (stated > 0)
    relative_diff = (recomputed - stated).abs() / stated

    flagged = comparable & (relative_diff > _BMI_FLAG_TOLERANCE)
    gross = comparable & (relative_diff > _BMI_NULL_TOLERANCE)

    df["bmi_inconsistent"] = flagged.fillna(False).astype("int8")
    if gross.any():
        df.loc[gross, "Physical-BMI"] = pd.NA

    return {"flagged": int(flagged.sum()), "nulled": int(gross.sum())}


# 4. Merge age-gated activity questionnaires
def _merge_physical_activity_questionnaires(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the age-gated PAQ_C / PAQ_A pair into one score + flags."""
    has_child = _PAQ_CHILD_TOTAL in df.columns
    has_adolescent = _PAQ_ADOLESCENT_TOTAL in df.columns
    if not (has_child or has_adolescent):
        return df

    df = df.copy()

    child_total = df[_PAQ_CHILD_TOTAL] if has_child else pd.Series(index=df.index, dtype="float64")
    adolescent_total = (
        df[_PAQ_ADOLESCENT_TOTAL] if has_adolescent else pd.Series(index=df.index, dtype="float64")
    )

    # Each participant has at most one of the two, so combine_first is unambiguous.
    df["PAQ_total"] = child_total.combine_first(adolescent_total)
    df["PAQ_missing"] = df["PAQ_total"].isna().astype("int8")

    version = pd.Series("none", index=df.index, dtype="object")
    if has_adolescent:
        version = version.mask(df[_PAQ_ADOLESCENT_TOTAL].notna(), "adolescent")
    if has_child:
        version = version.mask(df[_PAQ_CHILD_TOTAL].notna(), "child")
    df["PAQ_version"] = version

    child_season = (
        df[_PAQ_CHILD_SEASON] if _PAQ_CHILD_SEASON in df.columns else pd.Series(index=df.index, dtype="object")
    )
    adolescent_season = (
        df[_PAQ_ADOLESCENT_SEASON]
        if _PAQ_ADOLESCENT_SEASON in df.columns
        else pd.Series(index=df.index, dtype="object")
    )
    df["PAQ_season"] = child_season.combine_first(adolescent_season)

    df = df.drop(
        columns=[
            c
            for c in [
                _PAQ_CHILD_TOTAL,
                _PAQ_ADOLESCENT_TOTAL,
                _PAQ_CHILD_SEASON,
                _PAQ_ADOLESCENT_SEASON,
            ]
            if c in df.columns
        ]
    )
    return df


# 5. Instrument-level missingness indicators
def _instrument_columns(df: pd.DataFrame, instrument: str) -> list[str]:
    """Return columns belonging to one assessment block (``<instrument>-...``)."""
    return [c for c in df.columns if c.startswith(f"{instrument}-")]


def _add_instrument_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 flag per block, set when the whole block is missing"""
    indicators = {}
    for instrument in INDICATOR_INSTRUMENTS:
        columns = _instrument_columns(df, instrument)
        if not columns:
            continue
        indicators[f"{instrument}_not_administered"] = (
            df[columns].isna().all(axis=1).astype("int8")
        )
    if not indicators:
        return df
    return pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1)

def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame: # 6. Leakage
    """Remove PCIAT-* columns, which encode the target."""
    leakage = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    return df.drop(columns=leakage)


# Feature typing (documentation contract, dictionary-driven)
def feature_type_map(data_dict: pd.DataFrame) -> dict[str, str]:
    """Declared semantic type per field, from the data dictionary.

    Returns one of: ``identifier``, ``leakage``, ``categorical``, ``continuous``. 
    The choice of ordinal-vs-one-hot encoding for ``categorical`` fields is left to Layer B, not decided here.
    """
    mapping: dict[str, str] = {}
    for _, row in data_dict.iterrows():
        field = row.get("Field")
        type_text = row.get("Type")
        if not isinstance(field, str):
            continue
        if field == ID_COLUMN:
            mapping[field] = "identifier"
        elif field.startswith(LEAKAGE_PREFIX):
            mapping[field] = "leakage"
        elif isinstance(type_text, str) and (
            "categorical" in type_text or type_text.strip() == "str"
        ):
            mapping[field] = "categorical"
        else:
            mapping[field] = "continuous"
    return mapping


# Entry point
def build_features(
    df: pd.DataFrame,
    data_dict: pd.DataFrame | None = None,
    return_report: bool = False,
):
    """Apply Layer A to a raw train or test frame"""
    if data_dict is None:
        data_dict = load_data_dictionary()

    df = df.copy()
    report: dict[str, dict] = {}

    report["codebook"] = _apply_codebook(df, build_codebook(data_dict))
    report["domain"] = _apply_domain_bounds(df)
    report["bmi"] = _add_bmi_consistency(df)

    df = _merge_physical_activity_questionnaires(df)
    df = _add_instrument_missing_indicators(df)
    df = _drop_leakage_columns(df)

    if return_report:
        return df, report
    return df
