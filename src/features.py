"""Layer A: deterministic, row-local feature engineering.

Every transform operates on a single row, never reads statistics from other rows.

Design decisions:
1. Leakage columns (``PCIAT-*``) are dropped: ``sii`` is derived from
   ``PCIAT-PCIAT_Total`` and these fields would reveal the target.
2. 2 age-gated activity questionnaires are merged. ``PAQ_C`` is given
   to children and ``PAQ_A`` to adolescents.
3. Instrument-level missingness indicators are added. Missingness in this
   dataset is block-structured: a whole assessment is present or absent
   together. A binary "assessment not administered" flag preserves that
   information for models that would otherwise only see NaN.
"""

from __future__ import annotations

import pandas as pd

LEAKAGE_PREFIX = "PCIAT-"

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

# PAQ columns consumed by the merge below.
_PAQ_CHILD_TOTAL = "PAQ_C-PAQ_C_Total"
_PAQ_ADOLESCENT_TOTAL = "PAQ_A-PAQ_A_Total"
_PAQ_CHILD_SEASON = "PAQ_C-Season"
_PAQ_ADOLESCENT_SEASON = "PAQ_A-Season"


def _instrument_columns(df: pd.DataFrame, instrument: str) -> list[str]:
    """Return columns belonging to one assessment block (``<instrument>-...``)."""
    return [c for c in df.columns if c.startswith(f"{instrument}-")]


def _add_instrument_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 flag per block, set when the whole block is missing.

    The flag marks that the assessment was not administered at all. Occasional
    gaps inside an otherwise-present block are left as NaN and handled later by
    the imputer, so this does not duplicate per-column missingness.
    """
    indicators = {}
    for instrument in INDICATOR_INSTRUMENTS:
        columns = _instrument_columns(df, instrument)
        if not columns:
            continue
        # True where every column of the block is NaN for that participant.
        indicators[f"{instrument}_not_administered"] = (
            df[columns].isna().all(axis=1).astype("int8")
        )
    if not indicators:
        return df
    return pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1)


def _merge_physical_activity_questionnaires(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse age-gated PAQ_C / PAQ_A pair into one score + flags."""
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

    # PAQ_missing captures the structural gap separately from the merged score
    df["PAQ_missing"] = df["PAQ_total"].isna().astype("int8")

    # Which form was administered — a proxy for the child/adolescent split.
    version = pd.Series("none", index=df.index, dtype="object")
    if has_adolescent:
        version = version.mask(df[_PAQ_ADOLESCENT_TOTAL].notna(), "adolescent")
    if has_child:
        version = version.mask(df[_PAQ_CHILD_TOTAL].notna(), "child")
    df["PAQ_version"] = version

    # Merge season columns the same way so no PAQ-specific NaN remains.
    if has_child or has_adolescent:
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


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove PCIAT-* columns, which directly encode the target."""
    leakage = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    return df.drop(columns=leakage)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Layer A to train or test frame."""
    # Indicators are computed on the raw blocks before any column is dropped.
    df = _add_instrument_missing_indicators(df)
    df = _merge_physical_activity_questionnaires(df)
    df = _drop_leakage_columns(df)
    return df
