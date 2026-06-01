"""Feature engineering for the sales forecasting pipeline.

All time-dependent features (lags, rolling stats) are computed *within* each
(store, category) series and shifted so that a row never sees its own or any
future target value. This prevents target leakage into the training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_KEYS = ["store", "category"]
TARGET = "units_sold"

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"].dt
    df["year"] = d.year
    df["month"] = d.month
    df["day"] = d.day
    df["dayofweek"] = d.dayofweek
    df["weekofyear"] = d.isocalendar().week.astype(int)
    df["dayofyear"] = d.dayofyear
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["is_month_start"] = d.is_month_start.astype(int)
    df["is_month_end"] = d.is_month_end.astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(GROUP_KEYS + ["date"]).copy()
    g = df.groupby(GROUP_KEYS, observed=True)[TARGET]

    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    # Rolling stats use shift(1) so the current day is excluded (no leakage),
    # and transform keeps each statistic confined to its own series.
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
        df[f"roll_std_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).std())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature build. Drops the warm-up rows that lack lag history."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = df.dropna().reset_index(drop=True)

    # Treat store/category as native categoricals for the gradient booster.
    for col in GROUP_KEYS:
        df[col] = df[col].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"date", TARGET}
    return [c for c in df.columns if c not in drop]
