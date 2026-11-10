"""Stage 2, step 1: turn each hour into numbers a model can learn from.

The trading decision for day D is made at midnight at the start of day D, for
all 24 hours of that day. So every feature for any hour of day D must already
be known at that moment. In practice that means: calendar facts (always known)
and anything from day D minus 1 or earlier.

A 24 hour lag is safe for every hour of the day: for 11 PM on day D it points
at 11 PM on day D minus 1, which is before midnight. A 1 hour lag is NOT safe,
because at midnight we do not know the 4 PM price yet when predicting 5 PM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from powerbot.data import load_sqlite

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "market.db"

SPIKE_THRESHOLD = 300.0  # $/MWh: hours above this are "spikes"


def load_market(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Clean hourly table from stage 1, indexed by timestamp."""
    return load_sqlite(db_path).set_index("timestamp").sort_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar columns. Always known in advance, so always safe."""
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["dayofyear"] = idx.dayofyear
    # Cyclical encoding: tells the model 11 PM is next to midnight.
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags=(24, 48, 168)) -> pd.DataFrame:
    """Value of `col` from `lag` hours earlier. Only lags >= 24 are allowed."""
    df = df.copy()
    for lag in lags:
        if lag < 24:
            raise ValueError(f"lag {lag} would leak future data; use 24 or more")
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_yesterday_stats(df: pd.DataFrame, col: str = "price") -> pd.DataFrame:
    """Summary of the whole previous day (and week), the same for every hour of day D.

    We compute stats per calendar day, then shift by one day, so day D only
    ever sees day D minus 1 and earlier.
    """
    df = df.copy()
    day = df.index.normalize()
    daily = df[col].groupby(day).agg(["mean", "max", "min", "std"])
    daily["spike_hours"] = (df[col] > SPIKE_THRESHOLD).groupby(day).sum()
    daily["mean_7d"] = daily["mean"].rolling(7).mean()
    daily["max_7d"] = daily["max"].rolling(7).max()
    daily = daily.shift(1)  # the key line: yesterday's numbers only
    daily.columns = [f"{col}_yday_{c}" for c in daily.columns]
    looked_up = daily.reindex(day)
    looked_up.index = df.index
    return pd.concat([df, looked_up], axis=1)


def build_features(df: pd.DataFrame | None = None, dropna: bool = True) -> pd.DataFrame:
    """Full feature table, ready for the model."""
    if df is None:
        df = load_market()
    df = add_time_features(df)
    df = add_lag_features(df, "price")
    df = add_lag_features(df, "load_mw")
    df = add_yesterday_stats(df, "price")
    df = add_yesterday_stats(df, "load_mw")
    # Load is shifted too: we do not know tomorrow's actual load at midnight.
    df["is_spike"] = (df["price"] > SPIKE_THRESHOLD).astype(int)
    return df.dropna() if dropna else df


FEATURES = [
    "hour", "dayofweek", "month", "is_weekend", "dayofyear", "hour_sin", "hour_cos",
    "price_lag_24", "price_lag_48", "price_lag_168",
    "load_mw_lag_24", "load_mw_lag_48", "load_mw_lag_168",
    "price_yday_mean", "price_yday_max", "price_yday_min", "price_yday_std",
    "price_yday_spike_hours", "price_yday_mean_7d", "price_yday_max_7d",
    "load_mw_yday_mean", "load_mw_yday_max", "load_mw_yday_min", "load_mw_yday_std",
    "load_mw_yday_mean_7d",
]


def features_for_day(history: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Features for the 24 hours of `day`, using ONLY history before that day.

    This is what the live desk calls at midnight. We add 24 empty rows for the
    target day, run the normal feature code, and return those 24 rows.
    """
    day = pd.Timestamp(day).normalize()
    past = history[history.index < day]
    if past.empty or past.index.max() < day - pd.Timedelta(hours=1):
        raise ValueError(f"history must run up to {day - pd.Timedelta(hours=1)}")
    hours = pd.date_range(day, periods=24, freq="h")
    blank = pd.DataFrame({"price": np.nan, "load_mw": np.nan}, index=hours)
    feats = build_features(pd.concat([past[["price", "load_mw"]], blank]), dropna=False).loc[hours]
    missing = feats[FEATURES].isna().sum()
    if missing.any():
        raise ValueError(f"missing features for {day:%Y-%m-%d}: {missing[missing > 0].to_dict()}")
    return feats


if __name__ == "__main__":
    f = build_features()
    print(f[FEATURES].head())
    print(f.shape)
