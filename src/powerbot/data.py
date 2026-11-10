"""Stage 1: load, clean and store AESO market data.

The raw AESO files have a timestamp column, the pool price, Alberta Internal
Load (AIL), and hundreds of per generator columns. We only keep what the
trading agent needs, and we validate it the same way you will validate data
at NRCan: check types, ranges, duplicates and gaps.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# AESO price limits: the pool price is capped at $999.99/MWh and floored at $0.
PRICE_MIN = 0.0
PRICE_MAX = 999.99


def _find_column(columns: list[str], *keywords: str) -> str:
    """Find the first column whose lowercase name contains every keyword."""
    for col in columns:
        name = col.lower()
        if all(k in name for k in keywords):
            return col
    raise KeyError(f"No column containing {keywords}. Columns start with: {columns[:15]}")


def load_raw(path: Path) -> pd.DataFrame:
    """Read one AESO CSV and keep only timestamp, price and load."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    time_col = _find_column(header, "begin", "local") if any(
        "begin" in c.lower() and "local" in c.lower() for c in header
    ) else _find_column(header, "date")
    price_col = _find_column(header, "pool", "price")
    load_col = _find_column(header, "ail")

    df = pd.read_csv(path, usecols=[time_col, price_col, load_col])
    df = df.rename(columns={time_col: "timestamp", price_col: "price", load_col: "load_mw"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    return df


@dataclass
class QualityReport:
    rows_in: int = 0
    bad_timestamps: int = 0
    duplicates: int = 0
    out_of_range_prices: int = 0
    missing_hours: int = 0
    filled_values: int = 0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Data quality report",
            f"  rows read            {self.rows_in:>8}",
            f"  bad timestamps       {self.bad_timestamps:>8}",
            f"  duplicate hours      {self.duplicates:>8}",
            f"  prices out of range  {self.out_of_range_prices:>8}",
            f"  missing hours        {self.missing_hours:>8}",
            f"  values filled        {self.filled_values:>8}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Validate and clean an hourly price and load table."""
    rep = QualityReport(rows_in=len(df))

    rep.bad_timestamps = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    # Daylight saving time creates repeated local hours in the fall.
    rep.duplicates = int(df["timestamp"].duplicated().sum())
    df = df.groupby("timestamp", as_index=False).mean(numeric_only=True)

    bad = (df["price"] < PRICE_MIN) | (df["price"] > PRICE_MAX)
    rep.out_of_range_prices = int(bad.sum())
    df.loc[bad, "price"] = pd.NA

    df = df.set_index("timestamp").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    rep.missing_hours = len(full) - len(df)
    df = df.reindex(full)
    df.index.name = "timestamp"

    before = int(df[["price", "load_mw"]].isna().sum().sum())
    # Short gaps (up to 3 hours) are interpolated; longer gaps stay empty
    # so we never invent a long stretch of fake prices.
    df[["price", "load_mw"]] = df[["price", "load_mw"]].astype(float).interpolate(limit=3)
    after = int(df[["price", "load_mw"]].isna().sum().sum())
    rep.filled_values = before - after
    if after:
        rep.notes.append(f"{after} values left empty (gaps longer than 3 hours)")

    return df.reset_index(), rep


def save_sqlite(df: pd.DataFrame, db_path: Path, table: str = "market_hourly") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ts ON {table}(timestamp)")


def load_sqlite(db_path: Path, table: str = "market_hourly") -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(f"SELECT * FROM {table} ORDER BY timestamp", con, parse_dates=["timestamp"])
    return df
