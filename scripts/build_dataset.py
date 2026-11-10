"""Combine the raw AESO files, clean them, save to SQLite and print a summary.

    python scripts/build_dataset.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from powerbot.data import clean, load_raw, save_sqlite  # noqa: E402

RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "processed" / "market.db"


def main() -> None:
    files = sorted(RAW.glob("*.csv"))
    if not files:
        sys.exit("No CSV files in data/raw. Run scripts/download_data.py first.")

    frames = []
    for f in files:
        print(f"Reading {f.name}")
        frames.append(load_raw(f))
    raw = pd.concat(frames, ignore_index=True)

    df, report = clean(raw)
    print(report)

    save_sqlite(df, DB)
    print(f"\nSaved {len(df):,} hourly rows to {DB}")

    p = df["price"]
    print("\nPrice summary ($/MWh)")
    print(f"  from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  mean {p.mean():.2f}  median {p.median():.2f}  max {p.max():.2f}")
    print(f"  hours at the $999.99 cap: {(p >= 999.99).sum():,}")
    print(f"  hours at $0: {(p <= 0).sum():,}")
    print("\nNext: python scripts/explore.py")


if __name__ == "__main__":
    main()
