import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from powerbot.data import clean, load_raw, load_sqlite, save_sqlite  # noqa: E402


def _raw(rows):
    return pd.DataFrame(rows, columns=["timestamp", "price", "load_mw"]).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"])
    )


def test_out_of_range_price_removed():
    df = _raw([
        ("2024-01-01 00:00", 50.0, 9000),
        ("2024-01-01 01:00", 1500.0, 9100),  # above the cap: invalid
        ("2024-01-01 02:00", 60.0, 9200),
    ])
    out, rep = clean(df)
    assert rep.out_of_range_prices == 1
    assert out.loc[1, "price"] == 55.0  # filled by interpolation


def test_missing_hour_detected_and_filled():
    df = _raw([("2024-01-01 00:00", 40.0, 9000), ("2024-01-01 02:00", 60.0, 9200)])
    out, rep = clean(df)
    assert rep.missing_hours == 1
    assert len(out) == 3
    assert out.loc[1, "price"] == 50.0


def test_long_gap_not_invented():
    df = _raw([("2024-01-01 00:00", 40.0, 9000), ("2024-01-01 10:00", 60.0, 9200)])
    out, _ = clean(df)
    assert out["price"].isna().sum() > 0  # gaps over 3 hours stay empty


def test_duplicate_hours_merged():
    df = _raw([("2024-11-03 01:00", 40.0, 9000), ("2024-11-03 01:00", 60.0, 9000)])
    out, rep = clean(df)
    assert rep.duplicates == 1
    assert len(out) == 1 and out.loc[0, "price"] == 50.0


def test_load_raw_finds_aeso_columns(tmp_path):
    csv = tmp_path / "aeso.csv"
    csv.write_text(
        "Date_Begin_GMT,Date_Begin_Local,ACTUAL_POOL_PRICE,ACTUAL_AIL,SOME_PLANT\n"
        "2024-01-01 07:00,2024-01-01 00:00,45.2,9876,12\n"
    )
    df = load_raw(csv)
    assert list(df.columns) == ["timestamp", "price", "load_mw"]
    assert df.loc[0, "price"] == 45.2


def test_sqlite_round_trip(tmp_path):
    df, _ = clean(_raw([("2024-01-01 00:00", 40.0, 9000), ("2024-01-01 01:00", 42.0, 9100)]))
    db = tmp_path / "m.db"
    save_sqlite(df, db)
    back = load_sqlite(db)
    assert len(back) == 2 and back.loc[1, "price"] == 42.0
