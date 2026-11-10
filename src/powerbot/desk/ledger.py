"""The desk's book of record: one SQLite file per desk (live or replay).

Tables
  market    every hour of price and load the desk knows about
  plans     the 24 hour schedule for each traded day, with its forecasts
  days      one row per day: planned, settled, skipped or halted, and P&L
  audit     append only log of every decision, check and alert
  state     key/value flags such as the kill switch
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS market (
    ts TEXT PRIMARY KEY, price REAL, load_mw REAL);
CREATE TABLE IF NOT EXISTS plans (
    ts TEXT PRIMARY KEY, day TEXT, hour INTEGER,
    charge_mw REAL, discharge_mw REAL, soc_mwh REAL,
    f_mean REAL, f_p10 REAL, f_p50 REAL, f_p90 REAL, spike_prob REAL,
    charge_blocked INTEGER, model_version TEXT);
CREATE TABLE IF NOT EXISTS days (
    day TEXT PRIMARY KEY, status TEXT, reason TEXT,
    expected_pnl REAL, realized_pnl REAL, perfect_pnl REAL, mwh_out REAL,
    model_version TEXT, planned_at TEXT, settled_at TEXT);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, day TEXT,
    level TEXT, event TEXT, message TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(SCHEMA)

    def close(self) -> None:
        self.con.close()

    # market data
    def upsert_market(self, df: pd.DataFrame) -> int:
        rows = [(f"{ts:%Y-%m-%d %H:%M:%S}", float(r.price), float(r.load_mw))
                for ts, r in df.dropna().iterrows()]
        self.con.executemany("INSERT OR REPLACE INTO market VALUES (?,?,?)", rows)
        self.con.commit()
        return len(rows)

    def market(self, before: pd.Timestamp | None = None, since: pd.Timestamp | None = None) -> pd.DataFrame:
        """Hours with since <= ts < before (either bound optional)."""
        clauses, args = [], []
        if since is not None:
            clauses.append("ts >= ?"); args.append(f"{since:%Y-%m-%d %H:%M:%S}")
        if before is not None:
            clauses.append("ts < ?"); args.append(f"{before:%Y-%m-%d %H:%M:%S}")
        q = "SELECT ts, price, load_mw FROM market" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        df = pd.read_sql(q + " ORDER BY ts", self.con, params=args)
        df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S")
        return df.set_index("ts").rename_axis("timestamp")

    def last_market_hour(self) -> pd.Timestamp | None:
        v = self.con.execute("SELECT MAX(ts) FROM market").fetchone()[0]
        return pd.Timestamp(v) if v else None

    # plans and days
    def save_plan(self, day: pd.Timestamp, plan: pd.DataFrame, expected_pnl: float,
                  model_version: str, status: str = "planned", reason: str = "") -> None:
        d = f"{day:%Y-%m-%d}"
        self.con.execute("DELETE FROM plans WHERE day = ?", (d,))
        rows = [(f"{ts:%Y-%m-%d %H:%M:%S}", d, ts.hour, r.charge_mw, r.discharge_mw, r.soc_mwh,
                 r.f_mean, r.f_p10, r.f_p50, r.f_p90, r.spike_prob, int(r.charge_blocked), model_version)
                for ts, r in plan.iterrows()]
        self.con.executemany("INSERT INTO plans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.con.execute(
            "INSERT OR REPLACE INTO days (day, status, reason, expected_pnl, model_version, planned_at) "
            "VALUES (?,?,?,?,?,?)", (d, status, reason, expected_pnl, model_version, _utc_now()))
        self.con.commit()

    def mark_day(self, day: pd.Timestamp, status: str, reason: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO days (day, status, reason, planned_at) VALUES (?,?,?,?)",
                         (f"{day:%Y-%m-%d}", status, reason, _utc_now()))
        self.con.commit()

    def plan(self, day: pd.Timestamp) -> pd.DataFrame:
        df = pd.read_sql("SELECT * FROM plans WHERE day = ? ORDER BY ts", self.con,
                         params=(f"{day:%Y-%m-%d}",), parse_dates=["ts"])
        return df.set_index("ts")

    def unsettled_days(self) -> list[pd.Timestamp]:
        rows = self.con.execute("SELECT day FROM days WHERE status = 'planned' ORDER BY day").fetchall()
        return [pd.Timestamp(r[0]) for r in rows]

    def settle_day(self, day: pd.Timestamp, realized: float, perfect: float, mwh_out: float) -> None:
        self.con.execute(
            "UPDATE days SET status='settled', realized_pnl=?, perfect_pnl=?, mwh_out=?, settled_at=? "
            "WHERE day=?", (realized, perfect, mwh_out, _utc_now(), f"{day:%Y-%m-%d}"))
        self.con.commit()

    def days(self) -> pd.DataFrame:
        df = pd.read_sql("SELECT * FROM days ORDER BY day", self.con, parse_dates=["day"])
        return df.set_index("day")

    def settled_pnl(self) -> pd.Series:
        d = self.days()
        if d.empty:
            return pd.Series(dtype=float)
        return d.loc[d["status"] == "settled", "realized_pnl"].astype(float)

    # audit log
    def log(self, event: str, message: str, day: pd.Timestamp | None = None,
            level: str = "INFO", **detail) -> None:
        self.con.execute(
            "INSERT INTO audit (ts_utc, day, level, event, message, detail) VALUES (?,?,?,?,?,?)",
            (_utc_now(), f"{day:%Y-%m-%d}" if day is not None else None, level, event, message,
             json.dumps(detail, default=str)))
        self.con.commit()

    def audit(self, limit: int = 200) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM audit ORDER BY id DESC LIMIT ?", self.con, params=(limit,))

    # state flags
    def get(self, key: str, default: str | None = None) -> str | None:
        r = self.con.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def set(self, key: str, value: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, value))
        self.con.commit()
