"""The desk's daily cycle. Runs once just after midnight for trading day D.

    1. INGEST    pull new hours from the feed, clean them, store them
    2. SETTLE    pay out every finished day's plan at actual prices
    3. RISK      check loss limits; trip the kill switch if broken
    4. MODEL     retrain if a new month started, else load the current model
    5. CHECKS    fresh data? model current? (skip the day if not)
    6. FORECAST  24 hourly prices with ranges and spike odds
    7. OPTIMIZE  LP schedule, with the spike overlay blocking risky buys
    8. VERIFY    independent physics check and expected profit threshold
    9. RECORD    save the plan, write the reasons to the audit log, alert

Every step writes to the audit log, so any trade can be explained later:
"on this day the desk did X because the model said Y".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from powerbot.battery import Battery, optimize_day, settle
from powerbot.data import clean
from powerbot.desk.alerts import Alerter
from powerbot.desk.feeds import FeedError, check_quality
from powerbot.desk.ledger import Ledger
from powerbot.desk.model_store import load_models, needs_retrain, train_and_save
from powerbot.desk.risk import (RiskLimits, check_forecast, check_inputs, check_loss_limits,
                                check_plan)
from powerbot.features import features_for_day
from powerbot.forecast import predict_models


@dataclass
class Desk:
    ledger: Ledger
    feed: object
    models_dir: Path
    alerter: Alerter
    battery: Battery = field(default_factory=Battery)
    limits: RiskLimits = field(default_factory=RiskLimits)
    ingest_chunk_days: int = 31

    # 1. INGEST
    def ingest(self, until: pd.Timestamp) -> None:
        last = self.ledger.last_market_hour()
        start = (last + pd.Timedelta(hours=1)) if last is not None else until - pd.Timedelta(days=30)
        if start >= until:
            return
        total = 0
        chunk_start = start
        while chunk_start < until:
            chunk_end = min(chunk_start + pd.Timedelta(days=self.ingest_chunk_days), until)
            raw = self.feed.fetch(chunk_start, chunk_end)
            if len(raw):
                cleaned, _ = clean(raw.rename_axis("timestamp").reset_index())
                cleaned = cleaned.set_index("timestamp")
                total += self.ledger.upsert_market(cleaned)
            chunk_start = chunk_end
        problems = check_quality(self.ledger.market(since=start, before=until), start, until)
        self.ledger.log("ingest", f"stored {total} new hours from {start} to {until}",
                        level="WARN" if problems else "INFO", problems=problems)
        if problems:
            self.alerter.send("WARN", "Data quality: " + "; ".join(problems))

    # 2. SETTLE
    def settle_finished_days(self, today: pd.Timestamp) -> list[pd.Timestamp]:
        settled = []
        for day in self.ledger.unsettled_days():
            if day >= today:
                continue
            hours = pd.date_range(day, periods=24, freq="h")
            actual = self.ledger.market(since=day, before=day + pd.Timedelta(days=1)).reindex(hours)["price"]
            if actual.isna().any():
                self.ledger.log("settle", "waiting for missing actual prices", day=day, level="WARN")
                continue
            plan = self.ledger.plan(day)
            c, d = plan["charge_mw"].values, plan["discharge_mw"].values
            realized = settle(actual.values, c, d, self.battery)
            perfect = settle(actual.values, *optimize_day(actual.values, self.battery)[:2], self.battery)
            self.ledger.settle_day(day, realized, perfect, float(d.sum()))
            self.ledger.log("settle", f"realized ${realized:,.0f} (perfect foresight ${perfect:,.0f})",
                            day=day, realized=realized, perfect=perfect)
            settled.append(day)
        return settled

    # 3. RISK
    def kill_switch_on(self) -> bool:
        return self.ledger.get("kill_switch", "off") == "on"

    def trip_kill_switch(self, reason: str, day: pd.Timestamp) -> None:
        self.ledger.set("kill_switch", "on")
        self.ledger.set("kill_reason", reason)
        self.ledger.log("kill_switch", f"TRIPPED: {reason}", day=day, level="HALT")
        self.alerter.send("HALT", f"Kill switch tripped: {reason}. Trading stopped. "
                                  f"Review, then run: python scripts/desk.py resume")

    def resume(self, note: str = "") -> None:
        self.ledger.set("kill_switch", "off")
        self.ledger.log("kill_switch", f"resumed by operator. {note}".strip(), level="WARN")
        self.alerter.send("WARN", f"Kill switch reset by operator. {note}".strip())

    # the full cycle
    def run_cycle(self, day: pd.Timestamp) -> str:
        day = pd.Timestamp(day).normalize()
        self.ledger.log("cycle_start", f"cycle for trading day {day:%Y-%m-%d}", day=day)

        try:
            self.ingest(until=day)
        except FeedError as e:
            self.ledger.log("ingest", str(e), day=day, level="WARN")
            self.alerter.send("WARN", f"Feed error: {e}")

        newly = self.settle_finished_days(day)
        if newly:
            pnl = self.ledger.settled_pnl()
            breaches = check_loss_limits(pnl, self.limits)
            if breaches and not self.kill_switch_on():
                self.trip_kill_switch("; ".join(breaches), day)
            last = self.ledger.days().loc[newly[-1]]
            self.alerter.send("INFO", f"Settled {newly[-1]:%Y-%m-%d}: ${last['realized_pnl']:,.0f} "
                                      f"(perfect ${last['perfect_pnl']:,.0f}). "
                                      f"Total to date ${pnl.sum():,.0f}")

        if self.kill_switch_on():
            reason = f"kill switch on: {self.ledger.get('kill_reason')}"
            self.ledger.mark_day(day, "halted", reason)
            self.ledger.log("decision", "no trading today", day=day, level="HALT", reason=reason)
            return "halted"

        # Features only need the last few weeks (longest lookback: 7 day lag + 7 day average).
        history = self.ledger.market(since=day - pd.Timedelta(days=30), before=day)

        # 4. MODEL
        meta = None
        try:
            models, meta = load_models(self.models_dir)
        except FileNotFoundError:
            pass
        if needs_retrain(meta, day):
            version = train_and_save(self.ledger.market(before=day), self.models_dir, as_of=day)
            models, meta = load_models(self.models_dir, version)
            self.ledger.log("model", f"retrained {version} on {meta['train_rows']:,} hours", day=day,
                            trained_through=meta["trained_through"])

        # 5. CHECKS
        fails = check_inputs(history, day, meta, self.limits)
        if not fails.ok:
            return self._skip(day, "input checks failed: " + "; ".join(fails))

        # 6. FORECAST
        try:
            X = features_for_day(history, day)
        except ValueError as e:
            return self._skip(day, f"could not build features: {e}")
        fc = predict_models(models, X)
        fails = check_forecast(fc)
        if not fails.ok:
            return self._skip(day, "forecast checks failed: " + "; ".join(fails))

        # 7. OPTIMIZE with the spike overlay
        blocked = (fc["spike_prob"] > self.limits.spike_block_prob).values
        cap = np.where(blocked, 0.0, self.battery.power_mw)
        c, d, soc = optimize_day(fc["mean"].values, self.battery, charge_cap=cap)
        c, d = np.where(c < 1e-6, 0.0, c), np.where(d < 1e-6, 0.0, d)

        # 8. VERIFY
        fails = check_plan(c, d, self.battery)
        if not fails.ok:
            return self._skip(day, "PLAN FAILED PHYSICS CHECK: " + "; ".join(fails), level="HALT")
        expected = settle(fc["mean"].values, c, d, self.battery)
        reason = explain(fc, c, d, blocked, expected)
        if expected < self.limits.min_expected_profit:
            c, d = np.zeros(24), np.zeros(24)
            soc = np.full(24, self.battery.start_soc_mwh)
            reason = (f"expected profit ${expected:,.0f} below ${self.limits.min_expected_profit:,.0f} "
                      f"threshold: staying idle")
            expected = 0.0
        else:
            soc = self.battery.start_soc_mwh + np.cumsum(self.battery.eta * c - d / self.battery.eta)

        # 9. RECORD
        plan = pd.DataFrame({
            "charge_mw": c, "discharge_mw": d, "soc_mwh": soc,
            "f_mean": fc["mean"], "f_p10": fc["p10"], "f_p50": fc["p50"], "f_p90": fc["p90"],
            "spike_prob": fc["spike_prob"], "charge_blocked": blocked,
        }, index=fc.index)
        self.ledger.save_plan(day, plan, expected, meta["version"], reason=reason)
        self.ledger.log("decision", reason, day=day, expected_pnl=expected, model=meta["version"],
                        blocked_hours=[int(h) for h in np.where(blocked)[0]])
        self.alerter.send("INFO", f"Plan for {day:%Y-%m-%d}: {reason}")
        return "planned"

    def _skip(self, day: pd.Timestamp, reason: str, level: str = "WARN") -> str:
        self.ledger.mark_day(day, "skipped", reason)
        self.ledger.log("decision", f"skipped: {reason}", day=day, level=level)
        self.alerter.send(level, f"{day:%Y-%m-%d} skipped: {reason}")
        return "skipped"


def explain(fc: pd.DataFrame, c: np.ndarray, d: np.ndarray, blocked: np.ndarray, expected: float) -> str:
    """Human readable reason for the day's plan, stored in the audit log."""
    if d.sum() < 1e-6:
        msg = "no profitable spread after losses and wear: staying idle"
    else:
        ch, dh = np.where(c > 1e-6)[0], np.where(d > 1e-6)[0]
        buy = np.average(fc["mean"].values[ch], weights=c[ch])
        sell = np.average(fc["mean"].values[dh], weights=d[dh])
        msg = (f"buy {c.sum():.1f} MWh in hours {list(map(int, ch))} at forecast ${buy:.0f}, "
               f"sell {d.sum():.1f} MWh in hours {list(map(int, dh))} at forecast ${sell:.0f}; "
               f"expected profit ${expected:,.0f}")
    if blocked.any():
        msg += f"; spike risk blocked buying in hours {list(map(int, np.where(blocked)[0]))}"
    return msg
