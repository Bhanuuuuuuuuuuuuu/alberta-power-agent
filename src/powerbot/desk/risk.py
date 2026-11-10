"""Risk controls, the part of a desk that stops bad days becoming disasters.

Three layers:

1. Pre trade checks: is the data fresh, is the model current, are the
   forecasts sane, does the plan obey the battery's physics, is the expected
   profit worth the wear?
2. Spike overlay: never BUY in an hour the model thinks is likely to spike.
3. Loss limits and kill switch: after settlement, if daily, weekly or
   drawdown losses pass their limits, the kill switch trips and the desk stops
   trading until a human runs `desk.py resume`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powerbot.battery import Battery


@dataclass(frozen=True)
class RiskLimits:
    daily_loss_limit: float = -3000.0      # $: worst backtest day was about -2,700
    weekly_loss_limit: float = -5000.0     # $ over the last 7 settled days
    max_drawdown_limit: float = -7500.0    # $ below the best cumulative P&L so far
    min_expected_profit: float = 50.0      # $: skip days not worth the wear
    spike_block_prob: float = 0.5          # don't charge if spike odds are above this
    max_model_age_days: int = 45


class CheckResult(list):
    """List of failure messages. Empty means passed."""

    @property
    def ok(self) -> bool:
        return len(self) == 0


def check_inputs(history: pd.DataFrame, day: pd.Timestamp, model_meta: dict,
                 limits: RiskLimits) -> CheckResult:
    fails = CheckResult()
    need = day - pd.Timedelta(hours=1)
    last = history.index.max() if len(history) else None
    if last is None or last < need:
        fails.append(f"stale data: last hour {last}, need {need}")
    week = history[history.index >= day - pd.Timedelta(days=8)]
    expected = pd.date_range(day - pd.Timedelta(days=8), need, freq="h")
    gaps = expected.difference(week.dropna().index)
    if len(gaps):
        fails.append(f"{len(gaps)} missing hours in the last 8 days")
    age = (day - pd.Timestamp(model_meta["trained_through"])).days
    if age > limits.max_model_age_days:
        fails.append(f"model is {age} days old (limit {limits.max_model_age_days})")
    return fails


def check_forecast(fc: pd.DataFrame) -> CheckResult:
    fails = CheckResult()
    cols = ["p10", "p50", "p90", "mean", "spike_prob"]
    if len(fc) != 24:
        fails.append(f"forecast has {len(fc)} hours, expected 24")
    if fc[cols].isna().any().any():
        fails.append("forecast contains NaN")
    if ((fc[["p10", "p50", "p90", "mean"]] < 0) | (fc[["p10", "p50", "p90", "mean"]] > 999.99)).any().any():
        fails.append("forecast outside $0 to $999.99")
    if (fc["p10"] > fc["p90"] + 1e-6).any():
        fails.append("P10 above P90")
    return fails


def check_plan(charge: np.ndarray, discharge: np.ndarray, bat: Battery) -> CheckResult:
    """Independent physics check: recompute the battery state from scratch.

    The optimizer should never break these, but a desk never trusts a single
    component. If this fails, something upstream is broken.
    """
    fails = CheckResult()
    tol = 1e-6
    if (charge < -tol).any() or (charge > bat.power_mw + tol).any():
        fails.append("charge outside power limit")
    if (discharge < -tol).any() or (discharge > bat.power_mw + tol).any():
        fails.append("discharge outside power limit")
    soc = bat.start_soc_mwh + np.cumsum(bat.eta * charge - discharge / bat.eta)
    if (soc < -1e-4).any() or (soc > bat.capacity_mwh + 1e-4).any():
        fails.append("state of charge leaves 0 to capacity")
    if abs(soc[-1] - bat.start_soc_mwh) > 1e-4:
        fails.append("day does not end at the starting state of charge")
    if discharge.sum() / bat.eta > bat.max_cycles_per_day * bat.capacity_mwh + 1e-4:
        fails.append("more than the daily cycle limit")
    if ((charge > tol) & (discharge > tol)).any():
        fails.append("charging and discharging in the same hour")
    return fails


def check_loss_limits(settled_pnl: pd.Series, limits: RiskLimits) -> CheckResult:
    fails = CheckResult()
    if settled_pnl.empty:
        return fails
    last = settled_pnl.iloc[-1]
    if last < limits.daily_loss_limit:
        fails.append(f"daily loss ${last:,.0f} beyond limit ${limits.daily_loss_limit:,.0f}")
    week = settled_pnl.iloc[-7:].sum()
    if week < limits.weekly_loss_limit:
        fails.append(f"7 day loss ${week:,.0f} beyond limit ${limits.weekly_loss_limit:,.0f}")
    cum = settled_pnl.cumsum()
    dd = cum.iloc[-1] - max(cum.max(), 0.0)
    if dd < limits.max_drawdown_limit:
        fails.append(f"drawdown ${dd:,.0f} beyond limit ${limits.max_drawdown_limit:,.0f}")
    return fails
