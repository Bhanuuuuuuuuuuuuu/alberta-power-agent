"""Stage 4: replay history day by day and measure real profit and risk.

For every test day:
  1. at midnight, each strategy picks a 24 hour plan using only what it knew
  2. the plan is settled at the ACTUAL prices of that day
  3. we record profit

Strategies compared:
  rule        fixed hours (cheapest 3 night hours, priciest 3 evening hours),
              chosen from the TRAINING period only
  lp_naive    linear program on "tomorrow = today" prices
  lp_xgb      linear program on the XGBoost P50 (median) forecast
  lp_xgb_mean linear program on the XGBoost expected price forecast
  perfect     linear program on the real prices (impossible in real life:
              this is the upper bound, the most money anyone could make)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from powerbot.battery import Battery, optimize_day, rule_day, settle


def pick_rule_hours(train_prices: pd.Series, n: int = 3) -> tuple[list[int], list[int]]:
    """Cheapest n consecutive hours, then the priciest n consecutive hours after them."""
    profile = train_prices.groupby(train_prices.index.hour).mean().reindex(range(24)).values
    best = (None, None, -np.inf)
    for cs in range(0, 24 - 2 * n + 1):
        for ds in range(cs + n, 24 - n + 1):
            gain = profile[ds:ds + n].sum() - profile[cs:cs + n].sum()
            if gain > best[2]:
                best = (list(range(cs, cs + n)), list(range(ds, ds + n)), gain)
    return best[0], best[1]


def run_backtest(pred: pd.DataFrame, train_prices: pd.Series, bat: Battery = Battery()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (daily P&L per strategy, hourly decisions of the lp_xgb_mean strategy)."""
    ch_hours, dis_hours = pick_rule_hours(train_prices)
    rule_c, rule_d, _ = rule_day(ch_hours, dis_hours, bat)

    daily_rows, hourly = [], []
    for day, d in pred.groupby(pred.index.normalize()):
        if len(d) != 24:
            continue
        actual = d["price"].values
        row = {"date": day}
        plans = {
            "rule": (rule_c, rule_d),
            "lp_naive": optimize_day(d["naive"].values, bat)[:2],
            "lp_xgb": optimize_day(d["p50"].values, bat)[:2],
            "lp_xgb_mean": optimize_day(d["mean"].values, bat)[:2],
            "perfect": optimize_day(actual, bat)[:2],
        }
        for name, (c, dis) in plans.items():
            row[name] = settle(actual, c, dis, bat)
            row[f"{name}_mwh_out"] = dis.sum()
        daily_rows.append(row)
        c, dis = plans["lp_xgb_mean"]
        hourly.append(pd.DataFrame({"charge_mw": c, "discharge_mw": dis}, index=d.index))

    daily = pd.DataFrame(daily_rows).set_index("date")
    return daily, pd.concat(hourly)


STRATEGIES = ["rule", "lp_naive", "lp_xgb", "lp_xgb_mean", "perfect"]


def summarize(daily: pd.DataFrame, bat: Battery = Battery()) -> pd.DataFrame:
    """Profit and risk numbers a trading desk would look at."""
    years = len(daily) / 365.25
    rows = {}
    for s in STRATEGIES:
        pnl = daily[s]
        cum = pnl.cumsum()
        drawdown = (cum - cum.cummax()).min()
        worst5 = pnl[pnl <= pnl.quantile(0.05)].mean()
        rows[s] = {
            "total_profit_$": pnl.sum(),
            "profit_per_kw_year_$": pnl.sum() / (bat.power_mw * 1000) / years,
            "capture_vs_perfect_%": 100 * pnl.sum() / daily["perfect"].sum(),
            "sharpe_annual": pnl.mean() / pnl.std() * np.sqrt(365) if pnl.std() > 0 else np.nan,
            "max_drawdown_$": drawdown,
            "worst_5pct_day_avg_$": worst5,
            "losing_days_%": 100 * (pnl < 0).mean(),
            "cycles_per_day": daily[f"{s}_mwh_out"].mean() / (bat.capacity_mwh * bat.eta),
        }
    return pd.DataFrame(rows).T
