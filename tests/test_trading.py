"""Tests for the parts an interviewer will poke at: leakage and battery physics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from powerbot.backtest import pick_rule_hours  # noqa: E402
from powerbot.battery import Battery, optimize_day, rule_day, settle  # noqa: E402
from powerbot.features import add_lag_features, build_features  # noqa: E402


def _market(days=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    price = 50 + 40 * np.sin((idx.hour - 8) / 24 * 2 * np.pi) + rng.gamma(1.5, 10, len(idx))
    load = 9500 + 1000 * np.sin((idx.hour - 6) / 24 * 2 * np.pi)
    return pd.DataFrame({"price": price, "load_mw": load}, index=idx)


def test_lag_168_points_one_week_back():
    df = add_lag_features(_market(), "price")
    assert df["price_lag_168"].iloc[168] == df["price"].iloc[0]


def test_short_lags_are_refused():
    with pytest.raises(ValueError):
        add_lag_features(_market(), "price", lags=(1,))


def test_no_feature_changes_when_todays_prices_change():
    """The core no look ahead test: scrambling day D's prices must not change day D's features."""
    m = _market()
    f1 = build_features(m)
    m2 = m.copy()
    target_day = m2.index.normalize() == pd.Timestamp("2024-01-15")
    m2.loc[target_day, "price"] = 999.0
    m2.loc[target_day, "load_mw"] = 1.0
    f2 = build_features(m2)
    feature_cols = [c for c in f1.columns if c not in ("price", "load_mw", "is_spike")]
    day = f1.index.normalize() == pd.Timestamp("2024-01-15")
    pd.testing.assert_frame_equal(f1.loc[day, feature_cols], f2.loc[day, feature_cols])


def test_battery_respects_limits():
    bat = Battery()
    prices = np.r_[np.full(8, 10.0), np.full(8, 60.0), np.full(8, 200.0)]
    c, d, soc = optimize_day(prices, bat)
    tol = 1e-6
    assert (c >= -tol).all() and (c <= bat.power_mw + tol).all()
    assert (d >= -tol).all() and (d <= bat.power_mw + tol).all()
    assert (soc >= -tol).all() and (soc <= bat.capacity_mwh + tol).all()
    assert abs(soc[-1] - bat.start_soc_mwh) < 1e-6
    assert d.sum() / bat.eta <= bat.max_cycles_per_day * bat.capacity_mwh + tol


def test_battery_buys_low_sells_high():
    prices = np.r_[np.full(8, 10.0), np.full(8, 60.0), np.full(8, 200.0)]
    c, d, _ = optimize_day(prices)
    assert c[:8].sum() > 0 and d[16:].sum() > 0
    assert settle(prices, c, d) > 0


def test_battery_stays_idle_when_spread_too_small():
    """10% losses + wear cost: a $2 spread is not worth trading."""
    prices = np.r_[np.full(12, 50.0), np.full(12, 52.0)]
    c, d, _ = optimize_day(prices)
    assert d.sum() < 1e-6


def test_perfect_foresight_beats_any_plan():
    prices = _market(1)["price"].values
    best = settle(prices, *optimize_day(prices)[:2])
    c, d, _ = rule_day([2, 3, 4], [16, 17, 18])
    assert best >= settle(prices, c, d) - 1e-6


def test_rule_hours_charge_before_discharge():
    ch, dis = pick_rule_hours(_market()["price"])
    assert max(ch) < min(dis)
