"""Tests for the trading desk: feeds, risk controls, kill switch, full cycle."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from powerbot.battery import Battery, optimize_day  # noqa: E402
from powerbot.desk.alerts import Alerter  # noqa: E402
from powerbot.desk.engine import Desk  # noqa: E402
from powerbot.desk.feeds import FeedError, ReplayFeed, check_quality, parse_load, parse_pool_price  # noqa: E402
from powerbot.desk.ledger import Ledger  # noqa: E402
from powerbot.desk.risk import RiskLimits, check_forecast, check_loss_limits, check_plan  # noqa: E402
from powerbot.features import features_for_day  # noqa: E402


def _market(days=60, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    price = 50 + 60 * np.sin((idx.hour - 10) / 24 * 2 * np.pi) ** 2 + rng.gamma(1.5, 15, len(idx))
    price = np.where(rng.random(len(idx)) < 0.02, 900, price)
    load = 9500 + 1000 * np.sin((idx.hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 100, len(idx))
    df = pd.DataFrame({"price": np.clip(price, 0, 999.99), "load_mw": load}, index=idx)
    df.index.name = "timestamp"
    return df


# feeds
def test_replay_feed_never_shows_the_future():
    m = _market()
    feed = ReplayFeed(m)
    feed.now = pd.Timestamp("2024-01-10")
    got = feed.fetch(pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-20"))
    assert got.index.max() == pd.Timestamp("2024-01-09 23:00")


def test_parse_aeso_pool_price_payload():
    payload = {"return": {"Pool Price Report": [
        {"begin_datetime_utc": "2025-08-01 06:00", "begin_datetime_mpt": "2025-08-01 00:00",
         "pool_price": "45.10", "forecast_pool_price": "44.00"},
        {"begin_datetime_utc": "2025-08-01 07:00", "begin_datetime_mpt": "2025-08-01 01:00",
         "pool_price": "", "forecast_pool_price": "40.00"},  # not settled yet
    ]}}
    s = parse_pool_price(payload)
    assert list(s.index) == [pd.Timestamp("2025-08-01 00:00")]
    assert s.iloc[0] == pytest.approx(45.10)


def test_parse_aeso_load_payload_utc_fallback():
    payload = {"return": {"Actual Forecast Report": [
        {"begin_datetime_utc": "2025-08-01 06:00", "alberta_internal_load": "9876",
         "forecast_alberta_internal_load": "9900"}]}}
    s = parse_load(payload)
    assert s.index[0] == pd.Timestamp("2025-08-01 00:00")  # UTC-6 in summer
    assert s.iloc[0] == 9876


def test_bad_payload_raises_feed_error():
    with pytest.raises(FeedError):
        parse_pool_price({"message": "Access denied due to invalid subscription key"})


def test_quality_check_finds_gaps_and_bad_prices():
    m = _market(3).copy()
    m = m.drop(m.index[10:15])
    m.iloc[0, 0] = 1500
    problems = check_quality(m, m.index.min(), pd.Timestamp("2024-01-04"))
    assert any("missing" in p for p in problems)
    assert any("outside" in p for p in problems)


# features for a single live day
def test_features_for_day_uses_only_the_past():
    m = _market()
    day = pd.Timestamp("2024-02-01")
    f1 = features_for_day(m, day)
    m2 = m.copy()
    m2.loc[m2.index >= day, "price"] = 999.0  # change the future
    f2 = features_for_day(m2, day)
    pd.testing.assert_frame_equal(f1.drop(columns=["price", "load_mw", "is_spike"]),
                                  f2.drop(columns=["price", "load_mw", "is_spike"]))


def test_features_for_day_refuses_stale_history():
    m = _market()
    with pytest.raises(ValueError):
        features_for_day(m[m.index < "2024-01-31 20:00"], pd.Timestamp("2024-02-01"))


# risk
def test_physics_check_catches_broken_plan():
    bat = Battery()
    c = np.zeros(24); d = np.zeros(24)
    c[2] = 7.0  # above the 5 MW limit
    assert not check_plan(c, d, bat).ok
    c, d, _ = optimize_day(np.r_[np.full(12, 20.0), np.full(12, 200.0)], bat)
    assert check_plan(c, d, bat).ok


def test_forecast_check_catches_nonsense():
    fc = pd.DataFrame({"p10": [50.0] * 24, "p50": [60.0] * 24, "p90": [70.0] * 24,
                       "mean": [65.0] * 24, "spike_prob": [0.1] * 24})
    assert check_forecast(fc).ok
    fc.loc[3, "mean"] = np.nan
    assert not check_forecast(fc).ok


def test_loss_limits():
    limits = RiskLimits()
    assert check_loss_limits(pd.Series([500.0, 800.0, -200.0]), limits).ok
    assert not check_loss_limits(pd.Series([500.0, -3500.0]), limits).ok          # daily
    assert not check_loss_limits(pd.Series([-1500.0] * 4), limits).ok             # weekly
    assert not check_loss_limits(pd.Series([9000.0] + [-1400.0] * 6), limits).ok  # drawdown


def test_spike_overlay_blocks_buying():
    prices = np.r_[np.full(12, 20.0), np.full(12, 200.0)]
    cap = np.full(24, 5.0)
    cap[:12] = 0.0  # pretend all cheap hours are spike risky
    c, d, _ = optimize_day(prices, charge_cap=cap)
    assert c[:12].sum() < 1e-6


# full cycle
@pytest.fixture
def desk(tmp_path, monkeypatch):
    m = _market()
    start = pd.Timestamp("2024-02-10")
    feed = ReplayFeed(m)
    ledger = Ledger(tmp_path / "desk.db")
    ledger.upsert_market(m[m.index < start])
    d = Desk(ledger=ledger, feed=feed, models_dir=tmp_path / "models",
             alerter=Alerter(tmp_path / "alerts.log", enabled=False))
    # make training fast for tests
    import powerbot.forecast as fcmod
    for params in (fcmod.QUANTILE_PARAMS, fcmod.MEAN_PARAMS, fcmod.SPIKE_PARAMS):
        monkeypatch.setitem(params, "n_estimators", 20)
    yield d, feed, start
    ledger.close()


def test_full_cycle_plans_settles_and_logs(desk):
    d, feed, start = desk
    for day in pd.date_range(start, periods=3, freq="D"):
        feed.now = day
        assert d.run_cycle(day) in ("planned", "skipped")
    days = d.ledger.days()
    assert (days["status"] == "settled").sum() == 2      # first two days paid out
    assert len(d.ledger.plan(start)) == 24
    events = set(d.ledger.audit()["event"])
    assert {"cycle_start", "ingest", "model", "decision", "settle"} <= events


def test_kill_switch_stops_trading_until_resumed(desk):
    d, feed, start = desk
    feed.now = start
    d.run_cycle(start)
    d.trip_kill_switch("test", start)
    nxt = start + pd.Timedelta(days=1)
    feed.now = nxt
    assert d.run_cycle(nxt) == "halted"
    d.resume("checked")
    nxt2 = nxt + pd.Timedelta(days=1)
    feed.now = nxt2
    assert d.run_cycle(nxt2) == "planned"


def test_stale_data_skips_the_day(desk):
    d, feed, start = desk
    cutoff = start - pd.Timedelta(days=2)
    d.ledger.con.execute("DELETE FROM market WHERE ts >= ?", (f"{cutoff:%Y-%m-%d %H:%M:%S}",))
    d.ledger.con.commit()
    feed.now = cutoff  # the feed is stuck 2 days behind
    assert d.run_cycle(start) == "skipped"
    assert "stale" in d.ledger.days().loc[start, "reason"]
