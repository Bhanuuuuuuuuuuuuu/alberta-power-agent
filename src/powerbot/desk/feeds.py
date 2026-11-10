"""Market data feeds.

Two feeds with the same interface, so the desk code is identical in both modes:

* AESOFeed   live data from the AESO API gateway (needs a free API key)
* ReplayFeed historical data from market.db, released one day at a time,
             to rehearse the desk on the past exactly as if it were live

fetch(start, end) returns an hourly DataFrame indexed by local Alberta time
with columns price and load_mw, for hours start <= t < end.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

BASE_URL = "https://apimgw.aeso.ca/public"
POOL_PRICE_PATH = "poolprice-api/v1.1/price/poolPrice"
LOAD_PATH = "actualforecast-api/v1/load/albertaInternalLoad"


class FeedError(RuntimeError):
    pass


class ReplayFeed:
    """Serves history but refuses to reveal anything after `now`."""

    def __init__(self, market: pd.DataFrame):
        self.market = market
        self.now: pd.Timestamp | None = None

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self.now is not None and end > self.now:
            end = self.now  # cannot see the future, even by accident
        m = self.market
        return m[(m.index >= start) & (m.index < end)][["price", "load_mw"]].copy()


class AESOFeed:
    """Live AESO API client. Key is read from the AESO_API_KEY environment variable."""

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("AESO_API_KEY")
        if not self.api_key:
            raise FeedError("AESO_API_KEY is not set. See README: 'Getting an AESO API key'.")
        self.timeout = timeout

    def _get(self, path: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
        query = urllib.parse.urlencode({"startDate": f"{start:%Y-%m-%d}", "endDate": f"{end:%Y-%m-%d}"})
        req = urllib.request.Request(f"{BASE_URL}/{path}?{query}",
                                     headers={"API-KEY": self.api_key, "Cache-Control": "no-cache"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:  # network, auth, bad JSON
            raise FeedError(f"AESO API call failed for {path}: {e}") from e

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # AESO endDate is inclusive of whole days, so ask for the last day we need.
        last_day = (end - pd.Timedelta(hours=1)).normalize()
        price = parse_pool_price(self._get(POOL_PRICE_PATH, start, last_day))
        load = parse_load(self._get(LOAD_PATH, start, last_day))
        df = pd.concat([price, load], axis=1)
        return df[(df.index >= start) & (df.index < end)]


def _local_times(rows: list[dict]) -> pd.DatetimeIndex:
    """Prefer AESO's local (MPT) timestamp; fall back to converting UTC."""
    if rows and rows[0].get("begin_datetime_mpt"):
        return pd.DatetimeIndex(pd.to_datetime([r["begin_datetime_mpt"] for r in rows]))
    utc = pd.to_datetime([r["begin_datetime_utc"] for r in rows], utc=True)
    return pd.DatetimeIndex(utc.tz_convert("America/Edmonton").tz_localize(None))


def _series(rows: list[dict], field: str, name: str) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float, name=name)
    s = pd.Series(pd.to_numeric([r.get(field) for r in rows], errors="coerce"),
                  index=_local_times(rows), name=name)
    s = s.dropna()  # hours not settled yet come back empty
    return s[~s.index.duplicated(keep="first")]  # daylight saving repeat hour


def parse_pool_price(payload: dict) -> pd.Series:
    try:
        rows = payload["return"]["Pool Price Report"]
    except (KeyError, TypeError) as e:
        raise FeedError(f"unexpected pool price response: {str(payload)[:200]}") from e
    return _series(rows, "pool_price", "price")


def parse_load(payload: dict) -> pd.Series:
    try:
        rows = payload["return"]["Actual Forecast Report"]
    except (KeyError, TypeError) as e:
        raise FeedError(f"unexpected load response: {str(payload)[:200]}") from e
    return _series(rows, "alberta_internal_load", "load_mw")


def check_quality(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Problems with a batch of hourly data. Empty list = clean."""
    problems = []
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    missing = expected.difference(df.dropna().index)
    if len(missing):
        problems.append(f"{len(missing)} missing hours (first {missing[0]})")
    if len(df) and ((df["price"] < 0) | (df["price"] > 999.99)).any():
        problems.append("price outside AESO limits $0 to $999.99")
    if len(df) and (df["load_mw"] < 5000).any():
        problems.append("load below 5,000 MW: suspicious")
    return problems
