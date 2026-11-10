"""Interactive dashboard for the battery trading agent.

    pip install streamlit
    streamlit run app/dashboard.py

Reads the files made by train_forecast.py and run_backtest.py.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LABELS = {
    "rule": "Fixed hours rule",
    "lp_naive": "LP + naive forecast",
    "lp_xgb": "LP + XGBoost P50",
    "lp_xgb_mean": "LP + XGBoost expected price",
    "perfect": "Perfect foresight",
}

st.set_page_config(page_title="Alberta Battery Trading Agent", layout="wide")
st.title("Alberta Battery Trading Agent")
st.caption("A 10 MWh battery trading Alberta's AESO pool price, tested on data the models never saw.")

summary = pd.read_csv(REPORTS / "backtest_summary.csv", index_col=0)
daily = pd.read_csv(REPORTS / "daily_pnl.csv", index_col=0, parse_dates=True)
hourly = pd.read_csv(REPORTS / "hourly_decisions.csv", index_col=0, parse_dates=True)

best = summary.drop("perfect")["total_profit_$"].idxmax()
r = summary.loc[best]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total profit", f"${r['total_profit_$']:,.0f}")
c2.metric("Share of perfect foresight", f"{r['capture_vs_perfect_%']:.0f}%")
c3.metric("Sharpe ratio", f"{r['sharpe_annual']:.1f}")
c4.metric("Max drawdown", f"${r['max_drawdown_$']:,.0f}")
st.caption(f"Best strategy: {LABELS[best]}")

st.subheader("Cumulative profit")
cum = daily[list(LABELS)].cumsum().rename(columns=LABELS)
st.line_chart(cum)

st.subheader("Strategy comparison")
st.dataframe(summary.rename(index=LABELS).round(2), use_container_width=True)

st.subheader("Inspect any day")
days = sorted(hourly.index.normalize().unique())
day = st.select_slider("Day", options=days, value=days[len(days) // 2],
                       format_func=lambda d: d.strftime("%Y-%m-%d"))
d = hourly[hourly.index.normalize() == day]
left, right = st.columns(2)
left.markdown("**Prices: actual vs forecast ($/MWh)**")
left.line_chart(d[["price", "mean", "p10", "p90"]].rename(
    columns={"price": "Actual", "mean": "Forecast", "p10": "P10", "p90": "P90"}))
right.markdown("**Battery: sell (+) and buy (minus) in MW**")
right.bar_chart((d["discharge_mw"] - d["charge_mw"]).rename("MW"))
pnl = daily.loc[day]
st.write(f"Profit that day: **${pnl['lp_xgb_mean']:,.0f}** "
         f"(perfect foresight would have made ${pnl['perfect']:,.0f})")
