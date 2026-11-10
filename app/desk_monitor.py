"""Desk monitor: the screen a trader watches.

    streamlit run app/desk_monitor.py

Shows the kill switch, today's plan, running P&L against perfect foresight,
risk usage against limits, and the audit log. Switch between live and replay
in the sidebar.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from powerbot.desk.risk import RiskLimits  # noqa: E402

st.set_page_config(page_title="Battery Desk Monitor", layout="wide")
mode = st.sidebar.radio("Desk", ["replay", "live"])
db = ROOT / "data" / "desk" / f"{mode}.db"
if not db.exists():
    st.warning(f"No {mode} ledger yet. Run `python scripts/desk.py "
               f"{'replay --start 2024-01-01 --end 2024-03-31' if mode == 'replay' else 'init'}`.")
    st.stop()

con = sqlite3.connect(db)
days = pd.read_sql("SELECT * FROM days ORDER BY day", con, parse_dates=["day"]).set_index("day")
state = dict(con.execute("SELECT key, value FROM state").fetchall())
audit = pd.read_sql("SELECT ts_utc, day, level, event, message FROM audit ORDER BY id DESC LIMIT 300", con)
limits = RiskLimits()

st.title(f"Battery Desk Monitor ({mode})")

kill = state.get("kill_switch", "off") == "on"
if kill:
    st.error(f"🔴 KILL SWITCH ON: {state.get('kill_reason')}. "
             f"Review, then run `python scripts/desk.py resume{' --replay' if mode == 'replay' else ''}`.")
else:
    st.success("🟢 Trading enabled")

settled = days[days["status"] == "settled"]
pnl = settled["realized_pnl"].astype(float)
cum = pnl.cumsum()
dd = float((cum - cum.cummax().clip(lower=0)).iloc[-1]) if len(cum) else 0.0
week = float(pnl.iloc[-7:].sum()) if len(pnl) else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total P&L", f"${pnl.sum():,.0f}")
c2.metric("Last day", f"${pnl.iloc[-1]:,.0f}" if len(pnl) else "n/a")
perfect = settled["perfect_pnl"].sum()
c3.metric("Captured vs perfect", f"{100 * pnl.sum() / perfect:.0f}%" if perfect else "n/a")
c4.metric("Win rate", f"{100 * (pnl > 0).mean():.0f}%" if len(pnl) else "n/a")
c5.metric("Days traded", f"{len(settled)}")

st.subheader("Risk usage")
r1, r2, r3 = st.columns(3)
last = float(pnl.iloc[-1]) if len(pnl) else 0.0
for col, label, value, limit in [
    (r1, "Last day", last, limits.daily_loss_limit),
    (r2, "Last 7 days", week, limits.weekly_loss_limit),
    (r3, "Drawdown", dd, limits.max_drawdown_limit),
]:
    used = max(0.0, min(1.0, value / limit)) if value < 0 else 0.0
    shown = f"+${value:,.0f}" if value >= 0 else f"loss ${-value:,.0f}"
    col.write(f"**{label}**: {shown} (limit: loss ${-limit:,.0f}, {100 * used:.0f}% used)")
    col.progress(used)

st.subheader("Cumulative P&L")
if len(settled):
    st.line_chart(pd.DataFrame({"Desk": cum, "Perfect foresight": settled["perfect_pnl"].cumsum()}))

st.subheader("Latest plan")
latest = days.index.max()
plan = pd.read_sql("SELECT * FROM plans WHERE day = ? ORDER BY ts", con,
                   params=(f"{latest:%Y-%m-%d}",), parse_dates=["ts"]).set_index("ts")
st.write(f"**{latest:%Y-%m-%d}** ({days.loc[latest, 'status']}): {days.loc[latest, 'reason']}")
if len(plan):
    a, b = st.columns(2)
    a.markdown("Forecast with P10 to P90 range ($/MWh)")
    a.line_chart(plan[["f_mean", "f_p10", "f_p90"]].rename(columns={"f_mean": "Expected", "f_p10": "P10", "f_p90": "P90"}))
    b.markdown("Battery schedule (MW, sell + / buy minus) and spike odds")
    b.bar_chart((plan["discharge_mw"] - plan["charge_mw"]).rename("MW"))
    b.line_chart(plan["spike_prob"].rename("Spike probability"))

st.subheader("Days")
st.dataframe(days[["status", "expected_pnl", "realized_pnl", "perfect_pnl", "model_version", "reason"]]
             .sort_index(ascending=False).round(0), use_container_width=True)

st.subheader("Audit log")
lvl = st.multiselect("Levels", ["INFO", "WARN", "HALT"], default=["INFO", "WARN", "HALT"])
st.dataframe(audit[audit["level"].isin(lvl)], use_container_width=True, hide_index=True)
