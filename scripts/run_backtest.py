"""Stage 3 + 4: trade the battery on every test day and report profit and risk.

    python scripts/run_backtest.py

Needs data/processed/predictions.csv from train_forecast.py.
Saves reports/backtest_summary.csv, reports/daily_pnl.csv, charts, and
reports/RESULTS.md.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from powerbot.backtest import STRATEGIES, run_backtest, summarize  # noqa: E402
from powerbot.battery import Battery  # noqa: E402
from powerbot.features import load_market  # noqa: E402

PRED = ROOT / "data" / "processed" / "predictions.csv"
REPORTS = ROOT / "reports"
LABELS = {
    "rule": "Fixed hours rule",
    "lp_naive": "LP + naive forecast",
    "lp_xgb": "LP + XGBoost P50",
    "lp_xgb_mean": "LP + XGBoost expected price",
    "perfect": "Perfect foresight (upper bound)",
}


def main() -> None:
    pred = pd.read_csv(PRED, index_col=0, parse_dates=True)
    market = load_market()
    train_prices = market["price"][market.index < pred.index.min()]
    bat = Battery()

    print(f"Backtesting {pred.index.normalize().nunique()} days with a "
          f"{bat.capacity_mwh:.0f} MWh / {bat.power_mw:.0f} MW battery ...")
    daily, hourly = run_backtest(pred, train_prices, bat)
    summary = summarize(daily, bat)

    REPORTS.mkdir(exist_ok=True)
    daily.to_csv(REPORTS / "daily_pnl.csv")
    hourly.join(pred).to_csv(REPORTS / "hourly_decisions.csv")
    summary.to_csv(REPORTS / "backtest_summary.csv")

    pd.set_option("display.width", 160)
    print("\n" + summary.round(2).to_string())

    # Chart 1: cumulative profit
    fig, ax = plt.subplots(figsize=(11, 5))
    for s in STRATEGIES:
        ax.plot(daily.index, daily[s].cumsum() / 1000, label=LABELS[s],
                linestyle="--" if s == "perfect" else "-")
    ax.set_ylabel("Cumulative profit ($ thousands)")
    ax.set_title("Battery trading profit on unseen data (walk forward)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "5_cumulative_profit.png", dpi=120)

    # Chart 2: one example week of decisions
    week = hourly.join(pred).loc["2024-01-08":"2024-01-14"]
    if not week.empty:
        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(week.index, week["price"], label="Actual price", color="black")
        ax1.plot(week.index, week["mean"], label="Forecast (expected)", color="tab:blue")
        ax1.fill_between(week.index, week["p10"], week["p90"], alpha=0.2, label="P10 to P90 range")
        ax1.set_ylabel("$/MWh")
        ax2 = ax1.twinx()
        ax2.bar(week.index, week["discharge_mw"], width=0.04, color="tab:green", alpha=0.6, label="Sell (MW)")
        ax2.bar(week.index, -week["charge_mw"], width=0.04, color="tab:red", alpha=0.6, label="Buy (MW)")
        ax2.set_ylabel("Battery MW (sell up, buy down)")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")
        ax1.set_title("Example week: forecasts and battery decisions (Jan 2024 cold snap)")
        fig.tight_layout()
        fig.savefig(REPORTS / "6_example_week.png", dpi=120)

    # Chart 3: profit by year
    yearly = daily[STRATEGIES].groupby(daily.index.year).sum() / 1000
    ax = yearly.rename(columns=LABELS).plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("$ thousands")
    ax.set_title("Profit by year: falling prices shrink every strategy")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(REPORTS / "7_profit_by_year.png", dpi=120)

    write_results(summary, daily, bat)
    print(f"\nSaved charts and {REPORTS / 'RESULTS.md'}")
    print("Dashboard: streamlit run app/dashboard.py")


def write_results(summary: pd.DataFrame, daily: pd.DataFrame, bat: Battery) -> None:
    fm_path = REPORTS / "forecast_metrics.json"
    fm = json.loads(fm_path.read_text()) if fm_path.exists() else {}
    best = summary.drop("perfect")["total_profit_$"].idxmax()
    lines = [
        "# Results",
        "",
        f"Test period: {daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d} "
        f"({len(daily)} days), never seen during training. Models retrained every month.",
        f"Battery: {bat.capacity_mwh:.0f} MWh, {bat.power_mw:.0f} MW, "
        f"{bat.round_trip_eff:.0%} round trip efficiency, ${bat.wear_cost:.0f}/MWh wear cost, "
        f"max {bat.max_cycles_per_day:.0f} cycle per day.",
        "",
    ]
    if fm:
        lines += [
            "## Forecast accuracy",
            "",
            "| Metric | Naive | XGBoost |",
            "|---|---|---|",
            f"| Mean absolute error ($/MWh) | {fm['mae_naive']:.2f} | {fm['mae_model']:.2f} "
            f"({fm['mae_improvement_pct']:.1f}% better) |",
            f"| Hour ranking accuracy (Spearman) | {fm['rank_corr_naive']:.2f} | {fm['rank_corr_model']:.2f} |",
            f"| Actual price inside P10 to P90 | | {fm['p10_p90_coverage_pct']:.1f}% (target 80%) |",
            f"| Spike hours (> $300) caught | | {fm['spike_recall_pct']:.0f}% of {fm['spike_hours']} "
            f"(precision {fm['spike_precision_pct']:.0f}%) |",
            "",
        ]
    lines += ["## Trading results", "", "| Strategy | Profit | $/kW per year | % of perfect | Sharpe | "
              "Max drawdown | Worst 5% days (avg) | Losing days |", "|---|---|---|---|---|---|---|---|"]
    for s, r in summary.iterrows():
        lines.append(
            f"| {LABELS[s]} | ${r['total_profit_$']:,.0f} | ${r['profit_per_kw_year_$']:.0f} | "
            f"{r['capture_vs_perfect_%']:.0f}% | {r['sharpe_annual']:.1f} | ${r['max_drawdown_$']:,.0f} | "
            f"${r['worst_5pct_day_avg_$']:,.0f} | {r['losing_days_%']:.0f}% |")
    lines += ["", f"Best real strategy: **{LABELS[best]}**.", "",
              "![](5_cumulative_profit.png)", "", "![](6_example_week.png)", "", "![](7_profit_by_year.png)"]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
