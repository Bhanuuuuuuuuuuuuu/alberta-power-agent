"""Make the first charts: what does Alberta's power price actually look like?

    python scripts/explore.py

Saves PNG charts to reports/. Look at each one and write down what you notice
in reports/NOTES.md. Those observations are what your forecaster will use.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from powerbot.data import load_sqlite  # noqa: E402

DB = ROOT / "data" / "processed" / "market.db"
OUT = ROOT / "reports"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_sqlite(DB).set_index("timestamp")

    # 1. Daily average price over the whole history
    fig, ax = plt.subplots(figsize=(11, 4))
    df["price"].resample("D").mean().plot(ax=ax, lw=0.7)
    ax.set_title("Alberta pool price, daily average")
    ax.set_ylabel("$/MWh")
    fig.tight_layout()
    fig.savefig(OUT / "1_price_history.png", dpi=130)

    # 2. Average price by hour of day: the shape a battery trades on
    fig, ax = plt.subplots(figsize=(8, 4))
    df.groupby(df.index.hour)["price"].mean().plot(kind="bar", ax=ax)
    ax.set_title("Average price by hour of day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("$/MWh")
    fig.tight_layout()
    fig.savefig(OUT / "2_price_by_hour.png", dpi=130)

    # 3. Price vs load: does high demand mean high prices?
    sample = df.dropna().sample(min(20000, len(df.dropna())), random_state=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample["load_mw"], sample["price"], s=2, alpha=0.3)
    ax.set_title("Price vs Alberta Internal Load")
    ax.set_xlabel("Load (MW)")
    ax.set_ylabel("$/MWh")
    fig.tight_layout()
    fig.savefig(OUT / "3_price_vs_load.png", dpi=130)

    # 4. Price distribution: how often are there spikes?
    fig, ax = plt.subplots(figsize=(8, 4))
    df["price"].clip(upper=1000).hist(bins=100, ax=ax, log=True)
    ax.set_title("How often each price happens (log scale)")
    ax.set_xlabel("$/MWh")
    fig.tight_layout()
    fig.savefig(OUT / "4_price_distribution.png", dpi=130)

    print(f"Saved 4 charts to {OUT}")


if __name__ == "__main__":
    main()
