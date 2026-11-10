"""Command line for the trading desk.

Replay (rehearse on history, no API key needed):
    python scripts/desk.py replay --start 2024-01-01 --end 2024-03-31

Live paper trading (needs AESO_API_KEY):
    python scripts/desk.py init          # once: create the live ledger from market.db
    python scripts/desk.py run           # the daily cycle; schedule it just after midnight

Operations:
    python scripts/desk.py status        [--replay]
    python scripts/desk.py kill --reason "..."   [--replay]
    python scripts/desk.py resume --note "..."   [--replay]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from powerbot.desk.alerts import Alerter  # noqa: E402
from powerbot.desk.engine import Desk  # noqa: E402
from powerbot.desk.feeds import AESOFeed, FeedError, ReplayFeed  # noqa: E402
from powerbot.desk.ledger import Ledger  # noqa: E402
from powerbot.features import load_market  # noqa: E402

DESK_DIR = ROOT / "data" / "desk"
PATHS = {
    "live": dict(db=DESK_DIR / "live.db", models=ROOT / "models" / "live", log=ROOT / "logs" / "live_alerts.log"),
    "replay": dict(db=DESK_DIR / "replay.db", models=ROOT / "models" / "replay", log=ROOT / "logs" / "replay_alerts.log"),
}


def open_desk(mode: str, feed=None, discord: bool = True) -> Desk:
    p = PATHS[mode]
    return Desk(ledger=Ledger(p["db"]), feed=feed, models_dir=p["models"],
                alerter=Alerter(p["log"], enabled=discord))


def cmd_init(args) -> None:
    p = PATHS["live"]
    if p["db"].exists() and not args.force:
        sys.exit(f"{p['db']} already exists (use --force to rebuild)")
    if p["db"].exists():
        p["db"].unlink()
    desk = open_desk("live")
    n = desk.ledger.upsert_market(load_market())
    desk.ledger.log("init", f"live ledger created with {n:,} historical hours")
    print(f"Live ledger ready: {n:,} hours, last {desk.ledger.last_market_hour()}")
    print("Next: set AESO_API_KEY, then run: python scripts/desk.py run")


def cmd_run(args) -> None:
    try:
        feed = AESOFeed()
    except FeedError as e:
        sys.exit(str(e))
    desk = open_desk("live", feed)
    if desk.ledger.last_market_hour() is None:
        sys.exit("Live ledger is empty. Run: python scripts/desk.py init")
    day = pd.Timestamp(args.day) if args.day else pd.Timestamp.now().normalize()
    status = desk.run_cycle(day)
    print(f"{day:%Y-%m-%d}: {status}")
    print_status(desk, last_n=3)


def cmd_replay(args) -> None:
    p = PATHS["replay"]
    for f in [p["db"], p["log"]]:
        if f.exists():
            f.unlink()
    if p["models"].exists():
        for f in sorted(p["models"].rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
    market = load_market()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    feed = ReplayFeed(market)
    desk = open_desk("replay", feed, discord=False)
    desk.ledger.upsert_market(market[market.index < start])
    print(f"Replaying {start:%Y-%m-%d} to {end:%Y-%m-%d} as if live ...")
    t0 = time.time()
    for day in pd.date_range(start, end, freq="D"):
        feed.now = day  # the feed can't show anything from today onward
        if day.day == 1 or day == start:
            print("  (new month: retraining the model, this takes about a minute)", flush=True)
        status = desk.run_cycle(day)
        row = desk.ledger.days().iloc[-1]
        print(f"  {day:%Y-%m-%d} {status:8s} {row['reason']}", flush=True)
    # one last pass to settle the final day
    final = end + pd.Timedelta(days=1)
    feed.now = final
    desk.ingest(until=final)
    desk.settle_finished_days(final)
    print(f"Done in {time.time() - t0:.0f}s")
    print_status(desk)


def print_status(desk: Desk, last_n: int = 10) -> None:
    days = desk.ledger.days()
    settled = days[days["status"] == "settled"]
    print("\n=== Desk status ===")
    print(f"Kill switch: {desk.ledger.get('kill_switch', 'off').upper()}"
          + (f" ({desk.ledger.get('kill_reason')})" if desk.kill_switch_on() else ""))
    if len(settled):
        pnl = settled["realized_pnl"]
        cum = pnl.cumsum()
        print(f"Settled days: {len(settled)}   Total P&L: ${pnl.sum():,.0f}   "
              f"Perfect foresight: ${settled['perfect_pnl'].sum():,.0f} "
              f"({100 * pnl.sum() / settled['perfect_pnl'].sum():.0f}% captured)")
        print(f"Win rate: {100 * (pnl > 0).mean():.0f}%   Worst day: ${pnl.min():,.0f}   "
              f"Max drawdown: ${(cum - cum.cummax().clip(lower=0)).min():,.0f}")
    counts = days["status"].value_counts().to_dict()
    print(f"Days by status: {counts}")
    print(f"\nLast {last_n} days:")
    show = days.tail(last_n)[["status", "expected_pnl", "realized_pnl", "reason"]].copy()
    show["reason"] = show["reason"].str[:70]
    print(show.to_string())


def cmd_status(args) -> None:
    print_status(open_desk("replay" if args.replay else "live", discord=False))


def cmd_kill(args) -> None:
    desk = open_desk("replay" if args.replay else "live")
    desk.trip_kill_switch(f"manual: {args.reason}", pd.Timestamp.now().normalize())
    print("Kill switch ON. No trading until: python scripts/desk.py resume")


def cmd_resume(args) -> None:
    desk = open_desk("replay" if args.replay else "live")
    desk.resume(args.note)
    print("Kill switch OFF. Trading resumes at the next cycle.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Alberta battery trading desk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--force", action="store_true"); s.set_defaults(f=cmd_init)
    s = sub.add_parser("run"); s.add_argument("--day"); s.set_defaults(f=cmd_run)
    s = sub.add_parser("replay"); s.add_argument("--start", required=True); s.add_argument("--end", required=True)
    s.set_defaults(f=cmd_replay)
    s = sub.add_parser("status"); s.add_argument("--replay", action="store_true"); s.set_defaults(f=cmd_status)
    s = sub.add_parser("kill"); s.add_argument("--reason", required=True)
    s.add_argument("--replay", action="store_true"); s.set_defaults(f=cmd_kill)
    s = sub.add_parser("resume"); s.add_argument("--note", default="")
    s.add_argument("--replay", action="store_true"); s.set_defaults(f=cmd_resume)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
