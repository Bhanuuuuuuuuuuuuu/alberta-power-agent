"""Stage 2: walk forward forecast over the test period (takes about 10 minutes).

    python scripts/train_forecast.py

Saves data/processed/predictions.csv and prints forecast accuracy.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from powerbot.features import build_features  # noqa: E402
from powerbot.forecast import forecast_metrics, walk_forward  # noqa: E402

TEST_START = "2023-01-01"  # train on 2010 to 2022, test on 2023 onward
OUT = ROOT / "data" / "processed" / "predictions.csv"


def main() -> None:
    feats = build_features()
    print(f"Feature table: {feats.shape[0]:,} hours x {feats.shape[1]} columns")
    print(f"Walk forward from {TEST_START}, retraining every month:")
    pred = walk_forward(feats, TEST_START)
    pred.to_csv(OUT)

    m = forecast_metrics(pred)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "forecast_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print("\nForecast accuracy on unseen data")
    print(f"  MAE naive (tomorrow = today)   ${m['mae_naive']:.2f}/MWh")
    print(f"  MAE XGBoost P50                ${m['mae_model']:.2f}/MWh  "
          f"({m['mae_improvement_pct']:.1f}% better)")
    print(f"  Real price inside P10 to P90   {m['p10_p90_coverage_pct']:.1f}% of hours (target 80%)")
    print(f"  Hour ranking accuracy (Spearman, per day): naive {m['rank_corr_naive']:.2f}, "
          f"XGBoost {m['rank_corr_model']:.2f}")
    print(f"  Spikes > $300: {m['spike_hours']} hours, caught {m['spike_recall_pct']:.0f}%, "
          f"precision {m['spike_precision_pct']:.0f}%")
    print(f"\nSaved {OUT}\nNext: python scripts/run_backtest.py")


if __name__ == "__main__":
    main()
