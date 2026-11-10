"""Stage 2: forecast tomorrow's 24 hourly prices.

Three forecasters, from simplest to best:

1. Naive baseline: tomorrow at hour h = today at hour h (price_lag_24).
2. XGBoost quantile model: predicts the 10th, 50th and 90th percentile
   (P10, P50, P90) of the price, so we get a range, not just one number.
   P50 is our main forecast.
3. XGBoost mean model: the expected (average) price, which is what a
   trader should plan on, because spikes pull the average above the median.
4. XGBoost spike classifier: probability the hour goes above $300/MWh.

Walk forward testing: for each test month we retrain on ALL data before that
month, then predict that month. The model never sees the future, and it is
retrained the way a real desk would retrain as new data arrives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from powerbot.features import FEATURES

QUANTILES = [0.1, 0.5, 0.9]

QUANTILE_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=np.array(QUANTILES),
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
)

# Mean (expected value) model. P50 is the typical price, but a trader cares
# about the AVERAGE outcome, which spikes pull up. Squared error targets the mean.
MEAN_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
)

SPIKE_PARAMS = dict(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    n_jobs=-1,
)


def fit_quantile_model(train: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**QUANTILE_PARAMS)
    model.fit(train[FEATURES], train["price"])
    return model


def fit_spike_model(train: pd.DataFrame) -> xgb.XGBClassifier:
    # Spikes are rare (about 2% of hours), so weight them up.
    pos = train["is_spike"].sum()
    weight = (len(train) - pos) / max(pos, 1)
    model = xgb.XGBClassifier(**SPIKE_PARAMS, scale_pos_weight=weight)
    model.fit(train[FEATURES], train["is_spike"])
    return model


def fit_models(train: pd.DataFrame) -> dict:
    """Train the full model set used by the desk."""
    return {
        "quantile": fit_quantile_model(train),
        "mean": xgb.XGBRegressor(**MEAN_PARAMS).fit(train[FEATURES], train["price"]),
        "spike": fit_spike_model(train),
    }


def predict_models(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Turn features into the desk's forecast columns."""
    q = np.sort(models["quantile"].predict(X[FEATURES]), axis=1)
    return pd.DataFrame({
        "p10": np.clip(q[:, 0], 0, 999.99),
        "p50": np.clip(q[:, 1], 0, 999.99),
        "p90": np.clip(q[:, 2], 0, 999.99),
        "mean": np.clip(models["mean"].predict(X[FEATURES]), 0, 999.99),
        "spike_prob": models["spike"].predict_proba(X[FEATURES])[:, 1],
    }, index=X.index)


def walk_forward(feats: pd.DataFrame, test_start: str, test_end: str | None = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Monthly retrain, predict next month. Returns one row per test hour."""
    test_end = test_end or str(feats.index.max())
    months = pd.date_range(pd.Timestamp(test_start), pd.Timestamp(test_end), freq="MS")
    out = []
    for m in months:
        nxt = m + pd.offsets.MonthBegin(1)
        train = feats[feats.index < m]
        test = feats[(feats.index >= m) & (feats.index < nxt)]
        if test.empty:
            continue
        fc = predict_models(fit_models(train), test)  # P10 <= P50 <= P90 kept by sorting
        fc.insert(0, "naive", test["price_lag_24"])
        fc.insert(0, "price", test["price"])
        out.append(fc)
        if verbose:
            print(f"  {m:%Y-%m}: trained on {len(train):,} hours, predicted {len(test)}")
    return pd.concat(out)


def forecast_metrics(pred: pd.DataFrame) -> dict:
    """How good are the forecasts?"""
    y = pred["price"]
    mae_naive = (pred["naive"] - y).abs().mean()
    mae_model = (pred["p50"] - y).abs().mean()
    inside = ((y >= pred["p10"]) & (y <= pred["p90"])).mean()

    # Did we get the ORDER of hours right? That is what trading needs.
    def daily_rank_corr(col):
        g = pred.groupby(pred.index.normalize())
        return g.apply(lambda d: d[col].corr(d["price"], method="spearman")).mean()

    spikes = y > 300
    flagged = pred["spike_prob"] > 0.5
    tp = (spikes & flagged).sum()
    return {
        "mae_naive": mae_naive,
        "mae_model": mae_model,
        "mae_mean_model": (pred["mean"] - y).abs().mean(),
        "mae_improvement_pct": 100 * (1 - mae_model / mae_naive),
        "p10_p90_coverage_pct": 100 * inside,
        "rank_corr_naive": daily_rank_corr("naive"),
        "rank_corr_model": daily_rank_corr("p50"),
        "rank_corr_mean_model": daily_rank_corr("mean"),
        "spike_hours": int(spikes.sum()),
        "spike_recall_pct": 100 * tp / max(spikes.sum(), 1),
        "spike_precision_pct": 100 * tp / max(flagged.sum(), 1),
    }
