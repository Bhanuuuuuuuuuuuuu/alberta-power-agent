"""Model ops: train, save, load and version the forecasting models.

Every trained model set gets a version folder:

    models/v2025-08-01/
        quantile.json  mean.json  spike.json  meta.json

meta.json records what data it was trained on, so every trade in the ledger
can be traced back to the exact model that made it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

from powerbot.features import FEATURES, build_features
from powerbot.forecast import fit_models

MODEL_FILES = {"quantile": xgb.XGBRegressor, "mean": xgb.XGBRegressor, "spike": xgb.XGBClassifier}


def train_and_save(market: pd.DataFrame, models_dir: Path, as_of: pd.Timestamp) -> str:
    """Train on all data before `as_of` and save. Returns the version name."""
    as_of = pd.Timestamp(as_of).normalize()
    feats = build_features(market[market.index < as_of])
    models = fit_models(feats)
    version = f"v{as_of:%Y-%m-%d}"
    folder = models_dir / version
    folder.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.save_model(folder / f"{name}.json")
    meta = {
        "version": version,
        "trained_through": str(feats.index.max()),
        "train_rows": int(len(feats)),
        "features": FEATURES,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    (models_dir / "LATEST").write_text(version)
    return version


def latest_version(models_dir: Path) -> str | None:
    f = models_dir / "LATEST"
    return f.read_text().strip() if f.exists() else None


def load_models(models_dir: Path, version: str | None = None) -> tuple[dict, dict]:
    version = version or latest_version(models_dir)
    if version is None:
        raise FileNotFoundError("no trained models; run: python scripts/desk.py train")
    folder = models_dir / version
    meta = json.loads((folder / "meta.json").read_text())
    if meta["features"] != FEATURES:
        raise ValueError(f"model {version} was trained on different features; retrain it")
    models = {}
    for name, cls in MODEL_FILES.items():
        m = cls()
        m.load_model(folder / f"{name}.json")
        models[name] = m
    return models, meta


def needs_retrain(meta: dict | None, day: pd.Timestamp) -> bool:
    """Retrain on the first cycle of each new month (or if no model exists)."""
    if meta is None:
        return True
    trained = pd.Timestamp(meta["trained_through"])
    return (day.year, day.month) != ((trained + pd.Timedelta(hours=1)).year,
                                     (trained + pd.Timedelta(hours=1)).month)
