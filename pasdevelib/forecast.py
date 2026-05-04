"""Pre-calcule les previsions sur 7 jours glissants.

Schema de sortie (forecast_7d.parquet) :
- station_id, target_date (YYYY-MM-DD), hour (0-23)
- bikes_p10, bikes_p50, bikes_p90 (quantiles sur les voisins analogues)
- docks_p10, docks_p50, docks_p90
- prob_empty (P(num_bikes == 0) sur les voisins)
- proba_velib, proba_place (pour compat directe avec le webapp)
- p25, p50, p75 (alias bikes_p25/50/75 pour DailyCurve)
- n_neighbors
"""
from __future__ import annotations

import datetime as dt
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pasdevelib import calendar_feats, predict, storage, weather


FORECAST_ASSET = "forecast_7d.parquet"
CALENDAR_ASSET = "calendar.parquet"
HOURLY_ASSET = "hourly_history.parquet"


def _download_parquet(release: str, asset: str) -> pd.DataFrame:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _build_target_features(target_dates: list[dt.date]) -> pd.DataFrame:
    """Pour chaque date cible, construit les features (météo + calendrier)."""
    n_days = len(target_dates)
    if n_days == 0:
        return pd.DataFrame()

    # Météo : 1 appel global sur n_days jours, on agrège ensuite par jour
    weather_hourly = weather.fetch_forecast(days=n_days)
    weather_hourly["date"] = pd.to_datetime(weather_hourly["ts"]).dt.tz_convert("Europe/Paris").dt.date.astype(str)
    weather_daily = weather_hourly.groupby("date", as_index=False).agg(
        temp_avg=("temperature_2m", "mean"),
        precip_total=("precipitation", "sum"),
    )

    # Calendrier : on demande la fenêtre [min, max] des dates cibles
    start = min(target_dates)
    end = max(target_dates)
    cal = calendar_feats.build_calendar(start, end)
    cal["date"] = cal["date"].astype(str)

    # Merge et enrichissement
    target_dates_str = [d.isoformat() for d in target_dates]
    targets = pd.DataFrame({"date": target_dates_str})
    targets = targets.merge(cal, on="date", how="left")
    targets = targets.merge(weather_daily, on="date", how="left")

    # Renommage pour matcher predict.py (day_of_week, is_holiday, is_school_holiday, season)
    targets["day_of_week"] = targets["weekday"]
    targets["is_holiday"] = targets["is_ferie"]
    targets["is_school_holiday"] = targets["is_vacances"]
    targets["season"] = targets["month"].apply(_month_to_season)

    return targets


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def run() -> None:
    today = dt.date.today()
    target_dates = [today + dt.timedelta(days=i) for i in range(7)]

    print("[forecast] loading history...")
    hourly = _download_parquet(storage.RELEASE_AGGREGATES, HOURLY_ASSET)
    calendar_existing = _download_parquet(storage.RELEASE_AGGREGATES, CALENDAR_ASSET)
    print(f"[forecast] {len(hourly):,} historical rows")

    print("[forecast] building 7-day target features (weather + calendar)...")
    targets = _build_target_features(target_dates)
    print(f"[forecast] {len(targets)} target days")

    all_predictions = []
    for _, target in targets.iterrows():
        target_date = dt.date.fromisoformat(target["date"])
        print(f"[forecast] predicting {target_date.isoformat()}...")
        pred = predict.predict_day_with_quantiles(
            target_date=target_date,
            target_features=target,
            calendar_df=calendar_existing,
            hourly_history=hourly,
        )
        if not pred.empty:
            all_predictions.append(pred)

    if not all_predictions:
        print("[forecast] no predictions generated, skipping upload")
        return

    final = pd.concat(all_predictions, ignore_index=True)

    # Renommage / enrichissement pour matcher le schema HourlyForecast attendu cote webapp
    final = final.rename(columns={"date": "target_date"})

    # Compatibilite : alias p25/p50/p75 = bikes_p10/p50/p90
    final["p25"] = final["bikes_p10"]
    final["p50"] = final["bikes_p50"]
    final["p75"] = final["bikes_p90"]

    # Probas pour le routing
    final["proba_velib"] = 1.0 - final["prob_empty"]
    final["proba_place"] = (final["docks_p50"].clip(lower=0) / 3.0).clip(upper=1.0)
    final.loc[final["docks_p50"] <= 0, "proba_place"] = 0.0

    print(f"[forecast] uploading {FORECAST_ASSET} : {len(final):,} rows")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / FORECAST_ASSET
        final.to_parquet(path, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, path, FORECAST_ASSET)


if __name__ == "__main__":
    run()
