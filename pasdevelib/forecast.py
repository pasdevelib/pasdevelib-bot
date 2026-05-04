"""Pre-calcule les previsions sur 7 jours glissants.

Schema de sortie (forecast_7d.parquet), aligne sur HourlyForecast :
- station_id, target_date (YYYY-MM-DD), hour (0-23)
- proba_velib, proba_place : 0-1
- p25, p50, p75 : nombre de velos attendu (fill_rate * capacity)
- prob_empty : 1 - proba_velib
- n_neighbors
"""
from __future__ import annotations

import datetime as dt
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import requests

from pasdevelib import calendar_feats, predict, storage, weather


FORECAST_ASSET = "forecast_7d.parquet"
CALENDAR_ASSET = "calendar.parquet"
HOURLY_ASSET = "hourly_history.parquet"
STATIONS_ASSET = "stations.json"


def _download_parquet(release: str, asset: str) -> pd.DataFrame:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _download_json(release: str, asset: str) -> list:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _build_target_features(target_dates: list[dt.date]) -> pd.DataFrame:
    n_days = len(target_dates)
    if n_days == 0:
        return pd.DataFrame()

    weather_hourly = weather.fetch_forecast(days=n_days)
    weather_hourly["date"] = pd.to_datetime(weather_hourly["ts"]).dt.tz_convert("Europe/Paris").dt.date.astype(str)
    weather_daily = weather_hourly.groupby("date", as_index=False).agg(
        temp_avg=("temperature_2m", "mean"),
        precip_total=("precipitation", "sum"),
    )

    start = min(target_dates)
    end = max(target_dates)
    cal = calendar_feats.build_calendar(start, end)
    cal["date"] = cal["date"].astype(str)

    target_dates_str = [d.isoformat() for d in target_dates]
    targets = pd.DataFrame({"date": target_dates_str})
    targets = targets.merge(cal, on="date", how="left")
    targets = targets.merge(weather_daily, on="date", how="left")

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

    print("[forecast] loading stations metadata...")
    stations_raw = _download_json(storage.RELEASE_LIVE, STATIONS_ASSET)
    capacities = {
        str(s.get("station_id") or s.get("stationcode") or s.get("stationCode")): int(s.get("capacity", 0) or 0)
        for s in stations_raw
    }
    print(f"[forecast] {len(capacities)} station capacities loaded")

    print("[forecast] building 7-day target features...")
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

    # Convertit fill_rate (0-1) en nombre de velos via capacity de la station
    final["station_id_str"] = final["station_id"].astype(str)
    final["capacity"] = final["station_id_str"].map(capacities).fillna(0).astype(int)
    final["p25"] = (final["p25_fill"] * final["capacity"]).round().astype(int)
    final["p50"] = (final["p50_fill"] * final["capacity"]).round().astype(int)
    final["p75"] = (final["p75_fill"] * final["capacity"]).round().astype(int)

    # Schema final aligne sur HourlyForecast cote webapp
    output = final[[
        "station_id", "target_date", "hour",
        "proba_velib", "proba_place",
        "p25", "p50", "p75",
        "prob_empty", "n_neighbors",
    ]].copy()

    print(f"[forecast] uploading {FORECAST_ASSET} : {len(output):,} rows")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / FORECAST_ASSET
        output.to_parquet(path, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, path, FORECAST_ASSET)

    print("[forecast] DONE")


if __name__ == "__main__":
    run()
