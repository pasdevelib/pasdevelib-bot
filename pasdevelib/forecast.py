"""Pre-calcule les previsions sur 7 jours glissants avec quantiles."""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd
import requests

from pasdevelib import calendar_feats, predict, storage, weather


FORECAST_ASSET = "forecast_7d.parquet"
CALENDAR_ASSET = "calendar.parquet"
HOURLY_ASSET = "hourly_history.parquet"
WEATHER_ASSET = "weather.parquet"


def _download_parquet(release: str, asset: str) -> pd.DataFrame:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def run() -> None:
    today = dt.date.today()

    print("[forecast] loading history...")
    hourly = _download_parquet(storage.RELEASE_AGGREGATES, HOURLY_ASSET)
    calendar_existing = _download_parquet(storage.RELEASE_AGGREGATES, CALENDAR_ASSET)
    print(f"[forecast] {len(hourly):,} historical rows")

    print("[forecast] building 7-day targets (weather + calendar)...")
    target_dates = [today + dt.timedelta(days=i) for i in range(7)]
    forecast_weather = weather.fetch_forecast(target_dates)
    forecast_calendar = calendar_feats.build_features(target_dates)
    targets = forecast_calendar.merge(forecast_weather, on="date", how="left")
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
    print(f"[forecast] {FORECAST_ASSET} : {len(final):,} rows uploaded")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / FORECAST_ASSET
        final.to_parquet(path, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, path, FORECAST_ASSET)


if __name__ == "__main__":
    run()
