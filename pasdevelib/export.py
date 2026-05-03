"""Génère les bundles JSON consommés par le webapp Next.js.

Plutôt que de demander au front de parser des parquets, on prépare ici
trois fichiers servis comme assets de la release `aggregates` :

  - stations.json    : 1500 stations (lat/lon/name/capacity), change rarement
  - current.json     : état temps réel (refresh 5 min via export_current)
  - forecast.json    : prédiction 7 jours (refresh quotidien via export_forecast)

Le webapp les fetch via les URLs de release stables, servies par le CDN GitHub.
"""
from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path

import pandas as pd

from pasdevelib import storage, fetch
from pasdevelib.scrape import CURRENT_ASSET
from pasdevelib.forecast import FORECAST_ASSET


STATIONS_JSON = "stations.json"
CURRENT_JSON = "current.json"
FORECAST_JSON = "forecast.json"


def export_stations() -> None:
    """Produit `stations.json` à partir de l'API GBFS (donnée statique)."""
    info = fetch.fetch_station_information()
    stations = [
        {
            "id": str(row["station_id"]),
            "code": row.get("stationCode"),
            "name": row["name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "capacity": int(row["capacity"]),
        }
        for _, row in info.iterrows()
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / STATIONS_JSON
        path.write_text(json.dumps({
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "n_stations": len(stations),
            "stations": stations,
        }, ensure_ascii=False))
        storage.upload_asset(storage.RELEASE_AGGREGATES, path, STATIONS_JSON)
        print(f"[export] {STATIONS_JSON}: {len(stations)} stations")


def export_current() -> None:
    """Produit `current.json` : état temps réel de chaque station.

    Source : dernier snapshot dans `current_day.parquet` de la release `live`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        live_path = tmp_dir / CURRENT_ASSET
        if not storage.download_asset(storage.RELEASE_LIVE, CURRENT_ASSET, live_path):
            print("[export] no live data, abort current export")
            return

        df = pd.read_parquet(live_path)
        if df.empty:
            print("[export] live file empty")
            return

        # On garde uniquement le snapshot le plus récent par station
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
        latest = df.sort_values("fetched_at").groupby("station_id").tail(1)

        records = [
            {
                "id": row["station_id"],
                "num_bikes": int(row["num_bikes_available"]),
                "num_mech": int(row.get("num_bikes_mechanical", 0)),
                "num_ebike": int(row.get("num_bikes_ebike", 0)),
                "num_docks": int(row["num_docks_available"]),
                "is_renting": bool(row.get("is_renting", True)),
                "is_returning": bool(row.get("is_returning", True)),
            }
            for _, row in latest.iterrows()
        ]

        out = tmp_dir / CURRENT_JSON
        out.write_text(json.dumps({
            "fetched_at": latest["fetched_at"].max().isoformat(),
            "n_stations": len(records),
            "states": records,
        }, ensure_ascii=False))
        storage.upload_asset(storage.RELEASE_AGGREGATES, out, CURRENT_JSON)
        print(f"[export] {CURRENT_JSON}: {len(records)} stations")


def export_forecast() -> None:
    """Produit `forecast.json` à partir de `forecast_7d.parquet`.

    Format compact, pivoté pour faciliter le lookup côté front :
      { generated_at, forecast: { station_id: { "YYYY-MM-DD": [24 floats per metric] } } }
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        path = tmp_dir / FORECAST_ASSET
        if not storage.download_asset(storage.RELEASE_AGGREGATES, FORECAST_ASSET, path):
            print("[export] no forecast parquet, abort")
            return

        df = pd.read_parquet(path)
        if df.empty:
            print("[export] forecast empty")
            return

        # Pivot : station_id -> date -> liste 24h
        forecast: dict = {}
        for (sid, date), group in df.groupby(["station_id", "target_date"]):
            group = group.sort_values("hour")
            # On remplit à 24h avec nan pour les heures absentes
            hourly = {h: None for h in range(24)}
            for _, row in group.iterrows():
                hourly[int(row["hour"])] = {
                    "v": round(float(row["proba_velib"]), 3),
                    "p": round(float(row["proba_place"]), 3),
                    "f": round(float(row["fill_rate_p50"]), 3),
                    "lo": round(float(row["fill_rate_p25"]), 3),
                    "hi": round(float(row["fill_rate_p75"]), 3),
                }
            forecast.setdefault(sid, {})[date] = [hourly[h] for h in range(24)]

        out = tmp_dir / FORECAST_JSON
        out.write_text(json.dumps({
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "n_stations": len(forecast),
            "forecast": forecast,
        }, ensure_ascii=False))
        storage.upload_asset(storage.RELEASE_AGGREGATES, out, FORECAST_JSON)
        print(f"[export] {FORECAST_JSON}: {len(forecast)} stations forecasted")


def run_all() -> None:
    """Exécute les trois exports d'un coup (utile en local pour debug)."""
    storage.ensure_release(storage.RELEASE_AGGREGATES, "Prediction tables")
    export_stations()
    export_current()
    export_forecast()


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    storage.ensure_release(storage.RELEASE_AGGREGATES, "Prediction tables")
    match cmd:
        case "stations":
            export_stations()
        case "current":
            export_current()
        case "forecast":
            export_forecast()
        case _:
            run_all()
