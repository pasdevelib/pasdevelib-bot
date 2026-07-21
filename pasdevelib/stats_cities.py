"""stats_cities.py — Classements des stations (vides/pleines/fiables) par
ville, sur 3 fenetres glissantes : jour (derniere journee complete),
semaine (7 jours), mois (30 jours).

Alimente la page publique blog.pasdevelib.app/donnees. Tourne pour TOUTES
les villes (Paris inclus), avec la meme isolation par ville que
consolidate_cities.py / forecast_cities.py : un echec sur une ville ne
bloque jamais les autres.

Schema de sortie (stats_<ville>_<periode>.json, release "stats-cities") :
{
  "city_id", "period" ("day"|"week"|"month"),
  "generated_at", "window_start", "window_end",
  "city_avg_fill_rate": float,
  "top_empty":      [{station_id, name, pct_empty, n_obs}, ...] (20 max)
  "top_full":       [{station_id, name, pct_full,  n_obs}, ...] (20 max)
  "most_reliable":  [{station_id, name, pct_healthy, n_obs}, ...] (20 max)
  "worst":          [{station_id, name, pct_extreme, n_obs}, ...] (20 max)
}

"pct_healthy" = part du temps ou la station est ni quasi-vide ni quasi-
pleine (fill_rate entre 0.2 et 0.8) — sert de proxy de fiabilite.
Seuil MIN_OBS : une station avec trop peu d'observations sur la fenetre
est exclue des classements (evite qu'une station en panne 29 jours sur 30
mais "parfaite" le dernier jour ne remonte en tete par accident).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import requests

from pasdevelib import storage
from pasdevelib.cities import list_cities

RELEASE_STATS = "stats-cities"
MIN_OBS_RATIO = 0.3  # une station doit avoir des donnees au moins 30% du temps de la fenetre pour etre classee
TOP_N = 20

PERIODS = {
    "day": 1,
    "week": 7,
    "month": 30,
}


def _download_parquet(release: str, asset: str) -> pd.DataFrame | None:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _download_json(release: str, asset: str) -> list | None:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _load_history_and_names(city_id: str) -> tuple[pd.DataFrame, dict[str, str]] | None:
    if city_id == "paris":
        hourly = _download_parquet(storage.RELEASE_AGGREGATES, "hourly_history.parquet")
        stations_raw = _download_json(storage.RELEASE_LIVE, "stations.json")
        names = {str(s["station_id"]): s.get("name", str(s["station_id"])) for s in (stations_raw or [])}
    else:
        hourly = _download_parquet("cities-history", f"hourly_history_{city_id}.parquet")
        stations_raw = _download_json("cities-live", "stations_cities.json")
        names = {
            str(s["station_id"]): s.get("name", str(s["station_id"]))
            for s in (stations_raw or []) if s.get("city_id") == city_id
        }
    if hourly is None or hourly.empty:
        return None
    return hourly, names


def _rank(df: pd.DataFrame, sort_col: str, value_col: str, names: dict[str, str]) -> list[dict]:
    top = df.sort_values(sort_col, ascending=False).head(TOP_N)
    return [
        {
            "station_id": row.station_id,
            "name": names.get(row.station_id, row.station_id),
            value_col: round(float(getattr(row, sort_col)), 3),
            "n_obs": int(row.n_obs),
        }
        for row in top.itertuples()
    ]


def compute_period(hourly: pd.DataFrame, names: dict[str, str], days: int) -> dict:
    hourly = hourly.copy()
    hourly["date"] = pd.to_datetime(hourly["date"])
    window_end = hourly["date"].max()
    window_start = window_end - pd.Timedelta(days=days - 1)
    window = hourly[hourly["date"] >= window_start]

    expected_obs = days * 24  # au mieux, une observation par heure
    min_obs = max(3, int(expected_obs * MIN_OBS_RATIO))

    grouped = window.groupby("station_id").agg(
        mean_fill=("fill_rate", "mean"),
        pct_empty=("fill_rate", lambda s: float((s <= 0.05).mean())),
        pct_full=("fill_rate", lambda s: float((s >= 0.95).mean())),
        pct_healthy=("fill_rate", lambda s: float(((s >= 0.2) & (s <= 0.8)).mean())),
        n_obs=("fill_rate", "count"),
    ).reset_index()
    grouped["station_id"] = grouped["station_id"].astype(str)
    grouped["pct_extreme"] = grouped["pct_empty"] + grouped["pct_full"]

    eligible = grouped[grouped["n_obs"] >= min_obs]

    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "city_avg_fill_rate": round(float(window["fill_rate"].mean()), 3) if not window.empty else None,
        "top_empty": _rank(eligible, "pct_empty", "pct_empty", names),
        "top_full": _rank(eligible, "pct_full", "pct_full", names),
        "most_reliable": _rank(eligible, "pct_healthy", "pct_healthy", names),
        "worst": _rank(eligible, "pct_extreme", "pct_extreme", names),
    }


def run_city(city_id: str) -> None:
    print(f"[stats_cities] === {city_id} ===")
    loaded = _load_history_and_names(city_id)
    if loaded is None:
        print(f"[stats_cities] {city_id}: pas d'historique disponible, skip")
        return
    hourly, names = loaded

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for period, days in PERIODS.items():
            result = compute_period(hourly, names, days)
            result["city_id"] = city_id
            result["period"] = period
            out_name = f"stats_{city_id}_{period}.json"
            out_path = tmp_dir / out_name
            out_path.write_text(json.dumps(result, ensure_ascii=False))
            storage.upload_asset(RELEASE_STATS, out_path, out_name)
            print(f"[stats_cities] {city_id}: {out_name} uploade "
                  f"({len(result['top_empty'])} stations classees, fenetre {result['window_start']}..{result['window_end']})")


def run(city_ids: list[str] | None = None) -> None:
    if city_ids is None:
        city_ids = list_cities()  # Paris inclus, contrairement a forecast_cities.py

    storage.ensure_release(RELEASE_STATS, "Classements de stations par ville (vides / pleines / fiables)")

    for city_id in city_ids:
        try:
            run_city(city_id)
        except Exception as e:
            print(f"[stats_cities] {city_id}: ECHEC ({e}) — villes suivantes non affectees")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=None)
    args = parser.parse_args()
    run(args.cities)
