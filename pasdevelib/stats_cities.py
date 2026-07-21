"""stats_cities.py — Analyses de donnees par ville pour blog.pasdevelib.app/donnees :

1. Classements de stations (vides/pleines/fiables), 3 fenetres glissantes
   (jour/semaine/mois) — stats_<ville>_<periode>.json
2. Classement des quartiers/zones les plus problematiques (si geocodage
   disponible, cf. geocode_stations.py / geocode_cities.py) — inclus dans
   le fichier ci-dessus (cle "neighborhoods")
3. Evolution quotidienne du remplissage moyen du reseau, ~90 derniers
   jours — evolution_<ville>.json (pour un graphe de tendance)
4. Motif hebdomadaire (jour de semaine x heure), tout l'historique
   disponible — patterns_<ville>.json (pour une heatmap)
5. Records (meilleur/pire jour observe) — records_<ville>.json

Tourne pour TOUTES les villes (Paris inclus), avec la meme isolation par
ville que consolidate_cities.py / forecast_cities.py : un echec sur une
ville ne bloque jamais les autres.

Schema de sortie principal (stats_<ville>_<periode>.json, release
"stats-cities") :
{
  "city_id", "period" ("day"|"week"|"month"),
  "generated_at", "window_start", "window_end",
  "city_avg_fill_rate": float,
  "top_empty":      [{station_id, name, pct_empty, n_obs}, ...] (20 max)
  "top_full":       [{station_id, name, pct_full,  n_obs}, ...] (20 max)
  "most_reliable":  [{station_id, name, pct_healthy, n_obs}, ...] (20 max)
  "worst":          [{station_id, name, pct_extreme, n_obs}, ...] (20 max)
  "neighborhoods":  [{zone, pct_extreme, n_stations, n_obs}, ...] (10 max)
                    absent si aucune station de la ville n'a de zone geocodee.
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
TOP_N_ZONES = 10
EVOLUTION_DAYS = 90
MIN_ZONE_STATIONS = 3  # une zone avec moins de stations n'est pas assez representative pour etre classee

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


def _load_history_and_names(city_id: str) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]] | None:
    """Retourne (hourly_history, {station_id: name}, {station_id: zone}).

    Le dict de zones peut etre vide (geocodage pas encore passe, ou champ
    absent) — tout le code appelant doit tolerer ce cas sans planter.
    """
    if city_id == "paris":
        hourly = _download_parquet(storage.RELEASE_AGGREGATES, "hourly_history.parquet")
        stations_raw = _download_json(storage.RELEASE_LIVE, "stations.json")
        names = {str(s["station_id"]): s.get("name", str(s["station_id"])) for s in (stations_raw or [])}
        zones = {str(s["station_id"]): s["zone"] for s in (stations_raw or []) if s.get("zone")}
    else:
        hourly = _download_parquet("cities-history", f"hourly_history_{city_id}.parquet")
        stations_raw = _download_json("cities-live", "stations_cities.json")
        city_stations = [s for s in (stations_raw or []) if s.get("city_id") == city_id]
        names = {str(s["station_id"]): s.get("name", str(s["station_id"])) for s in city_stations}
        zones = {str(s["station_id"]): s["zone"] for s in city_stations if s.get("zone")}
    if hourly is None or hourly.empty:
        return None
    return hourly, names, zones


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


def _rank_zones(window: pd.DataFrame, zones: dict[str, str]) -> list[dict] | None:
    if not zones:
        return None
    df = window.copy()
    df["zone"] = df["station_id"].astype(str).map(zones)
    df = df[df["zone"].notna()]
    if df.empty:
        return None

    grouped = df.groupby("zone").agg(
        pct_empty=("fill_rate", lambda s: float((s <= 0.05).mean())),
        pct_full=("fill_rate", lambda s: float((s >= 0.95).mean())),
        n_obs=("fill_rate", "count"),
        n_stations=("station_id", "nunique"),
    ).reset_index()
    grouped["pct_extreme"] = grouped["pct_empty"] + grouped["pct_full"]
    grouped = grouped[grouped["n_stations"] >= MIN_ZONE_STATIONS]
    if grouped.empty:
        return None

    top = grouped.sort_values("pct_extreme", ascending=False).head(TOP_N_ZONES)
    return [
        {
            "zone": row.zone,
            "pct_extreme": round(float(row.pct_extreme), 3),
            "n_stations": int(row.n_stations),
            "n_obs": int(row.n_obs),
        }
        for row in top.itertuples()
    ]


def compute_period(hourly: pd.DataFrame, names: dict[str, str], zones: dict[str, str], days: int) -> dict:
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

    result = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "city_avg_fill_rate": round(float(window["fill_rate"].mean()), 3) if not window.empty else None,
        "top_empty": _rank(eligible, "pct_empty", "pct_empty", names),
        "top_full": _rank(eligible, "pct_full", "pct_full", names),
        "most_reliable": _rank(eligible, "pct_healthy", "pct_healthy", names),
        "worst": _rank(eligible, "pct_extreme", "pct_extreme", names),
    }
    neighborhoods = _rank_zones(window, zones)
    if neighborhoods is not None:
        result["neighborhoods"] = neighborhoods
    return result


def compute_evolution(hourly: pd.DataFrame) -> dict:
    """Serie quotidienne du remplissage moyen du reseau, ~90 derniers jours
    (pour un graphe de tendance sur la page Donnees)."""
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    end = df["date"].max()
    start = end - pd.Timedelta(days=EVOLUTION_DAYS - 1)
    window = df[df["date"] >= start]

    daily = window.groupby("date")["fill_rate"].mean().reset_index()
    daily = daily.sort_values("date")
    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "series": [
            {"date": row.date.date().isoformat(), "avg_fill_rate": round(float(row.fill_rate), 3)}
            for row in daily.itertuples()
        ],
    }


def compute_patterns(hourly: pd.DataFrame) -> dict:
    """Grille jour-de-semaine x heure (remplissage moyen), tout l'historique
    disponible — pour une heatmap "quel jour/heure est le pire" sur la page
    Donnees. weekday : 0=lundi ... 6=dimanche (convention pandas)."""
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.dayofweek

    grid = df.groupby(["weekday", "hour"])["fill_rate"].mean().reset_index()
    cells = [
        {"weekday": int(row.weekday), "hour": int(row.hour), "avg_fill_rate": round(float(row.fill_rate), 3)}
        for row in grid.itertuples()
    ]
    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_days_covered": int(df["date"].nunique()),
        "cells": cells,
    }


def compute_records(hourly: pd.DataFrame) -> dict:
    """Meilleur et pire jour observes (remplissage moyen du reseau sur
    l'historique disponible) — pour un encart "records" sur la page
    Donnees."""
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["fill_rate"].agg(["mean", "count"]).reset_index()
    # Un jour avec trop peu d'observations (scrape interrompu) fausserait
    # le record — meme logique de seuil que compute_period.
    daily = daily[daily["count"] >= 24 * 0.3]
    if daily.empty:
        return {"generated_at": dt.datetime.utcnow().isoformat() + "Z", "best_day": None, "worst_day": None}

    best = daily.loc[daily["mean"].idxmax()]
    worst = daily.loc[daily["mean"].idxmin()]
    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "best_day": {"date": best["date"].date().isoformat(), "avg_fill_rate": round(float(best["mean"]), 3)},
        "worst_day": {"date": worst["date"].date().isoformat(), "avg_fill_rate": round(float(worst["mean"]), 3)},
    }


def run_city(city_id: str) -> None:
    print(f"[stats_cities] === {city_id} ===")
    loaded = _load_history_and_names(city_id)
    if loaded is None:
        print(f"[stats_cities] {city_id}: pas d'historique disponible, skip")
        return
    hourly, names, zones = loaded

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for period, days in PERIODS.items():
            result = compute_period(hourly, names, zones, days)
            result["city_id"] = city_id
            result["period"] = period
            out_name = f"stats_{city_id}_{period}.json"
            out_path = tmp_dir / out_name
            out_path.write_text(json.dumps(result, ensure_ascii=False))
            storage.upload_asset(RELEASE_STATS, out_path, out_name)
            zone_note = f", {len(result['neighborhoods'])} quartiers" if "neighborhoods" in result else ""
            print(f"[stats_cities] {city_id}: {out_name} uploade "
                  f"({len(result['top_empty'])} stations classees{zone_note}, fenetre {result['window_start']}..{result['window_end']})")

        for name, compute_fn in [
            ("evolution", compute_evolution),
            ("patterns", compute_patterns),
            ("records", compute_records),
        ]:
            result = compute_fn(hourly)
            result["city_id"] = city_id
            out_name = f"{name}_{city_id}.json"
            out_path = tmp_dir / out_name
            out_path.write_text(json.dumps(result, ensure_ascii=False))
            storage.upload_asset(RELEASE_STATS, out_path, out_name)
            print(f"[stats_cities] {city_id}: {out_name} uploade")


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
