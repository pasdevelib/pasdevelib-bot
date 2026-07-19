"""consolidate_cities.py — Consolide les scrapes multi-villes en hourly_history par ville.

Produit un hourly_history_<city_id>.parquet par ville dans la release cities-history.
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import pandas as pd

from pasdevelib import storage
from pasdevelib.cities import list_cities

RELEASE_CITIES_LIVE = "cities-live"
RELEASE_CITIES_HISTORY = "cities-history"


def consolidate_city(city_id: str, tmp_dir: Path) -> None:
    """Consolide le current_day d'une ville en historique horaire."""
    asset_name = f"current_day_{city_id}.parquet"

    # Télécharger le current_day de la ville
    current_path = tmp_dir / asset_name
    if not storage.download_asset(RELEASE_CITIES_LIVE, asset_name, current_path):
        print(f"[consolidate_cities] {city_id}: current_day non trouvé, skip")
        return

    df = pd.read_parquet(current_path)
    if df.empty:
        print(f"[consolidate_cities] {city_id}: données vides, skip")
        return

    # Convertir en format hourly_history (fill_rate, has_velib, has_place)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
    df["date"] = df["fetched_at"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["fetched_at"].dt.hour

    # Calculer fill_rate depuis num_bikes_available / capacity
    df["fill_rate"] = df.apply(
        lambda r: r["num_bikes_available"] / r["capacity"] if r.get("capacity", 0) > 0 else 0.0,
        axis=1
    ).clip(0, 1)
    df["has_velib"] = (df["num_bikes_available"] > 0).astype(int)
    df["has_place"] = (df["num_docks_available"] > 0).astype(int)

    # Agréger par station_id + date + hour
    hourly = df.groupby(["station_id", "city_id", "date", "hour"]).agg(
        fill_rate=("fill_rate", "mean"),
        has_velib=("has_velib", "mean"),
        has_place=("has_place", "mean"),
        num_bikes_available=("num_bikes_available", "mean"),
        num_bikes_mechanical=("num_bikes_mechanical", "mean"),
        num_bikes_ebike=("num_bikes_ebike", "mean"),
    ).reset_index()

    # Charger l'historique existant si disponible
    history_name = f"hourly_history_{city_id}.parquet"
    history_path = tmp_dir / history_name
    if storage.download_asset(RELEASE_CITIES_HISTORY, history_name, history_path):
        existing = pd.read_parquet(history_path)
        combined = pd.concat([existing, hourly], ignore_index=True)
        combined = combined.drop_duplicates(subset=["station_id", "date", "hour"], keep="last")
        combined = combined.sort_values(["station_id", "date", "hour"])
    else:
        combined = hourly.sort_values(["station_id", "date", "hour"])

    combined.to_parquet(history_path, compression="snappy", index=False)
    storage.upload_asset(RELEASE_CITIES_HISTORY, history_path, history_name)
    print(f"[consolidate_cities] {city_id}: {len(combined)} lignes → {history_name}")


def run(city_ids: list[str] | None = None) -> None:
    if city_ids is None:
        city_ids = [c for c in list_cities() if c != "paris"]  # Paris géré séparément

    now = dt.datetime.now(dt.timezone.utc)
    print(f"[consolidate_cities] {now.isoformat()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for city_id in city_ids:
            # Isolation explicite (demande de Théo) : un plantage sur UNE
            # ville (donnee malformee, bug specifique a son format) ne doit
            # jamais empecher les autres villes de la meme execution
            # d'etre traitees. En pratique chaque ville tourne deja dans
            # son propre workflow GitHub Actions (voir consolidate-*.yml) —
            # ce try/except est une seconde couche de securite si jamais
            # cette fonction est un jour rappelee avec plusieurs villes.
            try:
                consolidate_city(city_id, tmp_dir)
            except Exception as e:
                print(f"[consolidate_cities] {city_id}: ECHEC ({e}) — villes suivantes non affectees")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=None,
                        help="IDs des villes à consolider (ex: bordeaux lyon)")
    args = parser.parse_args()
    run(args.cities)
