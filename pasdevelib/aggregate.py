"""Job d'agrégation hebdomadaire.

Construit les tables servant au modèle prédictif :
- medians.parquet : médiane et quantiles par (station_id, weekday, hour)
- analog_index.parquet : index des journées (date, weekday, météo, calendrier)
                         pour la recherche de journées analogues côté front

Les fichiers sont publiés sur la release `aggregates`.
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pasdevelib import storage, weather, calendar_feats


def _load_history(tmp_dir: Path) -> pd.DataFrame:
    """Télécharge tous les parquets de la release `history` et les concatène."""
    assets = storage.list_assets(storage.RELEASE_HISTORY)
    parquets = [a for a in assets if a.endswith(".parquet")]
    print(f"[aggregate] {len(parquets)} daily parquets to load")

    dfs = []
    for asset in parquets:
        path = tmp_dir / asset
        if storage.download_asset(storage.RELEASE_HISTORY, asset, path):
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_medians(history: pd.DataFrame) -> pd.DataFrame:
    """Médiane + quantiles par (station_id, weekday, hour).

    Sert pour la courbe attendue de la journée (graph type Solal).
    """
    df = history.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    paris_ts = df["fetched_at"].dt.tz_convert("Europe/Paris")
    df["weekday"] = paris_ts.dt.dayofweek
    df["hour"] = paris_ts.dt.hour

    # Taux de remplissage = vélos / capacité
    df["capacity"] = df["num_bikes_available"] + df["num_docks_available"]
    df = df[df["capacity"] > 0]
    df["fill_rate"] = df["num_bikes_available"] / df["capacity"]

    grouped = (
        df.groupby(["station_id", "weekday", "hour"])["fill_rate"]
        .agg([
            ("p25", lambda x: np.quantile(x, 0.25)),
            ("p50", lambda x: np.quantile(x, 0.50)),
            ("p75", lambda x: np.quantile(x, 0.75)),
            ("n_obs", "count"),
        ])
        .reset_index()
    )
    return grouped


def build_hourly_history(history: pd.DataFrame) -> pd.DataFrame:
    """Resample l'historique au pas horaire pour le forecast.

    Schéma : station_id, date, hour, fill_rate, has_velib, has_place
    Une ligne par (station_id, date, hour). On prend la médiane des
    snapshots dans l'heure pour lisser le bruit minute-par-minute.
    """
    df = history.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    paris_ts = df["fetched_at"].dt.tz_convert("Europe/Paris")
    df["date"] = paris_ts.dt.date
    df["hour"] = paris_ts.dt.hour

    df["capacity"] = df["num_bikes_available"] + df["num_docks_available"]
    df = df[df["capacity"] > 0]
    df["fill_rate"] = df["num_bikes_available"] / df["capacity"]
    df["has_velib"] = (df["num_bikes_available"] >= 1).astype(int)
    df["has_place"] = (df["num_docks_available"] >= 1).astype(int)

    grouped = df.groupby(["station_id", "date", "hour"]).agg(
        fill_rate=("fill_rate", "median"),
        has_velib=("has_velib", "max"),     # 1 si au moins un snapshot avec un vélo
        has_place=("has_place", "max"),
    ).reset_index()
    return grouped


def build_analog_index(
    history: pd.DataFrame,
    weather_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Index des journées : caractéristiques par jour pour la recherche analogue.

    Une ligne par jour avec :
    - date, weekday, month, is_ferie, is_vacances
    - mean_temperature, total_precipitation, mean_wind, has_rain
    """
    # Agrégation météo par jour
    w = weather_df.copy()
    w["date"] = w["ts"].dt.tz_convert("Europe/Paris").dt.date
    # Pluie cumulée sur fenêtres glissantes (features météo fines)
    w_sorted = w.sort_values(["date", "ts"])

    def precip_3h(grp: "pd.DataFrame") -> "pd.Series":
        return grp["precipitation"].rolling(3, min_periods=1).sum()

    w_sorted["precip_3h"] = w_sorted.groupby("date", group_keys=False).apply(precip_3h)
    w_sorted["precip_3h_max"] = w_sorted["precip_3h"]  # max sur la journée calculé après

    daily_weather = w_sorted.groupby("date").agg(
        mean_temperature=("temperature_2m", "mean"),
        max_temperature=("temperature_2m", "max"),
        mean_apparent_temperature=("apparent_temperature", "mean"),
        total_precipitation=("precipitation", "sum"),
        precip_3h_max=("precip_3h", "max"),   # pic de pluie sur 3h dans la journée
        mean_wind=("wind_speed_10m", "mean"),
    ).reset_index()
    daily_weather["has_rain"] = daily_weather["total_precipitation"] > 1.0
    daily_weather["has_heavy_rain"] = daily_weather["precip_3h_max"] > 5.0  # >5mm/3h = forte pluie

    merged = daily_weather.merge(calendar_df, on="date", how="inner")
    return merged


def run() -> None:
    storage.ensure_release(storage.RELEASE_AGGREGATES, "Prediction tables")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        history = _load_history(tmp_dir)
        if history.empty:
            print("[aggregate] no history yet, abort")
            return
        print(f"[aggregate] {len(history):,} rows of history")

        # 1. Médianes
        medians = build_medians(history)
        out = tmp_dir / "medians.parquet"
        medians.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] medians.parquet : {len(medians):,} rows")

        # 1bis. Historique horaire (pour le forecast.py)
        hourly = build_hourly_history(history)
        out = tmp_dir / "hourly_history.parquet"
        hourly.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] hourly_history.parquet : {len(hourly):,} rows")

        # 2. Météo + calendrier sur la même fenêtre
        history["fetched_at"] = pd.to_datetime(history["fetched_at"], utc=True)
        start = history["fetched_at"].min().date()
        end = history["fetched_at"].max().date()

        weather_df = weather.fetch_archive(start, end)
        out = tmp_dir / "weather.parquet"
        weather_df.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] weather.parquet : {len(weather_df):,} rows")

        calendar_df = calendar_feats.build_calendar(start, end + dt.timedelta(days=30))
        out = tmp_dir / "calendar.parquet"
        calendar_df.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] calendar.parquet : {len(calendar_df):,} rows")

        # 3. Index analogue (jointure météo + calendrier)
        analog = build_analog_index(history, weather_df, calendar_df)
        out = tmp_dir / "analog_index.parquet"
        analog.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] analog_index.parquet : {len(analog):,} rows")


if __name__ == "__main__":
    run()
