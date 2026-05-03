"""Pré-calcul des prédictions pour les 7 prochains jours.

Une fois par jour, ce module produit `forecast_7d.parquet` qui contient
pour chaque combinaison (station_id, target_date, hour) les probabilités
prédites par le modèle de journées analogues.

Le front peut alors faire un simple lookup au lieu de relancer un k-NN
à chaque requête utilisateur.

Volume estimé : 1500 stations × 7 jours × 24h = 252k lignes.
En zstd, environ 3-5 Mo. Téléchargeable en ~1s côté front.
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pasdevelib import storage, weather, calendar_feats
from pasdevelib.predict import TargetDay, find_analog_days


def _load_history(tmp_dir: Path) -> pd.DataFrame:
    assets = storage.list_assets(storage.RELEASE_HISTORY)
    parquets = [a for a in assets if a.endswith(".parquet")]
    dfs = []
    for asset in parquets:
        path = tmp_dir / asset
        if storage.download_asset(storage.RELEASE_HISTORY, asset, path):
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    paris_ts = df["fetched_at"].dt.tz_convert("Europe/Paris")
    df["date"] = paris_ts.dt.date
    df["hour"] = paris_ts.dt.hour
    df["capacity"] = df["num_bikes_available"] + df["num_docks_available"]
    df = df[df["capacity"] > 0]
    df["fill_rate"] = df["num_bikes_available"] / df["capacity"]
    df["has_velib"] = (df["num_bikes_available"] >= 1).astype(int)
    df["has_place"] = (df["num_docks_available"] >= 1).astype(int)
    return df


def _load_analog_index(tmp_dir: Path) -> pd.DataFrame:
    path = tmp_dir / "analog_index.parquet"
    if not storage.download_asset(storage.RELEASE_AGGREGATES, "analog_index.parquet", path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def _build_target_days() -> list[TargetDay]:
    """Construit la liste des 7 prochains jours avec leurs features cibles."""
    today = dt.date.today()
    horizon = [today + dt.timedelta(days=i) for i in range(7)]

    forecast_w = weather.fetch_forecast(days=7)
    forecast_w["date"] = forecast_w["ts"].dt.tz_convert("Europe/Paris").dt.date
    daily = forecast_w.groupby("date").agg(
        mean_temperature=("temperature_2m", "mean"),
        total_precipitation=("precipitation", "sum"),
        mean_wind=("wind_speed_10m", "mean"),
    ).reset_index()
    daily["has_rain"] = daily["total_precipitation"] > 1.0

    cal = calendar_feats.build_calendar(today, today + dt.timedelta(days=8))
    merged = daily.merge(cal, on="date", how="inner")

    targets = []
    for _, row in merged.iterrows():
        if row["date"] not in horizon:
            continue
        targets.append(TargetDay(
            date=row["date"],
            weekday=int(row["weekday"]),
            month=int(row["month"]),
            is_ferie=bool(row["is_ferie"]),
            is_vacances=bool(row["is_vacances"]),
            mean_temperature=float(row["mean_temperature"]),
            total_precipitation=float(row["total_precipitation"]),
            mean_wind=float(row["mean_wind"]),
            has_rain=bool(row["has_rain"]),
        ))
    return targets


def _predict_all_stations_for_day(
    target: TargetDay,
    history: pd.DataFrame,
    analog_index: pd.DataFrame,
    k: int = 20,
) -> pd.DataFrame:
    """Prédiction vectorisée pour TOUTES les stations à une date cible donnée.

    On cherche les K dates analogues une seule fois, puis on agrège
    l'historique de toutes les stations sur ces dates en un groupby.
    """
    neighbors = find_analog_days(target, analog_index, k=k)
    if neighbors.empty:
        return pd.DataFrame()

    neighbor_dates = set(neighbors["date"])
    sub = history[history["date"].isin(neighbor_dates)]
    if sub.empty:
        return pd.DataFrame()

    grouped = (
        sub.groupby(["station_id", "hour"])
        .agg(
            proba_velib=("has_velib", "mean"),
            proba_place=("has_place", "mean"),
            p25=("fill_rate", lambda x: np.quantile(x, 0.25)),
            p50=("fill_rate", lambda x: np.quantile(x, 0.50)),
            p75=("fill_rate", lambda x: np.quantile(x, 0.75)),
            n_neighbors=("date", "nunique"),
        )
        .reset_index()
    )
    grouped["target_date"] = target.date.isoformat()
    return grouped


def run() -> None:
    storage.ensure_release(storage.RELEASE_AGGREGATES, "Prediction tables")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        print("[forecast] loading history...")
        history = _load_history(tmp_dir)
        if history.empty:
            print("[forecast] no history, abort")
            return
        print(f"[forecast] {len(history):,} historical rows")

        print("[forecast] loading analog index...")
        analog_index = _load_analog_index(tmp_dir)
        if analog_index.empty:
            print("[forecast] no analog index, run aggregate.yml first")
            return

        print("[forecast] building 7-day targets (weather + calendar)...")
        targets = _build_target_days()
        if not targets:
            print("[forecast] no targets, abort")
            return
        print(f"[forecast] {len(targets)} target days")

        all_preds = []
        for target in targets:
            print(f"[forecast] predicting {target.date}...")
            preds = _predict_all_stations_for_day(target, history, analog_index)
            if not preds.empty:
                all_preds.append(preds)
                print(f"  -> {len(preds):,} (station, hour) predictions")

        if not all_preds:
            print("[forecast] nothing predicted, abort")
            return

        result = pd.concat(all_preds, ignore_index=True)
        result = result[[
            "station_id", "target_date", "hour",
            "proba_velib", "proba_place", "p25", "p50", "p75", "n_neighbors",
        ]]

        out = tmp_dir / "forecast_7d.parquet"
        result.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[forecast] forecast_7d.parquet : {len(result):,} rows uploaded")


if __name__ == "__main__":
    run()
