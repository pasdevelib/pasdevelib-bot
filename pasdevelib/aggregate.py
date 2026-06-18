"""Job d'agrégation hebdomadaire — V4.

Ajouts V4 :
- network_trend : % stations vides par heure à l'échelle Paris
- flux_graph : matrice de corrélation entre stations voisines (pour propagation spatiale)
- anomaly_stats : stats de référence pour détection d'anomalies
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pasdevelib import storage, weather, calendar_feats


def _load_history(tmp_dir: Path) -> pd.DataFrame:
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
    df = history.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    paris_ts = df["fetched_at"].dt.tz_convert("Europe/Paris")
    df["weekday"] = paris_ts.dt.dayofweek
    df["hour"] = paris_ts.dt.hour
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
        has_velib=("has_velib", "max"),
        has_place=("has_place", "max"),
    ).reset_index()
    return grouped


def build_dead_stations(hourly_history: pd.DataFrame, days_threshold: int = 2) -> pd.DataFrame:
    """Détecte les stations sans mouvement depuis days_threshold jours.

    Une station est "morte" si son fill_rate n'a pas changé de plus de 0.05
    sur les dernières 48h — probablement en maintenance ou hors service.

    Sortie : station_id, last_active_date, days_inactive, mean_fill_rate
    """
    df = hourly_history.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df.empty:
        return pd.DataFrame(columns=["station_id", "last_active_date", "days_inactive", "mean_fill_rate"])

    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=days_threshold)

    recent = df[df["date"] >= cutoff]
    dead = []

    for station_id, grp in recent.groupby("station_id"):
        std = grp["fill_rate"].std()
        mean = grp["fill_rate"].mean()
        # Mort si variance très faible ET remplissage fixe (pas de mouvement)
        if std < 0.03:
            last_active = df[df["station_id"] == station_id]["date"].max()
            days_inactive = (max_date - last_active).days
            dead.append({
                "station_id": str(station_id),
                "last_active_date": str(last_active.date()),
                "days_inactive": days_inactive,
                "mean_fill_rate": round(float(mean), 3),
            })

    return pd.DataFrame(dead)


def build_network_trend(hourly_history: pd.DataFrame) -> pd.DataFrame:
    """Tendance réseau global : % stations avec vélos par (weekday, hour).

    Permet de contextualiser la prédiction d'une station par rapport
    à l'état global du réseau à ce moment. Feature très corrélée aux
    pics de demande (concerts, météo, grèves).

    Sortie : weekday, hour, network_fill_rate, network_empty_rate
    """
    df = hourly_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.dayofweek

    trend = df.groupby(["weekday", "hour"]).agg(
        network_fill_rate=("fill_rate", "mean"),
        network_empty_rate=("has_velib", lambda x: 1 - x.mean()),
        n_stations=("station_id", "nunique"),
        n_obs=("fill_rate", "count"),
    ).reset_index()

    return trend


def build_flux_graph(hourly_history: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Graphe de flux entre stations : corrélations temporelles de fill_rate.

    Pour chaque station, identifie les N stations dont le fill_rate
    est le plus corrélé au sien (positivement ou négativement).
    Corrélation positive = flux simultanés (même quartier).
    Corrélation négative = flux complémentaires (départ/arrivée typiques).

    Sortie : station_a, station_b, correlation, lag_hours
    """
    df = hourly_history.copy()
    df["date"] = df["date"].astype(str)

    # Pivot : lignes = (date, hour), colonnes = station_id
    pivot = df.pivot_table(
        index=["date", "hour"],
        columns="station_id",
        values="fill_rate",
        aggfunc="mean",
    )

    # Limiter aux stations avec assez d'observations
    min_obs = len(pivot) * 0.3
    pivot = pivot.loc[:, pivot.count() >= min_obs]

    if pivot.shape[1] < 2:
        return pd.DataFrame(columns=["station_a", "station_b", "correlation"])

    # Corrélation par paires (limiter à 500 stations max pour la perf)
    stations = list(pivot.columns[:500])
    pivot_sub = pivot[stations].fillna(pivot[stations].mean())

    corr_matrix = pivot_sub.corr()

    rows = []
    for i, sta in enumerate(stations):
        # Top N corrélées (positif et négatif)
        corr_series = corr_matrix[sta].drop(sta).abs().nlargest(top_n)
        for stb, corr_val in corr_series.items():
            true_corr = corr_matrix.loc[sta, stb]
            rows.append({
                "station_a": str(sta),
                "station_b": str(stb),
                "correlation": round(float(true_corr), 4),
            })

    return pd.DataFrame(rows)


def build_anomaly_stats(hourly_history: pd.DataFrame) -> pd.DataFrame:
    """Stats de référence par (station_id, weekday, hour) pour détection d'anomalies.

    Retourne mean ± 2*std du fill_rate. Si la valeur temps réel
    dépasse ce seuil, on baisse la confiance de la prédiction.

    Sortie : station_id, weekday, hour, mean_fill, std_fill, q05_fill, q95_fill
    """
    df = hourly_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.dayofweek

    stats = df.groupby(["station_id", "weekday", "hour"])["fill_rate"].agg([
        ("mean_fill", "mean"),
        ("std_fill", "std"),
        ("q05_fill", lambda x: np.quantile(x, 0.05)),
        ("q95_fill", lambda x: np.quantile(x, 0.95)),
        ("n_obs", "count"),
    ]).reset_index()

    stats["std_fill"] = stats["std_fill"].fillna(0.1)
    return stats


def build_analog_index(
    history: pd.DataFrame,
    weather_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    w = weather_df.copy()
    w["date"] = w["ts"].dt.tz_convert("Europe/Paris").dt.date
    w_sorted = w.sort_values(["date", "ts"])

    def precip_3h(grp):
        return grp["precipitation"].rolling(3, min_periods=1).sum()

    w_sorted["precip_3h"] = w_sorted.groupby("date", group_keys=False).apply(precip_3h)

    daily_weather = w_sorted.groupby("date").agg(
        mean_temperature=("temperature_2m", "mean"),
        max_temperature=("temperature_2m", "max"),
        mean_apparent_temperature=("apparent_temperature", "mean"),
        total_precipitation=("precipitation", "sum"),
        precip_3h_max=("precip_3h", "max"),
        mean_wind=("wind_speed_10m", "mean"),
    ).reset_index()
    daily_weather["has_rain"] = daily_weather["total_precipitation"] > 1.0
    daily_weather["has_heavy_rain"] = daily_weather["precip_3h_max"] > 5.0

    merged = daily_weather.merge(calendar_df, on="date", how="inner")

    for col, default in [
        ("is_disruption_day", False),
        ("is_greve", False),
        ("is_event", False),
    ]:
        if col not in merged.columns:
            merged[col] = default

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

        # 2. Historique horaire
        hourly = build_hourly_history(history)
        out = tmp_dir / "hourly_history.parquet"
        hourly.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] hourly_history.parquet : {len(hourly):,} rows")

        # 3. Tendance réseau global — NOUVEAU
        network_trend = build_network_trend(hourly)
        out = tmp_dir / "network_trend.parquet"
        network_trend.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] network_trend.parquet : {len(network_trend):,} rows")

        # 4. Graphe de flux entre stations — NOUVEAU
        print("[aggregate] computing flux graph (may take a minute)...")
        flux_graph = build_flux_graph(hourly)
        out = tmp_dir / "flux_graph.parquet"
        flux_graph.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] flux_graph.parquet : {len(flux_graph):,} rows")

        # 5. Stats d'anomalies — NOUVEAU
        anomaly_stats = build_anomaly_stats(hourly)
        out = tmp_dir / "anomaly_stats.parquet"
        anomaly_stats.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] anomaly_stats.parquet : {len(anomaly_stats):,} rows")

        # 6. Météo + calendrier
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

        # 7. Index analogue
        analog = build_analog_index(history, weather_df, calendar_df)
        out = tmp_dir / "analog_index.parquet"
        analog.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] analog_index.parquet : {len(analog):,} rows")

        # 8. Profils de station
        from pasdevelib.predict import compute_station_profiles
        profiles = compute_station_profiles(hourly)
        out = tmp_dir / "station_profiles.parquet"
        profiles.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] station_profiles.parquet : {len(profiles):,} rows")

        # 9. Détection stations mortes (pas de changement depuis 48h)
        dead_stations = build_dead_stations(hourly)
        out = tmp_dir / "dead_stations.parquet"
        dead_stations.to_parquet(out, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, out)
        print(f"[aggregate] dead_stations.parquet : {len(dead_stations):,} stations mortes")


if __name__ == "__main__":
    run()
