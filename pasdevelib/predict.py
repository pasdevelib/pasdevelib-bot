"""Modèle de prédiction par journées analogues (k-NN temporel).

Méthode "comme dans les boutiques retail" :
on cherche dans l'historique les K journées les plus similaires
sur les axes calendaire + météo, puis on regarde ce qui s'est passé
à la station X à l'heure H sur ces journées-là.

Utilisable :
- offline (script Python qui produit un JSON pour le front)
- online (API Next.js qui charge les parquets et fait le k-NN)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TargetDay:
    """Conditions cibles pour la prédiction."""
    date: dt.date
    weekday: int
    month: int
    is_ferie: bool
    is_vacances: bool
    mean_temperature: float
    total_precipitation: float
    mean_wind: float
    has_rain: bool


@dataclass
class Prediction:
    station_id: str
    hour: int
    proba_velib: float          # P(num_bikes >= 1)
    proba_place: float          # P(num_docks >= 1)
    fill_rate_p25: float
    fill_rate_p50: float
    fill_rate_p75: float
    n_neighbors: int


def find_analog_days(
    target: TargetDay,
    analog_index: pd.DataFrame,
    k: int = 20,
    season_window_days: int = 15,
    temp_tolerance: float = 3.0,
) -> pd.DataFrame:
    """Retourne les K dates historiques les plus similaires.

    Filtres durs (must match):
    - même weekday
    - dans une fenêtre de ±season_window_days (toutes années confondues)
    - même statut férié, même statut vacances
    - même statut pluie (binaire)

    Tri par distance pondérée sur :
    - écart température
    - écart vent
    - écart précipitations
    """
    df = analog_index.copy()

    # Filtres durs
    df = df[df["weekday"] == target.weekday]
    df = df[df["is_ferie"] == target.is_ferie]
    df = df[df["is_vacances"] == target.is_vacances]
    df = df[df["has_rain"] == target.has_rain]

    # Fenêtre saisonnière (distance circulaire en jours sur l'année)
    target_doy = target.date.timetuple().tm_yday
    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear
    raw = (df["doy"] - target_doy).abs()
    df["doy_dist"] = np.minimum(raw, 365 - raw)
    df = df[df["doy_dist"] <= season_window_days]

    if df.empty:
        return df

    # Distance pondérée (z-score sur les 3 features continues)
    df["d_temp"] = (df["mean_temperature"] - target.mean_temperature).abs()
    df["d_wind"] = (df["mean_wind"] - target.mean_wind).abs()
    df["d_precip"] = (df["total_precipitation"] - target.total_precipitation).abs()
    df["score"] = df["d_temp"] / 3.0 + df["d_wind"] / 5.0 + df["d_precip"] / 5.0

    # Filtre tolérance température (soft)
    df = df[df["d_temp"] <= temp_tolerance * 1.5]

    return df.nsmallest(k, "score")


def predict_station(
    station_id: str,
    target: TargetDay,
    history: pd.DataFrame,
    analog_index: pd.DataFrame,
    hours: list[int] | None = None,
    k: int = 20,
) -> list[Prediction]:
    """Prédiction par heure pour une station donnée."""
    if hours is None:
        hours = list(range(24))

    neighbors = find_analog_days(target, analog_index, k=k)
    if neighbors.empty:
        return []

    neighbor_dates = set(neighbors["date"])

    # Filtrer l'historique sur la station + les dates analogues
    h = history[history["station_id"] == station_id].copy()
    h["fetched_at"] = pd.to_datetime(h["fetched_at"], utc=True)
    paris_ts = h["fetched_at"].dt.tz_convert("Europe/Paris")
    h["date"] = paris_ts.dt.date
    h["hour"] = paris_ts.dt.hour
    h = h[h["date"].isin(neighbor_dates)]

    if h.empty:
        return []

    h["capacity"] = h["num_bikes_available"] + h["num_docks_available"]
    h = h[h["capacity"] > 0]
    h["fill_rate"] = h["num_bikes_available"] / h["capacity"]
    h["has_velib"] = (h["num_bikes_available"] >= 1).astype(int)
    h["has_place"] = (h["num_docks_available"] >= 1).astype(int)

    out = []
    for hour in hours:
        sub = h[h["hour"] == hour]
        if sub.empty:
            continue
        out.append(Prediction(
            station_id=station_id,
            hour=hour,
            proba_velib=float(sub["has_velib"].mean()),
            proba_place=float(sub["has_place"].mean()),
            fill_rate_p25=float(np.quantile(sub["fill_rate"], 0.25)),
            fill_rate_p50=float(np.quantile(sub["fill_rate"], 0.50)),
            fill_rate_p75=float(np.quantile(sub["fill_rate"], 0.75)),
            n_neighbors=int(sub["date"].nunique()),
        ))
    return out
