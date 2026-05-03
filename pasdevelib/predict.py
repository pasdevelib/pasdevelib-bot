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

    Stratégie de fallback progressif si aucun voisin n'est trouvé :
    - Niveau 1 : tous les filtres durs (weekday + saison + ferié + vacances + pluie)
    - Niveau 2 : abandon de la fenêtre saisonnière (couvre les cas hors saison)
    - Niveau 3 : abandon du filtre pluie (n'importe quelle météo)
    - Niveau 4 : abandon des filtres calendaires (ferié/vacances)
    - Niveau 5 : abandon du filtre weekday (pire cas, n'importe quel jour)

    Le scoring par distance pondérée reste identique à chaque niveau,
    on relâche juste les filtres durs.
    """
    base = analog_index.copy()
    target_doy = target.date.timetuple().tm_yday
    base["doy"] = pd.to_datetime(base["date"]).dt.dayofyear
    raw = (base["doy"] - target_doy).abs()
    base["doy_dist"] = np.minimum(raw, 365 - raw)
    base["d_temp"] = (base["mean_temperature"] - target.mean_temperature).abs()
    base["d_wind"] = (base["mean_wind"] - target.mean_wind).abs()
    base["d_precip"] = (base["total_precipitation"] - target.total_precipitation).abs()
    base["score"] = base["d_temp"] / 3.0 + base["d_wind"] / 5.0 + base["d_precip"] / 5.0

    levels = [
        ("L1 strict", lambda df: df[
            (df["weekday"] == target.weekday)
            & (df["is_ferie"] == target.is_ferie)
            & (df["is_vacances"] == target.is_vacances)
            & (df["has_rain"] == target.has_rain)
            & (df["doy_dist"] <= season_window_days)
            & (df["d_temp"] <= temp_tolerance * 1.5)
        ]),
        ("L2 sans saison", lambda df: df[
            (df["weekday"] == target.weekday)
            & (df["is_ferie"] == target.is_ferie)
            & (df["is_vacances"] == target.is_vacances)
            & (df["has_rain"] == target.has_rain)
        ]),
        ("L3 sans pluie", lambda df: df[
            (df["weekday"] == target.weekday)
            & (df["is_ferie"] == target.is_ferie)
            & (df["is_vacances"] == target.is_vacances)
        ]),
        ("L4 sans calendrier", lambda df: df[df["weekday"] == target.weekday]),
        ("L5 tout", lambda df: df),
    ]

    for label, flt in levels:
        candidates = flt(base)
        if len(candidates) >= 3:  # au moins 3 voisins pour faire une stat
            chosen = candidates.nsmallest(k, "score")
            print(f"[predict] {target.date} → {label} : {len(chosen)} neighbors")
            return chosen

    return pd.DataFrame()


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
