"""Prediction module: k-NN journees analogues avec proba + quantiles fill_rate.

Le parquet hourly_history contient :
- station_id, date, hour
- fill_rate : taux de remplissage (0-1)
- has_velib : presence d'au moins un velo (bool)
- has_place : presence d'au moins une place (bool)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AnalogConfig:
    k: int = 7
    weight_temp: float = 1.0
    weight_rain: float = 2.0
    weight_dow: float = 3.0
    weight_holiday: float = 4.0
    weight_season: float = 1.5


def _row_distance(row_a: pd.Series, row_b: pd.Series, cfg: AnalogConfig) -> float:
    d = 0.0
    if pd.notna(row_a.get("temp_avg")) and pd.notna(row_b.get("temp_avg")):
        d += cfg.weight_temp * abs(row_a["temp_avg"] - row_b["temp_avg"]) / 30.0
    if pd.notna(row_a.get("precip_total")) and pd.notna(row_b.get("precip_total")):
        d += cfg.weight_rain * abs(row_a["precip_total"] - row_b["precip_total"]) / 20.0
    if row_a.get("day_of_week") != row_b.get("day_of_week"):
        d += cfg.weight_dow
    if row_a.get("is_holiday") != row_b.get("is_holiday"):
        d += cfg.weight_holiday
    if row_a.get("is_school_holiday") != row_b.get("is_school_holiday"):
        d += cfg.weight_holiday * 0.5
    if row_a.get("season") != row_b.get("season"):
        d += cfg.weight_season
    return d


def find_analog_days(
    target_features: pd.Series,
    candidates: pd.DataFrame,
    cfg: AnalogConfig | None = None,
) -> tuple[list[str], str]:
    cfg = cfg or AnalogConfig()
    levels = [
        ("L1 strict", cfg),
        ("L2 sans saison", AnalogConfig(k=cfg.k, weight_temp=cfg.weight_temp, weight_rain=cfg.weight_rain,
                                         weight_dow=cfg.weight_dow, weight_holiday=cfg.weight_holiday, weight_season=0)),
        ("L3 sans pluie", AnalogConfig(k=cfg.k, weight_temp=cfg.weight_temp, weight_rain=0,
                                        weight_dow=cfg.weight_dow, weight_holiday=cfg.weight_holiday, weight_season=0)),
        ("L4 sans calendrier", AnalogConfig(k=cfg.k, weight_temp=cfg.weight_temp, weight_rain=0,
                                             weight_dow=0, weight_holiday=0, weight_season=0)),
        ("L5 tout", AnalogConfig(k=max(20, cfg.k), weight_temp=0, weight_rain=0,
                                  weight_dow=0, weight_holiday=0, weight_season=0)),
    ]
    for label, cur_cfg in levels:
        distances = candidates.apply(lambda r: _row_distance(target_features, r, cur_cfg), axis=1)
        if len(distances) == 0:
            continue
        sorted_idx = distances.sort_values().index[: cur_cfg.k]
        if len(sorted_idx) >= 2:
            return list(candidates.loc[sorted_idx, "date"]), label
    return list(candidates["date"].head(min(20, len(candidates)))), "L5 fallback"


def predict_day_with_quantiles(
    target_date: dt.date,
    target_features: pd.Series,
    calendar_df: pd.DataFrame,
    hourly_history: pd.DataFrame,
) -> pd.DataFrame:
    """Predit (proba_velib, proba_place, fill_rate quantiles) pour chaque (station, hour).

    Sortie :
    - proba_velib : moyenne de has_velib sur les voisins
    - proba_place : moyenne de has_place sur les voisins
    - prob_empty : 1 - proba_velib
    - p25, p50, p75 : quantiles de fill_rate (0-1)
    - n_neighbors : nombre d'observations
    """
    candidates = calendar_df[calendar_df["date"] != target_date.isoformat()].copy()
    if len(candidates) == 0:
        return pd.DataFrame()

    analog_dates, level = find_analog_days(target_features, candidates)
    print(f"[predict] {target_date.isoformat()} -> {level} : {len(analog_dates)} neighbors")

    sub = hourly_history[hourly_history["date"].isin(analog_dates)]
    if sub.empty:
        return pd.DataFrame()

    grouped = sub.groupby(["station_id", "hour"]).agg(
        proba_velib=("has_velib", "mean"),
        proba_place=("has_place", "mean"),
        p25_fill=("fill_rate", lambda x: float(np.percentile(x, 25))),
        p50_fill=("fill_rate", lambda x: float(np.percentile(x, 50))),
        p75_fill=("fill_rate", lambda x: float(np.percentile(x, 75))),
        n_neighbors=("has_velib", "size"),
    ).reset_index()

    grouped["prob_empty"] = 1.0 - grouped["proba_velib"]
    grouped["target_date"] = target_date.isoformat()
    grouped["analog_level"] = level

    return grouped
