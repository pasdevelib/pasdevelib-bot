"""Prediction module: k-NN journees analogues avec quantiles p10/p50/p90."""
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
    """Distance pondérée entre deux journées."""
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
    """Trouve les k jours les plus proches avec fallback progressif."""
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
    """Predit (p10, p50, p90) pour chaque (station, hour) du jour cible.

    hourly_history columns: station_id, date, hour, num_bikes_available, num_docks_available
    """
    candidates = calendar_df[calendar_df["date"] != target_date.isoformat()].copy()
    if len(candidates) == 0:
        return pd.DataFrame()

    analog_dates, level = find_analog_days(target_features, candidates)
    print(f"[predict] {target_date.isoformat()} → {level} : {len(analog_dates)} neighbors")

    sub = hourly_history[hourly_history["date"].isin(analog_dates)]
    if sub.empty:
        return pd.DataFrame()

    grouped = sub.groupby(["station_id", "hour"]).agg(
        bikes_p10=("num_bikes_available", lambda x: float(np.percentile(x, 10))),
        bikes_p50=("num_bikes_available", lambda x: float(np.percentile(x, 50))),
        bikes_p90=("num_bikes_available", lambda x: float(np.percentile(x, 90))),
        docks_p10=("num_docks_available", lambda x: float(np.percentile(x, 10))),
        docks_p50=("num_docks_available", lambda x: float(np.percentile(x, 50))),
        docks_p90=("num_docks_available", lambda x: float(np.percentile(x, 90))),
        prob_empty=("num_bikes_available", lambda x: float((x == 0).mean())),
        n_neighbors=("num_bikes_available", "size"),
    ).reset_index()

    grouped["date"] = target_date.isoformat()
    grouped["analog_level"] = level
    return grouped
