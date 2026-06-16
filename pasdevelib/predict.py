"""Prediction module: k-NN journees analogues avec proba + quantiles fill_rate.

V2 — améliorations :
- Pondération temporelle : poids exponentiel par âge du voisin (favorise 2025-2026)
- Platt Scaling par tranche horaire : corrige le biais optimiste
- Shrinkage vers climatologie : quand peu de voisins récents
- k adaptatif : ajuste k selon la qualité des matchs

Le parquet hourly_history contient :
- station_id, date, hour
- fill_rate : taux de remplissage (0-1)
- has_velib : presence d'au moins un velo (bool)
- has_place : presence d'au moins une place (bool)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class AnalogConfig:
    k: int = 7
    weight_temp: float = 1.0
    weight_rain: float = 2.0
    weight_dow: float = 3.0
    weight_holiday: float = 4.0
    weight_season: float = 1.5
    # Pondération temporelle : demi-vie en jours (365 = poids x0.5 après 1 an)
    temporal_halflife_days: float = 365.0
    # Shrinkage : poids vers climatologie quand n_recent_neighbors < seuil
    shrinkage_threshold: int = 3
    shrinkage_alpha: float = 0.3   # 0 = full climatologie, 1 = full k-NN
    # Platt Scaling : paramètres par tranche horaire (a, b) dans sigmoid(a*p + b)
    # Calibrés sur données mai-juin 2026. Correction conservative pour l'instant.
    platt_by_slot: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "morning":   (1.0, 0.0),   # 6h-9h : biais faible, pas de correction
        "midday":    (1.0, 0.0),   # 10h-15h
        "peak":      (0.80, -0.15), # 16h-19h : biais optimiste → correction baissière
        "evening":   (0.90, -0.05), # 20h-22h
        "night":     (1.0, 0.0),   # 23h-5h
    })


def _hour_to_slot(hour: int) -> str:
    if 6 <= hour <= 9:
        return "morning"
    if 10 <= hour <= 15:
        return "midday"
    if 16 <= hour <= 19:
        return "peak"
    if 20 <= hour <= 22:
        return "evening"
    return "night"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _platt_scale(proba: float, hour: int, cfg: AnalogConfig) -> float:
    """Applique le Platt Scaling pour corriger le biais optimiste par tranche horaire."""
    slot = _hour_to_slot(hour)
    a, b = cfg.platt_by_slot.get(slot, (1.0, 0.0))
    if a == 1.0 and b == 0.0:
        return proba
    # Logit du proba brut → correction affine → sigmoid
    eps = 1e-6
    p = np.clip(proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    return float(_sigmoid(a * logit_p + b))


def _temporal_weight(date_str: str, today: dt.date, halflife_days: float) -> float:
    """Poids exponentiel selon l'âge de la journée analogue."""
    try:
        d = dt.date.fromisoformat(date_str)
        age_days = (today - d).days
        return float(np.exp(-age_days * np.log(2) / halflife_days))
    except Exception:
        return 1.0


# ── Distance ───────────────────────────────────────────────────────────────────

def _row_distance(row_a: pd.Series, row_b: pd.Series, cfg: AnalogConfig) -> float:
    d = 0.0
    # Température moyenne
    if pd.notna(row_a.get("temp_avg")) and pd.notna(row_b.get("temp_avg")):
        d += cfg.weight_temp * abs(row_a["temp_avg"] - row_b["temp_avg"]) / 30.0
    # Température ressentie (apparent) — poids 0.5x temp pour éviter redondance
    if pd.notna(row_a.get("mean_apparent_temperature")) and pd.notna(row_b.get("mean_apparent_temperature")):
        d += cfg.weight_temp * 0.5 * abs(row_a["mean_apparent_temperature"] - row_b["mean_apparent_temperature"]) / 30.0
    # Pluie totale
    if pd.notna(row_a.get("precip_total")) and pd.notna(row_b.get("precip_total")):
        d += cfg.weight_rain * abs(row_a["precip_total"] - row_b["precip_total"]) / 20.0
    # Pic de pluie sur 3h — plus discriminant que le total journalier
    if pd.notna(row_a.get("precip_3h_max")) and pd.notna(row_b.get("precip_3h_max")):
        d += cfg.weight_rain * 0.8 * abs(row_a["precip_3h_max"] - row_b["precip_3h_max"]) / 10.0
    # Jour de semaine
    if row_a.get("day_of_week") != row_b.get("day_of_week"):
        d += cfg.weight_dow
    # Jours fériés
    if row_a.get("is_holiday") != row_b.get("is_holiday"):
        d += cfg.weight_holiday
    if row_a.get("is_school_holiday") != row_b.get("is_school_holiday"):
        d += cfg.weight_holiday * 0.5
    # Saison
    if row_a.get("season") != row_b.get("season"):
        d += cfg.weight_season
    return d


# ── Matching ───────────────────────────────────────────────────────────────────

def _adaptive_k(distances: "pd.Series", cfg: AnalogConfig) -> int:
    """k adaptatif : on prend au moins cfg.k voisins mais on peut en prendre plus
    si les suivants sont proches en distance (qualité similaire).
    On s'arrête dès que le gap de distance dépasse 20% du premier voisin.
    """
    sorted_dists = distances.sort_values().values
    k_base = min(cfg.k, len(sorted_dists))
    if k_base <= 2:
        return k_base
    d0 = sorted_dists[0] + 1e-6  # éviter division par zéro
    k = k_base
    for i in range(k_base, min(len(sorted_dists), cfg.k * 2)):
        gap = (sorted_dists[i] - sorted_dists[i - 1]) / d0
        if gap > 0.20:
            break
        k = i + 1
    return k


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
        # k adaptatif : ajuste selon la qualité des matchs
        k_adaptive = _adaptive_k(distances, cur_cfg)
        sorted_idx = distances.sort_values().index[:k_adaptive]
        if len(sorted_idx) >= 2:
            return list(candidates.loc[sorted_idx, "date"]), f"{label} (k={k_adaptive})"
    return list(candidates["date"].head(min(20, len(candidates)))), "L5 fallback"


def _count_recent_neighbors(analog_dates: list[str], today: dt.date, recency_days: int = 365) -> int:
    """Compte le nombre de voisins datant de moins de recency_days."""
    cutoff = today - dt.timedelta(days=recency_days)
    count = 0
    for d in analog_dates:
        try:
            if dt.date.fromisoformat(d) >= cutoff:
                count += 1
        except Exception:
            pass
    return count


# ── Prédiction ─────────────────────────────────────────────────────────────────

def predict_day_with_quantiles(
    target_date: dt.date,
    target_features: pd.Series,
    calendar_df: pd.DataFrame,
    hourly_history: pd.DataFrame,
    cfg: AnalogConfig | None = None,
) -> pd.DataFrame:
    """Prédit (proba_velib, proba_place, fill_rate quantiles) pour chaque (station, hour).

    Sortie :
    - proba_velib : probabilité calibrée (Platt + shrinkage)
    - proba_place : probabilité calibrée
    - prob_empty : 1 - proba_velib
    - p25, p50, p75 : quantiles de fill_rate (0-1)
    - n_neighbors : nombre d'observations utilisées
    - analog_level : niveau de fallback utilisé
    """
    cfg = cfg or AnalogConfig()
    today = dt.date.today()

    candidates = calendar_df[calendar_df["date"] != target_date.isoformat()].copy()
    if len(candidates) == 0:
        return pd.DataFrame()

    analog_dates, level = find_analog_days(target_features, candidates, cfg)
    n_recent = _count_recent_neighbors(analog_dates, today)
    print(f"[predict] {target_date.isoformat()} -> {level} : {len(analog_dates)} neighbors "
          f"({n_recent} récents)")

    sub = hourly_history[hourly_history["date"].isin(analog_dates)].copy()
    if sub.empty:
        return pd.DataFrame()

    # ── Pondération temporelle ──────────────────────────────────────────────
    sub["_w"] = sub["date"].apply(
        lambda d: _temporal_weight(d, today, cfg.temporal_halflife_days)
    )

    # ── Agrégation pondérée ─────────────────────────────────────────────────
    def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
        w = weights.loc[series.index]
        wsum = w.sum()
        if wsum == 0:
            return float(series.mean())
        return float((series * w).sum() / wsum)

    def weighted_quantile(series: pd.Series, weights: pd.Series, q: float) -> float:
        s = series.sort_values()
        w = weights.loc[s.index].values
        cumw = np.cumsum(w)
        cutoff = cumw[-1] * q
        idx = np.searchsorted(cumw, cutoff)
        return float(s.iloc[min(idx, len(s) - 1)])

    grouped_rows = []
    for (station_id, hour), grp in sub.groupby(["station_id", "hour"]):
        w = grp["_w"]
        p_velib_raw = weighted_mean(grp["has_velib"].astype(float), w)
        p_place_raw = weighted_mean(grp["has_place"].astype(float), w)

        # Climatologie de fallback (moyenne simple, non pondérée)
        clim_velib = float(grp["has_velib"].mean())
        clim_place = float(grp["has_place"].mean())

        # Shrinkage vers climatologie si peu de voisins récents
        alpha = min(1.0, n_recent / cfg.shrinkage_threshold) if cfg.shrinkage_threshold > 0 else 1.0
        alpha = max(cfg.shrinkage_alpha, alpha)   # alpha minimum = shrinkage_alpha

        p_velib_shrunk = alpha * p_velib_raw + (1 - alpha) * clim_velib
        p_place_shrunk = alpha * p_place_raw + (1 - alpha) * clim_place

        # Platt Scaling par heure
        p_velib_cal = _platt_scale(p_velib_shrunk, int(hour), cfg)
        p_place_cal = _platt_scale(p_place_shrunk, int(hour), cfg)

        grouped_rows.append({
            "station_id": station_id,
            "hour": hour,
            "proba_velib": round(p_velib_cal, 4),
            "proba_place": round(p_place_cal, 4),
            "p25_fill": weighted_quantile(grp["fill_rate"], w, 0.25),
            "p50_fill": weighted_quantile(grp["fill_rate"], w, 0.50),
            "p75_fill": weighted_quantile(grp["fill_rate"], w, 0.75),
            "n_neighbors": len(grp),
            "n_recent_neighbors": n_recent,
        })

    if not grouped_rows:
        return pd.DataFrame()

    grouped = pd.DataFrame(grouped_rows)
    grouped["prob_empty"] = 1.0 - grouped["proba_velib"]
    grouped["target_date"] = target_date.isoformat()
    grouped["analog_level"] = level

    return grouped
