"""Prediction module: k-NN journees analogues avec proba + quantiles fill_rate.

V3 — améliorations :
- V2 : Platt Scaling, pondération temporelle, shrinkage, k adaptatif
- V3 : spatial layer (voisins géographiques), profil de station,
       grèves/événements dans la distance, poids is_disruption_day
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
    weight_disruption: float = 5.0   # NOUVEAU : grève/événement = distance forte
    # Pondération temporelle
    temporal_halflife_days: float = 365.0
    # Shrinkage
    shrinkage_threshold: int = 3
    shrinkage_alpha: float = 0.3
    # Spatial layer
    spatial_radius_m: float = 400.0  # rayon en mètres
    spatial_k: int = 5               # max voisines géographiques
    spatial_weight: float = 0.3      # poids du lissage spatial (0=désactivé, 1=full spatial)
    # Platt Scaling par tranche horaire
    platt_by_slot: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "morning":   (1.0, 0.0),
        "midday":    (1.0, 0.0),
        "peak":      (0.80, -0.15),
        "evening":   (0.90, -0.05),
        "night":     (1.0, 0.0),
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
    slot = _hour_to_slot(hour)
    a, b = cfg.platt_by_slot.get(slot, (1.0, 0.0))
    if a == 1.0 and b == 0.0:
        return proba
    eps = 1e-6
    p = np.clip(proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    return float(_sigmoid(a * logit_p + b))


def _temporal_weight(date_str: str, today: dt.date, halflife_days: float) -> float:
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
    # Température ressentie
    if pd.notna(row_a.get("mean_apparent_temperature")) and pd.notna(row_b.get("mean_apparent_temperature")):
        d += cfg.weight_temp * 0.5 * abs(row_a["mean_apparent_temperature"] - row_b["mean_apparent_temperature"]) / 30.0
    # Pluie totale
    if pd.notna(row_a.get("precip_total")) and pd.notna(row_b.get("precip_total")):
        d += cfg.weight_rain * abs(row_a["precip_total"] - row_b["precip_total"]) / 20.0
    # Pic de pluie 3h
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
    # Grève/événement NOUVEAU — poids fort car impact très différent
    if bool(row_a.get("is_disruption_day")) != bool(row_b.get("is_disruption_day")):
        d += cfg.weight_disruption
    elif bool(row_a.get("is_greve")) != bool(row_b.get("is_greve")):
        d += cfg.weight_disruption * 0.8  # grève seule = distance forte
    # Saison
    if row_a.get("season") != row_b.get("season"):
        d += cfg.weight_season
    return d


# ── Spatial layer ──────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points GPS."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def find_spatial_neighbors(
    station_id: str,
    stations_coords: pd.DataFrame,
    radius_m: float = 400.0,
    max_k: int = 5,
) -> list[str]:
    """Retourne les IDs des stations dans un rayon radius_m (hors la station elle-même).

    stations_coords : DataFrame avec colonnes station_id, lat, lon
    """
    row = stations_coords[stations_coords["station_id"] == station_id]
    if row.empty:
        return []
    lat, lon = float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])

    others = stations_coords[stations_coords["station_id"] != station_id].copy()
    others["dist_m"] = others.apply(
        lambda r: _haversine_m(lat, lon, r["lat"], r["lon"]), axis=1
    )
    nearby = others[others["dist_m"] <= radius_m].nsmallest(max_k, "dist_m")
    return list(nearby["station_id"])


# ── Profil de station ──────────────────────────────────────────────────────────

def compute_station_profiles(hourly_history: pd.DataFrame) -> pd.DataFrame:
    """Classe chaque station en : bureau, résidentiel, touristique, mixte.

    Méthode : ratio flux matin (7-9h) vs soir (17-19h) sur jours ouvrés.
    - bureau : forte sortie le matin (gens arrivent à vélo), forte arrivée le soir
      → fill_rate bas le matin, haut le soir = station "réceptrice matin"
    - résidentiel : inverse = station "émettrice matin"
    - touristique : flux élevés le week-end
    - mixte : pas de pattern dominant
    """
    df = hourly_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.dayofweek  # 0=lundi, 6=dimanche

    # Jours ouvrés uniquement pour le ratio bureau/résidentiel
    weekdays = df[df["weekday"] < 5]

    # Fill rate moyen par station aux heures clés
    morning = weekdays[weekdays["hour"].isin([7, 8, 9])].groupby("station_id")["fill_rate"].mean()
    evening = weekdays[weekdays["hour"].isin([17, 18, 19])].groupby("station_id")["fill_rate"].mean()
    weekend = df[df["weekday"] >= 5].groupby("station_id")["fill_rate"].mean()
    weekday_all = weekdays.groupby("station_id")["fill_rate"].mean()

    profiles = pd.DataFrame({
        "fill_morning": morning,
        "fill_evening": evening,
        "fill_weekend": weekend,
        "fill_weekday": weekday_all,
    }).dropna()

    def classify(row: pd.Series) -> str:
        ratio_me = row["fill_morning"] / (row["fill_evening"] + 0.01)
        we_boost = row["fill_weekend"] / (row["fill_weekday"] + 0.01)

        if we_boost > 1.3:
            return "touristique"
        if ratio_me < 0.7:
            return "bureau"  # matin vide, soir plein → zone de bureaux
        if ratio_me > 1.4:
            return "residentiel"  # matin plein, soir vide → zone résidentielle
        return "mixte"

    profiles["station_type"] = profiles.apply(classify, axis=1)
    return profiles[["station_type"]].reset_index()


# ── Matching ───────────────────────────────────────────────────────────────────

def _adaptive_k(distances: "pd.Series", cfg: AnalogConfig) -> int:
    sorted_dists = distances.sort_values().values
    k_base = min(cfg.k, len(sorted_dists))
    if k_base <= 2:
        return k_base
    d0 = sorted_dists[0] + 1e-6
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
                                         weight_dow=cfg.weight_dow, weight_holiday=cfg.weight_holiday,
                                         weight_disruption=cfg.weight_disruption, weight_season=0)),
        ("L3 sans pluie", AnalogConfig(k=cfg.k, weight_temp=cfg.weight_temp, weight_rain=0,
                                        weight_dow=cfg.weight_dow, weight_holiday=cfg.weight_holiday,
                                        weight_disruption=cfg.weight_disruption, weight_season=0)),
        ("L4 sans calendrier", AnalogConfig(k=cfg.k, weight_temp=cfg.weight_temp, weight_rain=0,
                                             weight_dow=0, weight_holiday=0, weight_disruption=0,
                                             weight_season=0)),
        ("L5 tout", AnalogConfig(k=max(20, cfg.k), weight_temp=0, weight_rain=0,
                                  weight_dow=0, weight_holiday=0, weight_disruption=0,
                                  weight_season=0)),
    ]
    for label, cur_cfg in levels:
        distances = candidates.apply(lambda r: _row_distance(target_features, r, cur_cfg), axis=1)
        if len(distances) == 0:
            continue
        k_adaptive = _adaptive_k(distances, cur_cfg)
        sorted_idx = distances.sort_values().index[:k_adaptive]
        if len(sorted_idx) >= 2:
            return list(candidates.loc[sorted_idx, "date"]), f"{label} (k={k_adaptive})"
    return list(candidates["date"].head(min(20, len(candidates)))), "L5 fallback"


def _count_recent_neighbors(analog_dates: list[str], today: dt.date, recency_days: int = 365) -> int:
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
    stations_coords: pd.DataFrame | None = None,
    station_profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prédit (proba_velib, proba_place, fill_rate quantiles) pour chaque (station, hour).

    Nouveautés V3 :
    - spatial_layer : lisse avec les 3-5 stations voisines dans 400m
    - station_profiles : utilise le profil (bureau/résidentiel/touristique) pour le contexte
    """
    cfg = cfg or AnalogConfig()
    today = dt.date.today()

    candidates = calendar_df[calendar_df["date"] != target_date.isoformat()].copy()
    if len(candidates) == 0:
        return pd.DataFrame()

    analog_dates, level = find_analog_days(target_features, candidates, cfg)
    n_recent = _count_recent_neighbors(analog_dates, today)
    print(f"[predict] {target_date.isoformat()} -> {level} : {len(analog_dates)} neighbors "
          f"({n_recent} récents)"
          + (" [GREVE]" if target_features.get("is_greve") else "")
          + (" [EVENT]" if target_features.get("is_event") else ""))

    sub = hourly_history[hourly_history["date"].isin(analog_dates)].copy()
    if sub.empty:
        return pd.DataFrame()

    # Pondération temporelle
    sub["_w"] = sub["date"].apply(
        lambda d: _temporal_weight(d, today, cfg.temporal_halflife_days)
    )

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

    # ── Calcul par station ────────────────────────────────────────────────────
    grouped_rows = []
    station_predictions: dict[str, dict[int, float]] = {}  # pour spatial layer

    for (station_id, hour), grp in sub.groupby(["station_id", "hour"]):
        w = grp["_w"]
        p_velib_raw = weighted_mean(grp["has_velib"].astype(float), w)
        p_place_raw = weighted_mean(grp["has_place"].astype(float), w)

        clim_velib = float(grp["has_velib"].mean())
        clim_place = float(grp["has_place"].mean())

        alpha = min(1.0, n_recent / cfg.shrinkage_threshold) if cfg.shrinkage_threshold > 0 else 1.0
        alpha = max(cfg.shrinkage_alpha, alpha)

        p_velib_shrunk = alpha * p_velib_raw + (1 - alpha) * clim_velib
        p_place_shrunk = alpha * p_place_raw + (1 - alpha) * clim_place

        p_velib_cal = _platt_scale(p_velib_shrunk, int(hour), cfg)
        p_place_cal = _platt_scale(p_place_shrunk, int(hour), cfg)

        row = {
            "station_id": station_id,
            "hour": hour,
            "proba_velib": round(p_velib_cal, 4),
            "proba_place": round(p_place_cal, 4),
            "p25_fill": weighted_quantile(grp["fill_rate"], w, 0.25),
            "p50_fill": weighted_quantile(grp["fill_rate"], w, 0.50),
            "p75_fill": weighted_quantile(grp["fill_rate"], w, 0.75),
            "n_neighbors": len(grp),
            "n_recent_neighbors": n_recent,
        }
        grouped_rows.append(row)

        # Mémoriser pour spatial layer
        sid = str(station_id)
        if sid not in station_predictions:
            station_predictions[sid] = {}
        station_predictions[sid][int(hour)] = p_velib_cal

    if not grouped_rows:
        return pd.DataFrame()

    # ── Spatial layer ─────────────────────────────────────────────────────────
    if stations_coords is not None and cfg.spatial_weight > 0:
        for row in grouped_rows:
            sid = str(row["station_id"])
            hour = int(row["hour"])
            neighbors = find_spatial_neighbors(
                sid, stations_coords, cfg.spatial_radius_m, cfg.spatial_k
            )
            neighbor_probas = [
                station_predictions[n][hour]
                for n in neighbors
                if n in station_predictions and hour in station_predictions[n]
            ]
            if neighbor_probas:
                spatial_mean = float(np.mean(neighbor_probas))
                # Lissage : (1 - w) * propre + w * voisines
                row["proba_velib"] = round(
                    (1 - cfg.spatial_weight) * row["proba_velib"]
                    + cfg.spatial_weight * spatial_mean,
                    4,
                )

    grouped = pd.DataFrame(grouped_rows)
    grouped["prob_empty"] = 1.0 - grouped["proba_velib"]
    grouped["target_date"] = target_date.isoformat()
    grouped["analog_level"] = level

    return grouped
