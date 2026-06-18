"""Module de recalibration automatique des paramètres Platt Scaling.

Calcule les paramètres (a, b) optimaux par tranche horaire
à partir des résidus des 30 derniers jours.

Utilisation dans eval_daily.py ou forecast.py :
    from pasdevelib.platt_calibrate import calibrate_platt
    params = calibrate_platt(hourly_history, target_dates_30j)
    # -> {"morning": (a, b), "peak": (a, b), ...}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid


def _logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _brier_loss(params: tuple[float, float], logit_p: np.ndarray, y: np.ndarray) -> float:
    a, b = params
    p_cal = sigmoid(a * logit_p + b)
    return float(np.mean((p_cal - y) ** 2))


def calibrate_platt_slot(
    proba_raw: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float]:
    """Calibre (a, b) pour sigmoid(a * logit(p) + b) minimisant le Brier score.

    Utilise une optimisation 2D simple (grid search + affinage).
    """
    if len(proba_raw) < 50:
        return (1.0, 0.0)  # pas assez de données

    logit_p = _logit(proba_raw)

    best = (1.0, 0.0)
    best_loss = _brier_loss(best, logit_p, y_true)

    # Grid search grossier
    for a in np.arange(0.5, 1.5, 0.1):
        for b in np.arange(-0.5, 0.5, 0.1):
            loss = _brier_loss((a, b), logit_p, y_true)
            if loss < best_loss:
                best_loss = loss
                best = (a, b)

    # Affinage local autour du meilleur point
    a0, b0 = best
    for a in np.arange(a0 - 0.15, a0 + 0.15, 0.02):
        for b in np.arange(b0 - 0.15, b0 + 0.15, 0.02):
            loss = _brier_loss((a, b), logit_p, y_true)
            if loss < best_loss:
                best_loss = loss
                best = (round(a, 3), round(b, 3))

    return best


def _hour_to_slot(hour: int) -> str:
    if 6 <= hour <= 9: return "morning"
    if 10 <= hour <= 15: return "midday"
    if 16 <= hour <= 19: return "peak"
    if 20 <= hour <= 22: return "evening"
    return "night"


def calibrate_platt(
    hourly_history: pd.DataFrame,
    analog_predictions: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """Calcule les paramètres Platt optimaux par tranche horaire.

    Args:
        hourly_history: DataFrame avec (station_id, date, hour, has_velib)
        analog_predictions: DataFrame avec (station_id, hour, date, proba_velib)
            -- les prédictions k-NN brutes (avant Platt) pour les jours évalués

    Returns:
        dict slot -> (a, b)
    """
    if analog_predictions.empty or hourly_history.empty:
        return {}

    merged = analog_predictions.merge(
        hourly_history[["station_id", "date", "hour", "has_velib"]],
        on=["station_id", "date", "hour"],
        how="inner",
    )

    if merged.empty:
        return {}

    merged["slot"] = merged["hour"].apply(_hour_to_slot)
    result: dict[str, tuple[float, float]] = {}

    for slot in ["morning", "midday", "peak", "evening", "night"]:
        sub = merged[merged["slot"] == slot]
        if len(sub) < 50:
            result[slot] = (1.0, 0.0)
            continue
        a, b = calibrate_platt_slot(
            sub["proba_velib"].values,
            sub["has_velib"].values.astype(float),
        )
        result[slot] = (a, b)
        print(f"[platt] {slot}: a={a:.3f}, b={b:.3f} (n={len(sub):,})")

    return result
