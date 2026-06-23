"""score.py — Score de qualité des stations ET prédiction par vélo.

Score station (0-100) :
  - Fiabilité     (25 pts) : % temps fill_rate > 0.1
  - Stabilité     (25 pts) : inverse de la volatilité
  - Tendance      (25 pts) : fill_rate récent vs ancien
  - Rotation      (25 pts) : fréquence de mouvements (proxy activité)

Score vélo prédit (0-100) :
  - Mouvement     (25 pts) : rotation de la station (vélos actifs = mieux entretenus)
  - Fraîcheur     (25 pts) : temps estimé en station (immobile trop longtemps = malus)
  - Batterie      (25 pts) : min(100, heures_en_station × 15) pour électriques
  - Héritage      (25 pts) : score de la station répercuté

Sortie : station_scores.json
{
  "10042": {
    "score": 73,
    "reliability": 0.82,
    "stability": 0.71,
    "trend": 0.65,
    "rotation": 0.60,
    "bike_score": {
      "predicted": 68,
      "movement": 0.60,
      "freshness": 0.55,
      "battery_hours": 3.2,
      "predicted_battery_pct": 48
    },
    "n_hours": 412,
    "updated_at": "2026-06-23T10:00:00Z"
  }
}
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pasdevelib import storage

HOURLY_HISTORY_ASSET = "hourly_history.parquet"
OUTPUT_ASSET = "station_scores.json"
RELEASE_HISTORY = "history"

RECENT_DAYS = 7
OLD_DAYS = 30
CHARGE_RATE_PER_HOUR = 15.0   # % batterie rechargée par heure sur borne Vélib
MAX_CHARGE_HOURS = 7.0         # ~100% en 7h


def compute_scores(df: pd.DataFrame, now: datetime) -> dict:
    scores = {}

    for station_id, g in df.groupby("station_id"):
        g = g.sort_values(["date", "hour"]).reset_index(drop=True)

        fill = g["fill_rate"].clip(0, 1).values
        n = len(fill)

        if n < 24:
            continue

        # ── 1. Fiabilité (25 pts) ───────────────────────────────
        reliability = float((fill > 0.10).mean())
        pts_reliability = reliability * 25

        # ── 2. Stabilité (25 pts) ───────────────────────────────
        mean_fill = fill.mean()
        if mean_fill > 0:
            cv = fill.std() / mean_fill
            stability = float(max(0.0, 1.0 - min(cv, 1.0)))
        else:
            stability = 0.0
        pts_stability = stability * 25

        # ── 3. Tendance (25 pts) ────────────────────────────────
        dates = pd.to_datetime(g["date"].astype(str))
        ts_now = pd.Timestamp(now)
        cutoff_recent = ts_now - pd.Timedelta(days=RECENT_DAYS)
        cutoff_old = ts_now - pd.Timedelta(days=OLD_DAYS)

        mask_recent = dates >= cutoff_recent
        mask_old = (dates >= cutoff_old) & (dates < cutoff_recent)

        recent_mean = float(fill[mask_recent.values].mean()) if mask_recent.sum() >= 12 else mean_fill
        old_mean = float(fill[mask_old.values].mean()) if mask_old.sum() >= 12 else mean_fill

        diff = recent_mean - old_mean
        trend = float(np.clip((diff + 0.1) / 0.2, 0.0, 1.0))
        pts_trend = trend * 25

        # ── 4. Rotation (25 pts) ────────────────────────────────
        # Détecte les transitions fill_rate montant/descendant comme proxy de mouvements
        diffs = np.abs(np.diff(fill))
        # Un mouvement significatif = variation > 0.05 (au moins 5% du remplissage)
        significant_moves = float((diffs > 0.05).mean())
        # Normaliser : 0.3 mouvements/heure ou plus = max
        rotation = float(np.clip(significant_moves / 0.3, 0.0, 1.0))
        pts_rotation = rotation * 25

        # ── Score station final ─────────────────────────────────
        station_score = int(round(min(100, pts_reliability + pts_stability + pts_trend + pts_rotation)))

        # ── Score vélo prédit ───────────────────────────────────
        # Estimer le temps depuis le dernier mouvement (fraîcheur)
        # On regarde les N dernières heures avec fill_rate stable
        recent_fill = fill[-48:] if n >= 48 else fill
        stable_streak = 0
        for i in range(len(recent_fill) - 1, 0, -1):
            if abs(recent_fill[i] - recent_fill[i-1]) <= 0.05:
                stable_streak += 1
            else:
                break
        hours_in_station = float(stable_streak)  # heures estimées sans mouvement

        # Fraîcheur : pénalité si immobile trop longtemps (>12h)
        # Optimal : 1-4h (juste rechargé et prêt)
        if hours_in_station <= 1:
            freshness = 0.6   # vient d'arriver, pas encore bien chargé
        elif hours_in_station <= 4:
            freshness = 1.0   # idéal
        elif hours_in_station <= 8:
            freshness = 0.85
        elif hours_in_station <= 12:
            freshness = 0.65
        elif hours_in_station <= 24:
            freshness = 0.45  # risque de problème mécanique
        else:
            freshness = 0.25  # immobile depuis très longtemps
        pts_freshness = freshness * 25

        # Mouvement (héritage de rotation station)
        pts_movement = rotation * 25

        # Batterie prédite (pour électriques)
        predicted_battery_pct = int(min(100, hours_in_station * CHARGE_RATE_PER_HOUR))
        pts_battery = (predicted_battery_pct / 100) * 25

        # Héritage station
        pts_heritage = (station_score / 100) * 25

        bike_score = int(round(min(100,
            pts_movement * 0.25 +
            pts_freshness * 0.25 +
            pts_battery * 0.25 +
            pts_heritage * 0.25
        )))
        # Recalcul correct : chaque composante est déjà sur 25
        bike_score = int(round(min(100,
            pts_movement + pts_freshness + pts_battery + pts_heritage
        )))

        scores[str(station_id)] = {
            "score": station_score,
            "reliability": round(reliability, 3),
            "stability": round(stability, 3),
            "trend": round(trend, 3),
            "rotation": round(rotation, 3),
            "bike_score": {
                "predicted": bike_score,
                "movement": round(rotation, 3),
                "freshness": round(freshness, 3),
                "hours_in_station": round(hours_in_station, 1),
                "predicted_battery_pct": predicted_battery_pct,
            },
            "has_data": True,
            "n_hours": n,
            "updated_at": now.isoformat(),
        }

    return scores


def run() -> None:
    now = datetime.now(timezone.utc)
    print(f"[score] {now.isoformat()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        history_path = tmp_dir / HOURLY_HISTORY_ASSET

        if not storage.download_asset(RELEASE_HISTORY, HOURLY_HISTORY_ASSET, history_path):
            print("[score] hourly_history.parquet introuvable")
            return

        df = pd.read_parquet(history_path)
        print(f"[score] {len(df)} lignes, {df['station_id'].nunique()} stations")

        scores = compute_scores(df, now)
        print(f"[score] {len(scores)} scores calculés")

        out_path = tmp_dir / OUTPUT_ASSET
        out_path.write_text(json.dumps(scores, ensure_ascii=False))
        storage.upload_asset(storage.RELEASE_LIVE, out_path, OUTPUT_ASSET)
        print(f"[score] {OUTPUT_ASSET} uploadé")


if __name__ == "__main__":
    run()
