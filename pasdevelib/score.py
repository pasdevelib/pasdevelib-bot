"""score.py — Calcule un score de qualité 0-100 par station.

Score = fiabilité (40) + stabilité (30) + tendance (20) + bonus disponibilité (10)
Basé uniquement sur hourly_history.parquet (nos propres données).

Format de sortie : station_scores.json
{
  "station_id": {
    "score": 73,
    "reliability": 0.82,   # % du temps avec fill_rate > 0.1
    "stability": 0.71,     # 1 - volatilité normalisée
    "trend": 0.65,         # fill_rate récent vs ancien
    "has_data": true,
    "n_hours": 412,
    "updated_at": "2026-06-22T10:00:00Z"
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

# Fenêtre pour la tendance : récent = 7j, ancien = 7-30j
RECENT_DAYS = 7
OLD_DAYS = 30

def compute_scores(df: pd.DataFrame, now: datetime) -> dict:
    scores = {}

    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("date")

        fill = g["fill_rate"].clip(0, 1)
        n = len(fill)

        if n < 24:  # Pas assez de données
            continue

        # ── Fiabilité (40 pts) ─────────────────────────────────────
        # % du temps où la station a au moins 10% de remplissage
        reliability = float((fill > 0.10).mean())
        pts_reliability = reliability * 40

        # ── Stabilité (30 pts) ─────────────────────────────────────
        # Inverse de la volatilité (std normalisé par la moyenne)
        mean_fill = fill.mean()
        if mean_fill > 0:
            cv = fill.std() / mean_fill  # Coefficient de variation
            stability = float(max(0, 1 - min(cv, 1)))
        else:
            stability = 0.0
        pts_stability = stability * 30

        # ── Tendance (20 pts) ──────────────────────────────────────
        # Compare fill_rate moyen récent (7j) vs ancien (7-30j)
        dates = pd.to_datetime(g["date"].astype(str))
        cutoff_recent = pd.Timestamp(now) - pd.Timedelta(days=RECENT_DAYS)
        cutoff_old = pd.Timestamp(now) - pd.Timedelta(days=OLD_DAYS)

        recent_fill = fill[dates >= cutoff_recent]
        old_fill = fill[(dates >= cutoff_old) & (dates < cutoff_recent)]

        if len(recent_fill) >= 12 and len(old_fill) >= 12:
            diff = float(recent_fill.mean()) - float(old_fill.mean())
            # +0.1 de diff → score max, -0.1 → score min
            trend = float(np.clip((diff + 0.1) / 0.2, 0, 1))
        else:
            trend = 0.5  # neutre si pas assez de données
        pts_trend = trend * 20

        # ── Bonus disponibilité (10 pts) ───────────────────────────
        # Bonus si has_velib est majoritairement True
        if "has_velib" in g.columns:
            avail = float(g["has_velib"].mean())
        else:
            avail = reliability
        pts_bonus = avail * 10

        # ── Score final ────────────────────────────────────────────
        total = pts_reliability + pts_stability + pts_trend + pts_bonus
        score = int(round(min(100, max(0, total))))

        scores[str(station_id)] = {
            "score": score,
            "reliability": round(reliability, 3),
            "stability": round(stability, 3),
            "trend": round(trend, 3),
            "availability": round(avail, 3),
            "has_data": True,
            "n_hours": n,
            "updated_at": now.isoformat(),
        }

    return scores


def run() -> None:
    now = datetime.now(timezone.utc)
    print(f"[score] Calcul des scores de qualit\u00e9 stations — {now.isoformat()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        history_path = tmp_dir / HOURLY_HISTORY_ASSET

        # T\u00e9l\u00e9charger hourly_history
        if not storage.download_asset(RELEASE_HISTORY, HOURLY_HISTORY_ASSET, history_path):
            print("[score] hourly_history.parquet introuvable, abandon")
            return

        df = pd.read_parquet(history_path)
        print(f"[score] {len(df)} lignes, {df['station_id'].nunique()} stations")

        scores = compute_scores(df, now)
        print(f"[score] {len(scores)} scores calcul\u00e9s")

        # Sauvegarder
        out_path = tmp_dir / OUTPUT_ASSET
        out_path.write_text(json.dumps(scores, ensure_ascii=False))
        storage.upload_asset(storage.RELEASE_LIVE, out_path, OUTPUT_ASSET)
        print(f"[score] {OUTPUT_ASSET} upload\u00e9")


if __name__ == "__main__":
    run()
