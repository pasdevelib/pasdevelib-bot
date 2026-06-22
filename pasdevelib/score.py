"""
Calcul du score de qualite par station.

Score 0-100 :
- Fiabilite   40pts — % temps avec fill_rate > 0.1
- Stabilite   30pts — inverse de la volatilite
- Tendance    20pts — evolution recente vs ancienne
- Anti-boomerang 10pts — penalite pour cycles rapides
"""
from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pasdevelib import storage


def compute_score(fr: pd.Series) -> float:
    if len(fr) < 3:
        return 50.0

    reliability = float((fr > 0.1).mean()) * 40

    volatility = float(fr.diff().abs().mean())
    stability  = max(0.0, 30.0 * (1 - min(volatility * 5, 1)))

    if len(fr) >= 12:
        recent = float(fr.tail(6).mean())
        older  = float(fr.head(6).mean())
        trend  = max(0.0, min(20.0, 10.0 + (recent - older) * 20))
    else:
        trend = 10.0

    changes  = fr.diff()
    boomerang = int(((changes > 0.05) & (changes.shift(-1) < -0.05)).sum())
    anti_boom = max(0.0, 10.0 - boomerang * 2)

    return round(min(100.0, max(0.0, reliability + stability + trend + anti_boom)), 1)


def run() -> None:
    print("[scores] Calcul des scores...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        day_path = tmp_dir / "current_day.parquet"
        if not storage.download_asset(storage.RELEASE_LIVE, "current_day.parquet", day_path):
            print("[scores] Pas de donnees")
            return

        df = pd.read_parquet(day_path)

        # Calculer fill_rate si absent
        if "fill_rate" not in df.columns:
            cap_path = tmp_dir / "stations.json"
            if storage.download_asset(storage.RELEASE_LIVE, "stations.json", cap_path):
                stations = pd.DataFrame(json.loads(cap_path.read_text()))
                stations["station_id"] = stations["station_id"].astype(str)
                df["station_id"] = df["station_id"].astype(str)
                df = df.merge(stations[["station_id", "capacity"]], on="station_id", how="left")
                df["fill_rate"] = df.get("num_bikes_available", 0) / df["capacity"].clip(lower=1)

        if "fill_rate" not in df.columns:
            print("[scores] Colonne fill_rate introuvable")
            return

        scores: dict[str, float] = {}
        for sid, grp in df.groupby("station_id"):
            fr = grp.sort_values("fetched_at")["fill_rate"].dropna()
            scores[str(sid)] = compute_score(fr)

        vals = list(scores.values())
        print(f"[scores] {len(scores)} stations | moy={np.mean(vals):.1f} min={min(vals)} max={max(vals)}")

        out = tmp_dir / "station_scores.json"
        out.write_text(json.dumps({
            "scores": scores,
            "computed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "stats": {
                "mean": round(float(np.mean(vals)), 1),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
        }, ensure_ascii=False))
        storage.upload_asset(storage.RELEASE_LIVE, out, "station_scores.json")
        print("[scores] Done")


if __name__ == "__main__":
    run()
