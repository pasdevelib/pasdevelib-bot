"""watchdog.py — Vérifie que le dernier scrape date de moins de 15 min.

Lit snapshot_prev.parquet depuis la release live-data et vérifie le timestamp.
Si le dernier scrape date de plus de 15 min, log une alerte.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from pasdevelib import storage


MAX_AGE_MINUTES = 15
ASSET = "snapshot_prev.parquet"


def run() -> None:
    import pandas as pd

    now = datetime.now(timezone.utc)
    print(f"[watchdog] {now.isoformat()}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ASSET
        if not storage.download_asset(storage.RELEASE_LIVE, ASSET, path):
            print("[watchdog] ⚠️  snapshot_prev.parquet introuvable — scraper possiblement en panne")
            sys.exit(1)

        df = pd.read_parquet(path)

        # Chercher la colonne de timestamp
        ts_col = None
        for col in ["fetched_at", "last_reported", "last_updated", "date"]:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None:
            print(f"[watchdog] colonnes: {list(df.columns)}")
            print("[watchdog] ⚠️  Aucune colonne timestamp trouvée")
            return

        last_ts = pd.to_datetime(df[ts_col]).max()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        age_minutes = (now - last_ts).total_seconds() / 60
        print(f"[watchdog] Dernier scrape : {last_ts.isoformat()} ({age_minutes:.1f} min)")

        if age_minutes > MAX_AGE_MINUTES:
            print(f"[watchdog] 🚨 ALERTE — dernier scrape il y a {age_minutes:.0f} min (seuil: {MAX_AGE_MINUTES} min)")
            sys.exit(1)
        else:
            print(f"[watchdog] ✅ OK — scraper actif")


if __name__ == "__main__":
    run()
