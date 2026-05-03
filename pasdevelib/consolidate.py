"""Job de consolidation quotidien (03:00 UTC).

Prend le `current_day.parquet` de la release `live`, en extrait les lignes
de la veille, les archive dans la release `history` sous `YYYY-MM-DD.parquet`,
puis purge ces lignes du fichier live.
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import pandas as pd

from pasdevelib import storage
from pasdevelib.scrape import CURRENT_ASSET


def run() -> None:
    storage.ensure_release(storage.RELEASE_HISTORY, "Historical daily snapshots")

    yesterday = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date()
    yesterday_str = yesterday.isoformat()

    with tempfile.TemporaryDirectory() as tmp:
        live_path = Path(tmp) / CURRENT_ASSET
        downloaded = storage.download_asset(storage.RELEASE_LIVE, CURRENT_ASSET, live_path)
        if not downloaded or not live_path.exists():
            print("[consolidate] no live file, nothing to do")
            return

        df = pd.read_parquet(live_path)

        # Sépare hier vs aujourd'hui
        is_yesterday = df["date"] == yesterday_str
        archive = df[is_yesterday].drop(columns=["date"])
        keep = df[~is_yesterday]

        if archive.empty:
            print(f"[consolidate] no data for {yesterday_str}, skip")
            return

        # Upload de l'archive
        archive_path = Path(tmp) / f"{yesterday_str}.parquet"
        archive.to_parquet(archive_path, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_HISTORY, archive_path)
        print(f"[consolidate] archived {len(archive)} rows to {yesterday_str}.parquet")

        # Re-upload du live nettoyé
        keep.to_parquet(live_path, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_LIVE, live_path, CURRENT_ASSET)
        print(f"[consolidate] live file shrunk to {len(keep)} rows")


if __name__ == "__main__":
    run()
