"""Job scrape : appelé toutes les 5 minutes par GitHub Actions.

1. Fetch GBFS station_status
2. Append au parquet du jour courant (release `live`)
3. (1x/jour) Refresh stations.json
"""
from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path

from pasdevelib import fetch, storage


CURRENT_ASSET = "current_day.parquet"
STATIONS_ASSET = "stations.json"


def run() -> None:
    snap = fetch.fetch_snapshot()
    today_utc = snap.fetched_at.date().isoformat()

    # On ajoute la date pour pouvoir détecter le rollover dans consolidate.py
    snap.status["date"] = today_utc

    storage.ensure_release(storage.RELEASE_LIVE, "Live data (current day)")

    # Append au parquet du jour
    storage.append_to_parquet(
        storage.RELEASE_LIVE,
        CURRENT_ASSET,
        snap.status,
    )

    # On rafraîchit stations.json une fois par jour (à minuit UTC)
    if snap.fetched_at.hour == 0 and snap.fetched_at.minute < 5:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / STATIONS_ASSET
            stations = snap.info.to_dict(orient="records")
            path.write_text(json.dumps(stations, ensure_ascii=False, indent=2))
            storage.upload_asset(storage.RELEASE_LIVE, path, STATIONS_ASSET)

    print(f"[scrape] {snap.fetched_at.isoformat()} | {len(snap.status)} stations | OK")


if __name__ == "__main__":
    run()
