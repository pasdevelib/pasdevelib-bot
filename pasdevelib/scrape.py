"""Job scrape : appele toutes les 5 minutes par GitHub Actions."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pasdevelib import fetch, storage


CURRENT_ASSET = "current_day.parquet"
STATIONS_ASSET = "stations.json"


def run() -> None:
    snap = fetch.fetch_snapshot()
    today_utc = snap.fetched_at.date().isoformat()
    snap.status["date"] = today_utc

    storage.ensure_release(storage.RELEASE_LIVE, "Live data (current day)")

    storage.append_to_parquet(
        storage.RELEASE_LIVE,
        CURRENT_ASSET,
        snap.status,
    )

    print(f"[scrape] info type: {type(snap.info)}")
    if snap.info is not None:
        print(f"[scrape] info len: {len(snap.info)}")

    if snap.info is not None and len(snap.info) > 0:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / STATIONS_ASSET
            stations = snap.info.to_dict(orient="records")
            path.write_text(json.dumps(stations, ensure_ascii=False, indent=2, default=str))
            print(f"[scrape] uploading {STATIONS_ASSET} ({path.stat().st_size} bytes, {len(stations)} stations)")
            storage.upload_asset(storage.RELEASE_LIVE, path, STATIONS_ASSET)
            print(f"[scrape] {STATIONS_ASSET} upload OK")
    else:
        print(f"[scrape] WARNING: snap.info is empty, skipping stations.json upload")

    print(f"[scrape] {snap.fetched_at.isoformat()} | {len(snap.status)} stations | OK")


if __name__ == "__main__":
    run()
