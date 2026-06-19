"""Job scrape : appele toutes les minutes par GitHub Actions."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from pasdevelib import fetch, storage
from pasdevelib.flux import run as run_flux

CURRENT_ASSET = "current_day.parquet"
STATIONS_ASSET = "stations.json"
PREV_ASSET = "snapshot_prev.parquet"


def run() -> None:
    snap = fetch.fetch_snapshot()
    today_utc = snap.fetched_at.date().isoformat()
    snap.status["date"] = today_utc

    storage.ensure_release(storage.RELEASE_LIVE, "Live data (current day)")

    # Charger le snapshot precedent
    prev_df = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        prev_path = tmp_dir / PREV_ASSET
        if storage.download_asset(storage.RELEASE_LIVE, PREV_ASSET, prev_path):
            prev_df = pd.read_parquet(prev_path)

    # Calculer les flux
    if not prev_df.empty:
        try:
            run_flux(prev_df, snap.status)
        except Exception as e:
            print(f"[scrape] flux error (non-blocking): {e}")

    # Sauvegarder snapshot courant comme precedent
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        prev_path = tmp_dir / PREV_ASSET
        snap.status.to_parquet(prev_path, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_LIVE, prev_path, PREV_ASSET)

    # Accumuler dans current_day.parquet
    storage.append_to_parquet(
        storage.RELEASE_LIVE,
        CURRENT_ASSET,
        snap.status,
    )

    # Mettre a jour stations.json
    if snap.info is not None and len(snap.info) > 0:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / STATIONS_ASSET
            stations = snap.info.to_dict(orient="records")
            path.write_text(json.dumps(stations, ensure_ascii=False,
                                       indent=2, default=str))
            storage.upload_asset(storage.RELEASE_LIVE, path, STATIONS_ASSET)
    else:
        print("[scrape] WARNING: snap.info is empty, skipping stations.json")

    print(f"[scrape] {snap.fetched_at.isoformat()} | "
          f"{len(snap.status)} stations | OK")


if __name__ == "__main__":
    run()
