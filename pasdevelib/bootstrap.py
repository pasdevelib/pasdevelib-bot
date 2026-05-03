"""Bootstrap one-shot : importe l'historique depuis lovasoa/historique-velib-opendata.

Le repo lovasoa publie un zip `stations.zip` sur sa release `latest`.
Le format historique est un CSV avec colonnes :
  capacity, available_mechanical, available_electrical,
  station_name, station_geo, operative, ts (en index)

On le transforme au schéma pasdevelib et on découpe en parquets quotidiens
qu'on uploade sur la release `history`.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

from pasdevelib import storage

LOVASOA_ZIP_URL = (
    "https://github.com/lovasoa/historique-velib-opendata/releases/"
    "download/latest/stations.zip"
)


def download_dump() -> bytes:
    print(f"[bootstrap] downloading {LOVASOA_ZIP_URL} ...")
    r = requests.get(LOVASOA_ZIP_URL, timeout=600, stream=True)
    r.raise_for_status()
    return r.content


def parse_csv(raw_csv: bytes) -> pd.DataFrame:
    """Parse le CSV historique lovasoa et normalise au schéma pasdevelib."""
    df = pd.read_csv(
        io.BytesIO(raw_csv),
        names=[
            "fetched_at", "capacity", "num_bikes_mechanical",
            "num_bikes_ebike", "name", "station_geo", "operative",
        ],
        parse_dates=["fetched_at"],
    )
    df["num_bikes_available"] = df["num_bikes_mechanical"] + df["num_bikes_ebike"]
    df["num_docks_available"] = (df["capacity"] - df["num_bikes_available"]).clip(lower=0)
    df["is_installed"] = True
    df["is_renting"] = df["operative"].astype(bool)
    df["is_returning"] = df["operative"].astype(bool)
    df["last_reported"] = df["fetched_at"]

    # station_id : lovasoa utilise le nom comme clé. On hash pour un id stable
    # (le mapping vers le station_id GBFS officiel se fera au join via le nom)
    df["station_id"] = df["name"].astype(str)

    df["date"] = df["fetched_at"].dt.date.astype(str)
    return df[[
        "station_id", "fetched_at", "num_bikes_available",
        "num_bikes_mechanical", "num_bikes_ebike", "num_docks_available",
        "is_installed", "is_renting", "is_returning", "last_reported", "date",
    ]]


def run() -> None:
    storage.ensure_release(storage.RELEASE_HISTORY, "Historical daily snapshots")

    blob = download_dump()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        print(f"[bootstrap] {len(names)} files in zip")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            all_dfs = []
            for n in names:
                if not n.endswith(".csv"):
                    continue
                with zf.open(n) as f:
                    all_dfs.append(parse_csv(f.read()))

            big = pd.concat(all_dfs, ignore_index=True)
            print(f"[bootstrap] {len(big):,} rows total")

            # Découpage par jour, upload de chaque parquet
            for day, group in big.groupby("date"):
                group = group.drop(columns=["date"])
                out = tmp_dir / f"{day}.parquet"
                group.to_parquet(out, compression="zstd", index=False)
                storage.upload_asset(storage.RELEASE_HISTORY, out)
                print(f"[bootstrap] uploaded {day}.parquet ({len(group):,} rows)")


if __name__ == "__main__":
    run()
