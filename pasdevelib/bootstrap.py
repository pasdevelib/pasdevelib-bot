"""Bootstrap one-shot : importe l'historique depuis lovasoa/historique-velib-opendata.

Le repo lovasoa snapshot l'API Vélib' toutes les 15 min depuis 2020 et publie
les données en zip dans sa release `latest`. Le nom des assets a changé :
ce ne sont plus `stations.zip` mais des `stations-YYYY-MM-DDTHHMMz.zip`
horodatés qui s'accumulent. On prend le plus récent.
"""
from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

from pasdevelib import storage

LOVASOA_REPO = "lovasoa/historique-velib-opendata"
LATEST_RELEASE_API = f"https://api.github.com/repos/{LOVASOA_REPO}/releases/latest"

USER_AGENT = "pasdevelib-bot/0.1 (+https://pasdevelib.fr)"


def find_latest_zip_url() -> str:
    """Interroge l'API GitHub pour trouver le zip le plus récent dans la release latest."""
    print(f"[bootstrap] querying {LATEST_RELEASE_API}")
    r = requests.get(
        LATEST_RELEASE_API,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
        timeout=30,
    )
    r.raise_for_status()
    release = r.json()
    assets = release.get("assets", [])
    # On garde les zips qui ressemblent à `stations-YYYY-MM-DDTHHMMz.zip` ou `stations.zip`
    zip_assets = [a for a in assets if a["name"].endswith(".zip") and a["name"].startswith("stations")]
    if not zip_assets:
        raise RuntimeError(f"No zip assets found in {LOVASOA_REPO} latest release")

    # Tri par date de création (plus récent en premier)
    zip_assets.sort(key=lambda a: a["created_at"], reverse=True)
    chosen = zip_assets[0]
    print(f"[bootstrap] chose asset {chosen['name']} ({chosen['size'] / 1e6:.1f} MB)")
    return chosen["browser_download_url"]


def download_dump(url: str) -> bytes:
    print(f"[bootstrap] downloading {url} ...")
    r = requests.get(url, timeout=600, stream=True, headers={"User-Agent": USER_AGENT})
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
    df["station_id"] = df["name"].astype(str)
    df["date"] = df["fetched_at"].dt.date.astype(str)
    return df[[
        "station_id", "fetched_at", "num_bikes_available",
        "num_bikes_mechanical", "num_bikes_ebike", "num_docks_available",
        "is_installed", "is_renting", "is_returning", "last_reported", "date",
    ]]


def run() -> None:
    storage.ensure_release(storage.RELEASE_HISTORY, "Historical daily snapshots")

    url = find_latest_zip_url()
    blob = download_dump(url)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        print(f"[bootstrap] {len(names)} CSV files in zip")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            all_dfs = []
            for n in names:
                with zf.open(n) as f:
                    try:
                        all_dfs.append(parse_csv(f.read()))
                    except Exception as e:
                        print(f"[bootstrap] skip {n}: {e}")
                        continue

            if not all_dfs:
                print("[bootstrap] no parsable CSV, abort")
                return

            big = pd.concat(all_dfs, ignore_index=True)
            print(f"[bootstrap] {len(big):,} rows total")

            for day, group in big.groupby("date"):
                group = group.drop(columns=["date"])
                out = tmp_dir / f"{day}.parquet"
                group.to_parquet(out, compression="snappy", index=False)
                storage.upload_asset(storage.RELEASE_HISTORY, out)
                print(f"[bootstrap] uploaded {day}.parquet ({len(group):,} rows)")


if __name__ == "__main__":
    run()
