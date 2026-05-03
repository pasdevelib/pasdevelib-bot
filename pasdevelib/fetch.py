"""Récupération des données Vélib' via l'API GBFS."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

GBFS_BASE = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole"
STATION_INFO_URL = f"{GBFS_BASE}/station_information.json"
STATION_STATUS_URL = f"{GBFS_BASE}/station_status.json"

USER_AGENT = "pasdevelib-bot/0.1 (+https://pasdevelib.fr)"
TIMEOUT = 30


@dataclass
class Snapshot:
    """Un snapshot complet à un instant t."""
    fetched_at: dt.datetime
    status: pd.DataFrame
    info: pd.DataFrame | None = None


def _http_get(url: str) -> dict[str, Any]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_station_information() -> pd.DataFrame:
    """Caractéristiques statiques des stations (nom, lat, lon, capacité)."""
    raw = _http_get(STATION_INFO_URL)
    rows = raw["data"]["stations"]
    df = pd.DataFrame(rows)
    df["station_id"] = df["station_id"].astype(str)
    return df[["station_id", "name", "lat", "lon", "capacity", "stationCode"]]


def fetch_station_status() -> pd.DataFrame:
    """État dynamique de chaque station (vélos dispo, bornes libres)."""
    raw = _http_get(STATION_STATUS_URL)
    last_updated = dt.datetime.fromtimestamp(raw["last_updated"], tz=dt.timezone.utc)
    rows = raw["data"]["stations"]

    records = []
    for s in rows:
        # num_bikes_available_types est une liste de dicts, on l'aplatit
        bikes_types = {k: v for d in s.get("num_bikes_available_types", []) for k, v in d.items()}
        records.append({
            "station_id": str(s["station_id"]),
            "fetched_at": last_updated,
            "num_bikes_available": s["num_bikes_available"],
            "num_bikes_mechanical": bikes_types.get("mechanical", 0),
            "num_bikes_ebike": bikes_types.get("ebike", 0),
            "num_docks_available": s["num_docks_available"],
            "is_installed": bool(s["is_installed"]),
            "is_renting": bool(s["is_renting"]),
            "is_returning": bool(s["is_returning"]),
            "last_reported": dt.datetime.fromtimestamp(
                s["last_reported"], tz=dt.timezone.utc
            ) if s.get("last_reported") else None,
        })
    return pd.DataFrame.from_records(records)


def fetch_snapshot() -> Snapshot:
    """Snapshot complet (status + info)."""
    return Snapshot(
        fetched_at=dt.datetime.now(dt.timezone.utc),
        status=fetch_station_status(),
        info=fetch_station_information(),
    )
