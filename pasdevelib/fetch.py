"""Récupération des données Vélib' via l'API GBFS, avec retry + fallback."""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

# Source primaire (officielle, GBFS 1.0, ~1500 stations Métropole)
GBFS_BASE = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole"
STATION_INFO_URL = f"{GBFS_BASE}/station_information.json"
STATION_STATUS_URL = f"{GBFS_BASE}/station_status.json"

# Fallback (Ville de Paris, OpenDataSoft, Paris intra-muros uniquement, ~900 stations)
FALLBACK_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "velib-disponibilite-en-temps-reel/exports/json"
)

USER_AGENT = "pasdevelib-bot/0.1 (+https://pasdevelib.fr)"
TIMEOUT = 30
MAX_RETRIES = 3


@dataclass
class Snapshot:
    fetched_at: dt.datetime
    status: pd.DataFrame
    info: pd.DataFrame | None = None


def _http_get_with_retry(url: str) -> dict[str, Any]:
    """GET avec backoff exponentiel : 1s, 2s, 4s entre tentatives."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                sleep_s = 2 ** attempt
                print(f"[fetch] attempt {attempt + 1}/{MAX_RETRIES} failed ({e}), retrying in {sleep_s}s")
                time.sleep(sleep_s)
    raise last_exc  # type: ignore[misc]


def fetch_station_information() -> pd.DataFrame:
    """Caractéristiques statiques des stations."""
    try:
        raw = _http_get_with_retry(STATION_INFO_URL)
        rows = raw["data"]["stations"]
        df = pd.DataFrame(rows)
        df["station_id"] = df["station_id"].astype(str)
        return df[["station_id", "name", "lat", "lon", "capacity", "stationCode"]]
    except Exception as e:
        print(f"[fetch] primary station_info failed: {e}, trying fallback")
        return _fallback_info()


def fetch_station_status() -> pd.DataFrame:
    """État dynamique de chaque station."""
    try:
        raw = _http_get_with_retry(STATION_STATUS_URL)
        last_updated = dt.datetime.fromtimestamp(raw["last_updated"], tz=dt.timezone.utc)
        rows = raw["data"]["stations"]

        records = []
        for s in rows:
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
    except Exception as e:
        print(f"[fetch] primary station_status failed: {e}, trying fallback")
        return _fallback_status()


def _fallback_payload() -> list[dict[str, Any]]:
    """Récupère le JSON OpenDataSoft (un seul endpoint qui combine info + status)."""
    return _http_get_with_retry(FALLBACK_URL)  # type: ignore[return-value]


def _fallback_info() -> pd.DataFrame:
    rows = _fallback_payload()
    records = []
    for r in rows:
        coord = r.get("coordonnees_geo", {}) or {}
        records.append({
            "station_id": str(r.get("stationcode", "")),
            "name": r.get("name", ""),
            "lat": coord.get("lat"),
            "lon": coord.get("lon"),
            "capacity": int(r.get("capacity", 0) or 0),
            "stationCode": str(r.get("stationcode", "")),
        })
    return pd.DataFrame(records)


def _fallback_status() -> pd.DataFrame:
    rows = _fallback_payload()
    now = dt.datetime.now(dt.timezone.utc)
    records = []
    for r in rows:
        records.append({
            "station_id": str(r.get("stationcode", "")),
            "fetched_at": now,
            "num_bikes_available": int(r.get("numbikesavailable", 0) or 0),
            "num_bikes_mechanical": int(r.get("mechanical", 0) or 0),
            "num_bikes_ebike": int(r.get("ebike", 0) or 0),
            "num_docks_available": int(r.get("numdocksavailable", 0) or 0),
            "is_installed": str(r.get("is_installed", "")).upper() == "OUI",
            "is_renting": str(r.get("is_renting", "")).upper() == "OUI",
            "is_returning": str(r.get("is_returning", "")).upper() == "OUI",
            "last_reported": pd.to_datetime(r.get("duedate"), errors="coerce"),
        })
    return pd.DataFrame.from_records(records)


def fetch_snapshot() -> Snapshot:
    return Snapshot(
        fetched_at=dt.datetime.now(dt.timezone.utc),
        status=fetch_station_status(),
        info=fetch_station_information(),
    )
