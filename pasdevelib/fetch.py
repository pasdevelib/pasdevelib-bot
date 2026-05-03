"""Récupération des données Vélib'.

Source primaire : OpenDataSoft de la Ville de Paris (stable depuis GitHub Actions).
Fallback       : API Smoove GBFS officielle (peut bloquer les IPs GH Actions).
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

# Source primaire : OpenDataSoft (Ville de Paris)
ODS_BASE = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records"
ODS_PAGE_SIZE = 100

# Fallback : API Smoove GBFS officielle
GBFS_BASE = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole"
STATION_INFO_URL = f"{GBFS_BASE}/station_information.json"
STATION_STATUS_URL = f"{GBFS_BASE}/station_status.json"

USER_AGENT = "pasdevelib-bot/0.1 (+https://pasdevelib.fr)"
TIMEOUT = 30
MAX_RETRIES = 3


@dataclass
class Snapshot:
    fetched_at: dt.datetime
    status: pd.DataFrame
    info: pd.DataFrame | None = None


def _http_get_with_retry(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET avec backoff exponentiel : 1s, 2s, 4s entre tentatives."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError, ValueError) as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                sleep_s = 2 ** attempt
                print(f"[fetch] {url} attempt {attempt + 1}/{MAX_RETRIES} failed ({type(e).__name__}), retrying in {sleep_s}s")
                time.sleep(sleep_s)
    raise last_exc  # type: ignore[misc]


# ============================================================================
# Source primaire : OpenDataSoft (paginée)
# ============================================================================

def _fetch_ods_all_records() -> list[dict[str, Any]]:
    """Récupère tous les enregistrements de l'API OpenDataSoft, en paginant."""
    # 1er appel pour avoir le total_count
    first = _http_get_with_retry(ODS_BASE, {"limit": ODS_PAGE_SIZE, "offset": 0})
    total = int(first.get("total_count", 0))
    results: list[dict[str, Any]] = list(first.get("results", []))

    # Pages suivantes
    offset = ODS_PAGE_SIZE
    while offset < total:
        page = _http_get_with_retry(ODS_BASE, {"limit": ODS_PAGE_SIZE, "offset": offset})
        results.extend(page.get("results", []))
        offset += ODS_PAGE_SIZE

    return results


def _ods_to_status_df(records: list[dict[str, Any]], fetched_at: dt.datetime) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "station_id": str(r.get("stationcode", "")),
            "fetched_at": fetched_at,
            "num_bikes_available": int(r.get("numbikesavailable", 0) or 0),
            "num_bikes_mechanical": int(r.get("mechanical", 0) or 0),
            "num_bikes_ebike": int(r.get("ebike", 0) or 0),
            "num_docks_available": int(r.get("numdocksavailable", 0) or 0),
            "is_installed": str(r.get("is_installed", "")).upper() in ("OUI", "YES", "TRUE"),
            "is_renting": str(r.get("is_renting", "")).upper() in ("OUI", "YES", "TRUE"),
            "is_returning": str(r.get("is_returning", "")).upper() in ("OUI", "YES", "TRUE"),
            "last_reported": pd.to_datetime(r.get("duedate"), errors="coerce", utc=True),
        })
    return pd.DataFrame.from_records(rows)


def _ods_to_info_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        coord = r.get("coordonnees_geo") or {}
        # OpenDataSoft renvoie parfois un dict {lat, lon}, parfois une liste [lat, lon]
        if isinstance(coord, list) and len(coord) == 2:
            lat, lon = coord[0], coord[1]
        elif isinstance(coord, dict):
            lat = coord.get("lat")
            lon = coord.get("lon")
        else:
            lat, lon = None, None

        rows.append({
            "station_id": str(r.get("stationcode", "")),
            "name": r.get("name", ""),
            "lat": lat,
            "lon": lon,
            "capacity": int(r.get("capacity", 0) or 0),
            "stationCode": str(r.get("stationcode", "")),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Fallback : Smoove GBFS
# ============================================================================

def _fetch_smoove_status() -> pd.DataFrame:
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


def _fetch_smoove_info() -> pd.DataFrame:
    raw = _http_get_with_retry(STATION_INFO_URL)
    rows = raw["data"]["stations"]
    df = pd.DataFrame(rows)
    df["station_id"] = df["station_id"].astype(str)
    return df[["station_id", "name", "lat", "lon", "capacity", "stationCode"]]


# ============================================================================
# Public API : essaie OpenDataSoft d'abord, fallback Smoove
# ============================================================================

def fetch_snapshot() -> Snapshot:
    fetched_at = dt.datetime.now(dt.timezone.utc)
    try:
        print("[fetch] trying OpenDataSoft (primary)...")
        records = _fetch_ods_all_records()
        print(f"[fetch] OpenDataSoft OK : {len(records)} stations")
        return Snapshot(
            fetched_at=fetched_at,
            status=_ods_to_status_df(records, fetched_at),
            info=_ods_to_info_df(records),
        )
    except Exception as e:
        print(f"[fetch] OpenDataSoft failed: {e}, trying Smoove fallback")
        return Snapshot(
            fetched_at=fetched_at,
            status=_fetch_smoove_status(),
            info=_fetch_smoove_info(),
        )


def fetch_station_status() -> pd.DataFrame:
    """Compatibilité avec l'ancienne API."""
    return fetch_snapshot().status


def fetch_station_information() -> pd.DataFrame:
    """Compatibilité avec l'ancienne API."""
    info = fetch_snapshot().info
    assert info is not None
    return info
