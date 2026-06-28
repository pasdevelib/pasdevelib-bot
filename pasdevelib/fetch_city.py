"""fetch_city.py — Scraper générique pour n'importe quel système GBFS.

Supporte : Paris (Vélib'), Bordeaux (Vcub), Lyon (Vélo'v)
et tout système suivant le standard GBFS v2.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Optional

import pandas as pd
import requests

from pasdevelib.cities import CityConfig, get_city

USER_AGENT = "pasdevelib-bot/2.0 (+https://pasdevelib.app)"
TIMEOUT = 30
MAX_RETRIES = 3


def _get(url: str, session: requests.Session) -> dict:
    """GET avec retry et User-Agent."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT,
                           headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def fetch_gbfs(city: CityConfig) -> Optional[pd.DataFrame]:
    """Scrape une ville via son endpoint GBFS standard.

    Retourne un DataFrame avec les colonnes :
    station_id, city_id, name, lat, lon,
    num_bikes_available, num_bikes_mechanical, num_bikes_ebike,
    num_docks_available, is_installed, is_renting, is_returning,
    last_reported, fetched_at
    """
    session = requests.Session()
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()

    try:
        # Station information (nom, coordonnées, capacité)
        info_url = f"{city.gbfs_base}/{city.station_info_path}"
        info_data = _get(info_url, session)
        stations = {
            s["station_id"]: s
            for s in info_data.get("data", {}).get("stations", [])
        }

        # Station status (disponibilités en temps réel)
        status_url = f"{city.gbfs_base}/{city.station_status_path}"
        status_data = _get(status_url, session)
        statuses = status_data.get("data", {}).get("stations", [])

        if not statuses:
            return None

        rows = []
        for s in statuses:
            sid = s.get("station_id", "")
            info = stations.get(sid, {})

            # Vélos mécaniques vs électriques
            # GBFS v2.2+ expose num_bikes_available par type
            vehicle_types = s.get("vehicle_types_available", [])
            n_mec = 0
            n_elec = 0
            for vt in vehicle_types:
                vtype = vt.get("vehicle_type_id", "")
                count = vt.get("count", 0)
                if "electric" in vtype.lower() or "ebike" in vtype.lower():
                    n_elec += count
                else:
                    n_mec += count

            # Fallback si vehicle_types non dispo
            n_total = s.get("num_bikes_available", 0)
            if n_mec == 0 and n_elec == 0:
                # Essayer num_docks_available et mechanical/ebike directs
                n_mec = s.get("num_bikes_available_types", {}).get("mechanical", n_total)
                n_elec = s.get("num_bikes_available_types", {}).get("ebike", 0)
                if n_mec + n_elec == 0:
                    n_mec = n_total  # tout mécanique par défaut

            rows.append({
                "station_id": f"{city.city_id}_{sid}",  # préfixe city_id pour éviter collisions
                "city_id": city.city_id,
                "name": info.get("name", s.get("name", "")),
                "lat": info.get("lat", s.get("lat", 0.0)),
                "lon": info.get("lon", s.get("lon", 0.0)),
                "capacity": info.get("capacity", s.get("capacity", 0)),
                "num_bikes_available": n_total,
                "num_bikes_mechanical": n_mec,
                "num_bikes_ebike": n_elec,
                "num_docks_available": s.get("num_docks_available", 0),
                "is_installed": int(s.get("is_installed", 1)),
                "is_renting": int(s.get("is_renting", 1)),
                "is_returning": int(s.get("is_returning", 1)),
                "last_reported": s.get("last_reported", 0),
                "fetched_at": fetched_at,
            })

        return pd.DataFrame(rows) if rows else None

    except Exception as e:
        print(f"[fetch_city] Erreur {city.city_id}: {e}")
        return None


def fetch_all_cities(city_ids: list[str]) -> pd.DataFrame:
    """Scrape toutes les villes et retourne un DataFrame combiné."""
    dfs = []
    for city_id in city_ids:
        city = get_city(city_id)
        print(f"[fetch_city] Scraping {city.city_name} ({city.operator})...")
        df = fetch_gbfs(city)
        if df is not None and not df.empty:
            dfs.append(df)
            print(f"  → {len(df)} stations")
        else:
            print(f"  → Échec ou données vides")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
