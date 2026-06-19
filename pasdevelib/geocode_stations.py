"""Géocode les stations Velib et enrichit stations.json avec le champ 'zone'.

Utilise api-adresse.data.gouv.fr en batch (50 par requête).

Usage : python -m pasdevelib.geocode_stations
"""
from __future__ import annotations

import csv
import io
import json
import time
import tempfile
from pathlib import Path

import requests

from pasdevelib import storage

STATIONS_ASSET = "stations.json"
CHUNK = 50
USER_AGENT = "pasdevelib-bot/0.1 (+https://pasdevelib.app)"
API_URL = "https://api-adresse.data.gouv.fr/reverse/csv/"


def get_zone(citycode: str, city: str) -> str:
    """Convertit citycode INSEE en label de zone."""
    citycode = citycode.strip()
    city = city.strip()
    if citycode.startswith("750") and len(citycode) == 5:
        arr = int(citycode[3:])
        suffix = "er" if arr == 1 else "e"
        return f"Paris {arr}{suffix}"
    return city if city else "Île-de-France"


def geocode_batch(stations: list[dict]) -> dict[str, str]:
    """Géocode un batch de stations via CSV, retourne {station_id: zone}."""
    result: dict[str, str] = {}

    # Construire le CSV avec header
    lines = ["longitude,latitude"]
    for s in stations:
        lon = s.get("lon", s.get("longitude", 0))
        lat = s.get("lat", s.get("latitude", 0))
        lines.append(f"{lon},{lat}")
    body = "\n".join(lines)

    try:
        r = requests.post(
            API_URL,
            params={"result_columns": "result_citycode,result_city"},
            data=body.encode("utf-8"),
            headers={"Content-Type": "text/csv", "User-Agent": USER_AGENT},
            timeout=30,
        )
        r.raise_for_status()

        # Parser le CSV proprement avec le module csv
        reader = csv.DictReader(io.StringIO(r.text))
        rows = list(reader)

        for i, row in enumerate(rows):
            if i >= len(stations):
                break
            s = stations[i]
            sid = str(s.get("station_id", s.get("stationcode", i)))
            citycode = row.get("result_citycode", "").strip()
            city = row.get("result_city", "").strip()
            result[sid] = get_zone(citycode, city)

        print(f"[geocode] batch ok — {len(rows)} lignes parsées")

    except Exception as e:
        print(f"[geocode] batch error: {e}")
        for s in stations:
            sid = str(s.get("station_id", s.get("stationcode", "")))
            result[sid] = "Paris"

    return result


def run() -> None:
    print("[geocode] downloading stations.json...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        path = tmp_dir / STATIONS_ASSET
        if not storage.download_asset(storage.RELEASE_LIVE, STATIONS_ASSET, path):
            print("[geocode] ERROR: could not download stations.json")
            return

        stations = json.loads(path.read_text())
        print(f"[geocode] {len(stations)} stations to geocode")

        zone_map: dict[str, str] = {}
        for i in range(0, len(stations), CHUNK):
            chunk = stations[i:i + CHUNK]
            zones = geocode_batch(chunk)
            zone_map.update(zones)
            print(f"[geocode] {min(i + CHUNK, len(stations))}/{len(stations)} done")
            time.sleep(0.3)

        # Injecter le champ zone
        for s in stations:
            sid = str(s.get("station_id", s.get("stationcode", "")))
            s["zone"] = zone_map.get(sid, "Paris")

        path.write_text(json.dumps(stations, ensure_ascii=False, indent=2))
        storage.upload_asset(storage.RELEASE_LIVE, path, STATIONS_ASSET)
        print("[geocode] done — stations.json enrichi")

        from collections import Counter
        zones = Counter(s["zone"] for s in stations)
        print("[geocode] top zones:")
        for zone, count in zones.most_common(20):
            print(f"  {zone}: {count}")


if __name__ == "__main__":
    run()
