"""geocode_cities.py — Enrichit stations_cities.json avec un champ 'zone'
(quartier/arrondissement/commune), pour les villes secondaires
(Bordeaux, Lyon, Toulouse, Lille, Rennes, Strasbourg, Nantes,
Montpellier...). Meme mecanique que geocode_stations.py (Paris), juste
generalisee a stations_cities.json (release cities-live) au lieu de
stations.json (release live).

N'a besoin de tourner que rarement (hebdomadaire) : la liste des
stations change peu, contrairement aux donnees temps reel.

Usage : python -m pasdevelib.geocode_cities
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pasdevelib import storage
from pasdevelib.geocode_stations import geocode_batch, CHUNK

RELEASE_CITIES = "cities-live"
STATIONS_ASSET = "stations_cities.json"


def run() -> None:
    print("[geocode_cities] downloading stations_cities.json...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        path = tmp_dir / STATIONS_ASSET
        if not storage.download_asset(RELEASE_CITIES, STATIONS_ASSET, path):
            print("[geocode_cities] ERROR: could not download stations_cities.json")
            return

        stations = json.loads(path.read_text())
        # Ne regeocode pas les stations qui ont deja une zone (evite de
        # re-taper l'API a chaque run pour rien — la position d'une
        # station ne bouge quasiment jamais).
        todo = [s for s in stations if not s.get("zone")]
        print(f"[geocode_cities] {len(stations)} stations, {len(todo)} a geocoder")

        zone_map: dict[str, str] = {}
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            zones = geocode_batch(chunk, fallback_zone="Zone inconnue")
            zone_map.update(zones)
            print(f"[geocode_cities] {min(i + CHUNK, len(todo))}/{len(todo)} done")

        for s in stations:
            sid = str(s.get("station_id", ""))
            if sid in zone_map:
                s["zone"] = zone_map[sid]

        out_path = tmp_dir / STATIONS_ASSET
        out_path.write_text(json.dumps(stations, ensure_ascii=False))
        storage.upload_asset(RELEASE_CITIES, out_path, STATIONS_ASSET)
        print(f"[geocode_cities] stations_cities.json mis a jour ({len(zone_map)} nouvelles zones)")


if __name__ == "__main__":
    run()
