"""cities.py — Configuration des systèmes de vélos en libre-service supportés.

Chaque ville a son propre endpoint GBFS (standard ouvert) et ses métadonnées.
Le scraper peut tourner sur toutes les villes en parallèle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CityConfig:
    """Configuration d'un système de vélos en libre-service."""
    city_id: str           # Identifiant unique ex: "paris", "bordeaux", "lyon"
    city_name: str         # Nom affiché
    country: str           # Code pays ISO
    operator: str          # Opérateur du service
    system_name: str       # Nom du système
    gbfs_base: str         # URL base GBFS
    station_info_path: str # Path station_information.json
    station_status_path: str # Path station_status.json
    # Source alternative (opendata si dispo)
    opendata_url: Optional[str] = None
    # Bbox approximative (lat_min, lon_min, lat_max, lon_max)
    bbox: Optional[tuple] = None
    # Timezone
    timezone: str = "Europe/Paris"


CITIES: dict[str, CityConfig] = {
    # ── Paris — Vélib' Métropole (Smovengo) ─────────────────────
    "paris": CityConfig(
        city_id="paris",
        city_name="Paris",
        country="FR",
        operator="Smovengo",
        system_name="Vélib' Métropole",
        gbfs_base="https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records",
        bbox=(48.7, 2.0, 49.0, 2.7),
        timezone="Europe/Paris",
    ),

    # ── Bordeaux — Vcub (JCDecaux / Keolis) ──────────────────────
    "bordeaux": CityConfig(
        city_id="bordeaux",
        city_name="Bordeaux",
        country="FR",
        operator="JCDecaux",
        system_name="Vcub",
        gbfs_base="https://transport.data.gouv.fr/gbfs/vcub-bordeaux-metropole",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://data.bordeaux-metropole.fr/api/datasets/1.0/ci_vcub_p/records",
        bbox=(44.75, -0.75, 44.95, -0.45),
        timezone="Europe/Paris",
    ),

    # ── Lyon — Vélo'v (JCDecaux / TCL) ──────────────────────────
    "lyon": CityConfig(
        city_id="lyon",
        city_name="Lyon",
        country="FR",
        operator="JCDecaux",
        system_name="Vélo'v",
        gbfs_base="https://transport.data.gouv.fr/gbfs/velov",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://download.data.grandlyon.com/wfs/rdata?SERVICE=WFS&VERSION=2.0.0&request=GetFeature&typename=jcd_jcdecaux.jcdvelovsimple&outputFormat=application/json&SRSNAME=urn:ogc:def:crs:EPSG::4171",
        bbox=(45.7, 4.75, 45.85, 5.0),
        timezone="Europe/Paris",
    ),

    # ── Toulouse — VélôToulouse (JCDecaux) ───────────────────────
    "toulouse": CityConfig(
        city_id="toulouse",
        city_name="Toulouse",
        country="FR",
        operator="JCDecaux",
        system_name="VélôToulouse",
        gbfs_base="https://transport.data.gouv.fr/gbfs/velo-toulouse",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://data.toulouse-metropole.fr/api/explore/v2.1/catalog/datasets/tiseo-arrets-et-stations/records",
        bbox=(43.55, 1.35, 43.65, 1.5),
        timezone="Europe/Paris",
    ),
}


def get_city(city_id: str) -> CityConfig:
    """Récupère la configuration d'une ville par son ID."""
    if city_id not in CITIES:
        raise ValueError(f"Ville inconnue: {city_id}. Villes disponibles: {list(CITIES.keys())}")
    return CITIES[city_id]


def list_cities() -> list[str]:
    """Retourne la liste des IDs de villes supportées."""
    return list(CITIES.keys())
