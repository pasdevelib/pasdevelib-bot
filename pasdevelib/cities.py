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
    # BUG CORRIGE : gbfs_base pointait vers transport.data.gouv.fr, dont le
    # proxy GBFS pour Bordeaux echoue systematiquement (404/timeout) —
    # confirme par 3 semaines de scrapes en echec silencieux (fetch_city.py
    # avale l'exception et renvoie None, jamais remonte). Bordeaux n'expose
    # de toute facon pas un vrai flux GBFS standard : c'est un JSON "Explore
    # v2" propre a l'opendata de Bordeaux Metropole (deja utilise et
    # verifie fonctionnel par app/api/cities-now/route.ts cote webapp) —
    # necessite un parseur dedie, cf. fetch_bordeaux() plus bas.
    "bordeaux": CityConfig(
        city_id="bordeaux",
        city_name="Bordeaux",
        country="FR",
        operator="JCDecaux",
        system_name="Vcub",
        gbfs_base="https://transport.data.gouv.fr/gbfs/vcub-bordeaux-metropole",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://opendata.bordeaux-metropole.fr/api/explore/v2.1/catalog/datasets/ci_vcub_p/exports/json?limit=500&timezone=Europe%2FParis",
        bbox=(44.75, -0.75, 44.95, -0.45),
        timezone="Europe/Paris",
    ),

    # ── Lyon — Vélo'v (JCDecaux / TCL) ──────────────────────────
    # Meme situation que Bordeaux : pas un vrai flux GBFS standard, un
    # GeoJSON OGC Features propre a data.grandlyon.com — parseur dedie,
    # cf. fetch_lyon() plus bas.
    "lyon": CityConfig(
        city_id="lyon",
        city_name="Lyon",
        country="FR",
        operator="JCDecaux",
        system_name="Vélo'v",
        gbfs_base="https://transport.data.gouv.fr/gbfs/velov",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://data.grandlyon.com/fr/geoserv/ogc/features/v1/collections/metropole-de-lyon:jcd_jcdecaux.jcdvelov/items?crs=EPSG:4171&f=application/geo%2Bjson&limit=500&startIndex=0&sortby=gid",
        bbox=(45.7, 4.75, 45.85, 5.0),
        timezone="Europe/Paris",
    ),

    # ── Toulouse — VélôToulouse (JCDecaux) ───────────────────────
    # BUG CORRIGE : gbfs_base pointait vers transport.data.gouv.fr (meme
    # probleme que Bordeaux/Lyon) — Toulouse expose pourtant un VRAI flux
    # GBFS standard, juste a la mauvaise adresse. Reste sur fetch_gbfs()
    # generique, aucun parseur dedie necessaire ici.
    "toulouse": CityConfig(
        city_id="toulouse",
        city_name="Toulouse",
        country="FR",
        operator="JCDecaux",
        system_name="VélôToulouse",
        gbfs_base="https://api.cyclocity.fr/contracts/toulouse/gbfs",
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
