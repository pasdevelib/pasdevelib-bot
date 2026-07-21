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

    # ── Lille — V'Lille (Ilévia / MEL) ───────────────────────────
    # Vrai GBFS standard (v2.3), verifie 100% disponible sur
    # transport.data.gouv.fr avant de s'engager — pas de parseur dedie
    # necessaire, fetch_gbfs() generique suffit (comme Toulouse).
    "lille": CityConfig(
        city_id="lille",
        city_name="Lille",
        country="FR",
        operator="Ilévia",
        system_name="V'Lille",
        gbfs_base="https://media.ilevia.fr/opendata",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://media.ilevia.fr/opendata/gbfs.json",
        bbox=(50.55, 2.95, 50.70, 3.20),
        timezone="Europe/Paris",
    ),

    # ── Rennes — le vélo STAR (Keolis Rennes) ────────────────────
    # Vrai GBFS standard (v1.0), verifie 100% disponible.
    "rennes": CityConfig(
        city_id="rennes",
        city_name="Rennes",
        country="FR",
        operator="Keolis Rennes",
        system_name="STAR",
        gbfs_base="https://eu.ftp.opendatasoft.com/star/gbfs",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://eu.ftp.opendatasoft.com/star/gbfs/gbfs.json",
        bbox=(48.05, -1.75, 48.15, -1.60),
        timezone="Europe/Paris",
    ),

    # ── Strasbourg — Vélhop (nextbike) ────────────────────────────
    # Vrai GBFS standard (v2.3), verifie 100% disponible. Attention :
    # systeme sans bornes classiques (cadenas sur arceau, boucle) — le
    # GBFS de nextbike remonte quand meme num_docks_available de facon
    # standard, aucune adaptation necessaire cote parsing.
    "strasbourg": CityConfig(
        city_id="strasbourg",
        city_name="Strasbourg",
        country="FR",
        operator="nextbike",
        system_name="Vélhop",
        # BUG CORRIGE ICI : nextbike sert ses fichiers GBFS sous un
        # sous-dossier de langue (/fr/), pas a la racine — confirme par
        # l'export CSV officiel data.strasbourg.eu qui liste les vraies
        # URLs (.../nextbike_ae/fr/station_information.json).
        gbfs_base="https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_ae/fr",
        station_info_path="station_information.json",
        station_status_path="station_status.json",
        opendata_url="https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_ae/gbfs.json",
        bbox=(48.50, 7.65, 48.65, 7.85),
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


def city_center_latlon(city_id: str) -> tuple[float, float]:
    """Point (lat, lon) représentatif d'une ville — centroïde de son bbox.

    Suffisant pour la météo (comme note Paris déjà : la météo varie peu
    a l'echelle d'une ville), evite d'avoir a maintenir un point dedie
    par ville en plus du bbox deja present.
    """
    city = get_city(city_id)
    if city.bbox is None:
        raise ValueError(f"Pas de bbox défini pour {city_id}, impossible de calculer un centre.")
    lat_min, lon_min, lat_max, lon_max = city.bbox
    return (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0
