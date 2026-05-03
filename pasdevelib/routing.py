"""Itinéraire vélo Vélib' : trouve un trajet A → B avec une station de départ
qui a un vélo et une station d'arrivée qui a une place.

Pipeline :
1. Pré-filtrage Haversine : N stations candidates autour de A et de B
2. Filtre par disponibilité (live) ou par prédiction (modèle analogue)
3. Raffinement OpenRouteService : distances de marche réelles
4. Scoring combiné : confiance × marche
5. Génération de deeplinks Apple Plans + Google Maps
"""
from __future__ import annotations

import datetime as dt
import math
import os
import urllib.parse
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Iterable

import requests

ORS_BASE = "https://api.openrouteservice.org"
ORS_API_KEY_ENV = "ORS_API_KEY"

# Vitesses moyennes pour estimation ETA
WALK_SPEED_M_PER_S = 1.4   # ~5 km/h
BIKE_SPEED_M_PER_S = 4.0   # ~14.5 km/h


class TripMode(str, Enum):
    NOW = "now"          # disponibilité temps réel
    LATER = "later"      # prédiction modèle analogue


@dataclass
class Coord:
    lat: float
    lon: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class StationCandidate:
    station_id: str
    name: str
    coord: Coord
    walk_distance_m: float        # raffinée par ORS si dispo, sinon Haversine
    proba: float                  # proba_velib pour départ, proba_place pour arrivée
    n_available: int              # vélos ou places (live si dispo)


@dataclass
class TripSegment:
    mode: str                     # "walk" | "bike"
    from_coord: Coord
    to_coord: Coord
    distance_m: float
    duration_s: float
    label: str                    # "Marcher jusqu'à la station Bastille"
    deeplink_apple: str
    deeplink_google: str


@dataclass
class TripPlan:
    start_station: StationCandidate
    end_station: StationCandidate
    segments: list[TripSegment]
    total_walk_m: float
    total_bike_m: float
    total_duration_s: float
    confidence: float             # proba_velib × proba_place
    score: float                  # critère de tri global

    def to_dict(self) -> dict:
        return {
            "start_station": asdict(self.start_station),
            "end_station": asdict(self.end_station),
            "segments": [asdict(s) for s in self.segments],
            "total_walk_m": self.total_walk_m,
            "total_bike_m": self.total_bike_m,
            "total_duration_s": self.total_duration_s,
            "confidence": self.confidence,
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# Géométrie
# ---------------------------------------------------------------------------

def haversine_m(a: Coord, b: Coord) -> float:
    """Distance vol d'oiseau en mètres."""
    R = 6_371_000
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# OpenRouteService
# ---------------------------------------------------------------------------

def _ors_headers() -> dict[str, str]:
    key = os.environ.get(ORS_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {ORS_API_KEY_ENV} env var to use ORS")
    return {
        "Authorization": key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def ors_walking_matrix(
    origins: list[Coord],
    destinations: list[Coord],
) -> list[list[float]]:
    """Matrice des distances piétonnes (mètres) entre origines et destinations.

    Un seul appel ORS pour O×D distances. Free tier : 500 matrices/jour,
    jusqu'à 50 points par matrice.
    """
    locations = [[c.lon, c.lat] for c in origins + destinations]
    n_origins = len(origins)
    sources = list(range(n_origins))
    dests = list(range(n_origins, n_origins + len(destinations)))

    body = {
        "locations": locations,
        "sources": sources,
        "destinations": dests,
        "metrics": ["distance"],
        "units": "m",
    }
    r = requests.post(
        f"{ORS_BASE}/v2/matrix/foot-walking",
        json=body, headers=_ors_headers(), timeout=30,
    )
    r.raise_for_status()
    return r.json()["distances"]


def ors_cycling_distance(start: Coord, end: Coord) -> tuple[float, float]:
    """Distance + durée vélo (m, s) via ORS cycling-regular."""
    body = {
        "coordinates": [[start.lon, start.lat], [end.lon, end.lat]],
        "instructions": False,
    }
    r = requests.post(
        f"{ORS_BASE}/v2/directions/cycling-regular",
        json=body, headers=_ors_headers(), timeout=30,
    )
    r.raise_for_status()
    summary = r.json()["routes"][0]["summary"]
    return summary["distance"], summary["duration"]


# ---------------------------------------------------------------------------
# Deeplinks
# ---------------------------------------------------------------------------

def google_maps_link(origin: Coord, dest: Coord, mode: str = "walking") -> str:
    """Deeplink Google Maps universel (web, iOS, Android)."""
    params = {
        "api": "1",
        "origin": f"{origin.lat},{origin.lon}",
        "destination": f"{dest.lat},{dest.lon}",
        "travelmode": mode,
    }
    return f"https://www.google.com/maps/dir/?{urllib.parse.urlencode(params)}"


def apple_maps_link(origin: Coord, dest: Coord, mode: str = "walking") -> str:
    """Deeplink Apple Plans. dirflg : w=walk, d=drive, r=transit, b=bike (iOS 18+)."""
    flag = {"walking": "w", "bicycling": "b", "driving": "d", "transit": "r"}[mode]
    params = {
        "saddr": f"{origin.lat},{origin.lon}",
        "daddr": f"{dest.lat},{dest.lon}",
        "dirflg": flag,
    }
    return f"http://maps.apple.com/?{urllib.parse.urlencode(params)}"


# ---------------------------------------------------------------------------
# Pré-filtrage
# ---------------------------------------------------------------------------

def nearby_stations(
    point: Coord,
    stations: list[dict],
    max_walk_m: int = 600,
    top_n: int = 10,
) -> list[tuple[dict, float]]:
    """Stations dans un rayon Haversine, triées par distance vol d'oiseau."""
    scored = []
    for s in stations:
        d = haversine_m(point, Coord(s["lat"], s["lon"]))
        if d <= max_walk_m:
            scored.append((s, d))
    scored.sort(key=lambda x: x[1])
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def plan_trip(
    origin: Coord,
    destination: Coord,
    stations: list[dict],
    departure_states: dict[str, dict],   # station_id -> {proba_velib, proba_place, num_bikes, num_docks}
    arrival_states: dict[str, dict] | None = None,   # même format, à l'heure d'arrivée estimée
    mode: TripMode = TripMode.NOW,
    target_time: dt.datetime | None = None,
    max_walk_m: int = 600,
    top_n: int = 10,
    n_results: int = 3,
    min_proba: float = 0.5,
    use_ors: bool = True,
) -> list[TripPlan]:
    """Renvoie les meilleurs itinéraires Vélib' A → B.

    `stations`           : liste de dicts `{station_id, name, lat, lon}` (depuis stations.json)
    `departure_states`   : état des stations au moment du départ (live ou prédit)
    `arrival_states`     : état au moment de l'arrivée (peut être identique en mode NOW)
    """
    if arrival_states is None:
        arrival_states = departure_states

    # 1. Pré-filtrage Haversine
    starts = nearby_stations(origin, stations, max_walk_m, top_n)
    ends = nearby_stations(destination, stations, max_walk_m, top_n)

    if not starts or not ends:
        return []

    # 2. Filtre disponibilité
    starts = [(s, d) for s, d in starts
              if departure_states.get(s["station_id"], {}).get("proba_velib", 0) >= min_proba]
    ends = [(s, d) for s, d in ends
            if arrival_states.get(s["station_id"], {}).get("proba_place", 0) >= min_proba]

    if not starts or not ends:
        return []

    # 3. Raffinement ORS (distances marche réelles)
    if use_ors and os.environ.get(ORS_API_KEY_ENV):
        try:
            origin_to_starts = ors_walking_matrix(
                [origin],
                [Coord(s["lat"], s["lon"]) for s, _ in starts],
            )[0]
            ends_to_dest = ors_walking_matrix(
                [Coord(s["lat"], s["lon"]) for s, _ in ends],
                [destination],
            )
            ends_to_dest = [row[0] for row in ends_to_dest]
        except Exception as e:
            print(f"[routing] ORS failed, fallback Haversine: {e}")
            origin_to_starts = [d for _, d in starts]
            ends_to_dest = [d for _, d in ends]
    else:
        origin_to_starts = [d for _, d in starts]
        ends_to_dest = [d for _, d in ends]

    # 4. Scoring combinatoire
    plans: list[TripPlan] = []
    for (sa, _), walk_in in zip(starts, origin_to_starts):
        for (sb, _), walk_out in zip(ends, ends_to_dest):
            if sa["station_id"] == sb["station_id"]:
                continue

            sa_state = departure_states.get(sa["station_id"], {})
            sb_state = arrival_states.get(sb["station_id"], {})
            proba_velib = sa_state.get("proba_velib", 0)
            proba_place = sb_state.get("proba_place", 0)
            confidence = proba_velib * proba_place

            sa_coord = Coord(sa["lat"], sa["lon"])
            sb_coord = Coord(sb["lat"], sb["lon"])
            bike_dist = haversine_m(sa_coord, sb_coord) * 1.3   # facteur réseau
            bike_dur = bike_dist / BIKE_SPEED_M_PER_S
            walk_in_dur = walk_in / WALK_SPEED_M_PER_S
            walk_out_dur = walk_out / WALK_SPEED_M_PER_S
            total_walk = walk_in + walk_out
            total_dur = walk_in_dur + bike_dur + walk_out_dur

            # Score : on veut maximiser confiance et minimiser durée totale
            score = confidence / max(total_dur, 60) * 1000

            # Construction des segments
            seg1 = TripSegment(
                mode="walk",
                from_coord=origin, to_coord=sa_coord,
                distance_m=walk_in, duration_s=walk_in_dur,
                label=f"Marcher jusqu'à {sa['name']}",
                deeplink_apple=apple_maps_link(origin, sa_coord, "walking"),
                deeplink_google=google_maps_link(origin, sa_coord, "walking"),
            )
            seg2 = TripSegment(
                mode="bike",
                from_coord=sa_coord, to_coord=sb_coord,
                distance_m=bike_dist, duration_s=bike_dur,
                label=f"Pédaler de {sa['name']} à {sb['name']}",
                deeplink_apple=apple_maps_link(sa_coord, sb_coord, "bicycling"),
                deeplink_google=google_maps_link(sa_coord, sb_coord, "bicycling"),
            )
            seg3 = TripSegment(
                mode="walk",
                from_coord=sb_coord, to_coord=destination,
                distance_m=walk_out, duration_s=walk_out_dur,
                label=f"Marcher de {sb['name']} à destination",
                deeplink_apple=apple_maps_link(sb_coord, destination, "walking"),
                deeplink_google=google_maps_link(sb_coord, destination, "walking"),
            )

            plans.append(TripPlan(
                start_station=StationCandidate(
                    station_id=sa["station_id"], name=sa["name"], coord=sa_coord,
                    walk_distance_m=walk_in, proba=proba_velib,
                    n_available=sa_state.get("num_bikes_available", 0),
                ),
                end_station=StationCandidate(
                    station_id=sb["station_id"], name=sb["name"], coord=sb_coord,
                    walk_distance_m=walk_out, proba=proba_place,
                    n_available=sb_state.get("num_docks_available", 0),
                ),
                segments=[seg1, seg2, seg3],
                total_walk_m=total_walk, total_bike_m=bike_dist,
                total_duration_s=total_dur, confidence=confidence, score=score,
            ))

    plans.sort(key=lambda p: p.score, reverse=True)
    return plans[:n_results]
