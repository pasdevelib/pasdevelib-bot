"""Récupération de la météo via Open-Meteo (Archive + Forecast).

Pas de clé API. Pour Paris, on prend un point unique (Châtelet) :
ça suffit largement, la météo varie peu intra-muros.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

PARIS_LAT = 48.8566
PARIS_LON = 2.3522

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "cloud_cover",
    "relative_humidity_2m",
]

USER_AGENT = "pasdevelib-bot/0.1 (+https://pasdevelib.fr)"


def _to_dataframe(payload: dict) -> pd.DataFrame:
    """Transforme la réponse Open-Meteo en DataFrame horaire."""
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.tz_localize("Europe/Paris", ambiguous="infer").dt.tz_convert("UTC")
    df = df.rename(columns={"time": "ts"})
    return df


def fetch_archive(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Météo historique (lag de ~5 jours par rapport à aujourd'hui)."""
    params = {
        "latitude": PARIS_LAT,
        "longitude": PARIS_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Paris",
    }
    r = requests.get(
        ARCHIVE_URL, params=params,
        headers={"User-Agent": USER_AGENT}, timeout=60,
    )
    r.raise_for_status()
    return _to_dataframe(r.json())


def fetch_forecast(days: int = 7) -> pd.DataFrame:
    """Prévisions horaires (jusqu'à 16 jours, on se limite à 7)."""
    params = {
        "latitude": PARIS_LAT,
        "longitude": PARIS_LON,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": min(days, 16),
        "timezone": "Europe/Paris",
    }
    r = requests.get(
        FORECAST_URL, params=params,
        headers={"User-Agent": USER_AGENT}, timeout=60,
    )
    r.raise_for_status()
    return _to_dataframe(r.json())


def fetch_yesterday() -> pd.DataFrame:
    """Récupère la météo de la veille (job quotidien d'enrichissement)."""
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    return fetch_archive(yesterday, yesterday)
