"""Features calendaires : jours fériés et vacances scolaires (zone C, Paris).

On précalcule sur la fenêtre 2023-2027 et on stocke en parquet.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

JOURS_FERIES_URL = "https://calendrier.api.gouv.fr/jours-feries/metropole.json"

# Vacances scolaires zone C (Paris, Versailles, Créteil, Bordeaux)
# Source : https://data.education.gouv.fr/explore/dataset/fr-en-calendrier-scolaire/
VACANCES_API = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"

USER_AGENT = "pasdevelib-bot/0.1 (+https://pasdevelib.fr)"


def fetch_jours_feries() -> pd.DataFrame:
    """Tous les jours fériés métropole."""
    r = requests.get(JOURS_FERIES_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = [{"date": pd.to_datetime(d).date(), "label": label} for d, label in data.items()]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fetch_vacances_zone_c() -> pd.DataFrame:
    """Vacances scolaires zone C, retourne intervalles (start, end, label)."""
    params = {
        "where": 'zones="Zone C" AND location="Paris"',
        "limit": 100,
        "order_by": "start_date",
    }
    r = requests.get(VACANCES_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    records = r.json().get("results", [])
    rows = []
    for rec in records:
        rows.append({
            "start": pd.to_datetime(rec["start_date"]).date(),
            "end": pd.to_datetime(rec["end_date"]).date(),
            "label": rec.get("description", ""),
        })
    return pd.DataFrame(rows)


def build_calendar(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Table jour-par-jour avec colonnes de features calendaires.

    Colonnes : date, weekday (0-6), month, is_ferie, is_vacances, ferie_label.
    """
    days = pd.date_range(start, end, freq="D").date
    df = pd.DataFrame({"date": days})
    df["weekday"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month

    feries = fetch_jours_feries()
    feries_set = set(feries["date"])
    df["is_ferie"] = df["date"].isin(feries_set)

    vacances = fetch_vacances_zone_c()
    vac_dates = set()
    for _, row in vacances.iterrows():
        for d in pd.date_range(row["start"], row["end"], freq="D").date:
            vac_dates.add(d)
    df["is_vacances"] = df["date"].isin(vac_dates)

    return df
