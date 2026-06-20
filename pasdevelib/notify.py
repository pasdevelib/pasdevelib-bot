"""
Bot de notifications push.
Tourne toutes les 5 minutes via GitHub Actions.
Vérifie les alertes actives et envoie des push si conditions remplies.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pywebpush import webpush, WebPushException

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
VAPID_PRIVATE = os.environ["VAPID_PRIVATE_KEY"]
VAPID_EMAIL = os.environ.get("VAPID_EMAIL", "mailto:hello@pasdevelib.app")
SITE_URL = os.environ.get("SITE_URL", "https://pasdevelib.app")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = "pasdevelib/pasdevelib-bot"
LIVE_RELEASE = "live"


def supabase_get(table: str, params: dict = {}) -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.get(url, params=params, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    r.raise_for_status()
    return r.json()


def get_live_status() -> dict[str, dict]:
    """Télécharge current_day.parquet et retourne le dernier status par station."""
    import pandas as pd
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{LIVE_RELEASE}"
    r = requests.get(url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
    r.raise_for_status()
    assets = r.json()["assets"]
    asset = next((a for a in assets if a["name"] == "current_day.parquet"), None)
    if not asset:
        raise RuntimeError("current_day.parquet not found in release")

    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        dl = requests.get(asset["browser_download_url"],
                         headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
        dl.raise_for_status()
        f.write(dl.content)
        f.flush()
        df = pd.read_parquet(f.name)

    # Garder le dernier snapshot par station
    latest = df.sort_values("fetched_at").groupby("station_id").last().reset_index()
    return {str(row["station_id"]): row.to_dict() for _, row in latest.iterrows()}


def is_in_schedule(schedules: list[dict], hour: int) -> bool:
    """Vérifie si l'heure actuelle est dans un des créneaux d'alerte."""
    for s in schedules:
        if s.get("start", 0) <= hour < s.get("end", 24):
            return True
    return False


def send_push(subscription_info: dict, title: str, body: str, url: str) -> bool:
    try:
        webpush(
            subscription_info=subscription_info,
            data=json.dumps({"title": title, "body": body, "url": url}),
            vapid_private_key=VAPID_PRIVATE,
            vapid_claims={"sub": VAPID_EMAIL},
        )
        return True
    except WebPushException as e:
        print(f"[push] WebPushException: {e}")
        # 410 Gone = subscription expirée
        if e.response and e.response.status_code == 410:
            return None  # type: ignore  # signal pour désactiver
        return False
    except Exception as e:
        print(f"[push] Error: {e}")
        return False


def run() -> None:
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour

    print(f"[notify] {now_utc.isoformat()} — heure UTC: {hour}h")

    # Charger les données live
    try:
        live = get_live_status()
        print(f"[notify] {len(live)} stations live")
    except Exception as e:
        print(f"[notify] Erreur chargement live: {e}")
        return

    # Charger les alertes actives avec endpoint push
    alerts = supabase_get("push_subscriptions", {
        "is_active": "eq.true",
        "select": "id,station_id,station_name,schedules,endpoint,p256dh,auth,user_id",
    })
    print(f"[notify] {len(alerts)} alertes actives")

    sent = 0
    for alert in alerts:
        station_id = str(alert["station_id"])
        schedules = alert.get("schedules") or []
        endpoint = alert.get("endpoint")
        p256dh = alert.get("p256dh")
        auth = alert.get("auth")

        # Vérifier le créneau
        if not is_in_schedule(schedules, hour):
            continue

        # Vérifier si on a les infos push
        if not endpoint or not p256dh or not auth:
            continue

        # Vérifier le statut de la station
        station_data = live.get(station_id)
        if not station_data:
            continue

        bikes = int(station_data.get("num_bikes_available", 0))
        is_renting = bool(station_data.get("is_renting", True))

        # Condition d'alerte : station vide ou hors service
        if not is_renting:
            title = f"⚠️ {alert['station_name']}"
            body = "Station hors service en ce moment"
        elif bikes == 0:
            title = f"🔴 {alert['station_name']}"
            body = "Aucun vélo disponible à cette station"
        elif bikes <= 2:
            title = f"🟡 {alert['station_name']}"
            body = f"Plus que {bikes} vélo{'s' if bikes > 1 else ''} disponible{'s' if bikes > 1 else ''}"
        else:
            # Station OK — pas d'alerte
            continue

        station_url = f"{SITE_URL}/station/{station_id}"
        sub_info = {
            "endpoint": endpoint,
            "keys": {"p256dh": p256dh, "auth": auth},
        }

        result = send_push(sub_info, title, body, station_url)
        if result is None:
            # Subscription expirée → désactiver
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/push_subscriptions",
                json={"is_active": False},
                params={"id": f"eq.{alert['id']}"},
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
            print(f"[notify] subscription expirée désactivée: {alert['id']}")
        elif result:
            sent += 1
            print(f"[notify] push envoyé → {alert['station_name']} ({bikes} vélos)")

    print(f"[notify] {sent} notifications envoyées")


if __name__ == "__main__":
    run()
