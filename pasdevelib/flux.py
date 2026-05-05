"""Pre-calcule le flux net horaire par station et jour de la semaine.

Schema de sortie (flux_hourly.json) consomme par le blog de la webapp :
{
  "generated_at": ISO timestamp,
  "n_days_used": int,
  "stations": [
    {
      "id": str,
      "name": str,
      "lat": float,
      "lon": float,
      "capacity": int,
      "flux": {
        "0": [null x6, value, ...],   # Lundi (pandas dayofweek), 24 valeurs
        ...
        "6": [...]                     # Dimanche
      }
    },
    ...
  ]
}

Methodologie :
  Pour chaque (station, jour_semaine, heure), on calcule la moyenne du delta
  du nombre de velos entre h et h-1, sur tous les jours de l'historique
  qui matchent ce jour de semaine.

    delta(d, h) = bikes(d, h) - bikes(d, h-1)
    flux(s, dow, h) = mean(delta) sur les jours d ou day_of_week(d) == dow

Mapping station_id :
  Le parquet contient un mix d'identifiants : IDs numeriques (donnees recentes)
  et noms de stations (donnees historiques bootstrap). On tente d'abord un
  match par ID, puis par nom (case et whitespace insensitive) pour maximiser
  la fraction des donnees utilisables.

Heures de nuit (0-5) masquees pour ne pas melanger flux humain et
rebalancing trucks Smovengo.
"""
from __future__ import annotations

import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from pasdevelib import storage


HOURLY_ASSET = "hourly_history.parquet"
STATIONS_ASSET = "stations.json"
FLUX_ASSET = "flux_hourly.json"

NIGHT_HOURS = set(range(0, 6))
ROUND_DECIMALS = 3


def _download_parquet(release: str, asset: str) -> pd.DataFrame:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _download_json(release: str, asset: str) -> list:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _build_stations_meta(stations_raw: list) -> dict:
    """Build a station_id -> metadata dict from stations.json."""
    out = {}
    for s in stations_raw:
        sid = str(
            s.get("station_id")
            or s.get("stationcode")
            or s.get("stationCode")
            or ""
        )
        if not sid:
            continue
        out[sid] = {
            "name": s.get("name") or s.get("station_name") or "",
            "lat": s.get("lat") or s.get("latitude"),
            "lon": s.get("lon") or s.get("longitude"),
            "capacity": int(s.get("capacity", 0) or 0),
        }
    return out


def _normalize_name(s: str) -> str:
    return str(s).strip().lower()


def _build_id_resolver(stations_meta: dict):
    """Return a function that resolves any parquet station_id to a canonical ID.

    Tries direct ID match first, then a name match (case + whitespace
    insensitive). Returns None if no match.
    """
    by_id = set(stations_meta.keys())
    by_name = {
        _normalize_name(meta.get("name", "")): sid
        for sid, meta in stations_meta.items()
        if meta.get("name")
    }

    def resolve(raw):
        s = str(raw).strip()
        if s in by_id:
            return s
        return by_name.get(_normalize_name(raw))

    return resolve


def _compute_flux(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean delta per (station_id, day_of_week, hour)."""
    df = df.copy()
    df["station_id"] = df["station_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek

    df = df.sort_values(["station_id", "date", "hour"]).reset_index(drop=True)

    df["n_bikes_prev"] = df.groupby(["station_id", "date"])["n_bikes"].shift(1)
    df["delta"] = df["n_bikes"] - df["n_bikes_prev"]

    df = df.dropna(subset=["delta"])

    agg = (
        df.groupby(["station_id", "day_of_week", "hour"])["delta"]
        .mean()
        .reset_index()
    )
    return agg


def _build_output(
    flux: pd.DataFrame,
    stations_meta: dict,
    n_days_used: int,
) -> dict:
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_days_used": int(n_days_used),
        "stations": [],
    }

    flux_by_station = dict(tuple(flux.groupby("station_id")))

    for sid, info in stations_meta.items():
        if info.get("lat") is None or info.get("lon") is None:
            continue
        if not info.get("capacity"):
            continue

        sub = flux_by_station.get(sid)
        if sub is None or sub.empty:
            continue

        flux_by_dow = {}
        for dow in range(7):
            arr = [None] * 24
            sub_dow = sub[sub["day_of_week"] == dow]
            for _, row in sub_dow.iterrows():
                hour = int(row["hour"])
                if hour in NIGHT_HOURS:
                    continue
                val = float(row["delta"])
                # Skip NaN means (when groupby aggregates over all-NaN deltas)
                if val != val:
                    continue
                arr[hour] = round(val, ROUND_DECIMALS)
            flux_by_dow[str(dow)] = arr

        output["stations"].append(
            {
                "id": sid,
                "name": info.get("name", ""),
                "lat": float(info["lat"]),
                "lon": float(info["lon"]),
                "capacity": int(info["capacity"]),
                "flux": flux_by_dow,
            }
        )

    return output


def run() -> None:
    print("[flux] downloading hourly history...")
    hourly = _download_parquet(storage.RELEASE_AGGREGATES, HOURLY_ASSET)
    print(f"[flux] {len(hourly):,} historical rows")

    print("[flux] downloading stations metadata...")
    stations_raw = _download_json(storage.RELEASE_LIVE, STATIONS_ASSET)
    stations_meta = _build_stations_meta(stations_raw)
    print(f"[flux] {len(stations_meta)} stations loaded")

    # Resolve station_ids: direct ID match, then name match
    print("[flux] resolving station_ids (id, then name fallback)...")
    resolve = _build_id_resolver(stations_meta)

    raw_ids = hourly["station_id"].astype(str).unique()
    id_map = {raw: resolve(raw) for raw in raw_ids}
    matched_by_id = sum(1 for raw, res in id_map.items() if res == raw.strip())
    matched_by_name = sum(
        1
        for raw, res in id_map.items()
        if res is not None and res != raw.strip()
    )
    unresolved = sum(1 for res in id_map.values() if res is None)
    print(
        f"[flux] resolution: {matched_by_id} by id, "
        f"{matched_by_name} by name, {unresolved} unresolved"
    )

    hourly["station_id"] = hourly["station_id"].astype(str).map(id_map)
    n_before = len(hourly)
    hourly = hourly.dropna(subset=["station_id"])
    print(
        f"[flux] dropped {n_before - len(hourly):,} unresolved rows "
        f"({len(hourly):,} remaining)"
    )

    # Convert fill_rate to bike count using capacity
    capacities = pd.Series(
        {sid: m.get("capacity", 0) for sid, m in stations_meta.items()}
    )
    hourly["capacity"] = hourly["station_id"].map(capacities).fillna(0)
    hourly["n_bikes"] = hourly["fill_rate"] * hourly["capacity"]

    n_days_used = (
        hourly["date"].nunique() if "date" in hourly.columns else 0
    )
    print(f"[flux] {n_days_used} unique dates after resolution")

    print("[flux] computing deltas and flux means...")
    flux = _compute_flux(hourly)
    print(f"[flux] {len(flux):,} (station, dow, hour) cells")

    print("[flux] building output payload...")
    output = _build_output(flux, stations_meta, n_days_used)
    print(f"[flux] {len(output['stations']):,} stations in output")

    # Quick stats on output coverage
    total_cells = 0
    filled_cells = 0
    for s in output["stations"]:
        for dow_arr in s["flux"].values():
            total_cells += len(dow_arr)
            filled_cells += sum(1 for v in dow_arr if v is not None)
    pct = (filled_cells / total_cells * 100) if total_cells else 0
    print(
        f"[flux] coverage: {filled_cells:,} / {total_cells:,} cells "
        f"filled ({pct:.1f}%)"
    )

    print(f"[flux] uploading {FLUX_ASSET}...")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / FLUX_ASSET
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, separators=(",", ":"))
        size_kb = path.stat().st_size / 1024
        print(f"[flux] payload size: {size_kb:.1f} KB")
        storage.upload_asset(storage.RELEASE_AGGREGATES, path, FLUX_ASSET)

    print("[flux] DONE")


if __name__ == "__main__":
    run()
