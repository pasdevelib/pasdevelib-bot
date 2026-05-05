"""
flux.py — Compute hourly net flux per station, day-of-week, hour.

Reads:
  - data/hourly_history.parquet (columns: station_id, date, hour, fill_rate, has_velib, has_place)
  - data/stations.json (with capacity, name, lat, lon for each station)

Writes:
  - data/output/flux_hourly.json (consumed by the webapp blog)

Methodology:
  For each (station, day_of_week, hour), we compute the mean delta of the bike count
  between hour h and hour h-1, averaged over all observed days that match.

  delta(d, h) = bikes(d, h) - bikes(d, h-1)
  flux_net(s, dow, h) = mean over d of delta(d, h) where day_of_week(d) == dow

  Hours 0-5 are masked to None to avoid mixing human trips with Smovengo
  rebalancing trucks. The webapp can offer a toggle to display them later.

Day-of-week convention:
  pandas dayofweek -> Monday=0, Tuesday=1, ..., Sunday=6
  The webapp aligns on this convention.

Usage:
  python -m pasdevelib.flux

Run frequency:
  Once per day after the daily consolidation step. Cheap to recompute.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PARQUET_PATH = Path("data/hourly_history.parquet")
STATIONS_PATH = Path("data/stations.json")
OUTPUT_PATH = Path("data/output/flux_hourly.json")

NIGHT_HOURS = set(range(0, 6))  # 0h to 5h excluded
ROUND_DECIMALS = 3


def load_stations(path: Path) -> dict:
    """Load stations.json defensively. Accept multiple known shapes."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Shape 1: list of station dicts at the top level
    if isinstance(raw, list):
        items = raw
    # Shape 2: GBFS-style { data: { stations: [...] } }
    elif isinstance(raw, dict) and "data" in raw and "stations" in raw["data"]:
        items = raw["data"]["stations"]
    # Shape 3: dict keyed by station_id
    elif isinstance(raw, dict):
        items = [{"station_id": k, **v} for k, v in raw.items()]
    else:
        raise ValueError(f"Unsupported stations.json shape: {type(raw)}")

    out = {}
    for s in items:
        sid = str(
            s.get("station_id")
            or s.get("id")
            or s.get("stationCode")
            or ""
        )
        if not sid:
            continue
        out[sid] = {
            "name": s.get("name") or s.get("station_name") or "",
            "lat": s.get("lat") or s.get("latitude"),
            "lon": s.get("lon") or s.get("longitude"),
            "capacity": (
                s.get("capacity")
                or s.get("total_docks")
                or s.get("totalDocks")
                or 0
            ),
        }
    return out


def compute_flux(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean delta per (station_id, day_of_week, hour)."""
    # Ensure types
    df = df.copy()
    df["station_id"] = df["station_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek

    # Sort to compute deltas correctly
    df = df.sort_values(["station_id", "date", "hour"]).reset_index(drop=True)

    # Compute previous hour bikes within each (station, date) group
    df["n_bikes_prev"] = df.groupby(["station_id", "date"])["n_bikes"].shift(1)
    df["delta"] = df["n_bikes"] - df["n_bikes_prev"]

    # Drop rows where we have no previous reference (first hour of a date)
    df = df.dropna(subset=["delta"])

    # Aggregate
    agg = (
        df.groupby(["station_id", "day_of_week", "hour"])["delta"]
        .mean()
        .reset_index()
    )
    return agg


def build_output(
    flux: pd.DataFrame,
    stations_meta: dict,
    n_days_used: int,
) -> dict:
    """Build the JSON payload consumed by the webapp."""
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_days_used": int(n_days_used),
        "stations": [],
    }

    # Index flux by station_id for fast lookup
    flux_by_station = dict(tuple(flux.groupby("station_id")))

    for sid, info in stations_meta.items():
        # Skip stations without coordinates or capacity
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
                    continue  # Keep None for night hours
                arr[hour] = round(float(row["delta"]), ROUND_DECIMALS)
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


def main() -> None:
    print(f"[flux] loading parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[flux] {len(df):,} rows loaded")

    print(f"[flux] loading stations: {STATIONS_PATH}")
    stations_meta = load_stations(STATIONS_PATH)
    print(f"[flux] {len(stations_meta):,} stations loaded")

    # Convert fill_rate to bike count using capacity
    df["station_id"] = df["station_id"].astype(str)
    capacities = pd.Series(
        {sid: m.get("capacity", 0) for sid, m in stations_meta.items()}
    )
    df["capacity"] = df["station_id"].map(capacities).fillna(0)
    df["n_bikes"] = df["fill_rate"] * df["capacity"]

    n_days_used = df["date"].nunique() if "date" in df.columns else 0
    print(f"[flux] {n_days_used} unique dates in history")

    print("[flux] computing deltas and flux means...")
    flux = compute_flux(df)
    print(f"[flux] {len(flux):,} (station, dow, hour) cells")

    print("[flux] building output...")
    output = build_output(flux, stations_meta, n_days_used)
    print(f"[flux] {len(output['stations']):,} stations in output")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"[flux] wrote {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
