"""
PasDeVélib — Calcul quotidien des métriques de précision.
Produit metrics__accuracy.json publié dans la GitHub Release du backup de la veille.

Usage (GitHub Actions) :
  python -m pasdevelib.eval.eval_daily
"""
from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pasdevelib import storage
from pasdevelib.predict import find_analog_days, AnalogConfig


def _dl(tag: str, asset: str) -> pd.DataFrame:
    url = f"https://github.com/{storage.REPO}/releases/download/{tag}/{asset}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _build_calendar(analog_idx: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Fusionne calendar + analog_index et ajoute les colonnes attendues par find_analog_days."""
    cal = calendar.merge(analog_idx, on="date", how="left", suffixes=("", "_ai"))
    for col in ["weekday", "month", "is_ferie", "is_vacances"]:
        if f"{col}_ai" in cal.columns:
            cal.drop(columns=[f"{col}_ai"], inplace=True)
    cal = cal.rename(columns={
        "mean_temperature": "temp_avg",
        "total_precipitation": "precip_total",
    })
    # Colonnes attendues par find_analog_days
    cal["day_of_week"] = cal["weekday"]
    cal["is_holiday"] = cal["is_ferie"]
    cal["is_school_holiday"] = cal["is_vacances"]
    cal["season"] = cal["month"].apply(_month_to_season)
    return cal


def compute_metrics(
    hourly: pd.DataFrame,
    target_dates: list[str],
    cal: pd.DataFrame,
) -> dict | None:
    cfg = AnalogConfig(k=7)
    records = []

    for target_str in target_dates:
        candidates = cal[cal["date"] < target_str].copy()
        if len(candidates) < 3:
            continue

        target_row = cal[cal["date"] == target_str]
        if target_row.empty:
            continue

        target_features = target_row.iloc[0]
        analog_dates, level = find_analog_days(target_features, candidates, cfg)
        if not analog_dates:
            continue

        obs = hourly[hourly["date"] == target_str][
            ["station_id", "hour", "has_velib", "fill_rate"]
        ].copy()
        if obs.empty:
            continue

        neighbors = hourly[hourly["date"].isin(analog_dates)]
        if neighbors.empty:
            print(f"  {target_str}: voisins {analog_dates[:3]} non trouvés dans historique, skip")
            continue

        pred = (
            neighbors.groupby(["station_id", "hour"])
            .agg(proba_velib=("has_velib", "mean"))
            .reset_index()
        )

        merged = pred.merge(obs, on=["station_id", "hour"], how="inner")
        if merged.empty:
            continue

        merged["target_date"] = target_str
        records.append(merged)
        print(f"  {target_str}: {len(merged):,} lignes ({level})")

    if not records:
        return None

    df = pd.concat(records, ignore_index=True)
    df["brier"] = (df["proba_velib"] - df["has_velib"]) ** 2
    df["pred_bin"] = (df["proba_velib"] >= 0.5).astype(int)
    df["correct"] = (df["pred_bin"] == df["has_velib"]).astype(int)
    df["fp"] = ((df["pred_bin"] == 1) & (df["has_velib"] == 0)).astype(int)
    df["baseline"] = df.groupby("hour")["has_velib"].transform("mean")
    df["brier_base"] = (df["baseline"] - df["has_velib"]) ** 2

    brier = float(df["brier"].mean())
    brier_base = float(df["brier_base"].mean())
    bss = float(1 - brier / brier_base) if brier_base > 0 else 0.0

    return {
        "accuracy": round(float(df["correct"].mean()), 4),
        "brier": round(brier, 4),
        "bss": round(bss, 4),
        "false_positive_rate": round(float(df["fp"].mean()), 4),
        "n_predictions": int(len(df)),
        "n_days": int(df["target_date"].nunique()),
    }


def main() -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    tag = f"backup-{yesterday.strftime('%Y%m%d')}"

    print(f"[eval_daily] Chargement depuis {tag}...")
    hourly = _dl(tag, "aggregates__hourly_history.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.strftime("%Y-%m-%d")

    calendar = _dl(tag, "aggregates__calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.strftime("%Y-%m-%d")

    analog_idx = _dl(tag, "aggregates__analog_index.parquet")
    analog_idx["date"] = pd.to_datetime(analog_idx["date"]).dt.strftime("%Y-%m-%d")

    cal = _build_calendar(analog_idx, calendar)
    all_dates = sorted(hourly["date"].unique())

    def dates_for_period(n_days: int) -> list[str]:
        cutoff = (yesterday - timedelta(days=n_days)).strftime("%Y-%m-%d")
        end = yesterday.strftime("%Y-%m-%d")
        return [d for d in all_dates if cutoff <= d <= end]

    print("[eval_daily] Calcul période J-1...")
    day_result = compute_metrics(hourly, [yesterday.strftime("%Y-%m-%d")], cal)

    print("[eval_daily] Calcul période 7j...")
    week_result = compute_metrics(hourly, dates_for_period(7), cal)

    print("[eval_daily] Calcul période 30j...")
    month_result = compute_metrics(hourly, dates_for_period(30), cal)

    result = {
        "computed_at": today.strftime("%Y-%m-%dT03:00:00Z"),
        "day": day_result,
        "week": week_result,
        "month": month_result,
    }

    out_path = Path("metrics__accuracy.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[eval_daily] Résultat :")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
