"""
PasDeVélib — Calcul quotidien des métriques de précision.
Produit metrics__accuracy.json publié dans la GitHub Release du jour.

Usage (GitHub Actions) :
  python -m pasdevelib.eval.eval_daily

Sortie JSON :
{
  "computed_at": "2026-06-15T03:00:00Z",
  "day":   { "accuracy": 0.94, "brier": 0.055, "bss": -0.28, "false_positive_rate": 0.035, "n_predictions": 25672, "n_days": 1 },
  "week":  { ... },
  "month": { ... }
}
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pasdevelib.storage import load_parquet_release


def compute_metrics(hourly: pd.DataFrame, target_dates: list[str], calendar: pd.DataFrame) -> dict | None:
    """Rejoue le k-NN sur les dates cibles et retourne les métriques agrégées."""
    from pasdevelib.predict import find_analog_days  # type: ignore

    records = []
    for target_str in target_dates:
        candidates = calendar[calendar["date"].astype(str) < target_str].copy()
        if len(candidates) < 3:
            continue

        target_row = calendar[calendar["date"].astype(str) == target_str]
        if target_row.empty:
            continue

        target_dict = target_row.iloc[0].to_dict()
        analog_dates = find_analog_days(target_dict, candidates, k=7)
        if not analog_dates:
            continue

        obs = hourly[hourly["date"].astype(str) == target_str][
            ["station_id", "hour", "has_velib", "fill_rate"]
        ].copy()
        if obs.empty:
            continue

        neighbors = hourly[hourly["date"].astype(str).isin([str(d) for d in analog_dates])]
        if neighbors.empty:
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
        "accuracy": float(df["correct"].mean()),
        "brier": round(brier, 4),
        "bss": round(bss, 4),
        "false_positive_rate": float(df["fp"].mean()),
        "n_predictions": int(len(df)),
        "n_days": int(df["target_date"].nunique()),
    }


def main() -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    tag = f"backup-{yesterday.strftime('%Y%m%d')}"

    print(f"[eval_daily] Chargement depuis {tag}...")
    hourly = load_parquet_release(tag, "aggregates__hourly_history.parquet")
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.strftime("%Y-%m-%d")

    calendar = load_parquet_release(tag, "aggregates__calendar.parquet")
    analog_idx = load_parquet_release(tag, "aggregates__analog_index.parquet")
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.strftime("%Y-%m-%d")
    analog_idx["date"] = pd.to_datetime(analog_idx["date"]).dt.strftime("%Y-%m-%d")

    cal = calendar.merge(analog_idx, on="date", how="left", suffixes=("", "_ai"))
    for col in ["weekday", "month", "is_ferie", "is_vacances"]:
        if f"{col}_ai" in cal.columns:
            cal.drop(columns=[f"{col}_ai"], inplace=True)
    cal = cal.rename(columns={"mean_temperature": "temp_avg", "total_precipitation": "precip_total"})

    all_dates = sorted(hourly["date"].unique())

    def dates_for_period(n_days: int) -> list[str]:
        cutoff = (yesterday - timedelta(days=n_days)).strftime("%Y-%m-%d")
        return [d for d in all_dates if cutoff <= d <= yesterday.strftime("%Y-%m-%d")]

    day_dates = [yesterday.strftime("%Y-%m-%d")]
    week_dates = dates_for_period(7)
    month_dates = dates_for_period(30)

    print(f"[eval_daily] Calcul sur 1j / 7j / 30j...")
    result = {
        "computed_at": today.strftime("%Y-%m-%dT03:00:00Z"),
        "day": compute_metrics(hourly, day_dates, cal),
        "week": compute_metrics(hourly, week_dates, cal),
        "month": compute_metrics(hourly, month_dates, cal),
    }

    out_path = Path("metrics__accuracy.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[eval_daily] Ecrit : {out_path.resolve()}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
