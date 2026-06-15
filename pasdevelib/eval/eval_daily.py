"""
PasDeVélib — Calcul quotidien des métriques de précision.
Produit metrics__accuracy.json publié dans la GitHub Release du backup de la veille.

Deux types d'analyse :
  binary  : proba_velib >= 0.5 vs has_velib  (dispo / vide)
  count   : p50 vélos prédits vs fill_rate * capacity  (MAE en vélos)

Usage : python -m pasdevelib.eval.eval_daily
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


def _dl(tag: str, asset: str) -> bytes:
    url = f"https://github.com/{storage.REPO}/releases/download/{tag}/{asset}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def _dl_parquet(tag: str, asset: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_dl(tag, asset)))


def _dl_json(tag: str, asset: str) -> list:
    return json.loads(_dl(tag, asset))


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2): return "winter"
    if month in (3, 4, 5):  return "spring"
    if month in (6, 7, 8):  return "summer"
    return "autumn"


def _build_calendar(analog_idx: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar.merge(analog_idx, on="date", how="left", suffixes=("", "_ai"))
    for col in ["weekday", "month", "is_ferie", "is_vacances"]:
        if f"{col}_ai" in cal.columns:
            cal.drop(columns=[f"{col}_ai"], inplace=True)
    cal = cal.rename(columns={
        "mean_temperature": "temp_avg",
        "total_precipitation": "precip_total",
    })
    cal["day_of_week"]       = cal["weekday"]
    cal["is_holiday"]        = cal["is_ferie"]
    cal["is_school_holiday"] = cal["is_vacances"]
    cal["season"]            = cal["month"].apply(_month_to_season)
    return cal


def compute_metrics(
    hourly: pd.DataFrame,
    target_dates: list[str],
    cal: pd.DataFrame,
    capacities: dict[str, int],
) -> dict | None:
    """
    Retourne un dict avec deux blocs :
      binary : accuracy, brier, bss, faux_positifs, faux_negatifs
      count  : mae_velib (MAE en vélos), rmse_velib, biais_velib
    """
    cfg = AnalogConfig(k=7)
    records = []

    for target_str in target_dates:
        candidates = cal[cal["date"] < target_str].copy()
        if len(candidates) < 3:
            continue
        target_row = cal[cal["date"] == target_str]
        if target_row.empty:
            continue

        analog_dates, level = find_analog_days(target_row.iloc[0], candidates, cfg)
        if not analog_dates:
            continue

        obs = hourly[hourly["date"] == target_str][
            ["station_id", "hour", "has_velib", "fill_rate"]
        ].copy()
        if obs.empty:
            continue

        neighbors = hourly[hourly["date"].isin(analog_dates)]
        if neighbors.empty:
            print(f"  {target_str}: voisins {analog_dates[:2]}... absents, skip")
            continue

        pred = (
            neighbors.groupby(["station_id", "hour"])
            .agg(
                proba_velib=("has_velib", "mean"),
                p25_fill=("fill_rate", lambda x: float(np.percentile(x, 25))),
                p50_fill=("fill_rate", lambda x: float(np.percentile(x, 50))),
                p75_fill=("fill_rate", lambda x: float(np.percentile(x, 75))),
            )
            .reset_index()
        )

        merged = pred.merge(obs, on=["station_id", "hour"], how="inner")
        if merged.empty:
            continue

        # Convertir fill_rate → nombre de vélos via capacité
        merged["capacity"] = merged["station_id"].astype(str).map(capacities).fillna(0).astype(int)
        merged["p50_bikes"]  = (merged["p50_fill"]  * merged["capacity"]).round()
        merged["real_bikes"] = (merged["fill_rate"] * merged["capacity"]).round()

        merged["target_date"] = target_str
        records.append(merged)
        print(f"  {target_str}: {len(merged):,} lignes OK ({level})")

    if not records:
        return None

    df = pd.concat(records, ignore_index=True)
    n = len(df)
    n_days = df["target_date"].nunique()

    # ── Analyse binaire (dispo / vide) ──────────────────────────────────────
    df["pred_bin"]  = (df["proba_velib"] >= 0.5).astype(int)
    df["correct"]   = (df["pred_bin"] == df["has_velib"]).astype(int)
    df["fp"]        = ((df["pred_bin"] == 1) & (df["has_velib"] == 0)).astype(int)
    df["fn"]        = ((df["pred_bin"] == 0) & (df["has_velib"] == 1)).astype(int)
    df["brier"]     = (df["proba_velib"] - df["has_velib"]) ** 2
    df["baseline"]  = df.groupby("hour")["has_velib"].transform("mean")
    df["brier_base"] = (df["baseline"] - df["has_velib"]) ** 2

    brier     = float(df["brier"].mean())
    brier_base = float(df["brier_base"].mean())
    bss       = float(1 - brier / brier_base) if brier_base > 0 else 0.0

    binary = {
        "accuracy":            round(float(df["correct"].mean()), 4),
        "brier":               round(brier, 4),
        "bss":                 round(bss, 4),
        "false_positive_rate": round(float(df["fp"].mean()), 4),
        "false_negative_rate": round(float(df["fn"].mean()), 4),
        "n_predictions":       n,
        "n_days":              n_days,
    }

    # ── Analyse nombre de vélos ──────────────────────────────────────────────
    # Exclure les stations sans capacité connue
    df_count = df[df["capacity"] > 0].copy()
    if df_count.empty:
        count = None
    else:
        df_count["err_abs"]  = (df_count["p50_bikes"] - df_count["real_bikes"]).abs()
        df_count["err_sq"]   = (df_count["p50_bikes"] - df_count["real_bikes"]) ** 2
        df_count["err_bias"] = df_count["p50_bikes"] - df_count["real_bikes"]

        # Précision à ±1 vélo et ±2 vélos
        df_count["within_1"] = (df_count["err_abs"] <= 1).astype(int)
        df_count["within_2"] = (df_count["err_abs"] <= 2).astype(int)

        # MAE par tranche de remplissage
        df_count["fill_bucket"] = pd.cut(
            df_count["fill_rate"],
            bins=[0, 0.1, 0.3, 0.7, 0.9, 1.01],
            labels=["vide (0-10%)", "bas (10-30%)", "moyen (30-70%)", "élevé (70-90%)", "plein (90-100%)"],
        )
        mae_by_fill = (
            df_count.groupby("fill_bucket", observed=True)["err_abs"]
            .mean()
            .round(2)
            .to_dict()
        )
        mae_by_fill = {str(k): v for k, v in mae_by_fill.items()}

        count = {
            "mae_bikes":        round(float(df_count["err_abs"].mean()), 2),
            "rmse_bikes":       round(float(df_count["err_sq"].mean() ** 0.5), 2),
            "bias_bikes":       round(float(df_count["err_bias"].mean()), 2),
            "within_1_bike":    round(float(df_count["within_1"].mean()), 4),
            "within_2_bikes":   round(float(df_count["within_2"].mean()), 4),
            "mae_by_fill":      mae_by_fill,
            "n_predictions":    int(len(df_count)),
            "n_days":           n_days,
        }

    return {"binary": binary, "count": count}


def main() -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    tag = f"backup-{yesterday.strftime('%Y%m%d')}"

    print(f"[eval_daily] Chargement depuis {tag}...")
    hourly    = _dl_parquet(tag, "aggregates__hourly_history.parquet")
    calendar  = _dl_parquet(tag, "aggregates__calendar.parquet")
    analog_idx = _dl_parquet(tag, "aggregates__analog_index.parquet")

    # Normaliser dates en string
    for df in [hourly, calendar, analog_idx]:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    cal = _build_calendar(analog_idx, calendar)

    # Charger capacités depuis stations.json
    stations_raw = _dl_json("live", "stations.json")
    capacities = {
        str(s.get("station_id") or s.get("stationcode") or s.get("stationCode")): int(s.get("capacity", 0) or 0)
        for s in stations_raw
    }
    print(f"[eval_daily] {len(capacities)} capacités chargées")

    all_dates = sorted(hourly["date"].unique())

    def dates_for_period(n_days: int) -> list[str]:
        cutoff = (yesterday - timedelta(days=n_days)).strftime("%Y-%m-%d")
        end    = yesterday.strftime("%Y-%m-%d")
        return [d for d in all_dates if cutoff <= d <= end]

    print("[eval_daily] Calcul J-1...")
    day = compute_metrics(hourly, [yesterday.strftime("%Y-%m-%d")], cal, capacities)

    print("[eval_daily] Calcul 7j...")
    week = compute_metrics(hourly, dates_for_period(7), cal, capacities)

    print("[eval_daily] Calcul 30j...")
    month = compute_metrics(hourly, dates_for_period(30), cal, capacities)

    result = {
        "computed_at": today.strftime("%Y-%m-%dT03:00:00Z"),
        "day":   day,
        "week":  week,
        "month": month,
    }

    out = Path("metrics__accuracy.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[eval_daily] Résultat :")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

