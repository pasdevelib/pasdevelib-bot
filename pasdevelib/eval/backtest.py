"""Honest backtesting: replay predict() on a past date using only
data strictly anterior to that date.

The current predict module excludes only `target_date` from candidates,
but for a real backtest we must exclude ALL dates >= target_date. Otherwise
the algorithm could pick a future day as an analog (which it would not have
been able to do in production at the moment of the prediction).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from pasdevelib import predict


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def build_target_features(
    target_date: dt.date,
    calendar_df: pd.DataFrame,
    weather_daily: pd.DataFrame | None,
) -> pd.Series:
    """Build a feature row for a single target date.

    Mirrors the logic in forecast._build_target_features but for a single day,
    using observed (past) weather instead of fetched forecasts.
    """
    target_str = target_date.isoformat()
    cal_row = calendar_df[calendar_df["date"] == target_str]
    if cal_row.empty:
        raise ValueError(f"No calendar entry for {target_str}")

    feat = cal_row.iloc[0].to_dict()

    # Optional weather attributes — gracefully missing if weather is not available
    if weather_daily is not None and not weather_daily.empty:
        w = weather_daily[weather_daily["date"] == target_str]
        if not w.empty:
            feat["temp_avg"] = w.iloc[0].get("temp_avg")
            feat["precip_total"] = w.iloc[0].get("precip_total")

    # Aliases expected by predict._row_distance
    feat["day_of_week"] = feat.get("weekday")
    feat["is_holiday"] = feat.get("is_ferie")
    feat["is_school_holiday"] = feat.get("is_vacances")
    if feat.get("month") is not None:
        feat["season"] = _month_to_season(int(feat["month"]))

    return pd.Series(feat)


def backtest_single_day(
    target_date: dt.date,
    hourly_history: pd.DataFrame,
    calendar_df: pd.DataFrame,
    weather_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay predict() for `target_date`, using only data strictly before it.

    Args:
        target_date: the day to backtest.
        hourly_history: full historical observations (station_id, date, hour,
                        fill_rate, has_velib, has_place).
        calendar_df:    full calendar features per date.
        weather_daily:  optional historical daily weather (date, temp_avg,
                        precip_total). If absent, predict will fall back to
                        levels L3+ which ignore weather.

    Returns:
        (predictions, ground_truth)
        predictions: DataFrame with station_id, hour, proba_velib, proba_place,
                     p25_fill, p50_fill, p75_fill, prob_empty, n_neighbors,
                     analog_level.
        ground_truth: DataFrame with station_id, hour, fill_rate, has_velib,
                      has_place (the actual observations on target_date).
    """
    target_str = target_date.isoformat()

    # 1. Strict temporal cutoff — no peeking at the target day or later
    hist_filtered = hourly_history[hourly_history["date"] < target_str]
    cal_filtered = calendar_df[calendar_df["date"] < target_str]

    if hist_filtered.empty:
        raise ValueError(f"No history before {target_str}, cannot backtest")
    if cal_filtered.empty:
        raise ValueError(f"No calendar entries before {target_str}, cannot backtest")

    # 2. Build target features using past calendar (the target row itself is
    #    looked up in the FULL calendar_df because we need its weekday, etc.)
    target_features = build_target_features(target_date, calendar_df, weather_daily)

    # 3. Run prediction with the truncated history
    predictions = predict.predict_day_with_quantiles(
        target_date=target_date,
        target_features=target_features,
        calendar_df=cal_filtered,
        hourly_history=hist_filtered,
    )

    # 4. Get ground truth (actual observations on target_date)
    ground_truth = hourly_history[hourly_history["date"] == target_str][
        ["station_id", "hour", "fill_rate", "has_velib", "has_place"]
    ].copy()

    return predictions, ground_truth
