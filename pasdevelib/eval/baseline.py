"""Climatology baseline: predict the historical mean per (station, hour),
ignoring all calendar/weather features.

This is the simplest possible baseline that still uses past data. If our
k-NN algorithm does not beat this baseline, it is not adding any value.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def predict_climatology(
    target_date: dt.date,
    hourly_history: pd.DataFrame,
) -> pd.DataFrame:
    """For each (station_id, hour), compute the mean and quantiles over all
    past days strictly before target_date. No feature matching at all.

    Returns a DataFrame with the same schema as predict_day_with_quantiles
    so it can be fed into compute_metrics() unchanged.
    """
    target_str = target_date.isoformat()
    hist = hourly_history[hourly_history["date"] < target_str]

    if hist.empty:
        return pd.DataFrame()

    grouped = hist.groupby(["station_id", "hour"]).agg(
        proba_velib=("has_velib", "mean"),
        proba_place=("has_place", "mean"),
        p25_fill=("fill_rate", lambda x: float(np.percentile(x, 25))),
        p50_fill=("fill_rate", lambda x: float(np.percentile(x, 50))),
        p75_fill=("fill_rate", lambda x: float(np.percentile(x, 75))),
        n_neighbors=("has_velib", "size"),
    ).reset_index()

    grouped["prob_empty"] = 1.0 - grouped["proba_velib"]
    grouped["target_date"] = target_str
    grouped["analog_level"] = "baseline_climatology"

    return grouped
