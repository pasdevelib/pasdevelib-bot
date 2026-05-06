"""Evaluation metrics for the prediction model.

NaN handling: rows with NaN in any required column are dropped before
computing aggregates. NaN can appear in predictions when a (station, hour)
pair has no analog observations (small sample, brand-new station, etc.).
"""
from __future__ import annotations

import pandas as pd


_PRED_REQUIRED = ["proba_velib", "proba_place", "p25_fill", "p50_fill", "p75_fill"]
_TRUTH_REQUIRED = ["fill_rate", "has_velib", "has_place"]


def _join_pred_truth(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (station_id, hour), with NaN rows dropped."""
    if predictions.empty or truth.empty:
        return pd.DataFrame()

    cols_pred = ["station_id", "hour"] + _PRED_REQUIRED
    cols_truth = ["station_id", "hour"] + _TRUTH_REQUIRED

    df = predictions[cols_pred].merge(
        truth[cols_truth],
        on=["station_id", "hour"],
        how="inner",
    )

    df = df.dropna(subset=_PRED_REQUIRED + _TRUTH_REQUIRED)

    return df


def compute_metrics(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Aggregate metrics over a single day's predictions."""
    df = _join_pred_truth(predictions, truth)
    if df.empty:
        return {"n": 0}

    has_velib_f = df["has_velib"].astype(float)
    has_place_f = df["has_place"].astype(float)

    mae_fill = float((df["p50_fill"] - df["fill_rate"]).abs().mean())

    pred_velib = (df["proba_velib"] > 0.5).astype(int)
    pred_place = (df["proba_place"] > 0.5).astype(int)
    acc_velib = float((pred_velib == df["has_velib"].astype(int)).mean())
    acc_place = float((pred_place == df["has_place"].astype(int)).mean())

    brier_velib = float(((df["proba_velib"] - has_velib_f) ** 2).mean())
    brier_place = float(((df["proba_place"] - has_place_f) ** 2).mean())

    in_interval = (df["fill_rate"] >= df["p25_fill"]) & (
        df["fill_rate"] <= df["p75_fill"]
    )
    coverage_50 = float(in_interval.mean())

    base_rate_velib = float(has_velib_f.mean())
    base_rate_place = float(has_place_f.mean())

    return {
        "n": int(len(df)),
        "mae_fill_rate": mae_fill,
        "decision_accuracy_velib": acc_velib,
        "decision_accuracy_place": acc_place,
        "brier_velib": brier_velib,
        "brier_place": brier_place,
        "coverage_50": coverage_50,
        "base_rate_velib": base_rate_velib,
        "base_rate_place": base_rate_place,
    }


def compute_per_station_metrics(
    predictions: pd.DataFrame, truth: pd.DataFrame
) -> pd.DataFrame:
    """Per-station metrics."""
    df = _join_pred_truth(predictions, truth)
    if df.empty:
        return pd.DataFrame()

    df = df.assign(
        abs_error=(df["p50_fill"] - df["fill_rate"]).abs(),
        correct_velib=(
            (df["proba_velib"] > 0.5).astype(int)
            == df["has_velib"].astype(int)
        ).astype(int),
        correct_place=(
            (df["proba_place"] > 0.5).astype(int)
            == df["has_place"].astype(int)
        ).astype(int),
    )

    return df.groupby("station_id", as_index=False).agg(
        mae_fill=("abs_error", "mean"),
        acc_velib=("correct_velib", "mean"),
        acc_place=("correct_place", "mean"),
        n=("station_id", "size"),
    )


def compute_calibration_buckets(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Calibration buckets for proba_velib."""
    df = _join_pred_truth(predictions, truth)
    if df.empty:
        return pd.DataFrame()

    df = df.assign(
        bucket=(df["proba_velib"] * n_buckets).clip(0, n_buckets - 1).astype(int)
    )

    return df.groupby("bucket", as_index=False).agg(
        predicted_mean=("proba_velib", "mean"),
        actual_rate=("has_velib", "mean"),
        n=("bucket", "size"),
    )
