"""Evaluation metrics for the prediction model.

All metrics operate on a join of (predictions, ground_truth) on
(station_id, hour). Predictions are in `fill_rate` units (0-1) for the
quantiles; ground_truth has the same fill_rate plus the binary has_velib /
has_place flags.

Three families of metrics:
    1. Regression on fill_rate:        MAE
    2. Binary decision (velib/place):  Decision Accuracy + Brier score
    3. Probabilistic calibration:      coverage of [p25, p75], buckets
"""
from __future__ import annotations

import pandas as pd


def _join_pred_truth(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (station_id, hour). Returns empty if no overlap."""
    if predictions.empty or truth.empty:
        return pd.DataFrame()

    cols_pred = [
        "station_id",
        "hour",
        "proba_velib",
        "proba_place",
        "p25_fill",
        "p50_fill",
        "p75_fill",
    ]
    cols_truth = ["station_id", "hour", "fill_rate", "has_velib", "has_place"]

    df = predictions[cols_pred].merge(
        truth[cols_truth],
        on=["station_id", "hour"],
        how="inner",
    )
    return df


def compute_metrics(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Aggregate metrics over a single day's predictions.

    Returns a dict ready to be JSON-serialized. Values are floats or ints,
    never numpy types.
    """
    df = _join_pred_truth(predictions, truth)
    if df.empty:
        return {"n": 0}

    # Cast bools to floats for numeric ops
    has_velib_f = df["has_velib"].astype(float)
    has_place_f = df["has_place"].astype(float)

    # 1. MAE on fill_rate, using p50 as the point prediction
    mae_fill = float((df["p50_fill"] - df["fill_rate"]).abs().mean())

    # 2. Decision accuracy: did we correctly predict has_velib / has_place?
    pred_velib = (df["proba_velib"] > 0.5).astype(int)
    pred_place = (df["proba_place"] > 0.5).astype(int)
    acc_velib = float((pred_velib == df["has_velib"].astype(int)).mean())
    acc_place = float((pred_place == df["has_place"].astype(int)).mean())

    # 3. Brier score (calibration of probabilistic predictions)
    #    Lower is better. 0 = perfect. 0.25 = predicting 0.5 always.
    brier_velib = float(((df["proba_velib"] - has_velib_f) ** 2).mean())
    brier_place = float(((df["proba_place"] - has_place_f) ** 2).mean())

    # 4. Coverage of [p25, p75] interval — should be ~50% if quantiles are
    #    well-calibrated.
    in_interval = (df["fill_rate"] >= df["p25_fill"]) & (
        df["fill_rate"] <= df["p75_fill"]
    )
    coverage_50 = float(in_interval.mean())

    # 5. Base rates (sanity check: what would a constant predictor get?)
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
    """Per-station metrics. Lets us identify which stations the algorithm
    handles well vs poorly.
    """
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
    """Bin predicted probabilities into n_buckets and compute the actual
    realization rate within each bucket.

    A perfectly calibrated model would have predicted == actual in every
    bucket (the diagonal on a calibration plot).
    """
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
