"""Orchestrates rolling and full backtests.

Two entry points:
    run_rolling(n_days=7):    re-runs the last n_days each call (idempotent),
                              uploads results to release `eval` overwriting
                              previous rolling assets.
    run_full(n_days=90):      one-shot baseline, uploaded to separate assets
                              that can be frozen as a long-term reference.

Releases produced under tag `eval`:
    rolling_metrics.json           — global summary, daily breakdown
    rolling_per_station.parquet    — per-station MAE / accuracy
    rolling_calibration.parquet    — calibration buckets
    baseline_<n>d_metrics.json     — full backtest summary (e.g. 90d)
    baseline_<n>d_per_station.parquet
"""
from __future__ import annotations

import datetime as dt
import io
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pasdevelib import storage
from pasdevelib.eval import backtest, baseline, metrics


# Release tag used to host all evaluation artifacts.
RELEASE_EVAL = "eval"

HOURLY_ASSET = "hourly_history.parquet"
CALENDAR_ASSET = "calendar.parquet"
WEATHER_ASSET = "weather.parquet"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _download_parquet(release: str, asset: str) -> pd.DataFrame:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _try_download_weather(release: str) -> pd.DataFrame | None:
    """Best-effort: returns daily weather features or None.

    The exact schema of weather.parquet is unknown to this module, so we try
    to coerce it into (date, temp_avg, precip_total). If anything fails,
    we return None and the algorithm gracefully falls back to L3+ levels.
    """
    try:
        w = _download_parquet(release, WEATHER_ASSET)
    except Exception as e:
        print(f"[eval.runner] no weather parquet ({e}); will run without weather features")
        return None

    # Heuristic mapping: try common columns
    candidate_ts = next(
        (c for c in ("ts", "timestamp", "datetime", "time") if c in w.columns), None
    )
    if candidate_ts is not None:
        try:
            ts = pd.to_datetime(w[candidate_ts])
            try:
                ts = ts.dt.tz_convert("Europe/Paris")
            except (TypeError, AttributeError):
                pass
            w = w.assign(date=ts.dt.date.astype(str))
        except Exception:
            pass

    if "date" not in w.columns:
        print("[eval.runner] weather parquet has no 'date' column; skipping")
        return None

    # Aggregate to daily features
    agg_cols = {}
    if "temperature_2m" in w.columns:
        agg_cols["temp_avg"] = ("temperature_2m", "mean")
    elif "temp_avg" in w.columns:
        agg_cols["temp_avg"] = ("temp_avg", "mean")

    if "precipitation" in w.columns:
        agg_cols["precip_total"] = ("precipitation", "sum")
    elif "precip_total" in w.columns:
        agg_cols["precip_total"] = ("precip_total", "sum")

    if not agg_cols:
        print("[eval.runner] weather parquet has no temp/precip columns; skipping")
        return None

    daily = w.groupby("date", as_index=False).agg(**agg_cols)
    print(f"[eval.runner] weather: {len(daily)} daily rows")
    return daily


def _ensure_release_exists(release: str, token: str) -> dict:
    """Get or create the release. Returns the release JSON object."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    # Try to fetch
    r = requests.get(
        f"https://api.github.com/repos/{storage.REPO}/releases/tags/{release}",
        headers=headers,
        timeout=30,
    )
    if r.status_code == 200:
        return r.json()
    if r.status_code != 404:
        r.raise_for_status()

    # Create
    print(f"[eval.runner] creating release {release}")
    r = requests.post(
        f"https://api.github.com/repos/{storage.REPO}/releases",
        headers=headers,
        json={
            "tag_name": release,
            "name": "Evaluation artifacts",
            "body": "Backtest metrics and per-station evaluation. Updated daily.",
            "draft": False,
            "prerelease": True,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------


def _backtest_window(
    target_dates: list[dt.date],
    hourly_history: pd.DataFrame,
    calendar_df: pd.DataFrame,
    weather_daily: pd.DataFrame | None,
) -> dict:
    """Run backtest on each target_date, compute metrics for both algorithm
    and climatology baseline. Aggregate into a single dict ready to be
    serialized.

    Returns:
        {
          "daily": [...one entry per date...],
          "global_algo": {...},
          "global_baseline": {...},
          "per_station": pd.DataFrame,
          "calibration": pd.DataFrame,
        }
    """
    daily = []
    per_station_frames = []
    calibration_frames = []

    for d in target_dates:
        try:
            preds, truth = backtest.backtest_single_day(
                target_date=d,
                hourly_history=hourly_history,
                calendar_df=calendar_df,
                weather_daily=weather_daily,
            )
        except Exception as e:
            print(f"[eval.runner] {d} skipped: {e}")
            traceback.print_exc()
            continue

        if preds.empty or truth.empty:
            print(f"[eval.runner] {d} skipped: empty preds or truth")
            continue

        m_algo = metrics.compute_metrics(preds, truth)

        # Climatology baseline on the same target day
        try:
            base_preds = baseline.predict_climatology(d, hourly_history)
            m_base = metrics.compute_metrics(base_preds, truth)
        except Exception as e:
            print(f"[eval.runner] {d} baseline failed: {e}")
            m_base = {"n": 0}

        daily.append({
            "date": d.isoformat(),
            "algo": m_algo,
            "baseline": m_base,
        })

        ps = metrics.compute_per_station_metrics(preds, truth)
        if not ps.empty:
            ps = ps.assign(date=d.isoformat())
            per_station_frames.append(ps)

        cb = metrics.compute_calibration_buckets(preds, truth)
        if not cb.empty:
            cb = cb.assign(date=d.isoformat())
            calibration_frames.append(cb)

        print(
            f"[eval.runner] {d}: "
            f"MAE={m_algo.get('mae_fill_rate', float('nan')):.4f}  "
            f"Acc={m_algo.get('decision_accuracy_velib', float('nan')):.3f}  "
            f"Brier={m_algo.get('brier_velib', float('nan')):.4f}  "
            f"(baseline MAE={m_base.get('mae_fill_rate', float('nan')):.4f})"
        )

    global_algo = _aggregate_global(daily, "algo")
    global_baseline = _aggregate_global(daily, "baseline")

    per_station = (
        pd.concat(per_station_frames, ignore_index=True)
        if per_station_frames
        else pd.DataFrame()
    )
    calibration = (
        pd.concat(calibration_frames, ignore_index=True)
        if calibration_frames
        else pd.DataFrame()
    )

    return {
        "daily": daily,
        "global_algo": global_algo,
        "global_baseline": global_baseline,
        "per_station": per_station,
        "calibration": calibration,
    }


def _aggregate_global(daily: list[dict], key: str) -> dict:
    """Aggregate per-day metrics into a single global summary, weighted by n."""
    rows = [d[key] for d in daily if d.get(key, {}).get("n", 0) > 0]
    if not rows:
        return {"n": 0}

    total_n = sum(r["n"] for r in rows)

    def w_mean(field: str) -> float:
        return float(sum(r[field] * r["n"] for r in rows) / total_n)

    return {
        "n": int(total_n),
        "n_days": int(len(rows)),
        "mae_fill_rate": w_mean("mae_fill_rate"),
        "decision_accuracy_velib": w_mean("decision_accuracy_velib"),
        "decision_accuracy_place": w_mean("decision_accuracy_place"),
        "brier_velib": w_mean("brier_velib"),
        "brier_place": w_mean("brier_place"),
        "coverage_50": w_mean("coverage_50"),
        "base_rate_velib": w_mean("base_rate_velib"),
        "base_rate_place": w_mean("base_rate_place"),
    }


# ---------------------------------------------------------------------------
# Per-station post-aggregation (top/bottom 10 across the window)
# ---------------------------------------------------------------------------


def _aggregate_per_station(per_station: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-station metrics across the window (mean of daily means,
    weighted by n)."""
    if per_station.empty:
        return pd.DataFrame()

    g = per_station.groupby("station_id").apply(
        lambda x: pd.Series({
            "mae_fill": float(np.average(x["mae_fill"], weights=x["n"])),
            "acc_velib": float(np.average(x["acc_velib"], weights=x["n"])),
            "acc_place": float(np.average(x["acc_place"], weights=x["n"])),
            "n": int(x["n"].sum()),
            "n_days": int(len(x)),
        }),
        include_groups=False,
    ).reset_index()
    return g


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _upload_json_asset(release: str, asset_name: str, payload: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / asset_name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        storage.upload_asset(release, path, asset_name)


def _upload_parquet_asset(
    release: str, asset_name: str, df: pd.DataFrame
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / asset_name
        df.to_parquet(path, compression="snappy", index=False)
        storage.upload_asset(release, path, asset_name)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_rolling(n_days: int = 7) -> None:
    """Backtest the last n_days. Overwrites rolling_* assets in release `eval`.

    Idempotent: rerunning produces the same result modulo the addition of
    the most recent day.
    """
    import os

    token = os.environ["GITHUB_TOKEN"]
    today = dt.date.today()

    # We backtest the n_days BEFORE today. Today itself is excluded because
    # ground truth (full day) is not yet known.
    target_dates = [today - dt.timedelta(days=i) for i in range(n_days, 0, -1)]
    print(
        f"[eval.runner] === Rolling backtest: {target_dates[0]} .. {target_dates[-1]} "
        f"({len(target_dates)} days) ==="
    )

    print("[eval.runner] downloading hourly_history…")
    hourly = _download_parquet(storage.RELEASE_AGGREGATES, HOURLY_ASSET)
    print(f"[eval.runner] hourly_history: {len(hourly):,} rows")

    print("[eval.runner] downloading calendar…")
    cal = _download_parquet(storage.RELEASE_AGGREGATES, CALENDAR_ASSET)
    cal["date"] = cal["date"].astype(str)
    print(f"[eval.runner] calendar: {len(cal):,} rows")

    weather_daily = _try_download_weather(storage.RELEASE_AGGREGATES)

    # Ensure target release exists
    _ensure_release_exists(RELEASE_EVAL, token)

    # Run the window
    result = _backtest_window(target_dates, hourly, cal, weather_daily)

    if not result["daily"]:
        print("[eval.runner] no daily results, nothing to upload")
        return

    # Aggregate per-station across the window for top/bottom views
    per_station_global = _aggregate_per_station(result["per_station"])

    # Build the JSON summary
    summary = {
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "rolling",
        "window_days": n_days,
        "global_algo": result["global_algo"],
        "global_baseline": result["global_baseline"],
        "daily": result["daily"],
        "top_10_best": per_station_global.nsmallest(10, "mae_fill")[
            ["station_id", "mae_fill", "acc_velib", "n"]
        ].to_dict(orient="records") if not per_station_global.empty else [],
        "top_10_worst": per_station_global.nlargest(10, "mae_fill")[
            ["station_id", "mae_fill", "acc_velib", "n"]
        ].to_dict(orient="records") if not per_station_global.empty else [],
    }

    # Upload
    print("[eval.runner] uploading rolling_metrics.json…")
    _upload_json_asset(RELEASE_EVAL, "rolling_metrics.json", summary)

    print("[eval.runner] uploading rolling_per_station.parquet…")
    _upload_parquet_asset(RELEASE_EVAL, "rolling_per_station.parquet", result["per_station"])

    print("[eval.runner] uploading rolling_calibration.parquet…")
    _upload_parquet_asset(RELEASE_EVAL, "rolling_calibration.parquet", result["calibration"])

    print("[eval.runner] DONE")
    print(
        f"[eval.runner] global ALGO     MAE={result['global_algo'].get('mae_fill_rate'):.4f}  "
        f"Acc(velib)={result['global_algo'].get('decision_accuracy_velib'):.3f}"
    )
    print(
        f"[eval.runner] global BASELINE MAE={result['global_baseline'].get('mae_fill_rate'):.4f}  "
        f"Acc(velib)={result['global_baseline'].get('decision_accuracy_velib'):.3f}"
    )


def run_full(n_days: int = 90) -> None:
    """One-shot full backtest. Uploads frozen baseline_<n>d_* assets.

    Use this once you are happy with rolling, to establish a long-term
    reference baseline that you can compare against after Phase 2 / Phase 3.
    """
    import os

    token = os.environ["GITHUB_TOKEN"]
    today = dt.date.today()
    target_dates = [today - dt.timedelta(days=i) for i in range(n_days, 0, -1)]

    print(
        f"[eval.runner] === Full backtest: {target_dates[0]} .. {target_dates[-1]} "
        f"({len(target_dates)} days) ==="
    )

    print("[eval.runner] downloading hourly_history…")
    hourly = _download_parquet(storage.RELEASE_AGGREGATES, HOURLY_ASSET)
    print(f"[eval.runner] hourly_history: {len(hourly):,} rows")

    print("[eval.runner] downloading calendar…")
    cal = _download_parquet(storage.RELEASE_AGGREGATES, CALENDAR_ASSET)
    cal["date"] = cal["date"].astype(str)

    weather_daily = _try_download_weather(storage.RELEASE_AGGREGATES)

    _ensure_release_exists(RELEASE_EVAL, token)

    result = _backtest_window(target_dates, hourly, cal, weather_daily)

    if not result["daily"]:
        print("[eval.runner] no daily results, nothing to upload")
        return

    per_station_global = _aggregate_per_station(result["per_station"])

    summary = {
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "full",
        "window_days": n_days,
        "global_algo": result["global_algo"],
        "global_baseline": result["global_baseline"],
        "daily": result["daily"],
        "top_10_best": per_station_global.nsmallest(10, "mae_fill")[
            ["station_id", "mae_fill", "acc_velib", "n"]
        ].to_dict(orient="records") if not per_station_global.empty else [],
        "top_10_worst": per_station_global.nlargest(10, "mae_fill")[
            ["station_id", "mae_fill", "acc_velib", "n"]
        ].to_dict(orient="records") if not per_station_global.empty else [],
    }

    suffix = f"baseline_{n_days}d"
    print(f"[eval.runner] uploading {suffix}_metrics.json…")
    _upload_json_asset(RELEASE_EVAL, f"{suffix}_metrics.json", summary)

    print(f"[eval.runner] uploading {suffix}_per_station.parquet…")
    _upload_parquet_asset(
        RELEASE_EVAL, f"{suffix}_per_station.parquet", result["per_station"]
    )

    print("[eval.runner] DONE")
