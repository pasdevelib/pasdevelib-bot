"""Backtesting and evaluation framework for the pasdevelib prediction model.

Public API:
    from pasdevelib.eval import runner
    runner.run_rolling(n_days=7)
    runner.run_full_baseline(n_days=90)
"""
from . import backtest, baseline, metrics, runner

__all__ = ["backtest", "baseline", "metrics", "runner"]
