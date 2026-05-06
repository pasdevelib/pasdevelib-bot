"""Daily rolling backtest entry point.

Re-runs the last 7 days each call. Idempotent: previous days' metrics are
overwritten with their (deterministic) values, plus the most recent day
gets added.

Triggered by .github/workflows/eval-rolling.yml at 06:00 UTC, after the
forecast workflow at 05:00 UTC.
"""
from __future__ import annotations

import sys

from pasdevelib.eval import runner


if __name__ == "__main__":
    runner.run_rolling(n_days=7)
    sys.exit(0)
