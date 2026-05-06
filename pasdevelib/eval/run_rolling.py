"""Daily rolling backtest entry point.

Re-runs the last 7 days each call. Idempotent.

Invoked from .github/workflows/eval-rolling.yml as:
    python -m pasdevelib.eval.run_rolling
"""
from __future__ import annotations

import sys

from pasdevelib.eval import runner


if __name__ == "__main__":
    runner.run_rolling(n_days=7)
    sys.exit(0)
