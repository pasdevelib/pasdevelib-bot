"""One-shot full backtest entry point (90 days by default).

Invoked from .github/workflows/eval-full.yml as:
    python -m pasdevelib.eval.run_full

Override the window via env var EVAL_FULL_DAYS (e.g. 30, 60, 180).
"""
from __future__ import annotations

import os
import sys

from pasdevelib.eval import runner


if __name__ == "__main__":
    n = int(os.environ.get("EVAL_FULL_DAYS", "90"))
    runner.run_full(n_days=n)
    sys.exit(0)
