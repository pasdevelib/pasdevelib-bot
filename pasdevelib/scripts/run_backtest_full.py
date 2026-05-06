"""One-shot full backtest entry point.

Triggered manually via workflow_dispatch. Produces a frozen baseline that
serves as a long-term reference point (Phase 0 baseline before Phase 2 / 3
algorithmic changes).

Default window is 90 days. Override via env var EVAL_FULL_DAYS.
"""
from __future__ import annotations

import os
import sys

from pasdevelib.eval import runner


if __name__ == "__main__":
    n = int(os.environ.get("EVAL_FULL_DAYS", "90"))
    runner.run_full(n_days=n)
    sys.exit(0)
