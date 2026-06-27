"""scrape_loop.py — Lance plusieurs scrapes en boucle dans un seul job GitHub Actions.

Usage : python -m pasdevelib.scrape_loop --duration 240 --interval 30
  --duration : durée totale de la boucle en secondes (défaut 240 = 4 min)
  --interval : intervalle entre deux scrapes en secondes (défaut 30)
"""
from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime, timezone

from pasdevelib.scrape import run as scrape_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=240)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    start = time.monotonic()
    n_ok = 0
    n_err = 0

    print(f"[scrape_loop] démarrage — durée={args.duration}s, intervalle={args.interval}s")

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= args.duration:
            break

        t0 = time.monotonic()
        try:
            scrape_once()
            n_ok += 1
        except Exception:
            n_err += 1
            print(f"[scrape_loop] ERREUR scrape :")
            traceback.print_exc()

        # Attendre jusqu'au prochain intervalle
        took = time.monotonic() - t0
        wait = max(0, args.interval - took)
        
        elapsed_after = time.monotonic() - start
        remaining = args.duration - elapsed_after
        if remaining <= 1:
            break
        
        actual_wait = min(wait, remaining - 1)
        if actual_wait > 0:
            time.sleep(actual_wait)

    total = time.monotonic() - start
    print(f"[scrape_loop] terminé — {n_ok} OK, {n_err} erreurs, {total:.0f}s écoulées")


if __name__ == "__main__":
    main()
