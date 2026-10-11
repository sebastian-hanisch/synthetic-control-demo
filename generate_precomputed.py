"""Erzeugt `precomputed_sweeps.json`: je Szenario und Zahl der Kontrollen sowie über die Stärke der neuen Regel die Kennzahlen der
vier Schätzer mit je 2 000 simulierten Welten (SplitMix64, Seeds fest).

    python generate_precomputed.py            # einige Minuten (parallel über die Zellen)"""

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import syn_constants as C
import syn_evaluation as E

OUT = Path(__file__).resolve().parent / "precomputed_sweeps.json"
WORLDS = 2000
CELLS = (
    [("scenario", s, j, C.DEFAULT_ZB) for s in C.SCENARIOS for j in C.J_CHOICES]
    + [("effect", "ruhig", j, zb) for j in (12, 30) for zb in (1.2, 1.4, 1.8, 2.0)]
)


def _cell(args):
    sweep, scenario, j, zb = args
    s = E.summarize(WORLDS, scenario, j, zb, tag=3)
    s["sweep"] = sweep
    return s


def main():
    t0 = time.time()
    with mp.Pool() as pool:
        cells = pool.map(_cell, CELLS, chunksize=1)
    meta = {"worlds": WORLDS, "periods": C.N_PERIODS, "t0": C.T0, "period_days": C.PERIOD_DAYS, "treated": C.TREATED,
            "shock_sd": C.SHOCK_SD, "trend_g": C.TREND_G}
    OUT.write_text(json.dumps({"meta": meta, "cells": cells}, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT.name}: {len(cells)} Zellen à {WORLDS} Welten, {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
