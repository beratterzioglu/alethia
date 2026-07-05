"""alethia in 30 seconds. Calibrate the instrument, then run three strategies through the gauntlet:
a real signal, pure noise, and a look-ahead leak, and watch the report card tell them apart.

    python examples/demo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alethia


def make(kind, n_bars=300, n_sym=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bars):
        t = pd.Timestamp("2025-01-01") + pd.Timedelta(days=b)
        p = rng.normal(size=n_sym)
        if kind == "signal":
            y = 0.003 * p + rng.normal(0, 0.015, n_sym)
        elif kind == "noise":
            y = rng.normal(0, 0.01, n_sym)
        elif kind == "leak":
            y = rng.normal(0, 0.01, n_sym)
            p = y.copy()
        for s in range(n_sym):
            rows.append({"time": t, "sym": s, "pred": float(p[s]), "y": float(y[s])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("STEP 1 - validate the instrument on known truth:\n")
    alethia.calibrate()
    for kind in ("signal", "noise", "leak"):
        print(f"\nSTEP 2 - reality-check a '{kind}' strategy:\n")
        print(alethia.reality_check(make(kind, seed=42), n_perm=150, cost_bps=2.0, n_trials=20))
