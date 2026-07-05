"""Validate the instrument on known-truth controls before trusting its verdicts. calibrate() runs
three checks: a planted signal must be detected, pure noise must not be certified real, and a
planted look-ahead must be flagged. If all pass, the gauntlet's verdict on real data is meaningful
because the same code just gave the right answer on cases where the truth was known."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gauntlet import reality_check


def _synthetic(kind: str, n_bars: int = 300, n_sym: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bars):
        t = pd.Timestamp("2025-01-01") + pd.Timedelta(days=b)
        p = rng.normal(size=n_sym)
        if kind == "signal":
            y = 0.003 * p + rng.normal(0, 0.015, n_sym)     # pred genuinely predicts y (realistic IC)
        elif kind == "noise":
            y = rng.normal(0, 0.01, n_sym)                  # y independent of pred
        elif kind == "leak":
            y = rng.normal(0, 0.01, n_sym)
            p = y.copy()                                    # pred is the future label
        for s in range(n_sym):
            rows.append({"time": t, "sym": s, "pred": float(p[s]), "y": float(y[s])})
    return pd.DataFrame(rows)


def calibrate(verbose: bool = True) -> bool:
    """Run the three controls. Returns True if the instrument is correctly calibrated."""
    sig = reality_check(_synthetic("signal", seed=1), n_perm=150, cost_bps=0.0)
    noi = reality_check(_synthetic("noise", seed=2), n_perm=150, cost_bps=0.0)
    lk = reality_check(_synthetic("leak", seed=3), n_perm=150, cost_bps=0.0)
    checks = [
        ("A. planted SIGNAL  -> detected", sig.permutation_p < 0.05 and "NOISE" not in sig.verdict, sig.permutation_p),
        ("B. pure NOISE      -> not certified real", "REAL" not in noi.verdict, noi.permutation_p),
        ("C. planted LEAK    -> flagged", "LEAK" in lk.verdict, lk.mean_ic),
    ]
    ok = all(passed for _n, passed, _v in checks)
    if verbose:
        print("alethia instrument calibration")
        for name, passed, val in checks:
            print(f"  {name:<40} {'PASS' if passed else 'FAIL'}   ({val:+.3f})")
        print(f"  => instrument {'CALIBRATED: verdicts are trustworthy' if ok else 'MIS-CALIBRATED: do not trust verdicts'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if calibrate() else 1)
