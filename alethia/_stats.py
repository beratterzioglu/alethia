"""Statistics core: Sharpe, PSR, Deflated Sharpe, PBO, block-bootstrap CI. numpy/scipy only.
Deflated Sharpe and PBO follow Bailey and Lopez de Prado."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649015329


def sharpe(returns, periods_per_year: float | None = None) -> float:
    """Annualised Sharpe ratio of a return series (per-observation if periods_per_year is None)."""
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    if r.size < 2 or r.std() == 0:
        return 0.0
    s = r.mean() / r.std()
    return float(s * np.sqrt(periods_per_year)) if periods_per_year else float(s)


def psr(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true per-obs Sharpe > sr_benchmark), correcting for sample
    length, skew and kurtosis (fat tails inflate an ordinary Sharpe's confidence)."""
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    n = r.size
    if n < 3 or r.std(ddof=1) == 0:
        return 0.5
    sr = r.mean() / r.std(ddof=1)
    sk = float(skew(r))
    ku = float(kurtosis(r, fisher=False))
    denom = np.sqrt(max(1e-12, 1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_trials_std: float) -> float:
    """E[max per-obs Sharpe] across n_trials iid random strategies (the bar a genuine edge must clear)."""
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_trials_std * ((1.0 - EULER) * a + EULER * b))


def deflated_sharpe(returns, n_trials: int, sr_trials_std: float) -> float:
    """Deflated Sharpe Ratio: PSR benchmarked against the expected max Sharpe under n_trials random
    trials. Corrects for selection bias and data mining. Above 0.95 survives deflation."""
    return psr(returns, sr_benchmark=expected_max_sharpe(n_trials, sr_trials_std))


def block_bootstrap_sharpe(returns, block: int = 10, n_boot: int = 1000, seed: int = 0,
                           periods_per_year: float | None = None) -> dict:
    """Autocorrelation-robust 90% CI for the Sharpe via the circular block bootstrap."""
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    n = r.size
    if n < block + 1:
        return {"ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    sr = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        sr[i] = sharpe(r[idx[:n]], periods_per_year)
    return {"ci_low": float(np.percentile(sr, 5)), "ci_high": float(np.percentile(sr, 95))}


def cscv_pbo(config_returns, n_splits: int = 10) -> float:
    """Probability of Backtest Overfitting (Combinatorially-Symmetric Cross-Validation). Takes a
    (T, C) matrix of per-bar returns for C candidate configs and returns the fraction of splits
    where the best in-sample config lands below the OOS median. Around 0.5 means the selection is
    overfit."""
    R = np.asarray(config_returns, dtype="float64")
    T, C = R.shape
    if C < 2 or n_splits % 2 != 0:
        return float("nan")
    chunks = np.array_split(np.arange(T), n_splits)
    below = total = 0
    for combo in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([chunks[j] for j in combo])
        oos_rows = np.concatenate([chunks[j] for j in range(n_splits) if j not in combo])
        is_sr = np.array([sharpe(R[is_rows, c]) for c in range(C)])
        oos_sr = np.array([sharpe(R[oos_rows, c]) for c in range(C)])
        best = int(np.argmax(is_sr))
        below += int(np.mean(oos_sr <= oos_sr[best]) < 0.5)
        total += 1
    return below / total if total else float("nan")
