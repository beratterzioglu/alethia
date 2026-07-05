"""Booking and information content: cross-sectional IC, a turnover-costed long/short book, and a
permutation test. IC (rank correlation) and book P&L (dollar-weighted) can disagree in sign; when
they do, the signal is non-monotone or heteroskedastic and does not translate to money."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cross_sectional_ic(df: pd.DataFrame) -> tuple[float, float]:
    """Mean per-bar Spearman IC of pred vs y, and its t-stat (df has columns time, pred, y)."""
    ics = []
    for _t, g in df.groupby("time"):
        if len(g) >= 8 and g["pred"].std() > 0 and g["y"].std() > 0:
            ic = spearmanr(g["pred"], g["y"]).correlation
            if np.isfinite(ic):
                ics.append(ic)
    ics = np.asarray(ics)
    if ics.size < 5:
        return 0.0, 0.0
    t = ics.mean() / ics.std() * np.sqrt(ics.size) if ics.std() > 0 else 0.0
    return float(ics.mean()), float(t)


def book_returns(df: pd.DataFrame, *, cost: float, n_side: int = 10) -> pd.Series:
    """Turnover-honest, dollar-neutral fixed-N long/short book from per-bar predictions.
    df columns: time, sym, pred, y (forward return). Longs the top n_side, shorts the bottom
    n_side, equal weight, gross 1.0, and charges cost times |weight change| each rebalance."""
    prev, out = {}, []
    for t, g in df.groupby("time"):
        if len(g) < 2 * n_side:
            out.append((t, 0.0))
            continue
        gg = g.sort_values("pred")
        S = gg.head(n_side)["sym"].tolist()
        L = gg.tail(n_side)["sym"].tolist()
        w = {s: 0.5 / n_side for s in L}
        w.update({s: -0.5 / n_side for s in S})
        fw = g.set_index("sym")["y"]
        pnl = float(sum(wt * float(fw.get(s, 0.0)) for s, wt in w.items()))
        turn = sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in set(w) | set(prev))
        out.append((t, pnl - turn * cost))
        prev = w
    return pd.Series(dict(out)).sort_index()


def turnover(df: pd.DataFrame, *, n_side: int = 10) -> float:
    """Average per-rebalance gross turnover of the fixed-N book (0 = hold, 2 = full flip)."""
    prev, ts = {}, []
    for _t, g in df.groupby("time"):
        if len(g) < 2 * n_side:
            continue
        gg = g.sort_values("pred")
        S = gg.head(n_side)["sym"].tolist()
        L = gg.tail(n_side)["sym"].tolist()
        w = {s: 0.5 / n_side for s in L}
        w.update({s: -0.5 / n_side for s in S})
        ts.append(sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in set(w) | set(prev)))
        prev = w
    return float(np.mean(ts)) if ts else 0.0


def permutation_pvalue(df: pd.DataFrame, *, cost: float, n_side: int, ppy: float,
                       n_perm: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """Shuffle y within each bar (destroying any real pred->return link) and rebuild the book many
    times. Returns (real gross Sharpe, shuffled mean Sharpe, p-value). p < 0.05 means the
    relationship is genuine, not a construction artifact or leak."""
    from ._stats import sharpe
    real = sharpe(book_returns(df, cost=0.0, n_side=n_side).to_numpy(), ppy)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for k in range(n_perm):
        d = df.copy()
        d["y"] = d.groupby("time")["y"].transform(lambda v: rng.permutation(v.to_numpy()))
        perm[k] = sharpe(book_returns(d, cost=0.0, n_side=n_side).to_numpy(), ppy)
    pval = float((np.sum(perm >= real) + 1) / (n_perm + 1))
    return real, float(perm.mean()), pval
