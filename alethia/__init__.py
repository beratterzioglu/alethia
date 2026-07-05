"""alethia: a trust-first check for cross-sectional trading signals.

Given out-of-sample predictions, it reports whether an edge is real and robust or an artifact of
data-mining, cost, or look-ahead, and it validates the check itself on known-truth controls first.

    import alethia
    alethia.calibrate()
    card = alethia.reality_check(predictions)    # predictions: columns time, sym, pred, y
    print(card)

y is the realised forward return. The gauntlet applies turnover-honest booking, block-bootstrap
confidence intervals, Deflated Sharpe (Bailey and Lopez de Prado), PSR, a permutation/leak test,
cross-sectional IC with its t-stat, an IC-vs-P&L sign check, and per-window robustness.
"""
from ._book import book_returns, cross_sectional_ic, permutation_pvalue, turnover
from ._stats import (block_bootstrap_sharpe, cscv_pbo, deflated_sharpe,
                     expected_max_sharpe, psr, sharpe)
from .calibration import calibrate
from .gauntlet import ReportCard, reality_check

__version__ = "0.1.0"
__all__ = [
    "calibrate", "reality_check", "ReportCard",
    "book_returns", "cross_sectional_ic", "permutation_pvalue", "turnover",
    "sharpe", "psr", "deflated_sharpe", "expected_max_sharpe", "block_bootstrap_sharpe", "cscv_pbo",
]
