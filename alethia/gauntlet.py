"""Reality report card. Runs out-of-sample cross-sectional predictions through the gauntlet
(booking, deflation, permutation, IC diagnostics) and returns a single verdict."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import _book as bk
from . import _stats as st

_REQUIRED = {"time", "sym", "pred", "y"}


def _infer_ppy(times) -> float:
    t = pd.to_datetime(pd.Series(sorted(pd.unique(times))))
    dt = t.diff().dropna().dt.total_seconds().median()
    return float(365 * 24 * 3600 / dt) if dt and dt > 0 else 252.0


@dataclass
class ReportCard:
    verdict: str
    net_sharpe: float
    ci_low: float
    ci_high: float
    deflated_sharpe: float
    psr: float
    permutation_p: float
    mean_ic: float
    ic_tstat: float
    ic_pnl_agree: bool
    turnover: float
    positive_windows: str
    n_obs: int
    n_trials: int
    notes: list = field(default_factory=list)

    def __str__(self) -> str:
        L = [
            "alethia reality report card",
            "===========================",
            f"verdict: {self.verdict}",
            "",
            f"  net sharpe         {self.net_sharpe:+.2f}   90% CI [{self.ci_low:+.2f}, {self.ci_high:+.2f}]",
            f"  deflated sharpe    {self.deflated_sharpe:.3f}   (>0.95 survives {self.n_trials}-trial deflation)",
            f"  psr [P(Sharpe>0)]  {self.psr:.3f}",
            f"  permutation p      {self.permutation_p:.3f}   (<0.05: genuine, not luck or leak)",
            f"  info coefficient   {self.mean_ic:+.4f}   (t={self.ic_tstat:+.1f})",
            f"  IC/P&L agree       {'yes' if self.ic_pnl_agree else 'NO (heteroskedastic / non-monotone)'}",
            f"  turnover/rebal     {self.turnover:.2f}",
            f"  positive windows   {self.positive_windows}",
            f"  observations       {self.n_obs}",
        ]
        for n in self.notes:
            L.append(f"  ! {n}")
        return "\n".join(L)

    def to_markdown(self, path: str | None = None) -> str:
        md = (f"### alethia Reality Report Card\n\n**Verdict: {self.verdict}**\n\n"
              f"| metric | value | pass-bar |\n|---|---|---|\n"
              f"| Net Sharpe | {self.net_sharpe:+.2f} (CI {self.ci_low:+.2f}..{self.ci_high:+.2f}) | CI_low>0 |\n"
              f"| Deflated Sharpe | {self.deflated_sharpe:.3f} | >0.95 |\n"
              f"| PSR | {self.psr:.3f} | >0.95 |\n"
              f"| Permutation p | {self.permutation_p:.3f} | <0.05 |\n"
              f"| IC (t-stat) | {self.mean_ic:+.4f} ({self.ic_tstat:+.1f}) | \\|t\\|>3 |\n"
              f"| IC vs P&L agree | {'yes' if self.ic_pnl_agree else 'NO'} | yes |\n"
              f"| Turnover | {self.turnover:.2f} | n/a |\n"
              f"| Positive windows | {self.positive_windows} | >=5/6 |\n")
        if path:
            with open(path, "w") as f:
                f.write(md)
        return md


def reality_check(predictions: pd.DataFrame, *, cost_bps: float = 5.5, periods_per_year: float | None = None,
                  n_side: int = 10, n_trials: int = 10, trial_sharpe_spread: float = 1.0,
                  n_windows: int = 6, n_perm: int = 200) -> ReportCard:
    """Run the full gauntlet on out-of-sample predictions.
    predictions must have columns time, sym, pred, y (y = realised forward return)."""
    df = predictions
    if not _REQUIRED.issubset(df.columns):
        raise ValueError(f"predictions must have columns {_REQUIRED}, got {set(df.columns)}")
    ppy = periods_per_year or _infer_ppy(df["time"])
    cost = cost_bps / 1e4

    net = bk.book_returns(df, cost=cost, n_side=n_side)
    r = net.to_numpy()
    n_obs = len(r)
    bs = st.block_bootstrap_sharpe(r, periods_per_year=ppy)
    net_sh = st.sharpe(r, ppy)
    sr_std_perobs = trial_sharpe_spread / np.sqrt(ppy)
    dsr = st.deflated_sharpe(r, n_trials, sr_std_perobs)
    psr = st.psr(r)
    mean_ic, ic_t = bk.cross_sectional_ic(df)
    real_g, _perm_mean, pval = bk.permutation_pvalue(df, cost=cost, n_side=n_side, ppy=ppy, n_perm=n_perm)
    tov = bk.turnover(df, n_side=n_side)
    idx = np.sort(net.index)
    wins = np.array_split(idx, n_windows)
    pos = sum(1 for w in wins if st.sharpe(net.reindex(w).dropna().to_numpy(), ppy) > 0)
    # a monotone edge has an IC whose sign matches the long/short book's; if they disagree the
    # signal is non-monotone or heteroskedastic and the significant IC will not translate to money.
    ic_pnl_agree = (np.sign(mean_ic) == np.sign(real_g)) or abs(ic_t) < 2

    notes = []
    leak = abs(mean_ic) > 0.40
    if leak:
        notes.append("IC above 0.40 is not plausible for a real cross-sectional signal (typical is under 0.10); "
                     "most likely look-ahead. Check that pred uses no information from after each bar.")
    if not ic_pnl_agree:
        notes.append("IC and book P&L disagree in sign: the signal is non-monotone or heteroskedastic, so a "
                     "significant IC does not translate into money.")
    if tov > 1.0 and net_sh < real_g - 0.5:
        notes.append("Turnover cost is the binding constraint; most of the gross signal is eaten by trading.")

    if leak:
        verdict = "LEAK SUSPECTED: IC too high to be real; fix look-ahead before trusting anything"
    elif pval > 0.10:
        verdict = "LIKELY NOISE: indistinguishable from random (permutation p high)"
    elif not ic_pnl_agree:
        verdict = "ARTIFACT: significant IC but untradeable (heteroskedastic / non-monotone)"
    elif dsr > 0.95 and bs["ci_low"] > 0 and pos >= n_windows - 1:
        verdict = "REAL & ROBUST: survives cost, deflation, permutation and windows"
    elif pval < 0.05:
        verdict = "REAL BUT WEAK: genuine relationship, not robust net-of-cost or after deflation"
    else:
        verdict = "INCONCLUSIVE: needs more data or a cleaner signal"

    return ReportCard(verdict, net_sh, bs["ci_low"], bs["ci_high"], dsr, psr, pval, mean_ic, ic_t,
                      bool(ic_pnl_agree), tov, f"{pos}/{n_windows}", n_obs, n_trials, notes)
