# Flagship study - An honest audit of cross-sectional alpha in crypto perpetuals

*Application of the `alethia` protocol to a two-year search for a tradeable predictive edge in
Binance USDT-M perpetual futures.*

This document is the flagship application of `alethia`: not a strategy pitch, but a worked example
of what happens when you subject a "promising" backtest to a calibrated, honest gauntlet. Every
claim below is reproducible from the accompanying code.

## Setup

- **Universe:** Binance USDT-M perpetuals, filtered to *established* names (>= ~1.7y of history) to
 remove the survivorship/listing-pump contamination discussed in section 2.
- **Signal:** cross-sectional, price/volume/flow features -> walk-forward Ridge -> rank -> market-
 neutral long/short book, purged and embargoed.
- **Verdict engine:** `alethia.reality_check`, after `alethia.calibrate` confirmed the instrument
 detects planted signals, rejects noise, and catches leaks.

## The headline number was mostly contamination

A naive first pass showed a market-neutral Sharpe of ~ **2.3**. One honest correction at a time:

| correction | Sharpe |
|---|---|
| naive backtest | ~ 2.3 |
| remove freshly-listed **hype coins** (~ 40% of a "top-by-volume" universe; 8-34 days of history) | ~ 1.4 |
| fix construction (net-directional drift, concentration) + honest cost | ~ 0.9 |
| **multiple-testing deflation** (Deflated Sharpe over dozens of configs) | CI covers 0 |

Every rigorous probe moved the number the same direction - **down**. Monotone convergence like this
is not a bug; it is the signature of a working instrument closing in on a weak true value.

## A t-stat of -7.7 that still loses money

The most instructive finding. One feature (a low-volatility signal) had a cross-sectional
**Information Coefficient of -0.084, t = -7.7** - overwhelmingly significant. Its tradeable book
**lost money.**

Reconciliation: the rank-IC and the dollar P&L had **opposite signs**. IC is an equal-weighted-
per-bar rank correlation; P&L is dollar-weighted, and dollars are dominated by high-dispersion
days. The edge was **heteroskedastic** - clean on calm days (small stakes), and it *reversed* on the
volatile days that drive the P&L (corr of daily P&L with dispersion: **+0.39**).

> **Statistical significance of an IC is necessary but not sufficient for a tradeable edge.**
> IC lives in rank-space; money lives in dollar-space, and they are not the same space.

`alethia` encodes this as the **IC vs P&L agreement** check.

## You cannot stack your way out of it

The natural rescue - a meta-model (meta-labeling) to filter the primary's wrong calls - was tested
honestly. Out-of-sample AUC: **0.502**. No filtering power, identical to the failure mode of an
earlier hand-built version.

The reason is information-theoretic. A meta-gate cannot extract skill that isn't in the data. If the
primary already captured the (tiny) predictable component, the meta-layer on the same information
space has only noise left - and noise is, by definition, the part no model can predict. A *flawless*
gate would have to predict noise; that is not difficult, it is impossible.

## Was there any signal at all?

Yes - a permutation test confirmed it: the real book's gross Sharpe (+2.06) sat far outside the
shuffled-target distribution (mean +0.10), **p = 0.02**. The relationship is genuine. It simply
wasn't *harvestable*: clean predictive skill existed only at a 15-minute horizon, where transaction
costs destroy it; at tradeable horizons it was a fragile, heteroskedastic tail effect.

## Conclusion

For a retail participant using public price data, there was no robustly harvestable *predictive*
edge in this universe - a result established across ~10 independent checks and, crucially, produced
by an instrument that had first proven itself on known truth. The participants who profit in this
space mostly aren't forecasting price at all; they earn structural risk premia (funding, basis),
provide liquidity, or arbitrage. Prediction from public data is the most competed, least forgiving
corner of the market.

**The most valuable output of a research process is a trustworthy "no."** It is what a discipline
produces that a hope cannot.

---
*Reproduce: `python -m alethia.calibration`, then run `reality_check` on your own walk-forward
predictions. The crypto pipeline that produced the numbers above lives in the parent project.*
