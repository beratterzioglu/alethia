# alethia

A small, dependency-light check for whether a cross-sectional trading signal is real or an artifact of data-mining, transaction cost, or look-ahead.

```python
import alethia

alethia.calibrate() # validate the check on known-truth controls
card = alethia.reality_check(predictions) # run your out-of-sample signal through the gauntlet
print(card)
```

`predictions` is a `DataFrame` with columns `time, sym, pred, y`, where `pred` is your model's cross-sectional prediction at each bar and `y` is the realised forward return.

The name is the Greek word for truth (ἀλήθεια).

## Calibrate first

A backtest is a measuring instrument, and most people never check that theirs gives the right answer on cases where the answer is known. `calibrate()` runs three controls:

```
alethia instrument calibration
  A. planted SIGNAL  -> detected              PASS   (+0.007)
  B. pure NOISE      -> not certified real    PASS   (+0.272)
  C. planted LEAK    -> flagged               PASS   (+1.000)
  => instrument CALIBRATED: verdicts are trustworthy
```

If the code detects a planted signal, refuses to certify pure noise, and catches a planted look-ahead, then its verdict on your real data means something.

## The report card

`reality_check` returns one verdict plus the numbers behind it:

```
alethia reality report card
===========================
verdict: REAL BUT WEAK: genuine relationship, not robust net-of-cost

  net sharpe         +0.91   90% CI [-0.19, +2.13]
  deflated sharpe    0.128   (>0.95 survives 20-trial deflation)
  psr [P(Sharpe>0)]  0.83
  permutation p      0.020   (<0.05: genuine, not luck or leak)
  info coefficient   +0.041   (t=+3.3)
  IC/P&L agree       yes
  turnover/rebal     1.51
  positive windows   3/6
```

What each line is checking:

| Check | Question |
|---|---|
| Net Sharpe + block-bootstrap CI | Does it clear zero after realistic cost, with autocorrelation-robust error bars? |
| Deflated Sharpe (Bailey and Lopez de Prado) | Does it survive the number of configs you tried? A high Sharpe out of 100 is expected under pure luck. |
| Permutation test | Shuffle the target; if the book still works, it is a bug or leak, not an edge. |
| IC and its t-stat | Is there any cross-sectional skill at all? |
| IC vs P&L agreement | A significant IC can still lose money if the relationship reverses in the high-dispersion periods that dominate dollar P&L. |
| Leak guard | A cross-sectional IC above ~0.4 is almost always look-ahead. |
| Positive windows | Does it hold across independent sub-periods, or lean on one lucky window? |

Verdicts: `REAL & ROBUST`, `REAL BUT WEAK`, `ARTIFACT` (significant IC but untradeable), `LIKELY NOISE`, `LEAK SUSPECTED`.

## Install

```bash
pip install -e .
```

Requires `numpy`, `pandas`, `scipy`.

## Try it

```bash
python examples/demo.py
python -m alethia.calibration
```

## Background

I'm not a quant by training. I spent about two years trying to find a real edge in crypto, learning most of it as I went. Early on I trusted a backtest that was quietly lying to me, and I lost real money believing it. After that I stopped chasing a winning strategy and started caring about one thing instead: being able to trust the answer when it was "no."

alethia is what came out of that. It checks itself on cases where the truth is already known before it says anything about your strategy, so that a "no" actually means something. If it saves one person the mistake I made, it did its job.

The full story, including a two-year search that honestly ended in "no edge," is in `STUDY.md`.

## References

- Bailey, Lopez de Prado, *The Deflated Sharpe Ratio* (2014)
- Bailey, Borwein, Lopez de Prado, Zhu, *The Probability of Backtest Overfitting* (2016)
- Lopez de Prado, *Advances in Financial Machine Learning* (2018)

MIT licensed.
