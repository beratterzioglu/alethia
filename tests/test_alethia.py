"""alethia self-tests: the instrument must calibrate, and give the right verdict on known truth."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alethia  # noqa: E402
from alethia.calibration import _synthetic  # noqa: E402


def test_calibrates():
    assert alethia.calibrate(verbose=False) is True


def test_detects_planted_signal():
    card = alethia.reality_check(_synthetic("signal", seed=5), n_perm=120, cost_bps=0.0)
    assert card.permutation_p < 0.05 and "NOISE" not in card.verdict


def test_never_certifies_noise_as_real():
    # the essential property: pure noise must NEVER earn a "REAL" verdict (it may be NOISE or INCONCLUSIVE)
    for seed in (2, 6, 10, 11):
        card = alethia.reality_check(_synthetic("noise", seed=seed), n_perm=120, cost_bps=0.0)
        assert "REAL" not in card.verdict, f"noise seed {seed} wrongly certified: {card.verdict}"


def test_flags_leak():
    card = alethia.reality_check(_synthetic("leak", seed=7), n_perm=120, cost_bps=0.0)
    assert "LEAK" in card.verdict and card.mean_ic > 0.4


def test_stats_sanity():
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    assert 0.0 <= alethia.psr(r) <= 1.0
    assert alethia.deflated_sharpe(r, 20, 0.05) <= alethia.psr(r) + 1e-9   # deflation only lowers confidence


def test_report_card_renders():
    card = alethia.reality_check(_synthetic("signal", seed=8), n_perm=60, cost_bps=0.0)
    assert "reality report card" in str(card)
    assert "Verdict" in card.to_markdown()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
