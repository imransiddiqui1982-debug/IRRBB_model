"""Tests for live SOFR / USD IRS market curve builder."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.market_curve import (  # noqa: E402
    MarketCurveSnapshot,
    build_live_curve_snapshot,
    get_live_yield_curve,
)
from src.calculator import IRRBBCalculator  # noqa: E402
from src.balance_sheet import get_instruments  # noqa: E402
from src.scenarios import SCENARIOS  # noqa: E402


def test_snapshot_to_yield_curve():
    snap = MarketCurveSnapshot(
        as_of="2026-09-01",
        points={0.0: 0.036, 0.25: 0.038, 1.0: 0.041, 5.0: 0.043, 10.0: 0.044},
    )
    curve = snap.to_yield_curve()
    assert len(curve.base_rates) == 19
    assert 0.035 < curve.base_rates[0] < 0.04


def test_live_curve_fetch_or_cache():
    try:
        curve, snap = get_live_yield_curve(use_cache_on_failure=True)
    except Exception as exc:
        pytest.skip(f"Network unavailable: {exc}")
    assert snap.points
    assert 0.0 in snap.points or min(snap.points) <= 0.01
    assert any(t >= 10.0 for t in snap.points)
    assert len(curve.base_rates) == 19


def test_eve_nii_recalc_with_live_like_curve():
    snap = MarketCurveSnapshot(
        as_of="test",
        points={
            0.0: 0.0366, 1 / 12: 0.0374, 0.25: 0.0384, 0.5: 0.0396,
            1.0: 0.0408, 2.0: 0.0421, 3.0: 0.0427, 5.0: 0.0428,
            7.0: 0.0433, 10.0: 0.0441,
        },
    )
    curve = snap.to_yield_curve()
    assets, liabilities = get_instruments()
    calc = IRRBBCalculator(assets, liabilities, tier1_capital=500, yield_curve=curve)
    results = calc.run_all(SCENARIOS)
    assert len(results) == 6
    assert all(isinstance(r.delta_eve, float) for r in results)
    assert all(isinstance(r.delta_nii, float) for r in results)
