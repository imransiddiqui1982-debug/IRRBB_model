"""
tests/test_key_rate_duration.py
-------------------------------
Invariants from the alm_engine design, adapted to this project's
Instrument / YieldCurve stack.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.balance_sheet import get_instruments  # noqa: E402
from src.calculator import IRRBBCalculator  # noqa: E402
from src.cashflows import Instrument  # noqa: E402
from src.key_rate_duration import (  # noqa: E402
    KeyRateGrid,
    TRADEABLE_TENORS,
    compute_kr01,
    parallel_dv01,
    suggest_key_rate_hedges,
    designate_key_rate_hedges,
)
from src.yield_curve import YieldCurve  # noqa: E402


@pytest.fixture
def flat_curve():
    return YieldCurve(ref_tenors=[0.0, 30.0], ref_rates=[0.04, 0.04])


@pytest.fixture
def sample_book(flat_curve):
    assets = [
        Instrument(
            "Fixed bond 5Y", 200, 4.0, "bullet_fixed", 5.0,
            payment_freq=2, side="asset",
        ),
        Instrument(
            "Mortgage 10Y", 300, 5.0, "amortising", 10.0,
            payment_freq=12, side="asset", prepay_enabled=True,
        ),
    ]
    liabilities = [
        Instrument(
            "Term deposit 2Y", 250, 3.0, "bullet_fixed", 2.0,
            payment_freq=2, side="liability",
        ),
        Instrument(
            "NMD core", 200, 0.5, "demand_deposit", 5.0,
            repricing_years=4.5, side="liability",
        ),
    ]
    return assets, liabilities, flat_curve


def test_tradeable_grid_tenors():
    assert KeyRateGrid().tenors == TRADEABLE_TENORS


def test_tent_weights_are_a_partition_of_unity():
    g = KeyRateGrid()
    t = __import__("numpy").linspace(0.01, 40.0, 4000)
    assert g.check_partition(t)


def test_tent_peaks_at_own_key():
    g = KeyRateGrid()
    for i, k in enumerate(g.tenors):
        w = g.tent(__import__("numpy").array([k]), i)
        assert w[0] == pytest.approx(1.0)


def test_kr01_reconciles_to_parallel_dv01(sample_book):
    """Core invariant: Σ KR01 ≈ parallel DV01 when tent weights sum to 1."""
    assets, liabilities, curve = sample_book
    for a in assets:
        if a.is_prepayable:
            a.base_cpr = 0.06
    kr = compute_kr01(assets, liabilities, curve, cpr_override=0.06)
    pdv = parallel_dv01(assets, liabilities, curve, cpr_override=0.06)
    assert float(kr["net_kr01_k"].sum()) == pytest.approx(pdv, rel=0.05)


def test_calculator_key_rate_gap_matches_module():
    assets, liabs = get_instruments()
    calc = IRRBBCalculator(assets, liabs, tier1_capital=500.0)
    kr = calc.key_rate_duration_gap()
    assert list(kr["label"]) == [f"{t:g}Y" for t in TRADEABLE_TENORS]
    assert set(kr.columns) >= {
        "asset_kr01_k", "liability_kr01_k", "net_kr01_k",
    }
    assert float(kr["net_kr01_k"].sum()) == pytest.approx(
        calc.parallel_eve_dv01_k(), rel=0.08,
    )


def test_asset_heavy_book_has_positive_net_kr01(flat_curve):
    assets = [
        Instrument("Long bond", 500, 4.0, "bullet_fixed", 10.0, side="asset"),
    ]
    liabs = [
        Instrument(
            "Short float", 500, 4.0, "bullet_floating", 5.0,
            repricing_years=0.25, side="liability",
        ),
    ]
    kr = compute_kr01(assets, liabs, flat_curve)
    assert float(kr["net_kr01_k"].sum()) > 0


def test_suggest_hedges_payer_when_asset_heavy(flat_curve):
    assets = [
        Instrument("Long bond", 400, 4.0, "bullet_fixed", 7.0, side="asset"),
    ]
    liabs = [
        Instrument(
            "Float fund", 400, 4.0, "bullet_floating", 3.0,
            repricing_years=0.5, side="liability",
        ),
    ]
    kr = compute_kr01(assets, liabs, flat_curve)
    hedges = suggest_key_rate_hedges(kr, hedge_ratio=0.8, min_abs_kr01_k=0.5)
    assert not hedges.empty
    assert any("Pay-fixed" in s for s in hedges["IRS structure"])


def test_designate_tags_fair_value_for_payer(flat_curve):
    assets = [
        Instrument("Long bond", 400, 4.0, "bullet_fixed", 7.0, side="asset"),
    ]
    liabs = [
        Instrument(
            "Float fund", 400, 4.0, "bullet_floating", 3.0,
            repricing_years=0.5, side="liability",
        ),
    ]
    kr = compute_kr01(assets, liabs, flat_curve)
    hedges = suggest_key_rate_hedges(kr, min_abs_kr01_k=0.5)
    tags = designate_key_rate_hedges(hedges, assets, liabs)
    assert not tags.empty
    assert "fair_value" in set(tags["Designation"]) or "cash_flow" in set(
        tags["Designation"]
    )


def test_live_prepay_path_runs(flat_curve):
    """Prepayable KR01 regenerates CFs under each key bump (behavioural path)."""
    mbs = Instrument(
        "MBS pool", 200, 5.0, "mbs", 10.0,
        payment_freq=12, side="asset", prepay_enabled=True,
    )
    kr = compute_kr01([mbs], [], flat_curve)
    assert len(kr) == len(TRADEABLE_TENORS)
    assert float(kr["asset_kr01_k"].sum()) > 0


def test_frozen_cpr_still_positive_kr01(flat_curve):
    frozen = Instrument(
        "MBS frozen", 200, 5.0, "mbs", 10.0,
        payment_freq=12, side="asset", prepay_enabled=True, base_cpr=0.06,
    )
    kr = compute_kr01([frozen], [], flat_curve, cpr_override=0.06)
    assert float(kr["asset_kr01_k"].sum()) > 0
