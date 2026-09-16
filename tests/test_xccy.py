"""Cross-currency IRS smoke tests (numpy path; no pandas/rateslib required)."""

from __future__ import annotations

from src.xccy.engine import XccyTradeSpec, interpret_user_request, price_xccy_numpy
from src.xccy.market_data import XccyMarketSnapshot


def test_interpret_user_request():
    spec = interpret_user_request("10mm EUR 5Y quarterly Act/360")
    assert spec.pair == "EURUSD"
    assert spec.notional == 10_000_000.0
    assert spec.tenor == "5Y"
    assert spec.frequency == "Q"


def test_numpy_xccy_fair_basis_near_zero_npv():
    snap = XccyMarketSnapshot(
        as_of="2026-01-01",
        spot=1.10,
        sofr_on=0.04,
        estr_on=0.02,
        usd_curve={"ON": 0.04, "1Y": 0.039, "2Y": 0.038, "5Y": 0.037, "10Y": 0.036},
        eur_curve={"ON": 0.02, "1Y": 0.019, "2Y": 0.018, "5Y": 0.017, "10Y": 0.016},
        fx_forward={"5Y": 1.20},
        fx_swap_pts={"5Y": 1000.0},
        xccy_basis_bp={"1Y": -10.0, "2Y": -12.0, "5Y": -15.0, "10Y": -18.0},
        source_notes=["unit test"],
        basis_is_live=False,
        forwards_are_cip=True,
    )
    trade = XccyTradeSpec(
        pair="EURUSD",
        notional_ccy="EUR",
        notional=10_000_000.0,
        tenor="5Y",
        frequency="Q",
        float_spread_bp=None,
    )
    res = price_xccy_numpy(snap, trade)
    assert res.engine == "numpy_fallback"
    assert abs(res.fair_basis_bp) < 250.0
    # At fair basis, NPV should be near zero
    trade2 = XccyTradeSpec(
        pair="EURUSD",
        notional_ccy="EUR",
        notional=10_000_000.0,
        tenor="5Y",
        frequency="Q",
        float_spread_bp=res.fair_basis_bp,
    )
    res2 = price_xccy_numpy(snap, trade2)
    assert abs(res2.npv_usd) < 50_000.0  # near-zero at fair basis on 10mm
