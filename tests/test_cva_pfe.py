"""Tests for hedge IRS pricing, EE/PFE, CVA, SA-CCR prototype."""

from __future__ import annotations

import numpy as np

from src.cva_pfe import (
    compute_cva,
    par_swap_rate,
    run_hedge_ccr,
    saccr_irs_portfolio,
    simulate_exposure,
    swap_mtm_m,
    trades_from_signed_notionals,
)
from src.cva_pfe.irs_pricing import IRSTrade
from src.yield_curve import YieldCurve


def test_par_swap_mtm_near_zero():
    curve = YieldCurve()
    k = par_swap_rate(curve, 5.0)
    tr = IRSTrade(tenor_years=5.0, notional_m=100.0, pay_fixed=True, fixed_rate=k)
    assert abs(swap_mtm_m(curve, tr)) < 1e-6


def test_pay_fixed_gains_when_rates_rise():
    curve = YieldCurve()
    k = par_swap_rate(curve, 5.0)
    tr = IRSTrade(tenor_years=5.0, notional_m=100.0, pay_fixed=True, fixed_rate=k)
    bumped = YieldCurve(
        ref_tenors=list(curve._ref_tenors),
        ref_rates=[r + 0.01 for r in curve._ref_rates],
    )
    assert swap_mtm_m(bumped, tr) > 0.0


def test_trades_from_notionals_and_run():
    curve = YieldCurve()
    notionals = {2.0: 50.0, 5.0: -30.0, 10.0: 20.0}
    trades = trades_from_signed_notionals(notionals, curve)
    assert len(trades) == 3
    assert trades[0].pay_fixed is True
    assert trades[1].pay_fixed is False

    report = run_hedge_ccr(
        curve,
        notionals,
        n_paths=80,
        n_steps=8,
        rate_vol_bp=60.0,
        cds_spread_bp=120.0,
    )
    assert not report.trades_df.empty
    assert report.summary["n_trades"] == 3
    assert report.saccr.ead_m >= 0.0
    assert report.cva.cva_m >= 0.0
    assert len(report.exposure.times) > 2
    assert np.all(report.exposure.pfe_m >= report.exposure.ee_m - 1e-9)


def test_saccr_nonzero_addon():
    trades = [
        IRSTrade(tenor_years=5.0, notional_m=100.0, pay_fixed=True, fixed_rate=0.04),
        IRSTrade(tenor_years=10.0, notional_m=50.0, pay_fixed=False, fixed_rate=0.04),
    ]
    res = saccr_irs_portfolio(trades, mark_to_market_m=1.0, collateral_m=0.0)
    assert res.addon_m > 0
    assert res.ead_m > res.replacement_cost_m


def test_exposure_profile_shapes():
    curve = YieldCurve()
    trades = trades_from_signed_notionals({5.0: 100.0}, curve)
    prof = simulate_exposure(curve, trades, horizon_years=5.0, n_steps=5, n_paths=50)
    cva = compute_cva(curve, prof, cds_spread_bp=100.0)
    assert cva.cva_m >= 0.0
    assert cva.hazard_rate > 0.0


def test_prospective_effectiveness_fixed_bond_vs_pay_fixed():
    from src.cashflows import Instrument
    from src.cva_pfe.hedge_effectiveness import prospective_effectiveness
    from src.cva_pfe.irs_pricing import IRSTrade, par_swap_rate

    curve = YieldCurve()
    bond = Instrument(
        name="Fixed 5Y Gov",
        notional=100.0,
        coupon_pct=4.5,
        instrument_type="bullet_fixed",
        maturity_years=5.0,
        payment_freq=2,
        side="asset",
    )
    bond.generate_cashflows()
    k = par_swap_rate(curve, 5.0)
    swap = IRSTrade(tenor_years=5.0, notional_m=100.0, pay_fixed=True, fixed_rate=k)
    res = prospective_effectiveness(curve, bond, [swap], designation="fair_value")
    assert res.n_obs >= 8
    assert len(res.progressive) >= 2
    assert 0.0 <= res.r_squared <= 1.0
    # Pay-fixed vs fixed asset should show positive offset-form slope
    assert res.slope > 0.0
    assert res.progressive[-1]["n_obs"] == res.n_obs


def test_progressive_expands_sample():
    from src.cashflows import Instrument
    from src.cva_pfe.hedge_effectiveness import prospective_effectiveness
    from src.cva_pfe.irs_pricing import IRSTrade, par_swap_rate

    curve = YieldCurve()
    bond = Instrument(
        "B", 50.0, 4.0, "bullet_fixed", 10.0, payment_freq=2, side="asset",
    )
    bond.generate_cashflows()
    k = par_swap_rate(curve, 10.0)
    res = prospective_effectiveness(
        curve,
        bond,
        [IRSTrade(10.0, 50.0, True, k)],
        shocks_bp=[-100, -50, 50, 100],
    )
    ns = [p["n_obs"] for p in res.progressive]
    assert ns == sorted(ns)
    assert ns[0] == 2
    assert ns[-1] == 4
