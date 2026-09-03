"""Tests for mortgage CPR / prepayment methodology."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.balance_sheet import get_instruments  # noqa: E402
from src.calculator import IRRBBCalculator  # noqa: E402
from src.cashflows import Instrument  # noqa: E402
from src.lcr_calculator import compute_cash_inflows, compute_lcr  # noqa: E402
from src.nsfr_calculator import compute_nsfr, _residual_maturity_years  # noqa: E402
from src.prepayment import (  # noqa: E402
    cpr_to_smm,
    enhance_cpr_with_hf_chronos,
    incentive_cpr,
    psa_cpr,
    s_curve_cpr,
    scenario_market_mortgage_rate,
)
from src.scenarios import SCENARIO_MAP  # noqa: E402
from src.yield_curve import YieldCurve  # noqa: E402


def test_cpr_to_smm_roundtrip():
    cpr = 0.06
    smm = cpr_to_smm(cpr, 12)
    assert 0 < smm < cpr
    assert abs((1 - (1 - smm) ** 12) - cpr) < 1e-9


def test_psa_ramp():
    assert psa_cpr(0) == 0.0
    assert abs(psa_cpr(15) - 0.03) < 1e-9
    assert abs(psa_cpr(30) - 0.06) < 1e-9
    assert abs(psa_cpr(60) - 0.06) < 1e-9


def test_s_curve_higher_when_in_the_money():
    low = s_curve_cpr(-0.02)
    mid = s_curve_cpr(0.005)
    high = s_curve_cpr(0.03)
    assert low < mid < high


def test_incentive_cpr_monotonic():
    assert incentive_cpr(5.0, 0.07, age_months=30) < incentive_cpr(5.0, 0.03, age_months=30)


def test_hf_chronos_fallback_without_package():
    base = 0.12
    out = enhance_cpr_with_hf_chronos([0.05] * 12, base)
    assert out == base


def test_amortising_with_cpr_principal_sums():
    inst = Instrument(
        "Mortgage Test", 120, 5.0, "amortising", 1.0,
        payment_freq=12, side="asset",
        prepay_enabled=True, base_cpr=0.12, age_months=30,
    )
    total = sum(
        cf.amount for cf in inst.cashflows
        if cf.cf_type in ("principal", "prepayment")
    )
    assert abs(total - 120) < 1e-6
    prep = sum(cf.amount for cf in inst.cashflows if cf.cf_type == "prepayment")
    assert prep > 0


def test_cpr_shortens_wal_vs_static():
    static = Instrument(
        "Static", 100, 5.0, "amortising", 10.0,
        payment_freq=12, side="asset", prepay_enabled=False,
    )
    prepaid = Instrument(
        "CPR", 100, 5.0, "amortising", 10.0,
        payment_freq=12, side="asset",
        prepay_enabled=True, base_cpr=0.20, age_months=30,
    )
    assert prepaid.wal_years() < static.wal_years()
    assert prepaid.effective_duration < static.effective_duration


def test_mortgage_auto_enables_prepay_by_name():
    inst = Instrument(
        "Fixed-Rate Mortgages (10Y)", 100, 5.2, "amortising", 10.0,
        payment_freq=12, side="asset",
    )
    assert inst.prepay_enabled is True
    assert any(cf.cf_type == "prepayment" for cf in inst.cashflows)


def test_eve_negative_convexity_rates_down_vs_static():
    curve = YieldCurve(ref_tenors=[0, 30], ref_rates=[0.05, 0.05])
    static = Instrument(
        "Mtg Static", 800, 5.2, "amortising", 10.0,
        payment_freq=12, side="asset", prepay_enabled=False,
    )
    prepaid = Instrument(
        "Mtg CPR", 800, 5.2, "amortising", 10.0,
        payment_freq=12, side="asset",
        prepay_enabled=True, age_months=36,
    )
    liab = Instrument(
        "Funding", 800, 1.0, "demand_deposit", 1 / 12,
        repricing_years=1 / 12, side="liability",
    )
    calc_static = IRRBBCalculator([static], [liab], tier1_capital=500, yield_curve=curve)
    calc_cpr = IRRBBCalculator([prepaid], [liab], tier1_capital=500, yield_curve=curve)
    down = SCENARIO_MAP["PS_DOWN"]
    eve_static, _, _ = calc_static.calc_eve(down)
    eve_cpr, _, _ = calc_cpr.calc_eve(down)
    assert eve_static > 0
    assert eve_cpr > 0
    assert eve_cpr < eve_static


def test_eve_extension_risk_rates_up():
    """
    Under parallel up, freezing CPR at the base-case speed understates the
    PV loss versus regenerating CPR (lower CPR → longer cash flows).
    """
    curve = YieldCurve(ref_tenors=[0, 30], ref_rates=[0.05, 0.05])
    prepaid = Instrument(
        "Mtg CPR", 800, 5.2, "amortising", 10.0,
        payment_freq=12, side="asset",
        prepay_enabled=True, age_months=36,
    )
    liab = Instrument(
        "Funding", 800, 1.0, "demand_deposit", 1 / 12,
        repricing_years=1 / 12, side="liability",
    )
    calc = IRRBBCalculator([prepaid], [liab], tier1_capital=500, yield_curve=curve)
    up = SCENARIO_MAP["PS_UP"]

    # Full behavioural EVE (CPR falls when rates rise)
    eve_behavioural, _, _ = calc.calc_eve(up)

    # Counterfactual: same discount shock but CPR frozen at base schedule
    mkt_base = scenario_market_mortgage_rate(curve.base_rates, None, 10.0)
    frozen_cfs = np.array([prepaid.bucket_cashflows_under_market_rate(mkt_base)])
    liab_cfs = calc._liab_cfs
    pv_base_a = calc._pv_matrix(calc._asset_cfs, None)
    pv_shock_a = calc._pv_matrix(frozen_cfs, up.shocks_bp)
    pv_base_l = calc._pv_matrix(liab_cfs, None)
    pv_shock_l = calc._pv_matrix(liab_cfs, up.shocks_bp)
    eve_frozen = float(np.sum(pv_shock_a - pv_base_a) - np.sum(pv_shock_l - pv_base_l))

    assert eve_behavioural < 0
    assert eve_frozen < 0
    # Extension risk: behavioural (lower CPR) loses at least as much
    assert eve_behavioural <= eve_frozen + 1e-6


def test_lcr_inflows_include_cpr_principal():
    inst = Instrument(
        "Fixed-Rate Mortgages", 800, 5.2, "amortising", 10.0,
        payment_freq=12, side="asset",
        prepay_enabled=True, base_cpr=0.24, age_months=30,
    )
    total, df = compute_cash_inflows([inst])
    assert total > 0
    assert not df.empty
    assert "CPR" in df.iloc[0]["Category"] or "Mortgage" in df.iloc[0]["Category"]


def test_nsfr_wal_uses_cpr():
    prepaid = Instrument(
        "Fixed-Rate Mortgages", 800, 5.2, "amortising", 10.0,
        payment_freq=12, side="asset",
        prepay_enabled=True, base_cpr=0.30, age_months=30,
    )
    static = Instrument(
        "Fixed-Rate Mortgages Static", 800, 5.2, "amortising", 10.0,
        payment_freq=12, side="asset", prepay_enabled=False,
    )
    assert _residual_maturity_years(prepaid) < _residual_maturity_years(static)


def test_sample_book_still_runs():
    assets, liabs = get_instruments()
    mtg = next(i for i in assets if "Mortgage" in i.name)
    assert mtg.prepay_enabled
    calc = IRRBBCalculator(assets, liabs, tier1_capital=500)
    results = calc.run_all([SCENARIO_MAP["PS_UP"], SCENARIO_MAP["PS_DOWN"]])
    assert len(results) == 2
    lcr = compute_lcr(assets, liabs)
    nsfr = compute_nsfr(assets, liabs, tier1_capital=500)
    assert lcr.lcr_pct > 0
    assert nsfr.nsfr_pct > 0


def test_scenario_market_rate_moves_with_shock():
    curve = YieldCurve()
    base = scenario_market_mortgage_rate(curve.base_rates, None, 10.0)
    up = scenario_market_mortgage_rate(curve.base_rates, [200] * 19, 10.0)
    assert up > base
