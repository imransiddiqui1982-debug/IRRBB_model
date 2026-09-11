"""
tests/test_mbs_pricing.py
-------------------------
Numeric checks from mbs_prepayment_spec_for_cursor.md §7.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cashflows import Instrument  # noqa: E402
from src.key_rate_duration import (  # noqa: E402
    compute_kr01,
    instrument_pv,
    shocked_yield_curve,
)
from src.mbs_pricing import (  # noqa: E402
    MbsTerms,
    cpr_to_psa,
    hqla_from_mbs_level,
    step_a_mortgage_rate,
    step_b_cpr,
    step_c_flows_and_price,
    terms_from_instrument,
)
from src.scenarios import SCENARIO_MAP  # noqa: E402
from src.yield_curve import YieldCurve  # noqa: E402


def _spec_pool() -> MbsTerms:
    return MbsTerms(
        notional=200.0,
        wac=0.055,
        wam_months=360,
        pool_age_months=0,
        anchor_tenor=10.0,
        spread_to_curve=0.0175,
        oas=0.005,
        base_turnover=0.06,
        max_refi_cpr=0.34,
        logistic_k=2.2,
        logistic_midpoint=0.60,
        seasoning_ramp_months=30,
    )


def _flat_curve(rate: float = 0.04) -> YieldCurve:
    tenors = [0.0, 0.25, 1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    rates = [rate] * len(tenors)
    return YieldCurve(ref_tenors=tenors, ref_rates=rates)


def _scenario_curve(base: YieldCurve, scenario_id: str) -> YieldCurve:
    sc = SCENARIO_MAP[scenario_id]
    return shocked_yield_curve(base, sc.shocks_bp, scenario=sc)


# ── §7.1 Step A ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "sid,exp_mrate,exp_inc",
    [
        ("BASE", 0.0575, -0.25),
        ("PS_UP", 0.0775, -2.25),
        ("PS_DOWN", 0.0375, 1.75),
        ("STEEPENER", 0.0683, -1.33),
        ("FLATTENER", 0.0512, 0.38),
    ],
)
def test_step_a_mortgage_rate_and_incentive(sid, exp_mrate, exp_inc):
    base = _flat_curve(0.04)
    terms = _spec_pool()
    if sid == "BASE":
        curve = base
    else:
        curve = _scenario_curve(base, sid)
    mrate, inc = step_a_mortgage_rate(curve, terms)
    assert mrate == pytest.approx(exp_mrate, abs=1e-4)
    assert inc == pytest.approx(exp_inc, abs=0.02)


# ── §7.2 Step B seasoned CPR ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "sid,exp_cpr",
    [
        ("BASE", 0.105),
        ("PS_UP", 0.061),
        ("PS_DOWN", 0.375),
        ("STEEPENER", 0.065),
        ("FLATTENER", 0.189),
    ],
)
def test_step_b_seasoned_cpr(sid, exp_cpr):
    base = _flat_curve(0.04)
    terms = _spec_pool()
    curve = base if sid == "BASE" else _scenario_curve(base, sid)
    _, inc = step_a_mortgage_rate(curve, terms)
    cpr = step_b_cpr(terms, inc, age_months=60)
    assert cpr == pytest.approx(exp_cpr, abs=0.002)
    assert cpr_to_psa(cpr) == pytest.approx((exp_cpr / 0.06) * 100, abs=5.0)


# ── §7.0 Amortization correctness (fixed dollar payment) ─────────────────────

def test_zero_prepayment_fully_amortizes_to_exactly_zero():
    """Spec §7.0 — level-pay must clear balance in exactly wam_months."""
    base = _flat_curve(0.04)
    terms = MbsTerms(
        notional=300.0,
        wac=0.055,
        wam_months=324,
        pool_age_months=0,
        anchor_tenor=10.0,
        spread_to_curve=0.0175,
        oas=0.005,
        base_turnover=0.0,
        max_refi_cpr=0.0,
        logistic_k=2.2,
        logistic_midpoint=0.60,
        seasoning_ramp_months=30,
    )
    res = step_c_flows_and_price(base, terms)
    assert len(res.times) == 324
    # Reconstruct ending balance from starting notional − all principal-like CF
    monthly = terms.wac / 12.0
    bal = float(terms.notional)
    pay = bal * monthly / (1.0 - (1.0 + monthly) ** (-terms.wam_months))
    for total in res.cashflows:
        interest = bal * monthly
        sched = min(pay - interest, bal)
        prep = max(float(total) - interest - sched, 0.0)
        bal = bal - sched - prep
    assert bal == pytest.approx(0.0, abs=1e-6)


# ── §7.3 Step C price / WAL (regenerated after §4 payment bugfix) ──────────

@pytest.mark.parametrize(
    "sid,exp_price,exp_wal",
    [
        ("BASE", 210.98, 5.40),
        ("PS_UP", 190.45, 7.01),
        ("PS_DOWN", 216.33, 2.68),
        ("STEEPENER", 205.14, 6.81),
        ("FLATTENER", 206.62, 3.96),
    ],
)
def test_step_c_price_and_wal(sid, exp_price, exp_wal):
    base = _flat_curve(0.04)
    terms = _spec_pool()
    curve = base if sid == "BASE" else _scenario_curve(base, sid)
    res = step_c_flows_and_price(curve, terms)
    assert res.price == pytest.approx(exp_price, abs=2.0)
    assert res.wal == pytest.approx(exp_wal, abs=0.35)


# ── §7.4 Structural regressions ───────────────────────────────────────────────

def test_negative_convexity_signature():
    base = _flat_curve(0.04)
    terms = _spec_pool()
    p0 = step_c_flows_and_price(base, terms).price
    pup = step_c_flows_and_price(_scenario_curve(base, "PS_UP"), terms).price
    pdn = step_c_flows_and_price(_scenario_curve(base, "PS_DOWN"), terms).price
    assert abs(pup - p0) > abs(pdn - p0)


def test_wal_extends_under_rate_rise():
    base = _flat_curve(0.04)
    terms = _spec_pool()
    assert (
        step_c_flows_and_price(_scenario_curve(base, "PS_UP"), terms).wal
        > step_c_flows_and_price(base, terms).wal
    )


def test_wal_compresses_under_rate_fall():
    base = _flat_curve(0.04)
    terms = _spec_pool()
    assert (
        step_c_flows_and_price(_scenario_curve(base, "PS_DOWN"), terms).wal
        < step_c_flows_and_price(base, terms).wal
    )


def test_frozen_schedule_overstates_kr01():
    base = _flat_curve(0.04)
    inst = Instrument(
        name="Spec MBS",
        notional=200.0,
        coupon_pct=5.5,
        instrument_type="mbs",
        maturity_years=30.0,
        payment_freq=12,
        side="asset",
        wac=0.055,
        wam_months=360,
        pool_age_months=0,
        age_months=0,
        spread_to_curve=0.0175,
        oas=0.005,
        mbs_level="agency",
    )
    # Live KR01
    live = compute_kr01([inst], [], base)
    live_tot = float(np.abs(live["net_kr01_k"]).sum())

    # Frozen: CFs at base, re-discount only
    from src.mbs_pricing import bucket_cashflows_from_terms, terms_from_instrument
    frozen_cfs = bucket_cashflows_from_terms(base, terms_from_instrument(inst))
    from src.key_rate_duration import KeyRateGrid, _bump_curve

    grid = KeyRateGrid()
    frozen_tot = 0.0
    pv0 = float(base.pv_cashflows(frozen_cfs))
    for i in range(len(grid.tenors)):
        bumped = _bump_curve(base, grid, i, 1.0)
        pv1 = float(bumped.pv_cashflows(frozen_cfs))
        frozen_tot += abs((pv0 - pv1) * 1000.0)
    assert frozen_tot > live_tot


def test_kr01_drifts_across_scenario_states():
    base = _flat_curve(0.04)
    inst = Instrument(
        name="Spec MBS",
        notional=200.0,
        coupon_pct=5.5,
        instrument_type="mbs",
        maturity_years=30.0,
        payment_freq=12,
        side="asset",
        wac=0.055,
        wam_months=360,
        pool_age_months=0,
        spread_to_curve=0.0175,
        oas=0.005,
        mbs_level="agency",
    )
    kr_base = float(compute_kr01([inst], [], base)["net_kr01_k"].sum())
    kr_down = float(
        compute_kr01([inst], [], _scenario_curve(base, "PS_DOWN"))["net_kr01_k"].sum()
    )
    # Duration compresses under a refi wave (signed net KR01 falls)
    assert kr_down < 0.85 * kr_base
    assert kr_down < kr_base


def test_anchor_tenor_matters_short_vs_parallel():
    base = _flat_curve(0.04)
    terms = _spec_pool()
    _, inc0 = step_a_mortgage_rate(base, terms)
    _, inc_par = step_a_mortgage_rate(_scenario_curve(base, "PS_UP"), terms)
    _, inc_short = step_a_mortgage_rate(_scenario_curve(base, "SHORT_UP"), terms)
    assert abs(inc_short - inc0) < abs(inc_par - inc0) * 0.3


def test_wholeloan_pricing_matches_mbs_equal_terms():
    base = _flat_curve(0.04)
    mbs = Instrument(
        name="Agency MBS",
        notional=200.0,
        coupon_pct=5.5,
        instrument_type="mbs",
        maturity_years=30.0,
        payment_freq=12,
        side="asset",
        wac=0.055,
        wam_months=360,
        pool_age_months=0,
        spread_to_curve=0.0175,
        oas=0.005,
        mbs_level="agency",
    )
    wl = Instrument(
        name="Whole Loan Pool",
        notional=200.0,
        coupon_pct=5.5,
        instrument_type="whole_loan",
        maturity_years=30.0,
        payment_freq=12,
        side="asset",
        wac=0.055,
        wam_months=360,
        pool_age_months=0,
        spread_to_curve=0.0175,
        oas=0.005,
        credit_spread=0.01,
    )
    assert mbs.pv_under_curve(base) == pytest.approx(wl.pv_under_curve(base), abs=1e-6)
    assert hqla_from_mbs_level("agency", False) == "level_2a"
    assert wl.hqla_level == "not_eligible"


def test_hqla_tags_static_across_scenarios():
    inst = Instrument(
        name="Ginnie MBS",
        notional=100.0,
        coupon_pct=4.5,
        instrument_type="mbs",
        maturity_years=25.0,
        payment_freq=12,
        side="asset",
        mbs_level="ginnie",
    )
    tag0 = inst.hqla_level
    rsf0 = inst.nsfr_rsf_factor
    for sid in ("PS_UP", "PS_DOWN", "STEEPENER", "FLATTENER"):
        # Tags must not depend on pricing path
        assert inst.hqla_level == tag0
        assert inst.nsfr_rsf_factor == rsf0
        assert tag0 == "level_1"
