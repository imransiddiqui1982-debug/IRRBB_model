"""Tests for nii_engine.py"""
import numpy as np
import pytest

from alm_engine import Curve, FixedBullet, FixedAmortising, Floater, NMD, MBS, Swap
from nii_engine import (NIIPortfolio, NIIScenario, BCBS_NII_SCENARIOS,
                        US_NII_SCENARIOS, project_instrument)

FLAT = Curve.flat(0.04)


# ----------------------------------------------------------------- fixed rate --

def test_fixed_bullet_nii_identical_across_all_scenarios_if_no_maturity_in_horizon():
    """The core sanity check: an instrument that is fixed-rate for its whole
    life and does NOT mature inside the horizon must earn EXACTLY the same
    interest regardless of what happens to rates. If this fails, something
    is leaking scenario-dependence into an instrument that has none."""
    bond = FixedBullet("10y bond", 100.0, coupon=0.05, years=10, freq=2)
    p = NIIPortfolio([bond], FLAT, horizon_months=12)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    up = p.run(NIIScenario("up", +200.0)).total_nii
    down = p.run(NIIScenario("down", -200.0)).total_nii
    assert base == pytest.approx(up, rel=1e-9)
    assert base == pytest.approx(down, rel=1e-9)
    assert base == pytest.approx(5.0, rel=1e-6)      # 100m * 5% for 1 year


def test_fixed_bullet_maturing_in_horizon_triggers_reinvestment():
    """A bullet maturing at month 6 should earn 6 months at the old rate,
    then (under constant balance sheet) 6 months at the NEW scenario rate."""
    bond = FixedBullet("6m bill", 100.0, coupon=0.03, years=0.5, freq=1)
    p = NIIPortfolio([bond], FLAT, horizon_months=12, constant_balance_sheet=True)
    up = p.run(NIIScenario("up", +200.0)).total_nii
    down = p.run(NIIScenario("down", -200.0)).total_nii
    # reinvested at a higher rate under "up" -> more income than under "down"
    assert up > down


def test_fixed_bullet_static_balance_sheet_no_reinvestment():
    """With constant_balance_sheet=False, runoff should NOT be replaced --
    NII after maturity should just stop, identical regardless of scenario."""
    bond = FixedBullet("6m bill", 100.0, coupon=0.03, years=0.5, freq=1)
    p = NIIPortfolio([bond], FLAT, horizon_months=12, constant_balance_sheet=False)
    up = p.run(NIIScenario("up", +200.0)).total_nii
    down = p.run(NIIScenario("down", -200.0)).total_nii
    assert up == pytest.approx(down, rel=1e-9)


def test_amortising_loan_interest_principal_split_reconciles():
    loan = FixedAmortising("5y loan", 100.0, coupon=0.06, years=5, freq=12)
    t, interest, principal = loan.flows_split(FLAT)
    t2, total_paid = loan.flows(FLAT)
    assert np.allclose(interest + principal, total_paid)
    assert principal.sum() == pytest.approx(100.0, rel=1e-6)  # fully amortizes


# --------------------------------------------------------------------- floaters --

def test_floater_moves_with_shock_after_reset():
    """A floater resetting at month 6 should show ZERO NII difference for
    the first 6 months and a clear difference for the remaining 6, matching
    the hand-calculated methodology from earlier in this build:
        dNII = notional * shock * (horizon - reset_month) / horizon"""
    frn = Floater("floating loan", 100.0, side=1, next_reset=0.5)
    p = NIIPortfolio([frn], FLAT, horizon_months=12)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    up = p.run(NIIScenario("up", +200.0)).total_nii
    expected_delta = 100.0 * 0.02 * 0.5     # notional * shock * 6/12
    assert (up - base) == pytest.approx(expected_delta, rel=0.05)


def test_floater_no_shock_before_reset_month():
    frn = Floater("floating loan", 100.0, side=1, next_reset=0.5)
    p = NIIPortfolio([frn], FLAT, horizon_months=12)
    r_up = p.run(NIIScenario("up", +200.0))
    r_base = p.run(NIIScenario("base", 0.0))
    # months 1-5 (index 0-4) must be identical regardless of shock
    assert np.allclose(r_up.monthly_net_interest[:5], r_base.monthly_net_interest[:5])
    # month 7+ (index 6+) must differ
    assert not np.isclose(r_up.monthly_net_interest[7], r_base.monthly_net_interest[7])


# -------------------------------------------------------------------------- nmd --

def test_nmd_higher_beta_means_larger_nii_sensitivity():
    """Beta drives NII sensitivity -- NOT decay/WAL. A higher-beta deposit
    book should show a LARGER swing in funding cost under a shock."""
    low_beta = NMD("low beta dep", 100.0, side=-1, beta=0.10, current_rate=0.01)
    high_beta = NMD("high beta dep", 100.0, side=-1, beta=0.60, current_rate=0.01)
    p_low = NIIPortfolio([low_beta], FLAT, horizon_months=12)
    p_high = NIIPortfolio([high_beta], FLAT, horizon_months=12)
    d_low = p_low.run(NIIScenario("up", 200)).total_nii - p_low.run(NIIScenario("base", 0)).total_nii
    d_high = p_high.run(NIIScenario("up", 200)).total_nii - p_high.run(NIIScenario("base", 0)).total_nii
    assert abs(d_high) > abs(d_low)


def test_nmd_decay_and_wal_do_not_affect_nii():
    """Sanity check for the beta-vs-decay separation: two NMDs with
    IDENTICAL beta but very different decay/WAL should show IDENTICAL NII
    sensitivity -- decay only matters for EVE slotting, never for NII."""
    fast_decay = NMD("fast", 100.0, side=-1, beta=0.35, decay=1.0, current_rate=0.01)
    slow_decay = NMD("slow", 100.0, side=-1, beta=0.35, decay=0.05, current_rate=0.01)
    p1 = NIIPortfolio([fast_decay], FLAT, horizon_months=12)
    p2 = NIIPortfolio([slow_decay], FLAT, horizon_months=12)
    r1 = p1.run(NIIScenario("up", 200)).total_nii
    r2 = p2.run(NIIScenario("up", 200)).total_nii
    assert r1 == pytest.approx(r2, rel=1e-9)


def test_nmd_asymmetric_beta_down_floors_at_zero():
    """A deposit rate cannot go negative. With current_rate near zero and a
    large down shock, the rate should floor at 0, not go negative."""
    dep = NMD("cheap dep", 100.0, side=-1, beta=0.20, beta_down=0.80,
             current_rate=0.005)
    p = NIIPortfolio([dep], FLAT, horizon_months=12)
    down = p.run(NIIScenario("big_down", -400.0))
    # interest (expense, negative for a liability) should not imply a
    # negative rate, i.e. magnitude should not exceed notional*0 -- expense
    # can be exactly zero at worst, never a "negative expense" (income)
    assert np.all(down.monthly_net_interest <= 1e-9)


def test_nmd_beta_down_defaults_to_beta_if_unset():
    dep1 = NMD("d1", 100.0, side=-1, beta=0.30, current_rate=0.02)  # beta_down unset
    dep2 = NMD("d2", 100.0, side=-1, beta=0.30, beta_down=0.30, current_rate=0.02)
    p1 = NIIPortfolio([dep1], FLAT, horizon_months=12)
    p2 = NIIPortfolio([dep2], FLAT, horizon_months=12)
    r1 = p1.run(NIIScenario("down", -100)).total_nii
    r2 = p2.run(NIIScenario("down", -100)).total_nii
    assert r1 == pytest.approx(r2, rel=1e-9)


# --------------------------------------------------------------------------- mbs --

def test_mbs_nii_reflects_prepayment_speed_change():
    """Faster prepayment under a down-shock should change the MBS's NII
    profile vs. base (via reinvestment timing), even though the loan's own
    coupon never changes."""
    pool = MBS("mbs", 100.0, wac=0.055, months=180, anchor_tenor=10.0,
              spread_to_curve=0.0175, oas=0.005)
    p = NIIPortfolio([pool], FLAT, horizon_months=24, constant_balance_sheet=True)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    down = p.run(NIIScenario("down", -200.0)).total_nii
    up = p.run(NIIScenario("up", +200.0)).total_nii
    assert not np.isclose(base, down)
    assert not np.isclose(base, up)


# ---------------------------------------------------------------------- swaps --

def test_pay_fixed_swap_nii_gains_when_floating_rate_rises():
    swap = Swap("pay fixed", 100.0, side=1, years=5, fixed_rate=0.04, next_reset=0.25)
    p = NIIPortfolio([swap], FLAT, horizon_months=12)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    up = p.run(NIIScenario("up", 200.0)).total_nii
    assert up > base


# ----------------------------------------------------------------- portfolio --

def test_grid_default_is_bcbs_two_scenarios():
    p = NIIPortfolio([FixedBullet("b", 100.0, coupon=0.05, years=10)], FLAT)
    g = p.grid()
    assert set(g.keys()) == {"par_up_200", "par_down_200"}


def test_us_scenario_set_has_eight_shocks_plus_two_ramps():
    assert len(US_NII_SCENARIOS) == 10
    ramps = [s for s in US_NII_SCENARIOS if s.ramp]
    assert len(ramps) == 2


def test_ramp_scenario_reaches_full_magnitude_only_at_ramp_end():
    s = NIIScenario("ramp", 200.0, ramp=True, ramp_months=12)
    assert s.shock_at_month(1) == pytest.approx(200.0/1e4/12, rel=1e-6)
    assert s.shock_at_month(12) == pytest.approx(200.0/1e4, rel=1e-6)
    assert s.shock_at_month(6) == pytest.approx(200.0/1e4 * 0.5, rel=1e-6)


def test_ramp_produces_smaller_impact_than_instantaneous_shock_of_same_size():
    """A ramped shock should have LESS effect on NII than an instantaneous
    one of the same final magnitude, since it takes time to build up."""
    frn = Floater("frn", 100.0, side=1, next_reset=0.1)
    p = NIIPortfolio([frn], FLAT, horizon_months=12)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    instant = p.run(NIIScenario("instant", 200.0, ramp=False)).total_nii
    ramped = p.run(NIIScenario("ramped", 200.0, ramp=True)).total_nii
    assert abs(ramped - base) < abs(instant - base)


def test_limit_status_reports_both_denominators():
    p = NIIPortfolio([FixedBullet("b", 100.0, coupon=0.05, years=10),
                      Floater("f", 100.0, side=-1, next_reset=0.5)],
                     FLAT, tier1=50.0)
    rows = p.limit_status(limit_pct_of_tier1=5.0, limit_pct_of_base_nii=10.0)
    assert all("pct_tier1" in r and "pct_base_nii" in r for r in rows)


def test_24_month_horizon_captures_more_reinvestment_than_12_month():
    """Longer horizon = more months of reinvestment effect for something
    maturing early, so the scenario spread (up minus down) in dollar terms
    should generally be larger over 24 months than 12 for a short instrument."""
    bill = FixedBullet("3m bill", 100.0, coupon=0.03, years=0.25, freq=1)
    p12 = NIIPortfolio([bill], FLAT, horizon_months=12, constant_balance_sheet=True)
    p24 = NIIPortfolio([bill], FLAT, horizon_months=24, constant_balance_sheet=True)
    spread12 = (p12.run(NIIScenario("up", 200)).total_nii
               - p12.run(NIIScenario("down", -200)).total_nii)
    spread24 = (p24.run(NIIScenario("up", 200)).total_nii
               - p24.run(NIIScenario("down", -200)).total_nii)
    assert spread24 > spread12


def test_by_instrument_breakdown_sums_to_total():
    positions = [FixedBullet("b", 100.0, coupon=0.05, years=10),
                Floater("f", 100.0, side=-1, next_reset=0.5),
                NMD("d", 50.0, side=-1, beta=0.3, current_rate=0.01)]
    p = NIIPortfolio(positions, FLAT, horizon_months=12)
    r = p.run(NIIScenario("up", 200))
    assert sum(r.by_instrument.values()) == pytest.approx(r.total_nii, rel=1e-9)
