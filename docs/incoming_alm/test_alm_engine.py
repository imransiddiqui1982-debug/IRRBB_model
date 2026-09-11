"""Tests for alm_engine.  Run:  python3 -m pytest test_alm_engine.py -q"""
import numpy as np
import pytest

from alm_engine import (Curve, KeyRateGrid, Portfolio, FixedBullet, FixedAmortising,
                        Floater, NMD, MBS, Swap, bcbs_shock, post_shock_floor,
                        solve_hedge, designate, SCENARIOS)

FLAT = Curve.flat(0.04)


# ----------------------------------------------------------- golden cases --

def test_par_bond_prices_at_par():
    """10y 4% annual bond on a flat 4% curve must price at exactly 100."""
    b = FixedBullet("bond", 100.0, coupon=0.04, years=10, freq=1)
    assert b.pv(FLAT) == pytest.approx(100.0, abs=1e-9)


def test_bond_dv01_matches_hand_calculation():
    """Hand-checked: sum(PV*t)*1e-4/(1+y) for a 10y 4% annual par bond."""
    b = FixedBullet("bond", 100.0, coupon=0.04, years=10, freq=1)
    t = np.arange(1, 11.0)
    c = np.full(10, 4.0); c[-1] += 100
    pv_t = np.sum(c / 1.04 ** t * t)
    expected = pv_t * 1e-4 / 1.04 * 1e6
    assert Portfolio([b], FLAT).dv01() == pytest.approx(expected, rel=2e-3)


def test_amortising_payment_formula():
    """5y 6% annual amortising $1m -> $237,396 level payment."""
    loan = FixedAmortising("loan", 1_000_000, coupon=0.06, years=5, freq=1)
    _, c = loan.flows(FLAT)
    assert c[0] == pytest.approx(237_396, abs=1.0)
    assert len(c) == 5


def test_floater_duration_is_to_next_reset_not_maturity():
    """A 10y floater resetting in 6m must behave like 6m paper."""
    f = Floater("frn", 100.0, next_reset=0.5)
    b = FixedBullet("bond", 100.0, coupon=0.04, years=10, freq=2)
    assert Portfolio([f], FLAT).dv01() < Portfolio([b], FLAT).dv01() / 15


def test_par_swap_prices_to_zero():
    s = Swap("swap", 100.0, side=1, years=10, fixed_rate=0.04, freq=1)
    assert abs(s.pv(FLAT)) < 0.6          # small residual from the floating stub


def test_pay_fixed_swap_gains_when_rates_rise():
    s = Swap("swap", 100.0, side=1, years=10, fixed_rate=0.04)
    up = FLAT.with_shift(lambda t: np.full_like(t, 0.01))
    assert s.pv(up) > s.pv(FLAT)


# ------------------------------------------------------------- invariants --

def test_tent_weights_are_a_partition_of_unity():
    g = KeyRateGrid()
    t = np.linspace(0.01, 40.0, 4000)
    assert g.check_partition(t)


def test_kr01_reconciles_to_dv01():
    """The core invariant. If this fails the tent weights are broken."""
    p = _sample_portfolio()
    assert p.kr01().sum() == pytest.approx(p.dv01(), rel=0.02)


def test_post_shock_floor():
    t = np.array([0.0, 10.0, 50.0, 60.0])
    r = post_shock_floor(np.full(4, -0.05), t)
    assert r[0] == pytest.approx(-0.01)
    assert r[1] == pytest.approx(-0.005)
    assert r[2] == pytest.approx(0.0)
    assert r[3] == pytest.approx(0.0)


def test_bcbs_scenario_shapes():
    t = np.array([0.25, 10.0])
    assert np.allclose(bcbs_shock(t, "par_up"), 0.02)
    up, dn = bcbs_shock(t, "short_up"), bcbs_shock(t, "short_down")
    assert np.allclose(up, -dn)
    assert up[0] > up[1]                                   # decays with tenor
    st = bcbs_shock(t, "steepener")
    assert st[0] < 0 < st[1]                               # short down, long up
    fl = bcbs_shock(t, "flattener")
    assert fl[0] > 0 > fl[1]
    with pytest.raises(KeyError):
        bcbs_shock(t, "nonsense")


def test_scenario_magnitudes_are_configurable():
    t = np.array([1.0])
    assert bcbs_shock(t, "par_up", r_par=0.025)[0] == pytest.approx(0.025)


# ------------------------------------------------------------ behavioural --

def test_nmd_core_respects_the_cap():
    d = NMD("dep", 1000.0, stable_pct=0.95, beta=0.05, core_cap=0.70)
    assert d.core == pytest.approx(700.0)          # raw 902.5, capped at 700
    assert d.flows(FLAT)[1].sum() == pytest.approx(1000.0)


def test_nmd_non_core_sits_in_the_front_bucket():
    d = NMD("dep", 1000.0, stable_pct=0.5, beta=0.5, core_cap=0.9)
    t, c = d.flows(FLAT)
    assert c[0] > 700                              # 750 non-core at month 1


def test_nmd_wal_is_capped():
    d = NMD("dep", 100.0, decay=0.05, wal_cap=4.5)  # implied WAL 20y
    assert d.wal == pytest.approx(4.5)


def test_slower_decay_lengthens_the_book():
    fast = Portfolio([NMD("d", 1000.0, decay=1.0, core_cap=0.9)], FLAT)
    slow = Portfolio([NMD("d", 1000.0, decay=0.20, core_cap=0.9)], FLAT)
    assert abs(slow.dv01()) > abs(fast.dv01())


def test_mbs_cpr_rises_as_rates_fall():
    m = MBS("mbs", 100.0)
    low = Curve.flat(0.02)
    hi = Curve.flat(0.07)
    inc_low = (m.wac - (low.anchor(7.0) + m.spread_to_curve)) * 100
    inc_hi = (m.wac - (hi.anchor(7.0) + m.spread_to_curve)) * 100
    assert m.cpr(inc_low, 60) > m.cpr(inc_hi, 60)


def test_mbs_has_negative_convexity():
    """Price gain from -100bp must be smaller than the loss from +100bp."""
    p = Portfolio([MBS("mbs", 100.0)], FLAT)
    p0 = p.pv()
    dn = p.pv(FLAT.with_shift(lambda t: np.full_like(t, -0.01))) - p0
    up = p.pv(FLAT.with_shift(lambda t: np.full_like(t, 0.01))) - p0
    assert dn < abs(up)


def test_frozen_cpr_would_overstate_kr01():
    """Live prepayment must reduce measured sensitivity vs a frozen schedule."""
    m = MBS("mbs", 200.0)
    live = Portfolio([m], FLAT).kr01().sum()
    frozen_t, frozen_c = m.flows(FLAT)
    frozen = FixedBullet("frozen", 0.0)
    frozen.flows = lambda curve, t=frozen_t, c=frozen_c: (t, c)
    assert live < Portfolio([frozen], FLAT).kr01().sum()


def test_convexity_sign_distinguishes_bullet_from_mbs():
    bullet = Portfolio([FixedBullet("b", 200.0, coupon=0.05, years=10)], FLAT)
    mbs = Portfolio([MBS("m", 200.0)], FLAT)
    assert bullet.attribution()["convexity"][0] > 0     # par_up, positive convexity
    assert mbs.attribution()["convexity"][0] < 0        # par_up, negative


def test_kr01_drifts_more_for_mbs_than_for_a_comparable_bullet():
    """Both a plain bullet bond and an MBS show SOME KR01 drift across rate
    states -- that's ordinary bond convexity (duration genuinely changes with
    yield level for any fixed-cash-flow instrument, which is why convexity
    is tracked as distinct from duration in the first place). The MBS's
    prepayment optionality is an ADDITIONAL source of drift on top of that
    baseline, so the correct test is a RELATIVE comparison -- MBS drift
    materially exceeds a comparable bullet's -- rather than an absolute
    threshold, which is fragile to exact discretization/parameter choices."""
    mbs = Portfolio([MBS("m", 200.0)], FLAT)
    bullet = Portfolio([FixedBullet("b", 200.0, coupon=0.05, years=10)], FLAT)
    def drift(p):
        b = p.kr01().sum()
        return abs(p.kr01_in_scenario("par_down").sum() - b) / abs(b)
    mbs_drift, bullet_drift = drift(mbs), drift(bullet)
    assert mbs_drift > bullet_drift * 1.15, (
        f"MBS drift ({mbs_drift:.3f}) should meaningfully exceed a comparable "
        f"bullet's convexity-driven drift ({bullet_drift:.3f}) -- if it "
        f"doesn't, the prepayment option is likely not responding correctly.")


# ----------------------------------------------------------- eve & hedging --

def test_eve_grid_covers_all_six_scenarios():
    g = _sample_portfolio().eve_grid()
    assert set(g) == set(SCENARIOS)


def test_attribution_sums_to_predicted():
    a = _sample_portfolio().attribution()
    assert np.allclose(a["contribution"].sum(axis=0), a["predicted"])
    assert np.allclose(a["actual"], a["predicted"] + a["convexity"])


def test_hedge_reduces_kr01():
    p = _sample_portfolio()
    h = solve_hedge(p)
    assert np.abs(h.residual_kr01).sum() < np.abs(h.pre_kr01).sum() * 0.1
    assert h.reduction_pct > 90


def test_hedge_to_a_target_leaves_that_target():
    p = _sample_portfolio()
    target = p.kr01() * 0.4                        # hedge to 40%, not to zero
    h = solve_hedge(p, target=target)
    assert np.allclose(h.residual_kr01, target, rtol=0.05, atol=50)


def test_max_notional_constraint_binds():
    p = _sample_portfolio()
    h = solve_hedge(p, max_notional=25.0)
    assert np.all(np.abs(h.notionals) <= 25.0 + 1e-9)


def test_designation_flags_nmd_limitation():
    p = _sample_portfolio()
    rows = designate(solve_hedge(p), p)
    assert any(r["warning"] for r in rows)
    assert {r["designation"] for r in rows} <= {"fair_value", "cash_flow", "undesignated"}


def test_pay_fixed_against_fixed_assets_is_a_fair_value_hedge():
    p = Portfolio([FixedBullet("mortgages", 500.0, coupon=0.05, years=10)], FLAT)
    rows = designate(solve_hedge(p, tenors=[10.0]), p)
    assert rows[0]["designation"] == "fair_value"
    assert "P&L" in rows[0]["mtm_to"]


def test_cash_flow_hedge_routes_to_aoci():
    p = Portfolio([Floater("cds", 300.0, side=-1, next_reset=0.25)], FLAT)
    rows = designate(solve_hedge(p, tenors=[5.0]), p)
    assert rows[0]["designation"] == "cash_flow"
    assert "AOCI" in rows[0]["mtm_to"]


# --------------------------------------------------------------- fixtures --

def _sample_portfolio() -> Portfolio:
    return Portfolio([
        Floater("floating loans", 200.0, next_reset=0.25),
        FixedBullet("2y auto loans", 150.0, coupon=0.045, years=2),
        FixedBullet("5y CRE", 250.0, coupon=0.055, years=5),
        MBS("30y MBS", 200.0),
        NMD("core deposits", 300.0, stable_pct=0.88, beta=0.35,
            core_cap=0.70, wal_cap=4.5, decay=0.33, side=-1),
        FixedBullet("6m CDs", 250.0, coupon=0.04, years=0.5, side=-1),
        FixedBullet("5y FHLB", 150.0, coupon=0.042, years=5, side=-1),
    ], FLAT, KeyRateGrid(), tier1=100.0)
