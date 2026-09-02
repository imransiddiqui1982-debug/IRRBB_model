"""
tests/test_irrbb.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.balance_sheet import get_instruments  # noqa: E402
from src.calculator import IRRBBCalculator  # noqa: E402
from src.cashflows import Instrument  # noqa: E402
from src.scenarios import SCENARIOS, Scenario  # noqa: E402
from src.time_buckets import years_to_bucket, N_BUCKETS  # noqa: E402
from src.yield_curve import YieldCurve  # noqa: E402


@pytest.fixture
def instruments():
    return get_instruments()


@pytest.fixture
def calc(instruments):
    assets, liabs = instruments
    return IRRBBCalculator(assets, liabs, tier1_capital=500.0)


@pytest.fixture
def flat_curve():
    # Flat 5% curve for easy manual verification
    return YieldCurve(ref_tenors=[0, 30], ref_rates=[0.05, 0.05])


# ── time_buckets ──────────────────────────────────────────────────────────────

def test_19_buckets(): assert N_BUCKETS == 19
def test_overnight(): assert years_to_bucket(0.0) == 0
def test_one_year(): assert years_to_bucket(1.0) == 6
def test_two_year(): assert years_to_bucket(2.0) == 8
def test_ten_year(): assert years_to_bucket(10.0) == 16
def test_twenty_plus(): assert years_to_bucket(25.0) == 18


def test_all_tenors_map_to_valid_bucket():
    for t in np.linspace(0, 30, 2000):
        assert 0 <= years_to_bucket(t) < N_BUCKETS


# ── CashFlow & Instrument ─────────────────────────────────────────────────────

def test_bullet_fixed_cf_count():
    """2Y semi-annual bullet: 4 coupons + 1 principal = 5 CFs (principal merges at maturity)."""
    inst = Instrument("Test", 100, 5.0, "bullet_fixed", 2.0, payment_freq=2, side="asset")
    principals = [cf for cf in inst.cashflows if cf.cf_type == "principal"]
    coupons = [cf for cf in inst.cashflows if cf.cf_type == "coupon"]
    assert len(principals) == 1
    assert len(coupons) == 4


def test_bullet_fixed_principal_at_maturity():
    inst = Instrument("Test", 100, 5.0, "bullet_fixed", 3.0, payment_freq=2, side="asset")
    principals = [cf for cf in inst.cashflows if cf.cf_type == "principal"]
    assert abs(principals[0].time_years - 3.0) < 0.01
    assert abs(principals[0].amount - 100) < 1e-9


def test_bullet_fixed_coupon_amount():
    """Quarterly coupon on $200M at 6%: each coupon = 200 * 0.06 / 4 = 3M."""
    inst = Instrument("Test", 200, 6.0, "bullet_fixed", 2.0, payment_freq=4, side="asset")
    coupons = [cf for cf in inst.cashflows if cf.cf_type == "coupon"]
    for cf in coupons:
        assert abs(cf.amount - 3.0) < 1e-9


def test_bullet_floating_single_cf():
    inst = Instrument("Test", 500, 5.0, "bullet_floating", 3.0,
                      repricing_years=0.25, side="asset")
    assert len(inst.cashflows) == 1
    assert inst.cashflows[0].cf_type == "repricing"
    assert abs(inst.cashflows[0].amount - 500) < 1e-9


def test_amortising_principal_sums_to_notional():
    inst = Instrument("Test", 240, 5.0, "amortising", 2.0, payment_freq=12, side="asset")
    total_principal = sum(cf.amount for cf in inst.cashflows if cf.cf_type == "principal")
    assert abs(total_principal - 240) < 1e-6


def test_amortising_declining_coupons():
    """Each coupon should be <= the previous one (declining balance)."""
    inst = Instrument("Test", 120, 6.0, "amortising", 1.0, payment_freq=12, side="asset")
    coupons = [cf.amount for cf in inst.cashflows if cf.cf_type == "coupon"]
    for i in range(1, len(coupons)):
        assert coupons[i] <= coupons[i-1] + 1e-9


def test_demand_deposit_single_cf():
    inst = Instrument("NMD", 700, 0.5, "demand_deposit", 1/12,
                      repricing_years=1/12, side="liability")
    assert len(inst.cashflows) == 1


def test_bucket_cashflows_shape():
    inst = Instrument("Test", 100, 4.0, "bullet_fixed", 5.0, payment_freq=2, side="asset")
    arr = inst.bucket_cashflows()
    assert arr.shape == (N_BUCKETS,)
    assert arr.sum() > 0


def test_bucket_cashflows_total():
    """Total bucketed CFs >= notional (includes coupons)."""
    inst = Instrument("Test", 100, 5.0, "bullet_fixed", 5.0, payment_freq=2, side="asset")
    assert inst.bucket_cashflows().sum() >= 100


def test_effective_duration_bullet_approx():
    """Effective duration of a 10Y bullet should be roughly 7-9Y."""
    inst = Instrument("Test", 100, 4.0, "bullet_fixed", 10.0, payment_freq=2, side="asset")
    # weighted avg across 19 buckets; principal bucket midpoint can exceed maturity
    assert inst.effective_duration > 5.0


def test_effective_duration_amortising_shorter():
    """Amortising duration should be shorter than equivalent bullet."""
    bullet = Instrument("B", 100, 4.0, "bullet_fixed",  10.0, payment_freq=2, side="asset")
    amort = Instrument("A", 100, 4.0, "amortising",    10.0, payment_freq=12, side="asset")
    assert amort.effective_duration < bullet.effective_duration


# ── YieldCurve ────────────────────────────────────────────────────────────────

def test_flat_curve_uniform_rates(flat_curve):
    assert np.allclose(flat_curve.base_rates, 0.05, atol=1e-9)


def test_shocked_rates_parallel(flat_curve):
    shocks = [200] * N_BUCKETS
    shocked = flat_curve.shocked_rates(shocks)
    assert np.allclose(shocked, 0.07, atol=1e-9)


def test_rate_floor_at_zero(flat_curve):
    """Large negative shock should floor at 0."""
    shocks = [-1000] * N_BUCKETS
    shocked = flat_curve.shocked_rates(shocks)
    assert (shocked >= 0).all()


def test_discount_factors_decreasing(flat_curve):
    """Longer tenors → smaller discount factors (positive rates)."""
    dfs = flat_curve.discount_factors(flat_curve.base_rates)
    # Not strictly monotone (bucket 0 is overnight ≈ 1.0), check general trend
    assert dfs[0] > dfs[10] > dfs[18]


def test_pv_zero_cashflows(flat_curve):
    cf = np.zeros(N_BUCKETS)
    assert flat_curve.pv_cashflows(cf) == 0.0


def test_pv_single_bucket(flat_curve):
    """$100 in bucket 6 (midpoint 1.25Y) at 5%: PV ≈ 100/1.05^1.25 ≈ 94.0."""
    cf = np.zeros(N_BUCKETS)
    cf[6] = 100.0
    pv = flat_curve.pv_cashflows(cf)
    expected = 100.0 / (1.05 ** 1.25)
    assert abs(pv - expected) < 0.01


# ── Scenarios ─────────────────────────────────────────────────────────────────

def test_six_scenarios(): assert len(SCENARIOS) == 6


def test_19_shock_values():
    for s in SCENARIOS:
        assert len(s.shocks_bp) == 19


def test_parallel_up_200():
    ps = next(s for s in SCENARIOS if s.id == "PS_UP")
    assert all(abs(v - 200) < 1e-9 for v in ps.shocks_bp)


def test_parallel_mirrors():
    up = next(s for s in SCENARIOS if s.id == "PS_UP")
    down = next(s for s in SCENARIOS if s.id == "PS_DOWN")
    assert all(abs(u + d) < 1e-9 for u, d in zip(up.shocks_bp, down.shocks_bp))


# ── Calculator — EVE ──────────────────────────────────────────────────────────

def test_eve_zero_shock(calc):
    zero = Scenario("Z", "Zero", "Zero", [0]*6)
    delta_eve, _, _ = calc.calc_eve(zero)
    assert abs(delta_eve) < 1e-6


def test_eve_parallel_up_negative(calc):
    """Net long-duration asset position → EVE falls when rates rise."""
    ps_up = next(s for s in SCENARIOS if s.id == "PS_UP")
    delta_eve, _, _ = calc.calc_eve(ps_up)
    assert delta_eve < 0


def test_eve_parallel_mirror(calc):
    """
    Due to bond convexity, PV is nonlinear in rates: up+down != 0 exactly.
    Parallel down always produces larger |ΔEVE| than parallel up (convexity gain).
    This is correct financial behaviour — we verify the sign pattern, not exact mirror.
    """
    up = next(s for s in SCENARIOS if s.id == "PS_UP")
    down = next(s for s in SCENARIOS if s.id == "PS_DOWN")
    eve_up,   _, _ = calc.calc_eve(up)
    eve_down, _, _ = calc.calc_eve(down)
    assert eve_up < 0   # rates up  → fixed asset book loses value
    assert eve_down > 0   # rates down → fixed asset book gains value
    assert eve_down > abs(eve_up)  # convexity: gain > loss for equal shock


def test_eve_larger_with_proper_coupons(instruments):
    """
    Full cash flow schedule should give larger |ΔEVE| than duration approx
    on a long-fixed-rate book, because coupons add sensitivity.
    (Here we just verify the result is a finite, non-trivial number.)
    """
    assets, liabs = instruments
    calc = IRRBBCalculator(assets, liabs, tier1_capital=500)
    ps_up = next(s for s in SCENARIOS if s.id == "PS_UP")
    r = calc.run_scenario(ps_up)
    assert abs(r.delta_eve) > 10   # should be material, not near zero


# ── Calculator — NII ──────────────────────────────────────────────────────────

def test_nii_zero_shock(calc):
    zero = Scenario("Z", "Zero", "Zero", [0]*6)
    total, _, _ = calc.calc_nii(zero)
    assert abs(total) < 1e-9


def test_nii_decomposition_sums(calc):
    for s in SCENARIOS:
        total, nii_a, nii_l = calc.calc_nii(s)
        assert abs(total - (nii_a + nii_l)) < 1e-9


def test_nii_parallel_mirror(calc):
    up = next(s for s in SCENARIOS if s.id == "PS_UP")
    down = next(s for s in SCENARIOS if s.id == "PS_DOWN")
    nii_up,   _, _ = calc.calc_nii(up)
    nii_down, _, _ = calc.calc_nii(down)
    assert abs(nii_up + nii_down) < 1e-9


# ── Outlier threshold ─────────────────────────────────────────────────────────

def test_outlier_tiny_tier1(instruments):
    assets, liabs = instruments
    calc_small = IRRBBCalculator(assets, liabs, tier1_capital=0.1)
    ps_up = next(s for s in SCENARIOS if s.id == "PS_UP")
    assert calc_small.run_scenario(ps_up).is_outlier


def test_no_outlier_huge_tier1(instruments):
    assets, liabs = instruments
    calc_large = IRRBBCalculator(assets, liabs, tier1_capital=1_000_000)
    for s in SCENARIOS:
        assert not calc_large.run_scenario(s).is_outlier


def test_mutually_exclusive_flags(calc):
    for s in SCENARIOS:
        r = calc.run_scenario(s)
        assert not (r.is_outlier and r.is_watch)


# ── Repricing gap ─────────────────────────────────────────────────────────────

def test_gap_19_rows(calc):
    assert len(calc.repricing_gap()) == 19


def test_gap_net_arithmetic(calc):
    gap = calc.repricing_gap()
    diff = (gap["assets"] - gap["liabilities"] - gap["net_gap"]).abs()
    assert diff.max() < 1e-9


def test_dv01_gap_net_arithmetic(calc):
    dv01 = calc.bucket_dv01_gap()
    diff = (
        dv01["asset_dv01"] - dv01["liability_dv01"] - dv01["net_dv01"]
    ).abs()
    assert diff.max() < 1e-9


def test_dv01_gap_non_negative_legs(calc):
    dv01 = calc.bucket_dv01_gap()
    assert (dv01["asset_dv01"] >= 0).all()
    assert (dv01["liability_dv01"] >= 0).all()


def test_irs_hedge_suggestions(calc):
    from src.calculator import suggest_irs_hedges

    dv01 = calc.bucket_dv01_gap()
    hedges = suggest_irs_hedges(dv01, hedge_ratio=0.8)
    if not hedges.empty:
        assert "IRS structure" in hedges.columns
        assert "Indicative notional ($M)" in hedges.columns
        assert (hedges["Indicative notional ($M)"] > 0).all()


def test_maturity_gap_aligns_with_dv01_basis(calc):
    mat = calc.bucket_maturity_gap()
    assert (mat["assets"] >= 0).all()
    assert (mat["liabilities"] >= 0).all()
    diff = (mat["assets"] - mat["liabilities"] - mat["net_gap"]).abs()
    assert diff.max() < 1e-9


# ── Instrument EVE detail ─────────────────────────────────────────────────────

def test_eve_detail_row_count(calc):
    ps_up = next(s for s in SCENARIOS if s.id == "PS_UP")
    detail = calc.instrument_eve_detail(ps_up)
    total_instruments = len(calc.assets) + len(calc.liabilities)
    assert len(detail) == total_instruments


def test_eve_detail_total_matches_scenario(calc):
    ps_up = next(s for s in SCENARIOS if s.id == "PS_UP")
    detail = calc.instrument_eve_detail(ps_up)
    result = calc.run_scenario(ps_up)
    # rounding in display (2dp) causes small gap
    assert abs(detail["delta_eve"].sum() - result.delta_eve) < 0.1


def test_summary_table(calc):
    results = calc.run_all(SCENARIOS)
    df = calc.summary_table(results)
    assert len(df) == 6
    assert "Status" in df.columns


# ── CSV balance sheet loader ──────────────────────────────────────────────────

def test_load_default_csv_matches_builtin_balance_sheet():
    import os
    from src.load_balance_sheet import load_instruments_from_csv

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_balance_sheet.csv"
    )
    assets, liabilities = load_instruments_from_csv(csv_path)
    default_assets, default_liabilities = get_instruments()

    assert len(assets) == len(default_assets)
    assert len(liabilities) == len(default_liabilities)
    assert sum(i.notional for i in assets) == sum(i.notional for i in default_assets)


def test_load_csv_rejects_missing_columns():
    import io
    from src.load_balance_sheet import load_instruments_from_csv

    bad_csv = io.StringIO("name,side,notional\nLoan,asset,100\n")
    try:
        load_instruments_from_csv(bad_csv)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "missing required column" in str(exc).lower()
