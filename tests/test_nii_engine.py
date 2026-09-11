"""Tests for src/nii_engine.py — NII accrual / sensitivity."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cashflows import Instrument  # noqa: E402
from src.nii_engine import (  # noqa: E402
    BCBS_NII_SCENARIOS,
    NIIPortfolio,
    NIIScenario,
    US_NII_SCENARIOS,
    portfolio_nii_delta,
)
from src.scenarios import SCENARIO_MAP  # noqa: E402
from src.yield_curve import YieldCurve  # noqa: E402


def _flat(rate: float = 0.04) -> YieldCurve:
    t = [0.0, 0.25, 1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    return YieldCurve(ref_tenors=t, ref_rates=[rate] * len(t))


FLAT = _flat()


def test_fixed_bullet_nii_identical_if_no_maturity_in_horizon():
    bond = Instrument(
        "10y bond", 100.0, 5.0, "bullet_fixed", 10.0, payment_freq=2, side="asset",
    )
    p = NIIPortfolio([bond], FLAT, horizon_months=12)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    up = p.run(NIIScenario("up", +200.0)).total_nii
    down = p.run(NIIScenario("down", -200.0)).total_nii
    assert base == pytest.approx(up, rel=1e-9)
    assert base == pytest.approx(down, rel=1e-9)
    assert base == pytest.approx(5.0, rel=1e-6)


def test_fixed_bullet_maturing_triggers_reinvestment():
    bond = Instrument(
        "6m bill", 100.0, 3.0, "bullet_fixed", 0.5, payment_freq=1, side="asset",
    )
    p = NIIPortfolio([bond], FLAT, horizon_months=12, constant_balance_sheet=True)
    up = p.run(NIIScenario("up", +200.0)).total_nii
    down = p.run(NIIScenario("down", -200.0)).total_nii
    assert up > down


def test_static_balance_sheet_no_reinvestment():
    bond = Instrument(
        "6m bill", 100.0, 3.0, "bullet_fixed", 0.5, payment_freq=1, side="asset",
    )
    p = NIIPortfolio([bond], FLAT, horizon_months=12, constant_balance_sheet=False)
    up = p.run(NIIScenario("up", +200.0)).total_nii
    down = p.run(NIIScenario("down", -200.0)).total_nii
    assert up == pytest.approx(down, rel=1e-9)


def test_floater_moves_after_reset():
    frn = Instrument(
        "floating loan", 100.0, 4.0, "bullet_floating", 5.0,
        payment_freq=12, repricing_years=0.5, side="asset", current_rate=0.04,
    )
    p = NIIPortfolio([frn], FLAT, horizon_months=12)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    up = p.run(NIIScenario("up", +200.0)).total_nii
    expected_delta = 100.0 * 0.02 * 0.5
    assert (up - base) == pytest.approx(expected_delta, rel=0.05)


def test_floater_no_shock_before_reset():
    frn = Instrument(
        "floating loan", 100.0, 4.0, "bullet_floating", 5.0,
        payment_freq=12, repricing_years=0.5, side="asset", current_rate=0.04,
    )
    p = NIIPortfolio([frn], FLAT, horizon_months=12)
    r_up = p.run(NIIScenario("up", +200.0))
    r_base = p.run(NIIScenario("base", 0.0))
    assert np.allclose(r_up.monthly_net_interest[:5], r_base.monthly_net_interest[:5])
    assert not np.isclose(r_up.monthly_net_interest[7], r_base.monthly_net_interest[7])


def test_nmd_moves_one_for_one_with_shock():
    """Deposit rate = current + full shock (no beta)."""
    dep = Instrument(
        "dep", 100.0, 1.0, "demand_deposit", 4.5, payment_freq=12, side="liability",
        current_rate=0.01,
    )
    p = NIIPortfolio([dep], FLAT, horizon_months=12)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    up = p.run(NIIScenario("up", +200.0)).total_nii
    assert (up - base) == pytest.approx(-2.0, abs=0.05)


def test_nmd_maturity_does_not_affect_nii():
    """WAL/maturity on NMD must not change NII (same rate path)."""
    a = Instrument(
        "fast", 100.0, 1.0, "demand_deposit", 1.0, payment_freq=12, side="liability",
        current_rate=0.01,
    )
    b = Instrument(
        "slow", 100.0, 1.0, "demand_deposit", 10.0, payment_freq=12, side="liability",
        current_rate=0.01,
    )
    p1 = NIIPortfolio([a], FLAT, horizon_months=12)
    p2 = NIIPortfolio([b], FLAT, horizon_months=12)
    assert p1.run(NIIScenario("up", 200)).total_nii == pytest.approx(
        p2.run(NIIScenario("up", 200)).total_nii, rel=1e-9
    )


def test_nmd_rate_floors_at_zero():
    dep = Instrument(
        "cheap", 100.0, 0.5, "demand_deposit", 4.0, payment_freq=12, side="liability",
        current_rate=0.005,
    )
    p = NIIPortfolio([dep], FLAT, horizon_months=12)
    down = p.run(NIIScenario("big_down", -400.0))
    assert np.all(down.monthly_net_interest <= 1e-9)


def test_mbs_nii_responds_to_prepay():
    pool = Instrument(
        "mbs", 100.0, 5.5, "mbs", 15.0, payment_freq=12, side="asset",
        wac=0.055, wam_months=180, pool_age_months=36,
        anchor_tenor=10.0, spread_to_curve=0.0175, oas=0.005,
    )
    p = NIIPortfolio([pool], FLAT, horizon_months=24, constant_balance_sheet=True)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    down = p.run(NIIScenario("down", -200.0)).total_nii
    up = p.run(NIIScenario("up", +200.0)).total_nii
    assert not np.isclose(base, down)
    assert not np.isclose(base, up)


def test_bcbs_and_us_grids():
    p = NIIPortfolio(
        [Instrument("b", 100.0, 5.0, "bullet_fixed", 10.0, side="asset")], FLAT
    )
    g = p.grid()
    assert set(g.keys()) == {"par_up_200", "par_down_200"}
    assert len(US_NII_SCENARIOS) == 10
    assert len([s for s in US_NII_SCENARIOS if s.ramp]) == 2


def test_ramp_smaller_than_instant():
    frn = Instrument(
        "frn", 100.0, 4.0, "bullet_floating", 5.0,
        payment_freq=12, repricing_years=0.1, side="asset", current_rate=0.04,
    )
    p = NIIPortfolio([frn], FLAT, horizon_months=12)
    base = p.run(NIIScenario("base", 0.0)).total_nii
    instant = p.run(NIIScenario("instant", 200.0, ramp=False)).total_nii
    ramped = p.run(NIIScenario("ramped", 200.0, ramp=True)).total_nii
    assert abs(ramped - base) < abs(instant - base)


def test_portfolio_nii_delta_matches_eve_ps_up():
    assets = [
        Instrument(
            "frn", 100.0, 4.0, "bullet_floating", 3.0,
            payment_freq=12, repricing_years=0.25, side="asset", current_rate=0.04,
        )
    ]
    liabs = [
        Instrument(
            "dep", 100.0, 1.0, "demand_deposit", 4.0, payment_freq=12, side="liability",
            current_rate=0.01,
        )
    ]
    total, da, dl = portfolio_nii_delta(
        assets, liabs, FLAT, SCENARIO_MAP["PS_UP"], horizon_months=12
    )
    assert abs(total - (da + dl)) < 1e-9
    assert da > 0  # floating asset benefits from +200
    assert dl < 0  # deposit cost rises


def test_by_instrument_sums():
    positions = [
        Instrument("b", 100.0, 5.0, "bullet_fixed", 10.0, side="asset"),
        Instrument(
            "f", 100.0, 4.0, "bullet_floating", 5.0,
            payment_freq=12, repricing_years=0.5, side="liability", current_rate=0.04,
        ),
    ]
    p = NIIPortfolio(positions, FLAT, horizon_months=12)
    r = p.run(NIIScenario("up", 200))
    assert sum(r.by_instrument.values()) == pytest.approx(r.total_nii, rel=1e-9)


def test_bcbs_scenario_tuple_length():
    assert len(BCBS_NII_SCENARIOS) == 2
