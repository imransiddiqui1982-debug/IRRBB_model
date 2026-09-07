"""
tests/test_kr01_attribution.py
------------------------------
KR01 → EVE bridge, scenario KR01, and hedge packages.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.balance_sheet import get_instruments  # noqa: E402
from src.calculator import IRRBBCalculator  # noqa: E402
from src.key_rate_duration import (  # noqa: E402
    eve_attribution_from_kr01,
    hedge_efficiency_table,
    shocks_at_key_tenors,
    compute_kr01_under_scenarios,
)
from src.scenarios import SCENARIOS  # noqa: E402


@pytest.fixture
def calc():
    assets, liabs = get_instruments()
    return IRRBBCalculator(assets, liabs, tier1_capital=500.0)


def test_shocks_at_keys_parallel_up():
    sc = next(s for s in SCENARIOS if s.id == "PS_UP")
    shocks = shocks_at_key_tenors(sc)
    assert np.allclose(shocks, 200.0)


def test_attribution_bridge_sign(calc):
    """Attribution matrix has one row per key and all scenarios."""
    results = calc.run_all(SCENARIOS)
    actual = {r.scenario.id: r.delta_eve for r in results}
    kr = calc.key_rate_duration_gap()
    attr = eve_attribution_from_kr01(kr, SCENARIOS, actual)
    par_up = next(s.name for s in SCENARIOS if s.id == "PS_UP")
    assert par_up in attr["predicted_m"]
    assert attr["contribution_m"].shape[0] == len(kr)


def test_convexity_equals_actual_minus_predicted(calc):
    results = calc.run_all(SCENARIOS)
    actual = {r.scenario.id: r.delta_eve for r in results}
    kr = calc.key_rate_duration_gap()
    attr = eve_attribution_from_kr01(kr, SCENARIOS, actual)
    for sc in SCENARIOS:
        pred = attr["predicted_m"][sc.name]
        act = attr["actual_m"][sc.name]
        conv = attr["convexity_m"][sc.name]
        assert conv == pytest.approx(act - pred, rel=1e-6, abs=1e-6)


def test_scenario_kr01_has_all_scenarios(calc):
    sk = compute_kr01_under_scenarios(
        calc.assets, calc.liabilities, calc.curve, SCENARIOS,
        cpr_for_scenario=calc._cpr_override_for,
    )
    assert "Base" in sk.columns
    for sc in SCENARIOS:
        assert sc.name in sk.columns


def test_hedge_efficiency_pay_fixed_helps_par_up():
    eff = hedge_efficiency_table(SCENARIOS)
    par_up = next(s.name for s in SCENARIOS if s.id == "PS_UP")
    row10 = eff[eff["Instrument"].str.contains("10Y")].iloc[0]
    assert row10[par_up] > 0
    assert row10["KR01 ($K/bp)"] < 0


def test_post_swap_improves_or_changes_par_up(calc):
    from src.key_rate_duration import proposed_swap_notionals, post_swap_eve_impact

    results = calc.run_all(SCENARIOS)
    actual = {r.scenario.id: r.delta_eve for r in results}
    kr = calc.key_rate_duration_gap()
    notionals = proposed_swap_notionals(kr, hedge_ratio=1.0)
    post = post_swap_eve_impact(kr, SCENARIOS, 500.0, actual, notionals)
    assert not post["summary"].empty
    assert list(post["contrib_after"].index) == list(post["contrib_before"].index)
    # After full hedge, predicted |ΔEVE| on parallel up should shrink vs before
    par = next(s.name for s in SCENARIOS if s.id == "PS_UP")
    before = abs(float(post["contrib_before"][par].sum()))
    after = abs(float(post["contrib_after"][par].sum()))
    assert after <= before + 1e-6


def test_treasury_alco_pack(calc):
    pack = calc.treasury_alco_pack(hedge_ratio=0.8)
    assert "limits" in pack and not pack["limits"].empty
    assert "attribution" in pack
    assert "efficiency" in pack
    assert "packages_pct" in pack
    assert "scenario_kr01" in pack
    assert "rationale" in pack
    assert "post_swap" in pack
    assert "summary" in pack["post_swap"]
    assert set(pack["limits"]["Status"]).issubset({"ok", "AMBER", "BREACH"})
