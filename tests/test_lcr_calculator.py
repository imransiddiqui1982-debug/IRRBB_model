"""
tests/test_lcr_calculator.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.balance_sheet import get_instruments  # noqa: E402
from src.cashflows import Instrument  # noqa: E402
from src.lcr_calculator import (  # noqa: E402
    compute_lcr,
    compute_lcr_from_nmd_allocation,
    compute_hqla_stock,
    _apply_level2_caps,
)


@pytest.fixture
def instruments():
    return get_instruments()


def test_hqla_from_sample_balance_sheet(instruments):
    assets, _ = instruments
    l1, l2a, l2b, df = compute_hqla_stock(assets)
    assert l1 > 0
    # Ginnie Mae MBS → Level 1; Fannie/Freddie → Level 2A
    assert l2a > 0
    assert l2b == 0.0
    assert not df.empty
    assert any("Ginnie" in n or "Cash" in n or "Gov" in n or "T-Bill" in n for n in df["Instrument"])
    assert any("Fannie" in n or "Freddie" in n or "Agency" in n for n in df["Instrument"])


def test_mbs_hqla_grid():
    ginnie = Instrument(
        "Ginnie Mae MBS", 100, 4.5, "mbs", 15.0,
        payment_freq=12, side="asset", mbs_level="ginnie",
    )
    agency = Instrument(
        "Fannie MBS", 100, 4.8, "mbs", 15.0,
        payment_freq=12, side="asset", mbs_level="agency",
    )
    private = Instrument(
        "Private-label RMBS", 100, 5.5, "mbs", 15.0,
        payment_freq=12, side="asset", mbs_level="private",
    )
    from src.lcr_calculator import _classify_hqla, HQLA_HAIRCUT
    assert _classify_hqla(ginnie) == "level_1"
    assert HQLA_HAIRCUT["level_1"] == 0.0
    assert _classify_hqla(agency) == "level_2a"
    assert HQLA_HAIRCUT["level_2a"] == 0.15
    assert _classify_hqla(private) is None


def test_lcr_sample_balance_sheet(instruments):
    assets, liabilities = instruments
    result = compute_lcr(assets, liabilities)
    assert result.hqla_stock > 0
    assert result.total_outflows > 0
    assert result.lcr_pct > 0
    assert isinstance(result.lcr_pass, bool)
    assert not result.summary.empty


def test_lcr_inflow_cap(instruments):
    assets, liabilities = instruments
    result = compute_lcr(assets, liabilities)
    assert result.capped_inflows <= result.total_outflows * 0.75 + 1e-9


def test_level2_caps():
    l1, l2a, l2b = _apply_level2_caps(100, 100, 100)
    pre_cap_total = 300.0
    assert l1 == 100
    assert l2b <= pre_cap_total * 0.15 + 1e-9
    assert l2a + l2b <= pre_cap_total * 0.40 + 1e-9


def test_nmd_allocation_lcr():
    result = compute_lcr_from_nmd_allocation(
        core_balance=800,
        non_core_balance=200,
        hqla_mb=577.52,
    )
    assert result["lcr_pct"] > 0
    assert result["total_outflow"] > 0
    assert "lcr_pass" in result


def test_nmd_refined_outflows():
    assets, liabilities = get_instruments()
    nmd_liabs = [
        Instrument(
            name="Retail — Core Sticky",
            notional=500, coupon_pct=1.0,
            instrument_type="demand_deposit",
            maturity_years=1 / 12, repricing_years=1 / 12,
            payment_freq=12, side="liability",
        ),
        Instrument(
            name="Retail — Rate Sensitive",
            notional=100, coupon_pct=1.5,
            instrument_type="demand_deposit",
            maturity_years=1 / 12, repricing_years=1 / 12,
            payment_freq=12, side="liability",
        ),
        Instrument(
            name="Retail — Non-Core Volatile",
            notional=150, coupon_pct=1.0,
            instrument_type="demand_deposit",
            maturity_years=1 / 12, repricing_years=1 / 12,
            payment_freq=12, side="liability",
        ),
    ]
    filtered = [l for l in liabilities if l.instrument_type != "demand_deposit"]
    result = compute_lcr(assets, filtered + nmd_liabs)
    outflow_names = set(result.outflow_breakdown["Instrument"])
    assert "Retail — Core Sticky" in outflow_names
    assert result.total_outflows > 0
