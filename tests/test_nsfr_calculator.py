"""
tests/test_nsfr_calculator.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.balance_sheet import get_instruments  # noqa: E402
from src.cashflows import Instrument  # noqa: E402
from src.liquidity_ratios import compute_liquidity_ratios  # noqa: E402
from src.nsfr_calculator import (  # noqa: E402
    compute_nsfr,
    compute_nsfr_from_nmd_allocation,
    compute_rsf,
    compute_asf,
    RSF_HQLA_LEVEL1,
    RSF_MORTGAGE,
    ASF_RETAIL_STABLE,
)


@pytest.fixture
def instruments():
    return get_instruments()


def test_nsfr_sample_balance_sheet(instruments):
    assets, liabilities = instruments
    result = compute_nsfr(assets, liabilities, tier1_capital=500.0)
    assert result.asf_total > 0
    assert result.rsf_total > 0
    assert result.nsfr_pct > 0
    assert isinstance(result.nsfr_pass, bool)
    assert not result.asf_breakdown.empty
    assert not result.rsf_breakdown.empty


def test_asf_includes_tier1(instruments):
    _, liabilities = instruments
    asf_with, _, _ = compute_asf(liabilities, tier1_capital=500.0)
    asf_without, _, _ = compute_asf(liabilities, tier1_capital=0.0)
    assert asf_with > asf_without
    assert asf_with - asf_without == pytest.approx(500.0, rel=1e-6)


def test_rsf_mortgage_factor(instruments):
    assets, _ = instruments
    _, rsf_df = compute_rsf(assets)
    mortgage = rsf_df[rsf_df["Instrument"].str.contains("Mortgage", case=False)]
    assert not mortgage.empty
    assert mortgage.iloc[0]["RSF Factor (%)"] == RSF_MORTGAGE * 100


def test_rsf_hqla_factor(instruments):
    assets, _ = instruments
    _, rsf_df = compute_rsf(assets)
    cash = rsf_df[rsf_df["Instrument"].str.contains("Cash", case=False)]
    assert not cash.empty
    assert cash.iloc[0]["RSF Factor (%)"] == 0.0

    gov = rsf_df[rsf_df["Instrument"].str.contains("Gov Bond", case=False)]
    assert not gov.empty
    assert gov.iloc[0]["RSF Factor (%)"] == RSF_HQLA_LEVEL1 * 100


def test_nmd_allocation_nsfr():
    result = compute_nsfr_from_nmd_allocation(
        core_balance=3850,
        non_core_balance=1189,
        alloc_result={"w_cat": 0.85, "w_liq": 0.15},
    )
    assert result["asf_total"] > 0
    assert result["rsf_total"] > 0
    assert result["nsfr_pct"] > 100
    assert result["nsfr_pass"]


def test_liquidity_ratios_combined(instruments):
    assets, liabilities = instruments
    result = compute_liquidity_ratios(assets, liabilities, tier1_capital=500.0)
    assert len(result.summary) == 2
    assert result.summary.iloc[0]["Metric"] == "LCR"
    assert result.summary.iloc[1]["Metric"] == "NSFR"


def test_nmd_refined_asf():
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
            name="Retail — Non-Core Volatile",
            notional=150, coupon_pct=1.0,
            instrument_type="demand_deposit",
            maturity_years=1 / 12, repricing_years=1 / 12,
            payment_freq=12, side="liability",
        ),
    ]
    filtered = [l for l in liabilities if l.instrument_type != "demand_deposit"]
    result = compute_nsfr(assets, filtered + nmd_liabs, tier1_capital=500.0)
    sources = set(result.asf_breakdown["Source"])
    assert "Retail — Core Sticky" in sources
    core_row = result.asf_breakdown[
        result.asf_breakdown["Source"] == "Retail — Core Sticky"
    ].iloc[0]
    assert core_row["ASF Factor (%)"] == ASF_RETAIL_STABLE * 100
