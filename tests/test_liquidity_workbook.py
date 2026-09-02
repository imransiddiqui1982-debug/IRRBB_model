"""
tests/test_liquidity_workbook.py
"""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.liquidity_ratios import compute_liquidity_from_workbook  # noqa: E402
from src.liquidity_workbook import (  # noqa: E402
    load_liquidity_workbook,
    workbook_has_liquidity_sheets,
)
from src.nmd_refinement import refine_nmd_deposits  # noqa: E402

V2_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "comprehensive_deposit_template_v2.xlsx"
)


@pytest.fixture
def v2_path():
    if not os.path.exists(V2_PATH):
        pytest.skip("v2 template not found — run scripts/extend_v2_liquidity_sheets.py")
    return V2_PATH


def test_workbook_has_liquidity_sheets(v2_path):
    assert workbook_has_liquidity_sheets(v2_path)


def test_load_liquidity_workbook(v2_path):
    wb = load_liquidity_workbook(v2_path)
    assert wb.has_liquidity_sheets
    assert len(wb.hqla_items) >= 5
    assert len(wb.lcr_outflows) >= 5
    assert len(wb.nsfr_asf) >= 5
    assert len(wb.nsfr_rsf) >= 5
    assert wb.params.tier1_capital_m > 0


def test_nmd_auto_fill_outflows(v2_path):
    nmd = refine_nmd_deposits(v2_path, deposit_name="Test NMD")
    wb = load_liquidity_workbook(v2_path, nmd_result=nmd)
    auto_rows = [o for o in wb.lcr_outflows if o.auto_from_nmd]
    assert auto_rows
    assert sum(o.unweighted_amount_m for o in auto_rows) > 0
    assert wb.nmd_buckets.core_sticky_m > 0


def test_compute_lcr_nsfr_from_workbook(v2_path):
    nmd = refine_nmd_deposits(v2_path, deposit_name="Test NMD")
    wb = load_liquidity_workbook(v2_path, nmd_result=nmd)
    result = compute_liquidity_from_workbook(wb)

    assert result.source == "workbook"
    assert result.lcr.hqla_stock > 0
    assert result.lcr.total_outflows > 0
    assert result.lcr.lcr_pct > 0
    assert result.nsfr.asf_total > 0
    assert result.nsfr.rsf_total > 0
    assert result.nsfr.nsfr_pct > 0
    assert not result.lcr.summary.empty
    assert not result.nsfr.summary.empty


def test_lcr_level2_caps_applied(v2_path):
    wb = load_liquidity_workbook(v2_path)
    result = compute_liquidity_from_workbook(wb)
    pre_cap = sum(
        h.unweighted_amount_m * (1 - h.haircut_pct / 100)
        for h in wb.hqla_items
    )
    assert result.lcr.hqla_stock <= pre_cap + 1e-6


def test_inflow_cap_respected(v2_path):
    nmd = refine_nmd_deposits(v2_path)
    wb = load_liquidity_workbook(v2_path, nmd_result=nmd)
    result = compute_liquidity_from_workbook(wb)
    assert result.lcr.capped_inflows <= (
        result.lcr.total_outflows * wb.params.inflow_cap_pct + 1e-6
    )


def test_bytesio_source(v2_path):
    with open(v2_path, "rb") as f:
        data = f.read()
    nmd = refine_nmd_deposits(io.BytesIO(data))
    wb = load_liquidity_workbook(io.BytesIO(data), nmd_result=nmd)
    result = compute_liquidity_from_workbook(wb)
    assert result.lcr.lcr_pct > 0
