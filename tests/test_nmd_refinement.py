
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.balance_sheet import get_instruments  # noqa: E402
from src.nmd_refinement import refine_nmd_deposits, merge_nmd_into_balance_sheet  # noqa: E402


TEMPLATE = os.path.join(
    os.path.dirname(__file__), "..", "data", "comprehensive_deposit_template.xlsx"
)
LEGACY_TEMPLATE = os.path.join(
    os.path.dirname(__file__), "..", "data", "nmd_deposit_template.xlsx"
)


def test_refine_customer_level_excel():
    result = refine_nmd_deposits(TEMPLATE, deposit_name="Test NMD")

    assert result.customer_count >= 1
    assert result.month_count == 36
    assert result.latest_balance_mb > 0
    assert 0 < result.stable_pct <= 1
    assert 0 <= result.beta <= 1
    assert result.wal_years > 0
    assert len(result.instruments) >= 1
    assert not result.segment_summary.empty
    assert abs(
        sum(i.notional for i in result.instruments) - result.latest_balance_mb
    ) < 1.0


def test_segment_caps_applied():
    result = refine_nmd_deposits(TEMPLATE)
    by_seg = {row["Segment"]: row for _, row in result.segment_summary.iterrows()}
    if "retail_transactional" in by_seg:
        assert by_seg["retail_transactional"]["Stable %"] <= 90.0 + 1e-6
    if "wholesale" in by_seg:
        assert by_seg["wholesale"]["Stable %"] <= 50.0 + 1e-6
        assert by_seg["wholesale"]["WAL (Y)"] <= 4.0 + 1e-6


def test_merge_replaces_demand_deposit_liabilities():
    assets, liabilities = get_instruments()
    nmd = refine_nmd_deposits(TEMPLATE)

    old_demand_count = sum(1 for i in liabilities if i.instrument_type == "demand_deposit")
    old_term_count = sum(1 for i in liabilities if "term deposit" in i.name.lower())
    assert old_demand_count >= 1

    _, new_liabilities = merge_nmd_into_balance_sheet(assets, liabilities, nmd)
    new_demand_count = sum(1 for i in new_liabilities if i.instrument_type == "demand_deposit")
    nmd_demand_count = sum(1 for i in nmd.instruments if i.instrument_type == "demand_deposit")
    assert new_demand_count == nmd_demand_count
    removed_terms = old_term_count if nmd.term_instruments else 0
    assert len(new_liabilities) == len(liabilities) - old_demand_count - removed_terms + len(nmd.instruments)


def test_comprehensive_template_has_term_deposits():
    result = refine_nmd_deposits(TEMPLATE, deposit_name="Comprehensive NMD")
    assert result.liquidity_summary is not None
    assert not result.liquidity_summary.empty
    assert len(result.term_instruments) >= 1
    term_names = " ".join(i.name for i in result.term_instruments).lower()
    assert "term deposit" in term_names


def test_legacy_template_still_loads():
    result = refine_nmd_deposits(LEGACY_TEMPLATE, deposit_name="Legacy NMD")
    assert result.customer_count >= 1


def test_nmd_rejects_incomplete_file():
    bad = io.StringIO("customer_id,segment\nC1,retail\n")
    try:
        refine_nmd_deposits(bad)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "unrecognized" in str(exc).lower() or "missing" in str(exc).lower()
