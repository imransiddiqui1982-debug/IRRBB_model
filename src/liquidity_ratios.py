"""
liquidity_ratios.py
-------------------
Combined LCR and NSFR reporting for IRRBB integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from .lcr_calculator import LcrResult, compute_lcr, compute_lcr_from_disclosure
from .liquidity_workbook import LiquidityWorkbookData, load_liquidity_workbook
from .nsfr_calculator import NsfrResult, compute_nsfr, compute_nsfr_from_disclosure

if TYPE_CHECKING:
    from .cashflows import Instrument
    from .nmd_refinement import NmdRefinementResult


@dataclass
class LiquidityRatiosResult:
    lcr: LcrResult
    nsfr: NsfrResult
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = "balance_sheet"
    workbook: LiquidityWorkbookData | None = None


def compute_liquidity_ratios(
    assets: list[Instrument],
    liabilities: list[Instrument],
    tier1_capital: float = 0.0,
    tier2_capital: float | None = None,
    nmd_result: NmdRefinementResult | None = None,
    stable_pct: float | None = None,
) -> LiquidityRatiosResult:
    """Compute LCR and NSFR together with a combined summary table."""
    lcr = compute_lcr(assets, liabilities, nmd_result=nmd_result, stable_pct=stable_pct)
    nsfr = compute_nsfr(
        assets,
        liabilities,
        tier1_capital=tier1_capital,
        tier2_capital=tier2_capital,
        nmd_result=nmd_result,
        stable_pct=stable_pct,
    )

    summary = pd.DataFrame([
        {
            "Metric": "LCR",
            "Numerator ($M)": round(lcr.hqla_stock, 2),
            "Denominator ($M)": round(lcr.net_cash_outflows, 2),
            "Ratio (%)": round(lcr.lcr_pct, 2),
            "Minimum (%)": 100,
            "Pass": lcr.lcr_pass,
        },
        {
            "Metric": "NSFR",
            "Numerator ($M)": round(nsfr.asf_total, 2),
            "Denominator ($M)": round(nsfr.rsf_total, 2),
            "Ratio (%)": round(nsfr.nsfr_pct, 2),
            "Minimum (%)": 100,
            "Pass": nsfr.nsfr_pass,
        },
    ])

    return LiquidityRatiosResult(lcr=lcr, nsfr=nsfr, summary=summary, source="balance_sheet")


def compute_liquidity_from_workbook(
    workbook: LiquidityWorkbookData,
) -> LiquidityRatiosResult:
    """Compute LCR and NSFR entirely from disclosure workbook line items."""
    lcr = compute_lcr_from_disclosure(workbook)
    nsfr = compute_nsfr_from_disclosure(workbook)

    summary = pd.DataFrame([
        {
            "Metric": "LCR",
            "Numerator ($M)": round(lcr.hqla_stock, 2),
            "Denominator ($M)": round(lcr.net_cash_outflows, 2),
            "Ratio (%)": round(lcr.lcr_pct, 2),
            "Minimum (%)": workbook.params.lcr_minimum_pct,
            "Pass": lcr.lcr_pass,
        },
        {
            "Metric": "NSFR",
            "Numerator ($M)": round(nsfr.asf_total, 2),
            "Denominator ($M)": round(nsfr.rsf_total, 2),
            "Ratio (%)": round(nsfr.nsfr_pct, 2),
            "Minimum (%)": workbook.params.nsfr_minimum_pct,
            "Pass": nsfr.nsfr_pass,
        },
    ])

    return LiquidityRatiosResult(
        lcr=lcr,
        nsfr=nsfr,
        summary=summary,
        source="workbook",
        workbook=workbook,
    )


def compute_liquidity_ratios_with_workbook(
    assets: list[Instrument],
    liabilities: list[Instrument],
    workbook_source,
    nmd_result: NmdRefinementResult | None = None,
    customers: pd.DataFrame | None = None,
    tier1_capital: float = 0.0,
    tier2_capital: float | None = None,
    stable_pct: float | None = None,
) -> LiquidityRatiosResult:
    """
    Prefer workbook-driven LCR/NSFR when liquidity sheets exist;
    otherwise fall back to balance-sheet heuristics.
    """
    if workbook_source is not None:
        wb = load_liquidity_workbook(
            workbook_source, nmd_result=nmd_result, customers=customers,
        )
        if wb.has_liquidity_sheets and (
            wb.hqla_items or wb.lcr_outflows or wb.nsfr_asf or wb.nsfr_rsf
        ):
            if wb.params.tier1_capital_m <= 0 and tier1_capital > 0:
                wb.params.tier1_capital_m = tier1_capital
            if tier2_capital is not None and wb.params.tier2_capital_m <= 0:
                wb.params.tier2_capital_m = tier2_capital
            return compute_liquidity_from_workbook(wb)

    return compute_liquidity_ratios(
        assets,
        liabilities,
        tier1_capital=tier1_capital,
        tier2_capital=tier2_capital,
        nmd_result=nmd_result,
        stable_pct=stable_pct,
    )
