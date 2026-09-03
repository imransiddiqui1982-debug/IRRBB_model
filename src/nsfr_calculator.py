"""
nsfr_calculator.py
------------------
Basel III Net Stable Funding Ratio (NSFR) engine for IRRBB integration.

NSFR = Available Stable Funding (ASF) / Required Stable Funding (RSF) >= 100%

Implements BCBS NSFR standard factors for:
  - ASF: regulatory capital, retail deposits (95%/90%), wholesale by tenor
  - RSF: HQLA (0–50%), loans (65–85%), other long-term assets (100%)

NMD-refined instruments map to behavioural ASF buckets when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from .cashflows import Instrument
from .lcr_calculator import (
    DEFAULT_STABLE_PCT,
    _classify_hqla,
    _stable_pct_from_nmd,
)

if TYPE_CHECKING:
    from .liquidity_workbook import LiquidityWorkbookData
    from .nmd_refinement import NmdRefinementResult

SIX_MONTHS = 6 / 12
ONE_YEAR = 1.0

# BCBS NSFR ASF factors (liabilities / equity)
ASF_REG_CAPITAL = 1.00
ASF_RETAIL_STABLE = 0.95
ASF_RETAIL_LESS_STABLE = 0.90
ASF_WHOLESALE_STABLE = 0.50
ASF_WHOLESALE_NON_OP = 0.50
ASF_SHORT_TERM = 0.00
ASF_MEDIUM_TERM = 0.50
ASF_LONG_TERM = 1.00

# BCBS NSFR RSF factors (assets)
RSF_CASH_RESERVES = 0.00
RSF_HQLA_LEVEL1 = 0.05
RSF_HQLA_LEVEL2A = 0.15
RSF_HQLA_LEVEL2B = 0.50
RSF_MORTGAGE = 0.65
RSF_PERFORMING_LOAN = 0.85
RSF_OTHER_LONG_TERM = 1.00


@dataclass
class NsfrResult:
    asf_total: float
    rsf_total: float
    nsfr_pct: float
    nsfr_pass: bool
    asf_capital: float
    asf_breakdown: pd.DataFrame
    rsf_breakdown: pd.DataFrame
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def _residual_maturity_years(inst: Instrument) -> float:
    """
    Residual maturity for NSFR ASF/RSF.

    Prepayable amortising assets use principal-weighted average life (WAL)
    so CPR shortens required stable funding tenor versus contractual maturity.
    """
    if (
        getattr(inst, "instrument_type", None) == "amortising"
        and getattr(inst, "prepay_enabled", False)
    ):
        return max(inst.wal_years(), 1 / 365)
    return max(inst.maturity_years, inst.repricing_years)


def _asf_maturity_factor(maturity_years: float) -> tuple[float, str]:
    if maturity_years < SIX_MONTHS:
        return ASF_SHORT_TERM, "Short-term (<6M)"
    if maturity_years < ONE_YEAR:
        return ASF_MEDIUM_TERM, "Medium-term (6M–1Y)"
    return ASF_LONG_TERM, "Long-term (≥1Y)"


def _classify_liability_asf(
    instrument: Instrument,
    stable_pct: float = DEFAULT_STABLE_PCT,
) -> tuple[float, str, float]:
    """
    Return (asf_factor, category_label, applicable_balance).
    """
    name = instrument.name.lower()
    notional = instrument.notional
    mat = _residual_maturity_years(instrument)

    if any(k in name for k in ("tier 2", "tier2", "subordinated")):
        return ASF_REG_CAPITAL, "Regulatory / other capital", notional

    if "core sticky" in name:
        return ASF_RETAIL_STABLE, "Stable retail (NMD core)", notional
    if "rate sensitive" in name:
        return ASF_RETAIL_LESS_STABLE, "Less stable retail (NMD sensitive)", notional
    if "non-core volatile" in name or "non_core" in name:
        return ASF_WHOLESALE_STABLE, "Non-core / volatile wholesale", notional

    if instrument.instrument_type == "demand_deposit":
        asf = (
            stable_pct * ASF_RETAIL_STABLE
            + (1 - stable_pct) * ASF_RETAIL_LESS_STABLE
        )
        return asf, "Retail demand deposit (blended)", notional

    if "term deposit" in name or "savings" in name:
        factor, tenor = _asf_maturity_factor(mat)
        if factor == ASF_LONG_TERM:
            stable = stable_pct * ASF_RETAIL_STABLE + (1 - stable_pct) * ASF_RETAIL_LESS_STABLE
            return stable, f"Retail term — {tenor}", notional
        return factor, f"Retail term — {tenor}", notional

    if "repo" in name or "secured" in name:
        factor, tenor = _asf_maturity_factor(mat)
        return factor, f"Secured funding — {tenor}", notional

    if any(k in name for k in ("senior debt", "covered bond", "borrowings", "floating")):
        factor, tenor = _asf_maturity_factor(mat)
        label = f"Wholesale funding — {tenor}"
        if factor == ASF_LONG_TERM:
            return ASF_LONG_TERM, label, notional
        if factor == ASF_MEDIUM_TERM:
            return ASF_MEDIUM_TERM, label, notional
        return ASF_WHOLESALE_NON_OP if mat >= SIX_MONTHS else ASF_SHORT_TERM, label, notional

    factor, tenor = _asf_maturity_factor(mat)
    return factor, f"Other liability — {tenor}", notional


def _rsf_factor_asset(instrument: Instrument) -> tuple[float, str]:
    """Return BCBS RSF factor and category for an asset."""
    name = instrument.name.lower()
    hqla = _classify_hqla(instrument)

    if hqla == "level_1":
        if any(k in name for k in ("cash", "central bank", "reserve")):
            return RSF_CASH_RESERVES, "Cash / central bank reserves"
        return RSF_HQLA_LEVEL1, "Unencumbered Level 1 HQLA"

    if hqla == "level_2a":
        return RSF_HQLA_LEVEL2A, "Level 2A HQLA"

    if hqla == "level_2b":
        return RSF_HQLA_LEVEL2B, "Level 2B HQLA"

    if "mortgage" in name:
        # BCBS: unencumbered residential mortgages ≤35% LTV often 65% RSF;
        # when WAL < 1Y after CPR, use performing-loan short factor path via maturity.
        mat = _residual_maturity_years(instrument)
        if mat < ONE_YEAR:
            return RSF_PERFORMING_LOAN, "Residential mortgage (WAL <1Y after CPR)"
        return RSF_MORTGAGE, "Residential mortgage"

    mat = _residual_maturity_years(instrument)
    if instrument.instrument_type in ("bullet_fixed", "bullet_floating", "amortising", "demand_deposit"):
        if mat >= ONE_YEAR:
            return RSF_PERFORMING_LOAN, "Performing loan / bond (≥1Y)"
        return RSF_PERFORMING_LOAN, "Performing loan / bond (<1Y)"

    if mat >= ONE_YEAR:
        return RSF_OTHER_LONG_TERM, "Other long-term asset"

    return RSF_PERFORMING_LOAN, "Other short-term asset"


def compute_asf(
    liabilities: list[Instrument],
    tier1_capital: float = 0.0,
    tier2_capital: float | None = None,
    nmd_result: NmdRefinementResult | None = None,
    stable_pct: float | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """
    Compute Available Stable Funding from capital and liabilities.

    Tier 2 is inferred from liability instruments unless tier2_capital is set.
    """
    if stable_pct is None:
        stable_pct = _stable_pct_from_nmd(nmd_result)

    rows = []
    asf_capital = tier1_capital
    asf_other = 0.0

    if tier2_capital is None:
        tier2_capital = sum(
            i.notional for i in liabilities
            if any(k in i.name.lower() for k in ("tier 2", "tier2", "subordinated"))
        )
    asf_capital += tier2_capital

    if tier1_capital > 0:
        rows.append({
            "Source": "Tier 1 Capital",
            "Category": "Regulatory capital",
            "Balance ($M)": tier1_capital,
            "ASF Factor (%)": ASF_REG_CAPITAL * 100,
            "ASF ($M)": round(tier1_capital * ASF_REG_CAPITAL, 2),
        })
    if tier2_capital > 0:
        rows.append({
            "Source": "Tier 2 / Subordinated",
            "Category": "Regulatory capital",
            "Balance ($M)": tier2_capital,
            "ASF Factor (%)": ASF_REG_CAPITAL * 100,
            "ASF ($M)": round(tier2_capital * ASF_REG_CAPITAL, 2),
        })

    counted_t2 = tier2_capital or 0.0
    for inst in liabilities:
        if any(k in inst.name.lower() for k in ("tier 2", "tier2", "subordinated")):
            continue
        factor, category, balance = _classify_liability_asf(inst, stable_pct)
        asf = balance * factor
        asf_other += asf
        rows.append({
            "Source": inst.name,
            "Category": category,
            "Balance ($M)": balance,
            "ASF Factor (%)": round(factor * 100, 1),
            "ASF ($M)": round(asf, 2),
        })

    total = asf_capital + asf_other
    return total, asf_capital, pd.DataFrame(rows)


def compute_rsf(assets: list[Instrument]) -> tuple[float, pd.DataFrame]:
    """Compute Required Stable Funding from asset instruments."""
    rows = []
    total = 0.0

    for inst in assets:
        factor, category = _rsf_factor_asset(inst)
        rsf = inst.notional * factor
        total += rsf
        rows.append({
            "Instrument": inst.name,
            "Category": category,
            "Notional ($M)": inst.notional,
            "RSF Factor (%)": round(factor * 100, 1),
            "RSF ($M)": round(rsf, 2),
        })

    return total, pd.DataFrame(rows)


def compute_nsfr(
    assets: list[Instrument],
    liabilities: list[Instrument],
    tier1_capital: float = 0.0,
    tier2_capital: float | None = None,
    nmd_result: NmdRefinementResult | None = None,
    stable_pct: float | None = None,
) -> NsfrResult:
    """
    Compute Basel III NSFR from balance sheet instruments.

    Args:
        assets: Asset-side instruments (RSF by BCBS asset categories).
        liabilities: Liability-side instruments (ASF by funding type / tenor).
        tier1_capital: Tier 1 capital included in ASF at 100%.
        tier2_capital: Optional Tier 2 override; inferred from liabilities if None.
        nmd_result: Optional NMD refinement for behavioural ASF splits.
        stable_pct: Override stable retail % for ASF.

    Returns:
        NsfrResult with ASF, RSF, ratio and breakdown tables.
    """
    asf_total, asf_capital, asf_df = compute_asf(
        liabilities,
        tier1_capital=tier1_capital,
        tier2_capital=tier2_capital,
        nmd_result=nmd_result,
        stable_pct=stable_pct,
    )
    rsf_total, rsf_df = compute_rsf(assets)

    nsfr_pct = (asf_total / rsf_total * 100) if rsf_total > 0 else float("inf")

    summary = pd.DataFrame([
        {
            "Component": "ASF — Regulatory Capital",
            "Value ($M)": round(asf_capital, 2),
            "Notes": "Tier 1 + Tier 2 at 100%",
        },
        {
            "Component": "ASF — Total",
            "Value ($M)": round(asf_total, 2),
            "Notes": "Capital + stable funding sources",
        },
        {
            "Component": "RSF — Total",
            "Value ($M)": round(rsf_total, 2),
            "Notes": "Required stable funding for assets",
        },
        {
            "Component": "NSFR (%)",
            "Value ($M)": round(nsfr_pct, 2),
            "Notes": "Minimum 100%",
        },
    ])

    return NsfrResult(
        asf_total=asf_total,
        rsf_total=rsf_total,
        nsfr_pct=nsfr_pct,
        nsfr_pass=nsfr_pct >= 100.0,
        asf_capital=asf_capital,
        asf_breakdown=asf_df,
        rsf_breakdown=rsf_df,
        summary=summary,
    )


def compute_nsfr_from_disclosure(
    workbook: LiquidityWorkbookData,
) -> NsfrResult:
    """
    Compute NSFR from U.S. disclosure-style workbook inputs (12 CFR 249.131).

    Uses NSFR ASF / NSFR RSF sheets plus Regulatory Parameters.
    """
    params = workbook.params
    asf_rows = []
    asf_total = 0.0
    asf_capital = params.tier1_capital_m + params.tier2_capital_m

    for item in workbook.nsfr_asf:
        factor = item.asf_factor_pct / 100.0 if item.asf_factor_pct > 1 else item.asf_factor_pct
        asf = item.unweighted_amount_m * factor
        asf_total += asf
        asf_rows.append({
            "Source": item.disclosure_item,
            "Category": item.maturity_bucket.replace("_", " "),
            "Balance ($M)": round(item.unweighted_amount_m, 2),
            "ASF Factor (%)": round(factor * 100, 1),
            "ASF ($M)": round(asf, 2),
            "Disclosure Row": item.row,
            "Source Type": "NMD auto" if item.auto_from_nmd else "User input",
        })
    asf_df = pd.DataFrame(asf_rows)

    rsf_rows = []
    rsf_pre_adj = 0.0
    for item in workbook.nsfr_rsf:
        factor = item.rsf_factor_pct / 100.0 if item.rsf_factor_pct > 1 else item.rsf_factor_pct
        rsf = item.unweighted_amount_m * factor
        rsf_pre_adj += rsf
        rsf_rows.append({
            "Instrument": item.disclosure_item,
            "Category": "RSF",
            "Notional ($M)": round(item.unweighted_amount_m, 2),
            "RSF Factor (%)": round(factor * 100, 1),
            "RSF ($M)": round(rsf, 2),
            "Disclosure Row": item.row,
        })
    rsf_df = pd.DataFrame(rsf_rows)

    rsf_total = rsf_pre_adj * (1.0 - params.rsf_adjustment_pct)
    nsfr_pct = (asf_total / rsf_total * 100) if rsf_total > 0 else float("inf")
    min_pct = params.nsfr_minimum_pct

    summary = pd.DataFrame([
        {"Component": "ASF — Regulatory Capital (row 2)", "Value ($M)": round(asf_capital, 2),
         "Notes": "Tier 1 + Tier 2 at 100%"},
        {"Component": "ASF — Total (row 15)", "Value ($M)": round(asf_total, 2),
         "Notes": "Weighted available stable funding"},
        {"Component": "RSF — Total pre-adjustment (row 37)", "Value ($M)": round(rsf_pre_adj, 2),
         "Notes": "Weighted required stable funding"},
        {"Component": f"RSF tailoring ({params.rsf_adjustment_pct:.0%} reduction, row 39)",
         "Value ($M)": round(rsf_total, 2),
         "Notes": "Category IV adjusted RSF"},
        {"Component": "NSFR (%) (row 40)", "Value ($M)": round(nsfr_pct, 2),
         "Notes": f"Minimum {min_pct:.0f}%"},
    ])

    return NsfrResult(
        asf_total=asf_total,
        rsf_total=rsf_total,
        nsfr_pct=nsfr_pct,
        nsfr_pass=nsfr_pct >= min_pct,
        asf_capital=asf_capital,
        asf_breakdown=asf_df,
        rsf_breakdown=rsf_df,
        summary=summary,
    )


def compute_nsfr_from_nmd_allocation(
    core_balance: float,
    non_core_balance: float,
    alloc_result: dict,
    stable_pct_nsfr: float = DEFAULT_STABLE_PCT,
    asf_factor_stable: float = ASF_RETAIL_STABLE,
    asf_factor_unstable: float = ASF_RETAIL_LESS_STABLE,
    asf_factor_noncash: float = ASF_WHOLESALE_STABLE,
    rsf_factor_cat: float = RSF_PERFORMING_LOAN,
    rsf_factor_hqla: float = RSF_HQLA_LEVEL1,
) -> dict:
    """
    Simplified NSFR from NMD allocation (nmd_model compatible).

    alloc_result expects keys notional_mb.cat and notional_mb.liq, or
    w_cat/w_liq with core_balance for notionals.
    """
    if "notional_mb" in alloc_result:
        cat_notional = alloc_result["notional_mb"].get("cat", 0.0)
        liq_notional = alloc_result["notional_mb"].get("liq", 0.0)
    else:
        w_cat = alloc_result.get("w_cat", 0.0)
        w_liq = alloc_result.get("w_liq", 0.0)
        cat_notional = w_cat * core_balance
        liq_notional = w_liq * core_balance

    asf_stable = core_balance * stable_pct_nsfr * asf_factor_stable
    asf_unstable = core_balance * (1 - stable_pct_nsfr) * asf_factor_unstable
    asf_noncash = non_core_balance * asf_factor_noncash
    asf_total = asf_stable + asf_unstable + asf_noncash

    rsf_cat = cat_notional * rsf_factor_cat
    rsf_hqla = liq_notional * rsf_factor_hqla
    rsf_total = rsf_cat + rsf_hqla

    nsfr_pct = (asf_total / rsf_total * 100) if rsf_total > 0 else float("inf")

    return {
        "asf_stable": float(asf_stable),
        "asf_unstable": float(asf_unstable),
        "asf_noncash": float(asf_noncash),
        "asf_total": float(asf_total),
        "rsf_cat": float(rsf_cat),
        "rsf_hqla": float(rsf_hqla),
        "rsf_total": float(rsf_total),
        "nsfr_pct": float(nsfr_pct),
        "nsfr_pass": nsfr_pct >= 100.0,
    }
