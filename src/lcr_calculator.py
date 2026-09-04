"""
lcr_calculator.py
-----------------
Basel III Liquidity Coverage Ratio (LCR) engine for IRRBB integration.

LCR = HQLA Stock / Total Net Cash Outflows (30-day stress) >= 100%

Implements BCBS liquidity standards aligned with the liquidity-lcr skill:
  - HQLA Level 1 / 2A / 2B with haircuts and Level-2 caps
  - Retail deposit runoff (stable 5%, less stable 10%)
  - Wholesale / non-core runoff (25–100%)
  - Cash inflows from maturing assets capped at 75% of outflows

When NMD refinement is available, deposit outflows use behavioural
core / sticky / non-core splits by segment instead of flat assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from .cashflows import Instrument

if TYPE_CHECKING:
    from .liquidity_workbook import LiquidityWorkbookData
    from .nmd_refinement import NmdRefinementResult

# ── HQLA haircuts (post-haircut value = notional × (1 - haircut)) ─────────────
HQLA_HAIRCUT = {
    "level_1": 0.00,
    "level_2a": 0.15,
    "level_2b": 0.50,
}

LEVEL2_CAP = 0.40
LEVEL2B_CAP = 0.15
INFLOW_CAP = 0.75
THIRTY_DAYS = 30 / 365

# BCBS standard 30-day runoff rates
OUTFLOW_RETAIL_STABLE = 0.05
OUTFLOW_RETAIL_LESS_STABLE = 0.10
OUTFLOW_NON_CORE = 0.25
OUTFLOW_WHOLESALE_NON_OP = 0.40
OUTFLOW_WHOLESALE_OPERATIONAL = 0.25
OUTFLOW_WHOLESALE_FIN = 1.00
OUTFLOW_SECURED = 0.00

INFLOW_PERF_LOAN = 0.50
INFLOW_SECURITIES = 1.00

DEFAULT_STABLE_PCT = 0.75


@dataclass
class LcrResult:
    hqla_level1: float
    hqla_level2a: float
    hqla_level2b: float
    hqla_level2a_adjusted: float
    hqla_level2b_adjusted: float
    hqla_stock: float
    total_outflows: float
    total_inflows: float
    capped_inflows: float
    net_cash_outflows: float
    lcr_pct: float
    lcr_pass: bool
    hqla_breakdown: pd.DataFrame
    outflow_breakdown: pd.DataFrame
    inflow_breakdown: pd.DataFrame
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def _classify_hqla(instrument: Instrument) -> str | None:
    """Map an asset to HQLA level, or None if not HQLA-eligible."""
    name = instrument.name.lower()
    if any(k in name for k in ("cash", "central bank", "reserve")):
        return "level_1"
    if "t-bill" in name or "tbill" in name:
        return "level_1"
    if "gov bond" in name or "government bond" in name or "sovereign" in name:
        return "level_1"
    if "covered bond" in name:
        return "level_2a"
    if "mbs" in name or "mortgage-backed" in name or "mortgage backed" in name:
        # Agency / GSE pass-through treated as Level 2A for this prototype
        return "level_2a"
    if instrument.instrument_type == "mbs":
        return "level_2a"
    if instrument.maturity_years <= THIRTY_DAYS and "bill" in name:
        return "level_1"
    return None


def _classify_liability_outflow(
    instrument: Instrument,
    stable_pct: float = DEFAULT_STABLE_PCT,
) -> tuple[str, float, float, float]:
    """
    Return (category, stable_balance, less_stable_balance, non_core_balance)
    for 30-day outflow calculation.
    """
    name = instrument.name.lower()
    notional = instrument.notional

    if "core sticky" in name:
        return "nmd_core_sticky", notional, 0.0, 0.0
    if "rate sensitive" in name:
        return "nmd_rate_sensitive", 0.0, notional, 0.0
    if "non-core volatile" in name or "non_core" in name:
        if "wholesale operational" in name or "operational" in name.split("—")[-1]:
            return "wholesale_operational", 0.0, 0.0, notional
        return "nmd_non_core", 0.0, 0.0, notional

    if instrument.instrument_type == "demand_deposit":
        stable = notional * stable_pct
        less_stable = notional * (1 - stable_pct)
        return "retail_demand", stable, less_stable, 0.0

    if "repo" in name or "secured" in name:
        return "secured", 0.0, 0.0, 0.0

    if any(k in name for k in ("subordinated", "tier 2", "tier2")):
        return "wholesale_financial", 0.0, 0.0, notional

    if any(k in name for k in ("senior debt", "covered bond", "borrowings", "floating")):
        return "wholesale_non_op", 0.0, 0.0, notional

    if "term deposit" in name or "savings" in name:
        if instrument.maturity_years <= THIRTY_DAYS:
            return "retail_term_short", 0.0, 0.0, 0.0
        stable = notional * stable_pct
        less_stable = notional * (1 - stable_pct)
        return "retail_term", stable, less_stable, 0.0

    return "wholesale_non_op", 0.0, 0.0, notional


def _outflow_rate(category: str) -> tuple[float, float, float]:
    """Return runoff multipliers for (stable, less_stable, non_core) portions."""
    rates = {
        "nmd_core_sticky": (OUTFLOW_RETAIL_STABLE, 0.0, 0.0),
        "nmd_rate_sensitive": (0.0, OUTFLOW_RETAIL_LESS_STABLE, 0.0),
        "nmd_non_core": (0.0, 0.0, OUTFLOW_NON_CORE),
        "wholesale_operational": (0.0, 0.0, OUTFLOW_WHOLESALE_OPERATIONAL),
        "retail_demand": (OUTFLOW_RETAIL_STABLE, OUTFLOW_RETAIL_LESS_STABLE, 0.0),
        "retail_term": (OUTFLOW_RETAIL_STABLE, OUTFLOW_RETAIL_LESS_STABLE, 0.0),
        "retail_term_short": (0.0, 0.0, 0.0),
        "secured": (OUTFLOW_SECURED, OUTFLOW_SECURED, OUTFLOW_SECURED),
        "wholesale_non_op": (0.0, 0.0, OUTFLOW_WHOLESALE_NON_OP),
        "wholesale_financial": (0.0, 0.0, OUTFLOW_WHOLESALE_FIN),
    }
    return rates.get(category, (0.0, 0.0, OUTFLOW_WHOLESALE_NON_OP))


def _apply_level2_caps(
    level1: float,
    level2a: float,
    level2b: float,
) -> tuple[float, float, float]:
    """Apply Level 2 (40%) and Level 2B (15%) caps to HQLA stock."""
    total_pre_cap = level1 + level2a + level2b
    if total_pre_cap <= 0:
        return level1, level2a, level2b

    l2b_cap = total_pre_cap * LEVEL2B_CAP
    l2b_adj = min(level2b, l2b_cap)

    l2_total_cap = total_pre_cap * LEVEL2_CAP
    l2a_adj = min(level2a, max(0.0, l2_total_cap - l2b_adj))

    return level1, l2a_adj, l2b_adj


def _stable_pct_from_nmd(nmd_result: NmdRefinementResult | None) -> float:
    if nmd_result is None:
        return DEFAULT_STABLE_PCT
    return nmd_result.stable_pct if nmd_result.stable_pct > 0 else DEFAULT_STABLE_PCT


def compute_hqla_stock(assets: list[Instrument]) -> tuple[float, float, float, pd.DataFrame]:
    """Compute HQLA by level with haircuts applied."""
    rows = []
    l1, l2a, l2b = 0.0, 0.0, 0.0

    for inst in assets:
        level = _classify_hqla(inst)
        if level is None:
            continue
        haircut = HQLA_HAIRCUT[level]
        adj = inst.notional * (1 - haircut)
        rows.append({
            "Instrument": inst.name,
            "Notional ($M)": inst.notional,
            "HQLA Level": level.replace("_", " ").upper(),
            "Haircut (%)": haircut * 100,
            "HQLA Value ($M)": round(adj, 2),
        })
        if level == "level_1":
            l1 += adj
        elif level == "level_2a":
            l2a += adj
        else:
            l2b += adj

    return l1, l2a, l2b, pd.DataFrame(rows)


def compute_cash_outflows(
    liabilities: list[Instrument],
    nmd_result: NmdRefinementResult | None = None,
    stable_pct: float | None = None,
) -> tuple[float, pd.DataFrame]:
    """Compute 30-day cash outflows from liability instruments."""
    if stable_pct is None:
        stable_pct = _stable_pct_from_nmd(nmd_result)

    rows = []
    total = 0.0

    for inst in liabilities:
        cat, stable, less, non_core = _classify_liability_outflow(inst, stable_pct)
        rs, rl, rn = _outflow_rate(cat)
        outflow = stable * rs + less * rl + non_core * rn
        total += outflow
        if outflow > 0 or cat.startswith("nmd") or inst.instrument_type == "demand_deposit":
            rows.append({
                "Instrument": inst.name,
                "Category": cat.replace("_", " ").title(),
                "Notional ($M)": inst.notional,
                "30d Outflow ($M)": round(outflow, 2),
                "Runoff Rate (%)": round(
                    (outflow / inst.notional * 100) if inst.notional > 0 else 0, 1
                ),
            })

    return total, pd.DataFrame(rows)


def compute_cash_inflows(assets: list[Instrument]) -> tuple[float, pd.DataFrame]:
    """
    Compute 30-day cash inflows from maturing / repricing assets.

    Prepayable amortising loans contribute scheduled + prepaid principal
    returned within 30 days (CPR-aware), not the full face amount.
    """
    rows = []
    total = 0.0

    for inst in assets:
        hqla = _classify_hqla(inst)

        if inst.is_prepayable:
            # Monthly mortgages/MBS pay at 1/12Y; LCR window is 30/365 ≈ 0.082Y.
            # Prorate the first payment period into the 30-day stress window.
            # (User shock-CPR table is EVE/NII only — LCR uses the base schedule.)
            period = 1.0 / max(int(inst.payment_freq), 1)
            first_period_prin = inst.principal_within_years(period)
            returned = first_period_prin * min(1.0, THIRTY_DAYS / period)
            if returned <= 1e-12:
                continue
            rate = INFLOW_PERF_LOAN
            inflow = returned * rate
            cat = "Mortgage / MBS (30d principal)"
            total += inflow
            rows.append({
                "Instrument": inst.name,
                "Category": cat,
                "Notional ($M)": round(returned, 2),
                "30d Inflow ($M)": round(inflow, 2),
                "Inflow Rate (%)": rate * 100,
            })
            continue

        if inst.maturity_years > THIRTY_DAYS and inst.repricing_years > THIRTY_DAYS:
            continue

        if hqla:
            rate = INFLOW_SECURITIES
            cat = "HQLA maturing"
        elif inst.side == "asset":
            rate = INFLOW_PERF_LOAN
            cat = "Performing loan maturing"
        else:
            continue

        inflow = inst.notional * rate
        if inflow <= 0:
            continue
        total += inflow
        rows.append({
            "Instrument": inst.name,
            "Category": cat,
            "Notional ($M)": inst.notional,
            "30d Inflow ($M)": round(inflow, 2),
            "Inflow Rate (%)": rate * 100,
        })

    return total, pd.DataFrame(rows)


def compute_lcr(
    assets: list[Instrument],
    liabilities: list[Instrument],
    nmd_result: NmdRefinementResult | None = None,
    stable_pct: float | None = None,
) -> LcrResult:
    """
    Compute Basel III LCR from balance sheet instruments.

    Args:
        assets: Asset-side instruments (HQLA classification by name/type).
        liabilities: Liability-side instruments (outflow classification).
        nmd_result: Optional NMD refinement for behavioural deposit splits.
        stable_pct: Override stable retail % (default 75% or from NMD).

    Returns:
        LcrResult with HQLA stock, outflows, inflows, LCR ratio and breakdowns.
    """
    l1, l2a, l2b, hqla_df = compute_hqla_stock(assets)
    l1_cap, l2a_cap, l2b_cap = _apply_level2_caps(l1, l2a, l2b)
    hqla_stock = l1_cap + l2a_cap + l2b_cap

    total_outflows, outflow_df = compute_cash_outflows(
        liabilities, nmd_result, stable_pct
    )
    total_inflows, inflow_df = compute_cash_inflows(assets)
    capped_inflows = min(total_inflows, total_outflows * INFLOW_CAP)
    net_outflows = max(total_outflows - capped_inflows, 0.0)

    lcr_pct = (hqla_stock / net_outflows * 100) if net_outflows > 0 else float("inf")

    summary = pd.DataFrame([
        {
            "Component": "HQLA Level 1",
            "Value ($M)": round(l1_cap, 2),
            "Notes": "Cash, reserves, sovereign (100%)",
        },
        {
            "Component": "HQLA Level 2A (capped)",
            "Value ($M)": round(l2a_cap, 2),
            "Notes": "Covered bonds etc. (85% after haircut)",
        },
        {
            "Component": "HQLA Level 2B (capped)",
            "Value ($M)": round(l2b_cap, 2),
            "Notes": "Max 15% of HQLA stock",
        },
        {
            "Component": "HQLA Stock",
            "Value ($M)": round(hqla_stock, 2),
            "Notes": "After Level 2 caps",
        },
        {
            "Component": "Total Cash Outflows (30d)",
            "Value ($M)": round(total_outflows, 2),
            "Notes": "BCBS stress runoff rates",
        },
        {
            "Component": "Total Cash Inflows (30d)",
            "Value ($M)": round(total_inflows, 2),
            "Notes": "Maturing assets",
        },
        {
            "Component": "Capped Inflows (75%)",
            "Value ($M)": round(capped_inflows, 2),
            "Notes": "Max 75% of outflows",
        },
        {
            "Component": "Net Cash Outflows",
            "Value ($M)": round(net_outflows, 2),
            "Notes": "Outflows − capped inflows",
        },
        {
            "Component": "LCR (%)",
            "Value ($M)": round(lcr_pct, 2),
            "Notes": "Minimum 100%",
        },
    ])

    return LcrResult(
        hqla_level1=l1_cap,
        hqla_level2a=l2a_cap,
        hqla_level2b=l2b_cap,
        hqla_level2a_adjusted=l2a_cap,
        hqla_level2b_adjusted=l2b_cap,
        hqla_stock=hqla_stock,
        total_outflows=total_outflows,
        total_inflows=total_inflows,
        capped_inflows=capped_inflows,
        net_cash_outflows=net_outflows,
        lcr_pct=lcr_pct,
        lcr_pass=lcr_pct >= 100.0,
        hqla_breakdown=hqla_df,
        outflow_breakdown=outflow_df,
        inflow_breakdown=inflow_df,
        summary=summary,
    )


def _hqla_from_workbook_items(
    items: list,
    level2_cap: float = LEVEL2_CAP,
    level2b_cap: float = LEVEL2B_CAP,
) -> tuple[float, float, float, pd.DataFrame]:
    """Compute post-haircut HQLA from disclosure line items."""
    rows = []
    l1, l2a, l2b = 0.0, 0.0, 0.0
    for item in items:
        net = max(item.unweighted_amount_m - item.encumbered_amount_m, 0.0)
        if net <= 0:
            continue
        haircut = item.haircut_pct / 100.0 if item.haircut_pct > 1 else item.haircut_pct
        adj = net * (1 - haircut)
        level = item.hqla_level.replace(" ", "_").lower()
        rows.append({
            "Instrument": item.disclosure_item,
            "Notional ($M)": round(net, 2),
            "HQLA Level": level.replace("_", " ").upper(),
            "Haircut (%)": round(haircut * 100, 1),
            "HQLA Value ($M)": round(adj, 2),
            "Disclosure Row": item.row,
        })
        if level == "level_1":
            l1 += adj
        elif level == "level_2a":
            l2a += adj
        else:
            l2b += adj
    l1_cap, l2a_cap, l2b_cap = _apply_level2_caps_custom(l1, l2a, l2b, level2_cap, level2b_cap)
    return l1_cap, l2a_cap, l2b_cap, pd.DataFrame(rows)


def _apply_level2_caps_custom(
    level1: float,
    level2a: float,
    level2b: float,
    level2_cap: float,
    level2b_cap: float,
) -> tuple[float, float, float]:
    total_pre_cap = level1 + level2a + level2b
    if total_pre_cap <= 0:
        return level1, level2a, level2b
    l2b_adj = min(level2b, total_pre_cap * level2b_cap)
    l2_total_cap = total_pre_cap * level2_cap
    l2a_adj = min(level2a, max(0.0, l2_total_cap - l2b_adj))
    return level1, l2a_adj, l2b_adj


def compute_lcr_from_disclosure(
    workbook: LiquidityWorkbookData,
) -> LcrResult:
    """
    Compute LCR from U.S. disclosure-style workbook inputs (12 CFR 249.91).

    Uses HQLA Stock, LCR Cash Outflows, LCR Cash Inflows sheets plus
    Regulatory Parameters (inflow cap, Level-2 caps, tailoring discount).
    """
    params = workbook.params
    l1, l2a, l2b, hqla_df = _hqla_from_workbook_items(
        workbook.hqla_items,
        level2_cap=params.level2_cap_pct,
        level2b_cap=params.level2b_cap_pct,
    )
    hqla_stock = l1 + l2a + l2b

    outflow_rows = []
    total_outflows = 0.0
    for item in workbook.lcr_outflows:
        rate = item.runoff_rate_pct / 100.0 if item.runoff_rate_pct > 1 else item.runoff_rate_pct
        weighted = item.unweighted_amount_m * rate
        total_outflows += weighted
        outflow_rows.append({
            "Instrument": item.disclosure_item,
            "Category": item.category.replace("_", " ").title() if item.category else "",
            "Notional ($M)": round(item.unweighted_amount_m, 2),
            "30d Outflow ($M)": round(weighted, 2),
            "Runoff Rate (%)": round(rate * 100, 1),
            "Disclosure Row": item.row,
            "Source": "NMD auto" if item.auto_from_nmd else "User input",
        })
    outflow_df = pd.DataFrame(outflow_rows)

    inflow_rows = []
    total_inflows = 0.0
    for item in workbook.lcr_inflows:
        rate = item.inflow_rate_pct / 100.0 if item.inflow_rate_pct > 1 else item.inflow_rate_pct
        weighted = item.unweighted_amount_m * rate
        total_inflows += weighted
        inflow_rows.append({
            "Instrument": item.disclosure_item,
            "Category": "Cash inflow",
            "Notional ($M)": round(item.unweighted_amount_m, 2),
            "30d Inflow ($M)": round(weighted, 2),
            "Inflow Rate (%)": round(rate * 100, 1),
            "Disclosure Row": item.row,
        })
    inflow_df = pd.DataFrame(inflow_rows)

    capped_inflows = min(total_inflows, total_outflows * params.inflow_cap_pct)
    net_pre_adj = max(total_outflows - capped_inflows, 0.0)
    net_outflows = net_pre_adj * (1.0 - params.outflow_adjustment_pct)

    lcr_pct = (hqla_stock / net_outflows * 100) if net_outflows > 0 else float("inf")
    min_pct = params.lcr_minimum_pct

    summary = pd.DataFrame([
        {"Component": "HQLA Level 1", "Value ($M)": round(l1, 2),
         "Notes": "Disclosure rows 2 — Level 1"},
        {"Component": "HQLA Level 2A (capped)", "Value ($M)": round(l2a, 2),
         "Notes": "Disclosure row 3"},
        {"Component": "HQLA Level 2B (capped)", "Value ($M)": round(l2b, 2),
         "Notes": "Disclosure row 4"},
        {"Component": "HQLA Stock (row 29)", "Value ($M)": round(hqla_stock, 2),
         "Notes": "After Level 2 caps"},
        {"Component": "Total Cash Outflows (row 19)", "Value ($M)": round(total_outflows, 2),
         "Notes": "Weighted 30-day outflows"},
        {"Component": "Total Cash Inflows (row 28)", "Value ($M)": round(total_inflows, 2),
         "Notes": "Weighted 30-day inflows"},
        {"Component": f"Capped Inflows ({params.inflow_cap_pct:.0%})", "Value ($M)": round(capped_inflows, 2),
         "Notes": "Max % of outflows"},
        {"Component": "Net Cash Outflows (pre-adjustment, row 30)", "Value ($M)": round(net_pre_adj, 2),
         "Notes": "Outflows − capped inflows"},
        {"Component": f"Tailoring discount ({params.outflow_adjustment_pct:.0%})", "Value ($M)": round(net_outflows, 2),
         "Notes": "Category IV adjusted net outflow (row 34)"},
        {"Component": "LCR (%) (row 35)", "Value ($M)": round(lcr_pct, 2),
         "Notes": f"Minimum {min_pct:.0f}%"},
    ])

    return LcrResult(
        hqla_level1=l1,
        hqla_level2a=l2a,
        hqla_level2b=l2b,
        hqla_level2a_adjusted=l2a,
        hqla_level2b_adjusted=l2b,
        hqla_stock=hqla_stock,
        total_outflows=total_outflows,
        total_inflows=total_inflows,
        capped_inflows=capped_inflows,
        net_cash_outflows=net_outflows,
        lcr_pct=lcr_pct,
        lcr_pass=lcr_pct >= min_pct,
        hqla_breakdown=hqla_df,
        outflow_breakdown=outflow_df,
        inflow_breakdown=inflow_df,
        summary=summary,
    )


def compute_lcr_from_nmd_allocation(
    core_balance: float,
    non_core_balance: float,
    hqla_mb: float,
    stable_pct_lcr: float = DEFAULT_STABLE_PCT,
    outflow_rate_stable: float = OUTFLOW_RETAIL_STABLE,
    outflow_rate_unstable: float = OUTFLOW_RETAIL_LESS_STABLE,
    outflow_rate_noncash: float = OUTFLOW_NON_CORE,
) -> dict:
    """
    Simplified LCR from NMD core/non-core split (nmd_model compatible).

    Used when only NMD allocation weights are available without full balance sheet.
    """
    outflow_stable = core_balance * stable_pct_lcr * outflow_rate_stable
    outflow_unstable = core_balance * (1 - stable_pct_lcr) * outflow_rate_unstable
    outflow_noncash = non_core_balance * outflow_rate_noncash
    total_outflow = outflow_stable + outflow_unstable + outflow_noncash
    lcr_pct = (hqla_mb / total_outflow * 100) if total_outflow > 0 else float("inf")

    return {
        "hqla": hqla_mb,
        "outflow_stable": outflow_stable,
        "outflow_unstable": outflow_unstable,
        "outflow_noncash": outflow_noncash,
        "total_outflow": total_outflow,
        "lcr_pct": lcr_pct,
        "min_liq_dynamic": total_outflow / core_balance if core_balance > 0 else 0.0,
        "lcr_pass": lcr_pct >= 100.0,
    }
