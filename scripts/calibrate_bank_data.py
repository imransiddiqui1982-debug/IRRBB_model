"""
Calibrate balance_sheet_template.csv and us_lcr_nsfr_deposit_model.xlsx
to a coherent ~$2.9B asset regional bank with LCR ~115% and NSFR ~125%.

Run: python scripts/calibrate_bank_data.py
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.liquidity_ratios import compute_liquidity_from_workbook  # noqa: E402
from src.liquidity_workbook import load_liquidity_workbook  # noqa: E402
from src.nmd_refinement import load_customer_nmd, refine_nmd_deposits  # noqa: E402

DESKTOP = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

BS_PATH = os.path.join(DESKTOP, "balance_sheet_template.csv")
XLSX_PATH = os.path.join(DESKTOP, "us_lcr_nsfr_deposit_model.xlsx")

TERM_PRODUCTS = {"term", "escrow"}
NMD_TARGET_M = 950.0
TERM_TARGET_M = 550.0
TIER1_M = 500.0
TIER2_M = 100.0


def build_balance_sheet() -> pd.DataFrame:
    """$2.9B assets; deposit funding supplied by NMD workbook."""
    return pd.DataFrame([
        ["Cash & Central Bank Reserves", "asset", 150, 0.5, "bullet_floating", 1 / 12, 12, 1 / 12],
        ["T-Bills (3M)", "asset", 100, 4.9, "bullet_fixed", 0.25, 1, 0.25],
        ["Floating Rate Notes (6M reset)", "asset", 200, 5.8, "bullet_floating", 3.0, 2, 0.5],
        ["Fixed Gov Bonds (2Y)", "asset", 200, 4.1, "bullet_fixed", 2.0, 2, 2.0],
        ["Floating Corp Loans (3Y, qtrly reset)", "asset", 600, 6.5, "bullet_floating", 3.0, 4, 0.25],
        ["Fixed Retail Loans (4Y, amortising)", "asset", 250, 5.8, "amortising", 4.0, 12, 4.0],
        ["Fixed Gov Bonds (5Y)", "asset", 300, 3.95, "bullet_fixed", 5.0, 2, 5.0],
        ["Fixed-Rate Mortgages (10Y, amortising)", "asset", 800, 5.2, "amortising", 10.0, 12, 10.0],
        ["Fixed Gov Bonds (15Y)", "asset", 200, 3.5, "bullet_fixed", 15.0, 2, 15.0],
        ["Fixed Infrastructure Bonds (20Y)", "asset", 100, 4.2, "bullet_fixed", 20.0, 2, 20.0],
        ["Overnight Repo", "liability", 100, 5.1, "bullet_floating", 1 / 365, 365, 1 / 365],
        ["Floating Senior Debt (3Y, semi reset)", "liability", 300, 5.8, "bullet_floating", 3.0, 2, 0.5],
        ["Fixed-Rate Covered Bonds (5Y)", "liability", 300, 4.3, "bullet_fixed", 5.0, 2, 5.0],
        ["Fixed-Rate Borrowings (7Y)", "liability", 200, 4.5, "bullet_fixed", 7.0, 2, 7.0],
        ["Subordinated Debt (10Y)", "liability", 200, 5.0, "bullet_fixed", 10.0, 2, 10.0],
        ["Tier 2 Capital Notes (15Y)", "liability", 100, 5.5, "bullet_fixed", 15.0, 2, 15.0],
    ], columns=[
        "name", "side", "notional", "coupon_pct", "instrument_type",
        "maturity_years", "payment_freq", "repricing_years",
    ])


def _month_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if str(c).startswith("20")], key=str)


def scale_customer_deposits(customers: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    month_cols = _month_cols(customers)
    if not month_cols:
        return customers

    is_term = customers["deposit_product"].astype(str).str.lower().isin(TERM_PRODUCTS)
    nmd = customers[~is_term]
    term = customers[is_term]

    nmd_latest = float(nmd[month_cols[-1]].sum(skipna=True))
    term_latest = float(term[month_cols[-1]].sum(skipna=True))

    nmd_factor = NMD_TARGET_M / nmd_latest if nmd_latest > 0 else 1.0
    term_factor = TERM_TARGET_M / term_latest if term_latest > 0 else 1.0

    for col in month_cols:
        customers.loc[~is_term, col] = (
            pd.to_numeric(customers.loc[~is_term, col], errors="coerce") * nmd_factor
        ).round(4)
        customers.loc[is_term, col] = (
            pd.to_numeric(customers.loc[is_term, col], errors="coerce") * term_factor
        ).round(4)

    return customers


def build_regulatory_params() -> pd.DataFrame:
    return pd.DataFrame([
        ["tier1_capital_m", TIER1_M, "Tier 1 capital — matches Streamlit sidebar default"],
        ["tier2_capital_m", TIER2_M, "Tier 2 from balance sheet"],
        ["outflow_adjustment_pct", 30, "Category IV tailoring discount on net outflows"],
        ["rsf_adjustment_pct", 30, "Category IV tailoring discount on RSF"],
        ["lcr_minimum_pct", 100, "Standard minimum LCR (%)"],
        ["nsfr_minimum_pct", 100, "Regulatory minimum NSFR (%)"],
        ["inflow_cap_pct", 75, "Max inflows as % of gross outflows"],
        ["level2_cap_pct", 40, "Level 2A+2B cap as % of total HQLA"],
        ["level2b_cap_pct", 15, "Level 2B cap as % of total HQLA"],
        ["auto_nmd_deposits", "Y", "Auto-fill deposit rows from NMD bifurcation"],
    ], columns=["parameter", "value", "notes"])


def build_hqla_stock() -> pd.DataFrame:
    return pd.DataFrame([
        [2, "Cash on deposit with central banks", "level_1", 150, 0, 0, "Matches BS Cash & Reserves"],
        [2, "U.S. Treasury securities & T-Bills", "level_1", 180, 0, 0, "Matches BS T-Bills + Treasuries"],
        [2, "GNMA / agency MBS (Level 1)", "level_1", 60, 0, 0, "Agency pass-through"],
        [3, "U.S. GSE debt securities", "level_2a", 100, 15, 0, "Level 2A"],
        [4, "Investment grade corporate debt", "level_2b", 35, 50, 0, "Level 2B cap bucket"],
    ], columns=[
        "row", "disclosure_item", "hqla_level", "unweighted_amount_m",
        "haircut_pct", "encumbered_amount_m", "notes",
    ])


def build_lcr_outflows() -> pd.DataFrame:
    return pd.DataFrame([
        [6, "Stable retail deposit outflow", "stable_retail", 0, 5, "Y", "core_sticky", "NMD auto"],
        [7, "Other retail funding", "less_stable_retail", 0, 10, "Y", "rate_sensitive", "NMD auto"],
        [7, "Non-core volatile retail deposits", "non_core_retail", 0, 25, "Y", "non_core_volatile", "NMD auto"],
        [8, "Brokered deposit outflow", "brokered", 80, 10, "N", "", "Brokered retail"],
        [10, "Operational deposit outflow", "operational", 0, 25, "Y", "wholesale_operational", "NMD auto"],
        [11, "Non-operational wholesale funding", "wholesale_non_op", 0, 40, "Y", "wholesale_non_op", "NMD auto"],
        [12, "Unsecured debt outflow", "unsecured_debt", 200, 100, "N", "", "Maturing wholesale debt"],
        [13, "Secured wholesale funding outflow", "secured", 100, 0, "N", "", "Overnight repo"],
        [15, "Derivative collateral and margin outflow", "derivatives", 45, 100, "N", "", "Net 30d derivative outflow"],
        [16, "Credit and liquidity facilities", "facilities", 250, 40, "N", "", "Undrawn commitments"],
        [17, "Other contractual funding obligations", "other_contractual", 40, 100, "N", "", ""],
        [18, "Other contingent funding obligations", "contingent", 60, 5, "N", "", ""],
    ], columns=[
        "row", "disclosure_item", "category", "unweighted_amount_m", "runoff_rate_pct",
        "auto_from_nmd", "nmd_bucket", "notes",
    ])


def build_lcr_inflows() -> pd.DataFrame:
    return pd.DataFrame([
        [20, "Secured lending and asset exchange inflow", 35, 100, "Reverse repos / securities lending"],
        [21, "Retail cash inflow", 25, 50, "Maturing retail loans ≤30d"],
        [22, "Unsecured wholesale cash inflow", 30, 50, "Wholesale maturities"],
        [24, "Net derivative cash inflow", 25, 100, "Net derivative inflows"],
        [25, "Securities cash inflow", 20, 100, "Non-HQLA securities maturing ≤30d"],
    ], columns=["row", "disclosure_item", "unweighted_amount_m", "inflow_rate_pct", "notes"])


def build_nsfr_asf() -> pd.DataFrame:
    return pd.DataFrame([
        [2, "NSFR regulatory capital elements", "perpetual", 0, 100, "Y", "regulatory_capital", "Tier 1 + Tier 2"],
        [3, "Other capital elements and securities", "gte_1y", 250, 100, "N", "", "Sub debt + covered bonds ≥1Y"],
        [5, "Stable retail deposits", "open", 0, 95, "Y", "core_sticky", "NMD auto"],
        [6, "Less stable retail deposits", "open", 0, 90, "Y", "rate_sensitive", "NMD auto"],
        [6, "Non-core volatile retail", "open", 0, 50, "Y", "non_core_volatile", "NMD auto"],
        [10, "Operational deposits", "open", 0, 50, "Y", "wholesale_operational", "NMD auto"],
        [11, "Other wholesale funding < 6 months", "lt_6m", 100, 0, "N", "", "Overnight repo / CP"],
        [11, "Other wholesale funding 6M–1Y", "m6_1y", 200, 50, "N", "", "Short wholesale"],
        [11, "Other wholesale funding ≥ 1 year", "gte_1y", 350, 100, "N", "", "Senior / covered / borrowings"],
    ], columns=[
        "row", "disclosure_item", "maturity_bucket", "unweighted_amount_m", "asf_factor_pct",
        "auto_from_nmd", "nmd_bucket", "notes",
    ])


def build_nsfr_rsf() -> pd.DataFrame:
    return pd.DataFrame([
        [17, "Level 1 liquid assets", 0, 5, "Y", "level_1", "From HQLA Stock"],
        [18, "Level 2A liquid assets", 0, 15, "Y", "level_2a", ""],
        [19, "Level 2B liquid assets", 0, 50, "Y", "level_2b", ""],
        [20, "Zero percent RSF assets (excl. L1)", 50, 0, "N", "", "Settlement balances"],
        [22, "Loans to retail customers", 1150, 65, "N", "", "Retail loans + mortgages (BS)"],
        [23, "Loans to wholesale customers", 950, 85, "N", "", "Corp floating + FRN (BS)"],
        [24, "Securities (non-HQLA)", 850, 50, "N", "", "Gov bonds FRN etc. (BS)"],
        [35, "All other assets", 150, 100, "N", "", "Infrastructure / other (BS)"],
        [36, "Undrawn credit and liquidity facilities", 750, 5, "N", "", "Off-balance commitments"],
    ], columns=[
        "row", "disclosure_item", "unweighted_amount_m", "rsf_factor_pct",
        "link_hqla", "hqla_level", "notes",
    ])


def calibrate_xlsx(path: str) -> dict:
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheets = {name: pd.read_excel(path, sheet_name=name, engine="openpyxl")
              for name in xl.sheet_names}

    sheets["Customer Deposits"] = scale_customer_deposits(sheets["Customer Deposits"])
    sheets["Regulatory Parameters"] = build_regulatory_params()
    sheets["HQLA Stock"] = build_hqla_stock()
    sheets["LCR Cash Outflows"] = build_lcr_outflows()
    sheets["LCR Cash Inflows"] = build_lcr_inflows()
    sheets["NSFR ASF"] = build_nsfr_asf()
    sheets["NSFR RSF"] = build_nsfr_rsf()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    with open(path, "rb") as f:
        data = f.read()
    nmd = refine_nmd_deposits(io.BytesIO(data))
    cust = load_customer_nmd(io.BytesIO(data)).customers
    wb = load_liquidity_workbook(io.BytesIO(data), nmd_result=nmd, customers=cust)
    res = compute_liquidity_from_workbook(wb)

    month_cols = _month_cols(cust)
    latest = month_cols[-1] if month_cols else None
    is_term = cust["deposit_product"].astype(str).str.lower().isin(TERM_PRODUCTS)
    nmd_bal = float(cust.loc[~is_term, latest].sum(skipna=True)) if latest else 0
    term_bal = float(cust.loc[is_term, latest].sum(skipna=True)) if latest else 0

    return {
        "nmd_balance_m": round(nmd_bal, 1),
        "term_balance_m": round(term_bal, 1),
        "lcr_pct": round(res.lcr.lcr_pct, 1),
        "nsfr_pct": round(res.nsfr.nsfr_pct, 1),
        "hqla_m": round(res.lcr.hqla_stock, 1),
        "net_outflows_m": round(res.lcr.net_cash_outflows, 1),
        "asf_m": round(res.nsfr.asf_total, 1),
        "rsf_m": round(res.nsfr.rsf_total, 1),
    }


def main() -> None:
    os.makedirs(DATA, exist_ok=True)

    bs = build_balance_sheet()
    bs.to_csv(BS_PATH, index=False)
    bs.to_csv(os.path.join(DATA, "balance_sheet_template.csv"), index=False)
    print(f"Balance sheet: assets ${bs[bs.side=='asset'].notional.sum():,.0f}M, "
          f"non-deposit liabilities ${bs[bs.side=='liability'].notional.sum():,.0f}M")

    if not os.path.exists(XLSX_PATH):
        alt = os.path.join(DATA, "comprehensive_deposit_template_v2.xlsx")
        if os.path.exists(alt):
            import shutil
            shutil.copy2(alt, XLSX_PATH)
        else:
            raise FileNotFoundError(f"Deposit model not found: {XLSX_PATH}")

    stats = calibrate_xlsx(XLSX_PATH)
    import shutil
    shutil.copy2(XLSX_PATH, os.path.join(DATA, "us_lcr_nsfr_deposit_model.xlsx"))
    shutil.copy2(BS_PATH, os.path.join(DATA, "balance_sheet_template.csv"))

    print(f"Deposits: NMD ${stats['nmd_balance_m']:,.0f}M + term ${stats['term_balance_m']:,.0f}M")
    print(f"LCR: {stats['lcr_pct']}% (HQLA ${stats['hqla_m']:,.0f}M / net outflows ${stats['net_outflows_m']:,.0f}M)")
    print(f"NSFR: {stats['nsfr_pct']}% (ASF ${stats['asf_m']:,.0f}M / RSF ${stats['rsf_m']:,.0f}M)")
    print(f"Updated: {BS_PATH}")
    print(f"Updated: {XLSX_PATH}")


if __name__ == "__main__":
    main()
