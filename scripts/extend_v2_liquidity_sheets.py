"""
Extend comprehensive_deposit_template_v2.xlsx with U.S. LCR / NSFR disclosure input sheets.

Aligned with 12 CFR 249.91 (LCR) and 249.131 (NSFR) — same structure as HSBC NA public disclosures.
Run: python scripts/extend_v2_liquidity_sheets.py
"""

from __future__ import annotations

import os
import shutil

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
V2_PATH = os.path.join(BASE, "comprehensive_deposit_template_v2.xlsx")
DESKTOP_V2 = os.path.join(
    os.path.expanduser("~"), "OneDrive", "Desktop", "comprehensive_deposit_template_v2.xlsx"
)


def build_regulatory_params() -> pd.DataFrame:
    return pd.DataFrame([
        ["tier1_capital_m", 8500, "Tier 1 capital included in NSFR ASF row 2 ($M)"],
        ["tier2_capital_m", 1200, "Tier 2 / subordinated debt ($M)"],
        ["outflow_adjustment_pct", 30, "Category IV tailoring — % reduction of net outflows (row 33-34)"],
        ["rsf_adjustment_pct", 30, "Category IV tailoring — % reduction of RSF (row 38-39)"],
        ["lcr_minimum_pct", 70, "Category IV minimum LCR (%); use 100 for full LCR banks"],
        ["nsfr_minimum_pct", 100, "Regulatory minimum NSFR (%)"],
        ["inflow_cap_pct", 75, "Max inflows as % of gross outflows"],
        ["level2_cap_pct", 40, "Level 2A+2B cap as % of total HQLA"],
        ["level2b_cap_pct", 15, "Level 2B cap as % of total HQLA"],
        ["auto_nmd_deposits", "Y", "Auto-fill deposit rows from NMD bifurcation"],
    ], columns=["parameter", "value", "notes"])


def build_hqla_stock() -> pd.DataFrame:
    return pd.DataFrame([
        [2, "Cash on deposit with central banks", "level_1", 12500, 0, 0, "Fed reserves / nostro"],
        [2, "U.S. Treasury securities", "level_1", 18000, 0, 500, "Unencumbered Treasuries"],
        [2, "GNMA / agency MBS (Level 1)", "level_1", 4200, 0, 0, "GNMA pass-through"],
        [3, "U.S. GSE debt securities", "level_2a", 6500, 15, 0, "Fannie/Freddie debt"],
        [3, "Covered bonds", "level_2a", 800, 15, 0, ""],
        [4, "Investment grade corporate debt", "level_2b", 2200, 50, 0, "Non-financial IG"],
        [4, "Listed equity (RSF cap bucket)", "level_2b", 450, 50, 0, "Exchange-traded equities"],
    ], columns=[
        "row", "disclosure_item", "hqla_level", "unweighted_amount_m",
        "haircut_pct", "encumbered_amount_m", "notes",
    ])


def build_lcr_outflows() -> pd.DataFrame:
    return pd.DataFrame([
        [6, "Stable retail deposit outflow", "stable_retail", 0, 5, "Y", "core_sticky",
         "NMD core sticky → row 6; leave amount blank when auto=Y"],
        [7, "Other retail funding", "less_stable_retail", 0, 10, "Y", "rate_sensitive",
         "NMD rate-sensitive retail"],
        [7, "Non-core volatile retail deposits", "non_core_retail", 0, 25, "Y", "non_core_volatile",
         "NMD volatile retail portion"],
        [8, "Brokered deposit outflow", "brokered", 350, 10, "N", "", "Fully brokered retail"],
        [10, "Operational deposit outflow", "operational", 0, 25, "Y", "wholesale_operational",
         "Wholesale is_operational=Y balances"],
        [11, "Non-operational wholesale funding", "wholesale_non_op", 0, 40, "Y", "wholesale_non_op",
         "Remaining wholesale funding"],
        [12, "Unsecured debt outflow", "unsecured_debt", 2800, 100, "N", "", "Maturing wholesale debt"],
        [13, "Secured wholesale funding outflow", "secured", 5200, 0, "N", "", "Repos / secured funding"],
        [15, "Derivative collateral and margin outflow", "derivatives", 4100, 100, "N", "",
         "Net 30d derivative outflow (pre-weighted amount × rate)"],
        [16, "Credit and liquidity facilities", "facilities", 1900, 40, "N", "", "Undrawn commitments"],
        [17, "Other contractual funding obligations", "other_contractual", 600, 100, "N", "", ""],
        [18, "Other contingent funding obligations", "contingent", 250, 5, "N", "", ""],
    ], columns=[
        "row", "disclosure_item", "category", "unweighted_amount_m", "runoff_rate_pct",
        "auto_from_nmd", "nmd_bucket", "notes",
    ])


def build_lcr_inflows() -> pd.DataFrame:
    return pd.DataFrame([
        [20, "Secured lending and asset exchange inflow", 1800, 100, "Reverse repos / securities lending"],
        [21, "Retail cash inflow", 120, 50, "Maturing retail loans"],
        [22, "Unsecured wholesale cash inflow", 950, 50, "Wholesale maturities"],
        [24, "Net derivative cash inflow", 3200, 100, "Net derivative inflows (can offset row 15)"],
        [25, "Securities cash inflow", 800, 100, "Non-HQLA securities maturing ≤30d"],
        [27, "Other cash inflow", 150, 50, ""],
    ], columns=["row", "disclosure_item", "unweighted_amount_m", "inflow_rate_pct", "notes"])


def build_nsfr_asf() -> pd.DataFrame:
    return pd.DataFrame([
        [2, "NSFR regulatory capital elements", "perpetual", 0, 100, "Y", "regulatory_capital",
         "Tier 1 + Tier 2 from Regulatory Parameters"],
        [3, "Other capital elements and securities", "gte_1y", 1500, 100, "N", "",
         "Long-term debt ≥1Y"],
        [5, "Stable retail deposits", "open", 0, 95, "Y", "core_sticky", "NMD core sticky"],
        [6, "Less stable retail deposits", "open", 0, 90, "Y", "rate_sensitive", "NMD rate sensitive"],
        [6, "Non-core volatile retail", "open", 0, 50, "Y", "non_core_volatile", "NMD volatile retail"],
        [10, "Operational deposits", "open", 0, 50, "Y", "wholesale_operational", ""],
        [11, "Other wholesale funding < 1 year", "lt_6m", 4200, 0, "N", "", "Short wholesale CP"],
        [11, "Other wholesale funding 6M–1Y", "m6_1y", 3100, 50, "N", "", ""],
        [11, "Other wholesale funding ≥ 1 year", "gte_1y", 8500, 100, "N", "", "Senior / covered bonds"],
    ], columns=[
        "row", "disclosure_item", "maturity_bucket", "unweighted_amount_m", "asf_factor_pct",
        "auto_from_nmd", "nmd_bucket", "notes",
    ])


def build_nsfr_rsf() -> pd.DataFrame:
    return pd.DataFrame([
        [17, "Level 1 liquid assets", 0, 5, "Y", "level_1", "Auto-linked to HQLA Stock sheet"],
        [18, "Level 2A liquid assets", 0, 15, "Y", "level_2a", ""],
        [19, "Level 2B liquid assets", 0, 50, "Y", "level_2b", ""],
        [20, "Zero percent RSF assets (excl. L1)", 800, 0, "N", "", "Settlement balances"],
        [22, "Loans to retail customers", 28500, 65, "N", "", "Residential / consumer"],
        [23, "Loans to wholesale customers", 35200, 85, "N", "", "Corporate / commercial"],
        [24, "Securities (non-HQLA)", 4200, 50, "N", "", "Non-HQLA debt / equity"],
        [35, "All other assets", 6800, 100, "N", "", "Fixed assets, intangibles, other"],
        [36, "Undrawn credit and liquidity facilities", 9500, 5, "N", "", "Off-balance-sheet commitments"],
    ], columns=[
        "row", "disclosure_item", "unweighted_amount_m", "rsf_factor_pct",
        "link_hqla", "hqla_level", "notes",
    ])


def build_lcr_disclosure_summary() -> pd.DataFrame:
    return pd.DataFrame([
        [1, "Total eligible HQLA", "Calculated", "Sum rows 2-4 after caps"],
        [19, "Total cash outflows", "Calculated", "Sum weighted outflows rows 5-18"],
        [28, "Total cash inflows", "Calculated", "Sum weighted inflows rows 20-27"],
        [29, "HQLA amount", "Calculated", "Eligible HQLA stock"],
        [30, "Net cash outflows (excl. mismatch add-on)", "Calculated", "Row 19 − capped row 28"],
        [34, "Total adjusted net cash outflows", "Calculated", "After tailoring discount"],
        [35, "Liquidity Coverage Ratio (%)", "Calculated", "Row 29 / Row 34"],
    ], columns=["row", "disclosure_item", "source", "notes"])


def build_nsfr_disclosure_summary() -> pd.DataFrame:
    return pd.DataFrame([
        [1, "Total ASF", "Calculated", "Sum weighted ASF rows 1-15"],
        [16, "Total HQLA (RSF)", "Calculated", "Sum rows 17-19"],
        [37, "Total RSF", "Calculated", "Sum weighted RSF rows 16-36"],
        [39, "Total adjusted RSF", "Calculated", "After tailoring discount"],
        [40, "Net Stable Funding Ratio (%)", "Calculated", "Row 1 / Row 39"],
    ], columns=["row", "disclosure_item", "source", "notes"])


LIQUIDITY_SHEETS = {
    "Regulatory Parameters": build_regulatory_params(),
    "HQLA Stock": build_hqla_stock(),
    "LCR Cash Outflows": build_lcr_outflows(),
    "LCR Cash Inflows": build_lcr_inflows(),
    "NSFR ASF": build_nsfr_asf(),
    "NSFR RSF": build_nsfr_rsf(),
    "LCR Disclosure Summary": build_lcr_disclosure_summary(),
    "NSFR Disclosure Summary": build_nsfr_disclosure_summary(),
}


def style_sheet(ws) -> None:
    header_fill = PatternFill("solid", fgColor="1A3A6B")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    accent_fill = PatternFill("solid", fgColor="EEF0F7")
    thin = Side(style="thin", color="D0D5E8")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            cell.border = border
            if cell.column <= 3:
                cell.fill = accent_fill
    ws.freeze_panes = "A2"
    for col in range(1, ws.max_column + 1):
        letter = get_column_letter(col)
        max_len = max(len(str(cell.value or "")) for cell in ws[letter])
        ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 36)


def extend_workbook(path: str) -> None:
    existing = pd.ExcelFile(path, engine="openpyxl")
    all_sheets = {name: pd.read_excel(path, sheet_name=name, engine="openpyxl")
                  for name in existing.sheet_names}

    for name, df in LIQUIDITY_SHEETS.items():
        all_sheets[name] = df

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    wb = load_workbook(path)
    for name in LIQUIDITY_SHEETS:
        if name in wb.sheetnames:
            style_sheet(wb[name])
    wb.save(path)
    print(f"Extended: {path}")


def main() -> None:
    os.makedirs(BASE, exist_ok=True)
    src = V2_PATH if os.path.exists(V2_PATH) else DESKTOP_V2
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"v2 template not found at {V2_PATH} or {DESKTOP_V2}. "
            "Copy comprehensive_deposit_template_v2.xlsx to data/ first."
        )
    if src != V2_PATH:
        shutil.copy2(src, V2_PATH)
    extend_workbook(V2_PATH)
    if os.path.exists(DESKTOP_V2) and os.path.abspath(DESKTOP_V2) != os.path.abspath(V2_PATH):
        shutil.copy2(V2_PATH, DESKTOP_V2)
        print(f"Copied to Desktop: {DESKTOP_V2}")


if __name__ == "__main__":
    main()
