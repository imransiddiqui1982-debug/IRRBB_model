"""
Create comprehensive deposit data Excel template for NMD + LCR + NSFR.

Sheets:
  Instructions       — field guide and workflow
  Customer Deposits  — customer-level input (NMD history + liquidity tags)
  Market Rates       — monthly market / policy rate
  BCBS Segment Caps  — IRRBB NMD caps by segment
  LCR Outflow Rates  — BCBS 30-day runoff reference
  NSFR ASF Rates     — BCBS stable funding reference
  Bifurcation Map    — how NMD outputs feed LCR / NSFR
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
COMPREHENSIVE_PATH = os.path.join(BASE, "comprehensive_deposit_template.xlsx")
LEGACY_PATH = os.path.join(BASE, "nmd_deposit_template.xlsx")

MONTHS = pd.date_range("2022-01-31", periods=36, freq="ME")
MONTH_COLS = [d.strftime("%Y-%m") for d in MONTHS]

META_COLS = [
    "customer_id",
    "segment",
    "deposit_product",
    "currency",
    "start_date",
    "maturity_date",
    "deposit_rate",
    "is_insured",
    "is_operational",
    "lcr_bucket",
    "nsfr_bucket",
    "branch_id",
    "relationship_years",
]


def _customer_path(start_idx: int, start_bal: float, decay: float, noise: float, rng) -> list[float]:
    vals = []
    bal = start_bal
    for i in range(36):
        if i < start_idx:
            vals.append(np.nan)
        else:
            if i > start_idx:
                bal = max(bal * (1 - decay) + rng.normal(0, noise), 0.0)
            vals.append(round(bal, 4))
    return vals


def build_sample_customers(rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    for i in range(1, 9):
        start_idx = int(rng.integers(0, 6))
        path = _customer_path(start_idx, rng.uniform(0.8, 2.5), 0.008, 0.02, rng)
        rows.append({
            "customer_id": f"RT-{i:03d}",
            "segment": "retail_transactional",
            "deposit_product": "demand",
            "currency": "USD",
            "start_date": MONTHS[start_idx].strftime("%Y-%m-%d"),
            "maturity_date": "",
            "deposit_rate": round(float(rng.uniform(0.004, 0.012)), 4),
            "is_insured": "Y",
            "is_operational": "N",
            "lcr_bucket": "auto",
            "nsfr_bucket": "auto",
            "branch_id": f"BR-{rng.integers(1, 5)}",
            "relationship_years": round(float(rng.uniform(1, 15)), 1),
            **{m: v for m, v in zip(MONTH_COLS, path)},
        })

    for i in range(1, 6):
        start_idx = int(rng.integers(0, 10))
        path = _customer_path(start_idx, rng.uniform(1.0, 4.0), 0.015, 0.05, rng)
        rows.append({
            "customer_id": f"RN-{i:03d}",
            "segment": "retail_non_transactional",
            "deposit_product": "savings",
            "currency": "USD",
            "start_date": MONTHS[start_idx].strftime("%Y-%m-%d"),
            "maturity_date": "",
            "deposit_rate": round(float(rng.uniform(0.010, 0.025)), 4),
            "is_insured": "Y",
            "is_operational": "N",
            "lcr_bucket": "auto",
            "nsfr_bucket": "auto",
            "branch_id": f"BR-{rng.integers(1, 5)}",
            "relationship_years": round(float(rng.uniform(2, 20)), 1),
            **{m: v for m, v in zip(MONTH_COLS, path)},
        })

    for i in range(1, 4):
        start_idx = int(rng.integers(0, 8))
        path = _customer_path(start_idx, rng.uniform(5.0, 15.0), 0.025, 0.2, rng)
        operational = "Y" if i == 1 else "N"
        rows.append({
            "customer_id": f"WS-{i:03d}",
            "segment": "wholesale",
            "deposit_product": "demand",
            "currency": "USD",
            "start_date": MONTHS[start_idx].strftime("%Y-%m-%d"),
            "maturity_date": "",
            "deposit_rate": round(float(rng.uniform(0.020, 0.040)), 4),
            "is_insured": "N",
            "is_operational": operational,
            "lcr_bucket": "wholesale_operational" if operational == "Y" else "auto",
            "nsfr_bucket": "auto",
            "branch_id": "HQ",
            "relationship_years": round(float(rng.uniform(3, 12)), 1),
            **{m: v for m, v in zip(MONTH_COLS, path)},
        })

    for i, mat_offset in enumerate([18, 24, 30], start=1):
        start_idx = 0
        bal = round(float(rng.uniform(2.0, 8.0)), 4)
        path = []
        for j in range(36):
            if j < start_idx:
                path.append(np.nan)
            else:
                path.append(bal)
        mat_date = MONTHS[min(mat_offset, 35)].strftime("%Y-%m-%d")
        rows.append({
            "customer_id": f"TD-{i:03d}",
            "segment": "retail_non_transactional",
            "deposit_product": "term",
            "currency": "USD",
            "start_date": MONTHS[0].strftime("%Y-%m-%d"),
            "maturity_date": mat_date,
            "deposit_rate": round(float(rng.uniform(0.025, 0.035)), 4),
            "is_insured": "Y",
            "is_operational": "N",
            "lcr_bucket": "auto",
            "nsfr_bucket": "auto",
            "branch_id": f"BR-{i}",
            "relationship_years": 1.0,
            **{m: v for m, v in zip(MONTH_COLS, path)},
        })

    return pd.DataFrame(rows)


def build_market_rates(rng: np.random.Generator) -> pd.DataFrame:
    rate = 0.045
    rates = []
    for d in MONTHS:
        rate = float(np.clip(rate + rng.normal(0, 0.0015), 0.01, 0.10))
        rates.append({"date": d.strftime("%Y-%m-%d"), "market_rate": round(rate, 5)})
    return pd.DataFrame(rates)


def build_instructions() -> pd.DataFrame:
    return pd.DataFrame([
        ["1", "Workflow", "Fill Customer Deposits + Market Rates, upload in Streamlit sidebar"],
        ["2", "NMD engine", "HP filter + beta regression → core / rate-sensitive / non-core per segment"],
        ["3", "LCR engine", "30-day outflows: core sticky 5%, rate sensitive 10%, non-core 25%"],
        ["4", "NSFR engine", "ASF: stable 95%, less stable 90%, non-core 50%; term by maturity"],
        ["5", "Term deposits", "deposit_product=term uses maturity_date for NSFR tenor (not NMD cohort)"],
        ["", "", ""],
        ["Field", "Required", "Description / allowed values"],
        ["customer_id", "Yes", "Unique account or customer ID"],
        ["segment", "Yes", "retail_transactional | retail_non_transactional | wholesale"],
        ["deposit_product", "Yes", "demand | savings | term | notice | escrow"],
        ["currency", "Yes", "ISO code, e.g. USD"],
        ["start_date", "Yes", "Relationship start date (YYYY-MM-DD)"],
        ["maturity_date", "Term only", "Blank for NMD; required for term/notice (YYYY-MM-DD)"],
        ["deposit_rate", "Yes", "Rate paid to customer as decimal (0.01 = 1%)"],
        ["is_insured", "Yes", "Y/N — deposit insurance (retail typically Y)"],
        ["is_operational", "Wholesale", "Y/N — operational wholesale deposits (LCR 25% vs 40%)"],
        ["lcr_bucket", "Optional", "auto | stable_retail | less_stable_retail | non_core | wholesale_operational | wholesale_non_op"],
        ["nsfr_bucket", "Optional", "auto | stable_retail | less_stable_retail | non_core | short | medium | long"],
        ["branch_id", "Optional", "Branch or cost centre"],
        ["relationship_years", "Optional", "Years of relationship (behavioural context)"],
        ["YYYY-MM columns", "Yes", "36 end-of-month balances in USD millions; blank before start_date"],
    ], columns=["Step", "Topic", "Detail"])


def build_segment_caps() -> pd.DataFrame:
    return pd.DataFrame([
        ["retail_transactional", "90%", "5.0", "5%", "10%", "25%", "95%", "90%", "50%"],
        ["retail_non_transactional", "70%", "4.5", "5%", "10%", "25%", "95%", "90%", "50%"],
        ["wholesale", "50%", "4.0", "5%", "10%", "25%", "95%", "90%", "50%"],
    ], columns=[
        "Segment", "Max Core % (IRRBB)", "Max WAL (Y)",
        "LCR Core Sticky", "LCR Rate Sensitive", "LCR Non-core",
        "NSFR Stable ASF", "NSFR Less Stable ASF", "NSFR Non-core ASF",
    ])


def build_lcr_reference() -> pd.DataFrame:
    return pd.DataFrame([
        ["Core Sticky (NMD output)", "5%", "Stable retail portion after beta split"],
        ["Rate Sensitive (NMD output)", "10%", "Rate-pass-through portion of core"],
        ["Non-Core Volatile (NMD output)", "25%", "HP filter volatile portion"],
        ["Wholesale operational", "25%", "is_operational = Y on wholesale deposits"],
        ["Wholesale non-operational", "40%", "Standard wholesale funding"],
        ["Secured funding", "0%", "Fully secured short-term"],
        ["Term deposit < 30 days", "0%", "Maturing within stress horizon"],
    ], columns=["LCR Category", "30d Runoff Rate", "Notes"])


def build_nsfr_reference() -> pd.DataFrame:
    return pd.DataFrame([
        ["Tier 1 / Tier 2 capital", "100%", "Regulatory capital in ASF"],
        ["Stable retail (core sticky)", "95%", "NMD core sticky portion"],
        ["Less stable retail (rate sensitive)", "90%", "NMD rate-sensitive portion"],
        ["Non-core / volatile wholesale", "50%", "NMD non-core portion"],
        ["Funding ≥ 1 year", "100%", "Term deposits / long wholesale"],
        ["Funding 6M – 1Y", "50%", "Medium-term funding"],
        ["Funding < 6M", "0%", "Short-term funding"],
    ], columns=["NSFR ASF Category", "ASF Factor", "Notes"])


def build_bifurcation_map() -> pd.DataFrame:
    return pd.DataFrame([
        ["Core Sticky", "HP filter stable × (1 − beta)", "5%", "95%", "demand_deposit"],
        ["Rate Sensitive", "HP filter stable × beta", "10%", "90%", "bullet_floating"],
        ["Non-Core Volatile", "HP filter volatile portion", "25%", "50%", "demand_deposit"],
        ["Term Deposits", "deposit_product = term", "By maturity", "By maturity", "bullet_fixed"],
        ["Wholesale Operational", "is_operational = Y", "25%", "50%", "demand_deposit"],
    ], columns=[
        "Output Bucket", "NMD / Input Rule", "LCR Outflow", "NSFR ASF", "IRRBB Instrument Type",
    ])


def style_workbook(path: str) -> None:
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1A3A6B")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    accent_fill = PatternFill("solid", fgColor="EEF0F7")
    thin = Side(style="thin", color="D0D5E8")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=ws.max_column):
            for cell in row:
                cell.border = border
        if sheet_name == "Customer Deposits":
            meta_count = len(META_COLS)
            for col in range(1, meta_count + 1):
                letter = get_column_letter(col)
                for cell in ws[letter][1:]:
                    if cell.row > 1:
                        cell.fill = accent_fill
            ws.freeze_panes = "N2"
        else:
            ws.freeze_panes = "A2"
        for col in range(1, ws.max_column + 1):
            letter = get_column_letter(col)
            max_len = max(len(str(cell.value or "")) for cell in ws[letter])
            ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 22)
    wb.save(path)


def write_templates(customers: pd.DataFrame, market: pd.DataFrame) -> None:
    os.makedirs(BASE, exist_ok=True)
    sheets = {
        "Instructions": build_instructions(),
        "Customer Deposits": customers,
        "Market Rates": market,
        "BCBS Segment Caps": build_segment_caps(),
        "LCR Outflow Rates": build_lcr_reference(),
        "NSFR ASF Rates": build_nsfr_reference(),
        "Bifurcation Map": build_bifurcation_map(),
    }
    for path in (COMPREHENSIVE_PATH, LEGACY_PATH):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            if path == LEGACY_PATH:
                customers.to_excel(writer, sheet_name="Customer Balances", index=False)
                market.to_excel(writer, sheet_name="Market Rates", index=False)
                build_instructions().to_excel(writer, sheet_name="Instructions", index=False)
                build_segment_caps().to_excel(writer, sheet_name="Segment Caps", index=False)
            else:
                for name, df in sheets.items():
                    df.to_excel(writer, sheet_name=name, index=False)
        style_workbook(path)


def main() -> None:
    rng = np.random.default_rng(42)
    customers = build_sample_customers(rng)
    market = build_market_rates(rng)
    write_templates(customers, market)
    print(f"Created: {COMPREHENSIVE_PATH}")
    print(f"Created: {LEGACY_PATH}")


if __name__ == "__main__":
    main()
