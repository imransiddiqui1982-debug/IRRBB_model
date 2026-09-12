"""
ui_helpers.py
-------------
Streamlit UX helpers: templates, validation, heatmaps, export packs, what-if.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from .load_balance_sheet import REQUIRED_COLUMNS, VALID_TYPES, load_instruments_from_csv
from .scenarios import REF_LABELS, REF_TENORS, Scenario

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass(frozen=True)
class TemplatePack:
    key: str
    label: str
    description: str
    bs_path: Path
    nmd_path: Path | None


def available_packs() -> list[TemplatePack]:
    bs_default = DATA / "balance_sheet_template.csv"
    if not bs_default.exists():
        bs_default = DATA / "sample_balance_sheet.csv"
    nmd = DATA / "us_lcr_nsfr_deposit_model.xlsx"
    if not nmd.exists():
        nmd = DATA / "comprehensive_deposit_template_v2.xlsx"
    nmd_p = nmd if nmd.exists() else None

    packs = [
        TemplatePack(
            "commercial",
            "Commercial / mixed bank",
            "Diversified assets & liabilities — default BCBS demo book.",
            bs_default,
            nmd_p,
        ),
    ]
    mbs = DATA / "balance_sheet_template_mbs30.csv"
    packs.append(
        TemplatePack(
            "mortgage",
            "Mortgage / MBS heavy",
            "Whole loans + agency MBS for OA prepay / NII / KR01 stress.",
            mbs if mbs.exists() else bs_default,
            nmd_p,
        ),
    )
    packs.append(
        TemplatePack(
            "wholesale",
            "Wholesale / rate-sensitive",
            "Same base template — filter floaters / wholesale in the editor.",
            bs_default,
            nmd_p,
        ),
    )
    return packs


def validate_balance_sheet_bytes(csv_bytes: bytes) -> tuple[pd.DataFrame | None, list[dict]]:
    """Return (dataframe, issues). issues: severity, column, message."""
    issues: list[dict] = []
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        return None, [{"severity": "error", "column": "", "message": f"Cannot parse CSV: {exc}"}]

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for c in missing:
        issues.append({"severity": "error", "column": c, "message": f"Required column missing: {c}"})
    if missing:
        return df, issues

    if df.empty:
        issues.append({"severity": "error", "column": "", "message": "CSV has no data rows"})
        return df, issues

    for col in ("notional", "coupon_pct", "maturity_years"):
        bad = df[col].isna() | ~pd.to_numeric(df[col], errors="coerce").notna()
        if bad.any():
            issues.append({
                "severity": "error",
                "column": col,
                "message": f"{int(bad.sum())} row(s) have invalid {col}",
            })

    sides = set(df["side"].astype(str).str.lower().unique())
    bad_sides = sides - {"asset", "liability"}
    if bad_sides:
        issues.append({
            "severity": "error",
            "column": "side",
            "message": f"Invalid side values: {sorted(bad_sides)}",
        })

    types = set(df["instrument_type"].astype(str).str.lower().unique())
    bad_types = types - VALID_TYPES
    if bad_types:
        issues.append({
            "severity": "error",
            "column": "instrument_type",
            "message": f"Invalid instrument_type: {sorted(bad_types)}",
        })

    if "asset" not in set(df["side"].astype(str).str.lower()):
        issues.append({"severity": "error", "column": "side", "message": "Need at least one asset"})
    if "liability" not in set(df["side"].astype(str).str.lower()):
        issues.append({"severity": "error", "column": "side", "message": "Need at least one liability"})

    if "wac" in df.columns:
        mbs_mask = df["instrument_type"].astype(str).str.lower().isin(["mbs", "whole_loan"])
        missing_wac = mbs_mask & df["wac"].isna()
        if missing_wac.any():
            issues.append({
                "severity": "warning",
                "column": "wac",
                "message": f"{int(missing_wac.sum())} MBS/whole_loan row(s) missing wac (coupon used)",
            })

    try:
        load_instruments_from_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        issues.append({"severity": "error", "column": "", "message": str(exc)})

    return df, issues


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def scenario_heatmap_frame(results) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Scenario": r.scenario.name,
            "ΔNII ($M)": round(r.delta_nii, 2),
            "ΔEVE ($M)": round(r.delta_eve, 2),
            "|ΔEVE|/T1 %": round(r.delta_eve_pct, 2),
            "Status": r.status,
        })
    return pd.DataFrame(rows)


def exception_rows(results) -> pd.DataFrame:
    rows = []
    for r in results:
        flag = "OUTLIER" if r.is_outlier else ("WATCH" if r.is_watch else "OK")
        rows.append({
            "Scenario": r.scenario.name,
            "Flag": flag,
            "ΔEVE ($M)": round(r.delta_eve, 2),
            "|ΔEVE|/T1 %": round(r.delta_eve_pct, 2),
            "ΔNII ($M)": round(r.delta_nii, 2),
        })
    out = pd.DataFrame(rows)
    # Prefer breaches / watches first
    order = {"OUTLIER": 0, "WATCH": 1, "OK": 2}
    out["_o"] = out["Flag"].map(order)
    return out.sort_values(["_o", "|ΔEVE|/T1 %"], ascending=[True, False]).drop(columns=["_o"])


def custom_scenario(name: str, shocks_bp: Sequence[float]) -> Scenario:
    shocks = [int(round(float(x))) for x in shocks_bp]
    if len(shocks) != len(REF_TENORS):
        raise ValueError(f"Need {len(REF_TENORS)} pillar shocks, got {len(shocks)}")
    return Scenario(
        id="CUSTOM",
        name=name or "Custom",
        description="User-defined pillar shocks",
        ref_shocks_bp=shocks,
    )


def style_delta_columns(df: pd.DataFrame, cols: list[str]):
    def _color(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return ""
        if x > 0:
            return "color: #1a7a4a"
        if x < 0:
            return "color: #c0392b"
        return ""

    sty = df.style
    for c in cols:
        if c in df.columns:
            sty = sty.map(_color, subset=[c])
    return sty


def build_export_zip(
    *,
    instruments_df: pd.DataFrame,
    results_df: pd.DataFrame,
    curve_df: pd.DataFrame | None,
    kr01_df: pd.DataFrame | None,
    nii_df: pd.DataFrame | None,
    meta: dict,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("instruments.csv", instruments_df.to_csv(index=False))
        zf.writestr("eve_nii_scenarios.csv", results_df.to_csv(index=False))
        if curve_df is not None and not curve_df.empty:
            zf.writestr("yield_curve.csv", curve_df.to_csv(index=False))
        if kr01_df is not None and not kr01_df.empty:
            zf.writestr("kr01.csv", kr01_df.to_csv(index=False))
        if nii_df is not None and not nii_df.empty:
            zf.writestr("nii_sensitivity.csv", nii_df.to_csv(index=False))
        meta = dict(meta)
        meta["exported_at"] = datetime.now(timezone.utc).isoformat()
        zf.writestr("run_meta.json", json.dumps(meta, indent=2, default=str))
        lines = [
            "IRRBB ALCO one-pager",
            f"Exported: {meta['exported_at']}",
            f"Tier 1: {meta.get('tier1')}",
            f"Curve: {meta.get('curve_source')}",
            f"PMMS: {meta.get('pmms')}",
            f"Outliers: {meta.get('outliers')}",
            f"Worst |ΔEVE|/T1: {meta.get('worst_eve_pct')}",
            "",
            "Scenario results:",
        ]
        for _, row in results_df.iterrows():
            lines.append(
                f"  {row.get('Scenario', '')}: ΔEVE={row.get('ΔEVE ($M)', '')} "
                f"ΔNII={row.get('ΔNII ($M)', '')} status={row.get('Status', '')}"
            )
        zf.writestr("alco_one_pager.txt", "\n".join(lines))
    return buf.getvalue()


def apply_whatif_wac(csv_bytes: bytes, instrument_name: str, wac_delta_bp: float) -> bytes:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    mask = df["name"].astype(str) == str(instrument_name)
    if not mask.any():
        return csv_bytes
    delta = float(wac_delta_bp) / 10_000.0
    if "wac" in df.columns:
        w = pd.to_numeric(df.loc[mask, "wac"], errors="coerce")

        def _bump(x):
            if pd.isna(x):
                return x
            if float(x) > 1.0:
                return float(x) + wac_delta_bp / 100.0
            return float(x) + delta

        df.loc[mask, "wac"] = w.map(_bump)
    coup = pd.to_numeric(df.loc[mask, "coupon_pct"], errors="coerce")
    df.loc[mask, "coupon_pct"] = coup + (wac_delta_bp / 100.0)
    return dataframe_to_csv_bytes(df)


def filter_instruments_df(
    df: pd.DataFrame,
    *,
    side: str = "All",
    itype: str = "All",
    name_query: str = "",
) -> pd.DataFrame:
    out = df.copy()
    if side and side != "All" and "side" in out.columns:
        out = out[out["side"].astype(str).str.lower() == side.lower()]
    if itype and itype != "All" and "instrument_type" in out.columns:
        out = out[out["instrument_type"].astype(str).str.lower() == itype.lower()]
    if name_query and "name" in out.columns:
        out = out[out["name"].astype(str).str.contains(name_query, case=False, na=False)]
    return out


DOCS_MARKDOWN = """
## How to use this model

1. **Choose a template pack** (or upload your own CSV + LCR/NSFR workbook).
2. **Validate** the balance sheet — fix errors in the editor on the **Inputs** tab.
3. Set **assumptions** in the sidebar (Tier 1, curve, NII horizon, shocks).
4. Click **Apply edits & Run** after changing the grid.
5. Read the **Run status** strip, then drill into tabs. Enable **Expert mode** for full tables.
6. **Export pack** downloads instruments, EVE/NII, curve, KR01, and an ALCO one-pager.

### EVE vs NII
- **EVE**: discount cash flows (bucket midpoints for most instruments; exact monthly dates + OAS for OA MBS/loans).
- **NII**: accrue interest over 12/24 months, constant balance sheet, **no discounting**. Deposits reprice 1:1 with the shock.

### Inputs that matter most
| Input | Affects |
|-------|---------|
| Live SOFR/IRS curve | Discounting + mortgage anchor |
| PMMS (FRED) | Step A mortgage rate level |
| WAC / coupon | Income & prepay incentive |
| OAS | MBS/loan PV only |
| Reset / maturity | Floater NII & gap |
| Tier 1 | Outlier / watch % |

### BCBS 19 buckets
Cash flows are generated at payment dates, then slotted into the 19 Annex 2 buckets for gap reporting and non-OA EVE discounting.

## MBS / mortgage prepay — how the model works

Live EVE / KR01 for each option-adjusted MBS or whole loan follows **Steps A → B → C** in `mbs_pricing`. The path re-runs on every curve (including KR01 bumps). There is no portfolio sidebar CPR/PSA override.

### Step A — Mortgage rate & refi incentive
- Live **FRED PMMS 30Y** anchors the mortgage-rate *level*: each OA pool’s `spread_to_curve` is set so curve(anchor) + spread ≈ PMMS on the base curve. BCBS / KR01 bumps then move the rate 1:1 with the anchor tenor.
- Refi incentive (pp) = (WAC − mortgage rate) × 100 (WAC and rates as decimals; result in **percentage points**).
- Positive = in-the-money to refinance; negative = lock-in.
- Fallback: balance-sheet `spread_to_curve` if PMMS fetch fails.

### Step B — CPR (logistic × seasoning)
- Logistic refi response (parameters from `data/calibrated_prepayment_params.json` when present — fit via `python -m src.calibrate_prepayment` on loan-level history; otherwise illustrative defaults):
  - refi_response = max_refi_cpr / (1 + e^(−k × (incentive_pp − midpoint)))
- **Seasoning ramp** (calibrated `seasoning_ramp_months`, else 30):
  - seasoning = min((pool_age + m) / ramp_months, 1) for forecast month *m*
  - Young pools get lower CPR; at full seasoning the ramp is 1.
- CPR = (base_turnover + refi_response) × seasoning
- Monthly SMM from CPR drives scheduled principal + prepay cash flows.
- PSA is only a **reporting label** of that CPR (100 PSA ≡ 6% seasoned CPR). PSA is **not** a pricing input.

### Step C — Price / EVE — where OAS is used
- Discount each month’s cash flow at curve(t) + OAS.
- **OAS does not change CPR or PSA.** It only shifts the discount rate, so it moves PV, EVE, and KR01 for a given prepay schedule.
- Default OAS ≈ 50 bp if blank on the instrument.

### Calibration
Re-fit Step B with:
`python -m src.calibrate_prepayment --data your_loans.csv --out data/calibrated_prepayment_params.json`

## Hedge & CCR (prototype)

The **Hedge & CCR** tab prices the 2Y/5Y/10Y ALCO IRS ladder on the active curve and estimates:

| Output | Method |
|--------|--------|
| MtM / DV01 | Annual-pay vanilla IRS on the YieldCurve (ATM fixed rate) |
| EE / PFE | Monte Carlo parallel rate shocks; PFE = percentile of positive MtM |
| CVA | Unilateral: (1−R) Σ EE·DF·ΔPD from CDS≈λ·LGD hazard |
| SA-CCR EAD | BCBS 279 IR hedge-set Add-on; EAD = 1.4×(RC+PFE) |

Economic PFE ≠ SA-CCR PFE. Designation hints reuse the KR01 playbook (indicative only).
"""
