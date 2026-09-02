"""
liquidity_workbook.py
---------------------
Load U.S. LCR / NSFR public-disclosure style inputs from the deposit workbook.

Sheets (HSBC / 12 CFR 249.91 & 249.131 aligned):
  Regulatory Parameters — tailoring, capital, caps
  HQLA Stock           — LCR rows 2–4 detail
  LCR Cash Outflows    — LCR rows 5–19
  LCR Cash Inflows     — LCR rows 20–28
  NSFR ASF             — NSFR rows 1–15
  NSFR RSF             — NSFR rows 16–36

When ``auto_from_nmd = Y`` on an outflow/ASF row, balances are populated from
NMD behavioural bifurcation (core sticky, rate sensitive, non-core, operational).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, BinaryIO, TextIO, Union

import pandas as pd

from .file_io import is_excel_content, read_binary

if TYPE_CHECKING:
    from .nmd_refinement import NmdRefinementResult

WorkbookSource = Union[str, TextIO, BinaryIO, io.BytesIO]

LIQUIDITY_SHEETS = {
    "Regulatory Parameters",
    "HQLA Stock",
    "LCR Cash Outflows",
    "LCR Cash Inflows",
    "NSFR ASF",
    "NSFR RSF",
}

NMD_BUCKET_KEYS = {
    "core_sticky",
    "rate_sensitive",
    "non_core_volatile",
    "wholesale_operational",
    "wholesale_non_op",
    "term_deposit",
    "term_deposit_lt_30d",
}


@dataclass
class RegulatoryParams:
    tier1_capital_m: float = 0.0
    tier2_capital_m: float = 0.0
    outflow_adjustment_pct: float = 0.30
    lcr_minimum_pct: float = 100.0
    nsfr_minimum_pct: float = 100.0
    inflow_cap_pct: float = 0.75
    level2_cap_pct: float = 0.40
    level2b_cap_pct: float = 0.15
    rsf_adjustment_pct: float = 0.30
    auto_nmd_deposits: bool = True


@dataclass
class HqlaLineItem:
    row: int
    disclosure_item: str
    hqla_level: str
    unweighted_amount_m: float
    haircut_pct: float
    encumbered_amount_m: float = 0.0
    notes: str = ""


@dataclass
class LcrOutflowLineItem:
    row: int
    disclosure_item: str
    category: str
    unweighted_amount_m: float
    runoff_rate_pct: float
    auto_from_nmd: bool = False
    nmd_bucket: str = ""
    notes: str = ""


@dataclass
class LcrInflowLineItem:
    row: int
    disclosure_item: str
    unweighted_amount_m: float
    inflow_rate_pct: float
    notes: str = ""


@dataclass
class NsfrAsfLineItem:
    row: int
    disclosure_item: str
    maturity_bucket: str
    unweighted_amount_m: float
    asf_factor_pct: float
    auto_from_nmd: bool = False
    nmd_bucket: str = ""
    notes: str = ""


@dataclass
class NsfrRsfLineItem:
    row: int
    disclosure_item: str
    unweighted_amount_m: float
    rsf_factor_pct: float
    link_hqla: bool = False
    notes: str = ""


@dataclass
class NmdDepositBuckets:
    core_sticky_m: float = 0.0
    rate_sensitive_m: float = 0.0
    non_core_volatile_m: float = 0.0
    wholesale_operational_m: float = 0.0
    wholesale_non_op_m: float = 0.0
    term_deposit_m: float = 0.0
    term_deposit_lt_30d_m: float = 0.0


@dataclass
class LiquidityWorkbookData:
    params: RegulatoryParams
    hqla_items: list[HqlaLineItem] = field(default_factory=list)
    lcr_outflows: list[LcrOutflowLineItem] = field(default_factory=list)
    lcr_inflows: list[LcrInflowLineItem] = field(default_factory=list)
    nsfr_asf: list[NsfrAsfLineItem] = field(default_factory=list)
    nsfr_rsf: list[NsfrRsfLineItem] = field(default_factory=list)
    nmd_buckets: NmdDepositBuckets = field(default_factory=NmdDepositBuckets)
    has_liquidity_sheets: bool = False


def _read_sheet(source: WorkbookSource, sheet: str) -> pd.DataFrame | None:
    try:
        data = read_binary(source)
        if not is_excel_content(data):
            return None
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine="openpyxl")
    except ValueError:
        return None
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _read_all_liquidity_sheets(source: WorkbookSource) -> dict[str, pd.DataFrame | None]:
    """Read liquidity sheets once (safe for BytesIO sources)."""
    try:
        data = read_binary(source)
        if not is_excel_content(data):
            return {}
        xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    except Exception:
        return {}
    out: dict[str, pd.DataFrame | None] = {}
    for sheet in LIQUIDITY_SHEETS | {"LCR Disclosure Summary", "NSFR Disclosure Summary"}:
        if sheet in xl.sheet_names:
            df = xl.parse(sheet)
            if df is not None and not df.empty:
                df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
                out[sheet] = df
            else:
                out[sheet] = None
        else:
            out[sheet] = None
    return out


def _to_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    return str(value).strip().upper() in {"Y", "YES", "TRUE", "1"}


def _pct_to_decimal(value, default: float) -> float:
    v = _to_float(value, default)
    if v > 1.0:
        return v / 100.0
    return v


def compute_nmd_deposit_buckets(
    nmd_result: NmdRefinementResult,
    customers: pd.DataFrame | None = None,
) -> NmdDepositBuckets:
    """Aggregate NMD bifurcation balances into disclosure bucket keys."""
    buckets = NmdDepositBuckets()

    for seg in nmd_result.segment_results:
        is_wholesale = seg.segment == "wholesale"
        if is_wholesale:
            buckets.core_sticky_m += seg.core_balance_mb
            buckets.rate_sensitive_m += seg.rate_sensitive_balance_mb
            op_bal = 0.0
            if customers is not None and not customers.empty:
                month_cols = sorted(
                    [c for c in customers.columns if _is_month_col(c)],
                    key=_month_key,
                )
                if month_cols:
                    latest = month_cols[-1]
                    ws = customers[customers["segment"].astype(str).str.lower() == "wholesale"]
                    for _, row in ws.iterrows():
                        bal = _to_float(row.get(latest), 0.0)
                        if bal > 0 and _to_bool(row.get("is_operational", "N")):
                            op_bal += bal
            else:
                op_bal = seg.latest_balance_mb * seg.operational_pct
            op_bal = min(op_bal, seg.latest_balance_mb)
            buckets.wholesale_operational_m += op_bal
            buckets.wholesale_non_op_m += max(seg.latest_balance_mb - op_bal, 0.0)
        else:
            buckets.core_sticky_m += seg.core_balance_mb
            buckets.rate_sensitive_m += seg.rate_sensitive_balance_mb
            buckets.non_core_volatile_m += seg.non_core_balance_mb

    for inst in nmd_result.term_instruments:
        if inst.maturity_years < 30 / 365:
            buckets.term_deposit_lt_30d_m += inst.notional
        else:
            buckets.term_deposit_m += inst.notional

    return buckets


def _is_month_col(col: str) -> bool:
    return bool(__import__("re").match(r"^\d{4}-\d{2}$", str(col)))


def _month_key(col: str) -> pd.Timestamp:
    return pd.Timestamp(f"{col}-01")


def _resolve_nmd_amount(bucket: str, nmd: NmdDepositBuckets) -> float:
    mapping = {
        "core_sticky": nmd.core_sticky_m,
        "rate_sensitive": nmd.rate_sensitive_m,
        "non_core_volatile": nmd.non_core_volatile_m,
        "wholesale_operational": nmd.wholesale_operational_m,
        "wholesale_non_op": nmd.wholesale_non_op_m,
        "term_deposit": nmd.term_deposit_m,
        "term_deposit_lt_30d": nmd.term_deposit_lt_30d_m,
    }
    return mapping.get(bucket, 0.0)


def load_regulatory_params(df: pd.DataFrame | None) -> RegulatoryParams:
    if df is None or df.empty:
        return RegulatoryParams()
    if "parameter" in df.columns:
        lookup = {
            str(r["parameter"]).strip().lower(): r.get("value")
            for _, r in df.iterrows()
            if pd.notna(r.get("parameter"))
        }
        return RegulatoryParams(
            tier1_capital_m=_to_float(lookup.get("tier1_capital_m"), 0.0),
            tier2_capital_m=_to_float(lookup.get("tier2_capital_m"), 0.0),
            outflow_adjustment_pct=_pct_to_decimal(
                lookup.get("outflow_adjustment_pct"), 0.30
            ),
            lcr_minimum_pct=_to_float(lookup.get("lcr_minimum_pct"), 100.0),
            nsfr_minimum_pct=_to_float(lookup.get("nsfr_minimum_pct"), 100.0),
            inflow_cap_pct=_pct_to_decimal(lookup.get("inflow_cap_pct"), 0.75),
            level2_cap_pct=_pct_to_decimal(lookup.get("level2_cap_pct"), 0.40),
            level2b_cap_pct=_pct_to_decimal(lookup.get("level2b_cap_pct"), 0.15),
            rsf_adjustment_pct=_pct_to_decimal(lookup.get("rsf_adjustment_pct"), 0.30),
            auto_nmd_deposits=_to_bool(lookup.get("auto_nmd_deposits"), True),
        )
    return RegulatoryParams()


def load_hqla_stock(df: pd.DataFrame | None) -> list[HqlaLineItem]:
    if df is None or df.empty:
        return []
    items = []
    for _, r in df.iterrows():
        item = str(r.get("disclosure_item", "")).strip()
        if not item or item.lower() == "nan":
            continue
        items.append(HqlaLineItem(
            row=int(_to_float(r.get("row"), 0)),
            disclosure_item=item,
            hqla_level=str(r.get("hqla_level", "level_1")).strip().lower(),
            unweighted_amount_m=_to_float(r.get("unweighted_amount_m"), 0.0),
            haircut_pct=_to_float(r.get("haircut_pct"), 0.0),
            encumbered_amount_m=_to_float(r.get("encumbered_amount_m"), 0.0),
            notes=str(r.get("notes", "") or ""),
        ))
    return items


def load_lcr_outflows(
    df: pd.DataFrame | None,
    nmd: NmdDepositBuckets,
    auto_nmd: bool,
) -> list[LcrOutflowLineItem]:
    if df is None or df.empty:
        return []
    items = []
    for _, r in df.iterrows():
        item = str(r.get("disclosure_item", "")).strip()
        if not item or item.lower() == "nan":
            continue
        auto = _to_bool(r.get("auto_from_nmd"), False)
        bucket = str(r.get("nmd_bucket", "") or "").strip().lower()
        amount = _to_float(r.get("unweighted_amount_m"), 0.0)
        if auto and auto_nmd and bucket:
            amount = _resolve_nmd_amount(bucket, nmd)
        items.append(LcrOutflowLineItem(
            row=int(_to_float(r.get("row"), 0)),
            disclosure_item=item,
            category=str(r.get("category", "")).strip().lower(),
            unweighted_amount_m=amount,
            runoff_rate_pct=_to_float(r.get("runoff_rate_pct"), 0.0),
            auto_from_nmd=auto,
            nmd_bucket=bucket,
            notes=str(r.get("notes", "") or ""),
        ))
    return items


def load_lcr_inflows(df: pd.DataFrame | None) -> list[LcrInflowLineItem]:
    if df is None or df.empty:
        return []
    items = []
    for _, r in df.iterrows():
        item = str(r.get("disclosure_item", "")).strip()
        if not item or item.lower() == "nan":
            continue
        items.append(LcrInflowLineItem(
            row=int(_to_float(r.get("row"), 0)),
            disclosure_item=item,
            unweighted_amount_m=_to_float(r.get("unweighted_amount_m"), 0.0),
            inflow_rate_pct=_to_float(r.get("inflow_rate_pct"), 0.0),
            notes=str(r.get("notes", "") or ""),
        ))
    return items


def load_nsfr_asf(
    df: pd.DataFrame | None,
    nmd: NmdDepositBuckets,
    params: RegulatoryParams,
    auto_nmd: bool,
) -> list[NsfrAsfLineItem]:
    if df is None or df.empty:
        return []
    items = []
    for _, r in df.iterrows():
        item = str(r.get("disclosure_item", "")).strip()
        if not item or item.lower() == "nan":
            continue
        auto = _to_bool(r.get("auto_from_nmd"), False)
        bucket = str(r.get("nmd_bucket", "") or "").strip().lower()
        amount = _to_float(r.get("unweighted_amount_m"), 0.0)
        if bucket == "regulatory_capital":
            amount = params.tier1_capital_m + params.tier2_capital_m
        elif auto and auto_nmd and bucket:
            amount = _resolve_nmd_amount(bucket, nmd)
        items.append(NsfrAsfLineItem(
            row=int(_to_float(r.get("row"), 0)),
            disclosure_item=item,
            maturity_bucket=str(r.get("maturity_bucket", "open")).strip().lower(),
            unweighted_amount_m=amount,
            asf_factor_pct=_to_float(r.get("asf_factor_pct"), 0.0),
            auto_from_nmd=auto,
            nmd_bucket=bucket,
            notes=str(r.get("notes", "") or ""),
        ))
    return items


def load_nsfr_rsf(
    df: pd.DataFrame | None,
    hqla_items: list[HqlaLineItem],
) -> list[NsfrRsfLineItem]:
    if df is None or df.empty:
        return []
    hqla_by_level = {"level_1": 0.0, "level_2a": 0.0, "level_2b": 0.0}
    for h in hqla_items:
        net = max(h.unweighted_amount_m - h.encumbered_amount_m, 0.0)
        level = h.hqla_level.replace(" ", "_").lower()
        if level in hqla_by_level:
            hqla_by_level[level] += net

    items = []
    for _, r in df.iterrows():
        item = str(r.get("disclosure_item", "")).strip()
        if not item or item.lower() == "nan":
            continue
        link = _to_bool(r.get("link_hqla"), False)
        amount = _to_float(r.get("unweighted_amount_m"), 0.0)
        if link:
            level_key = str(r.get("hqla_level", "")).strip().lower().replace(" ", "_")
            if level_key in hqla_by_level:
                amount = hqla_by_level[level_key]
        items.append(NsfrRsfLineItem(
            row=int(_to_float(r.get("row"), 0)),
            disclosure_item=item,
            unweighted_amount_m=amount,
            rsf_factor_pct=_to_float(r.get("rsf_factor_pct"), 0.0),
            link_hqla=link,
            notes=str(r.get("notes", "") or ""),
        ))
    return items


def load_liquidity_workbook(
    source: WorkbookSource,
    nmd_result: NmdRefinementResult | None = None,
    customers: pd.DataFrame | None = None,
) -> LiquidityWorkbookData:
    """Load liquidity disclosure inputs; apply NMD auto-fill when enabled."""
    sheets = _read_all_liquidity_sheets(source)
    has_any = bool(sheets)

    params_df = sheets.get("Regulatory Parameters")
    params = load_regulatory_params(params_df)
    nmd = (
        compute_nmd_deposit_buckets(nmd_result, customers)
        if nmd_result is not None
        else NmdDepositBuckets()
    )
    auto_nmd = params.auto_nmd_deposits and nmd_result is not None

    hqla_items = load_hqla_stock(sheets.get("HQLA Stock"))
    return LiquidityWorkbookData(
        params=params,
        hqla_items=hqla_items,
        lcr_outflows=load_lcr_outflows(sheets.get("LCR Cash Outflows"), nmd, auto_nmd),
        lcr_inflows=load_lcr_inflows(sheets.get("LCR Cash Inflows")),
        nsfr_asf=load_nsfr_asf(sheets.get("NSFR ASF"), nmd, params, auto_nmd),
        nsfr_rsf=load_nsfr_rsf(sheets.get("NSFR RSF"), hqla_items),
        nmd_buckets=nmd,
        has_liquidity_sheets=has_any,
    )


def workbook_has_liquidity_sheets(source: WorkbookSource) -> bool:
    """Return True if the workbook contains at least one liquidity input sheet."""
    try:
        data = read_binary(source)
        if not is_excel_content(data):
            return False
        xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
        return bool(LIQUIDITY_SHEETS & set(xl.sheet_names))
    except Exception:
        return False
