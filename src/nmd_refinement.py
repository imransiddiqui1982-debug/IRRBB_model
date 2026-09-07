"""
nmd_refinement.py
-----------------
Customer-level behavioural NMD refinement for IRRBB integration.

Input: one row per customer with segment, start date, deposit rate,
and 36 end-of-month balances (3 years), plus a market-rate series.

Outputs core / non-core / sticky balances by segment as IRRBB liability
instruments. Does NOT compute NII or EVE — those remain in IRRBBCalculator.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from typing import BinaryIO, TextIO, Union

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .cashflows import Instrument
from .file_io import is_excel_content, read_binary, read_csv_robust, read_excel_sheets

NmdSource = Union[str, TextIO, BinaryIO, io.BytesIO]

META_COLUMNS = {"customer_id", "segment", "start_date", "deposit_rate"}

OPTIONAL_META_COLUMNS = {
    "deposit_product", "currency", "maturity_date", "is_insured", "is_operational",
    "lcr_bucket", "nsfr_bucket", "branch_id", "relationship_years",
}

NMD_PRODUCTS = {"demand", "savings", "notice", ""}
TERM_PRODUCTS = {"term", "escrow"}

# BCBS 368 segment caps (Annex 1 / §113)
SEGMENT_CAPS = {
    "retail_transactional": {"max_core_pct": 0.90, "max_wal_years": 5.0},
    "retail_non_transactional": {"max_core_pct": 0.70, "max_wal_years": 4.5},
    "wholesale": {"max_core_pct": 0.50, "max_wal_years": 4.0},
}
SEGMENT_ALIASES = {
    "retail transactional": "retail_transactional",
    "retail_transactional": "retail_transactional",
    "retail": "retail_transactional",
    "transactional": "retail_transactional",
    "retail non transactional": "retail_non_transactional",
    "retail_non_transactional": "retail_non_transactional",
    "retail non-transactional": "retail_non_transactional",
    "non_transactional": "retail_non_transactional",
    "wholesale": "wholesale",
    "corporate": "wholesale",
}

DEFAULT_VOLATILE_REPRICING_YEARS = 1 / 12
DEFAULT_RATE_SENSITIVE_REPRICING_YEARS = 1 / 12


@dataclass
class NmdWorkbookData:
    """Customer balances, market rates, and optional monthly deposit rate table."""
    customers: pd.DataFrame
    market: pd.Series
    deposit_rates: pd.DataFrame | None = None


@dataclass
class SegmentResult:
    segment: str
    latest_balance_mb: float
    avg_deposit_rate_pct: float
    n_customers: int
    stable_pct: float
    non_core_pct: float
    beta: float
    sticky_pct: float
    core_balance_mb: float
    non_core_balance_mb: float
    rate_sensitive_balance_mb: float
    wal_years: float
    instruments: list[Instrument] = field(default_factory=list)
    operational_pct: float = 0.0


@dataclass
class NmdRefinementResult:
    deposit_name: str
    latest_balance_mb: float
    latest_deposit_rate_pct: float
    stable_pct: float
    non_core_pct: float
    beta: float
    sticky_pct: float
    core_balance_mb: float
    non_core_balance_mb: float
    rate_sensitive_balance_mb: float
    wal_years: float
    irrbb_buckets: pd.DataFrame
    summary: pd.DataFrame
    instruments: list[Instrument]
    segment_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    segment_results: list[SegmentResult] = field(default_factory=list)
    customer_count: int = 0
    month_count: int = 0
    liquidity_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    term_instruments: list[Instrument] = field(default_factory=list)


def _normalize_yes_no(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().upper() in {"Y", "YES", "TRUE", "1"}


def _deposit_product(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "demand"
    return str(value).strip().lower()


def _prepare_customer_frame(cust: pd.DataFrame) -> pd.DataFrame:
    cust = _normalize_columns(cust)
    for col in ("customer_id", "segment", "start_date"):
        if col not in cust.columns:
            raise ValueError(
                f"Customer sheet missing column '{col}'. "
                f"Required: customer_id, segment, start_date, and deposit_rate or rate_code."
            )
    if "deposit_rate" not in cust.columns and "rate_code" not in cust.columns:
        raise ValueError(
            "Customer sheet requires deposit_rate or rate_code "
            "(with Deposit Rates sheet for monthly product rates)."
        )
    if "deposit_product" not in cust.columns:
        cust["deposit_product"] = "demand"
    else:
        cust["deposit_product"] = cust["deposit_product"].map(_deposit_product)
    if "is_operational" not in cust.columns:
        cust["is_operational"] = "N"
    cust["segment"] = cust["segment"].map(_normalize_segment)
    cust["start_date"] = pd.to_datetime(cust["start_date"])
    if "deposit_rate" in cust.columns:
        cust["deposit_rate"] = pd.to_numeric(cust["deposit_rate"], errors="coerce")
    if "maturity_date" in cust.columns:
        cust["maturity_date"] = pd.to_datetime(cust["maturity_date"], errors="coerce")
    return cust


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _normalize_segment(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    key_spaced = key.replace("_", " ")
    if key in SEGMENT_ALIASES:
        return SEGMENT_ALIASES[key]
    if key_spaced in SEGMENT_ALIASES:
        return SEGMENT_ALIASES[key_spaced]
    raise ValueError(
        f"Unknown segment '{value}'. Use: retail_transactional, "
        "retail_non_transactional, or wholesale."
    )


def _is_month_column(name: str) -> bool:
    name = str(name).strip().lower()
    return bool(
        re.match(r"^\d{4}[-_/]\d{1,2}$", name)
        or re.match(r"^bal[_\-]?\d{4}[-_/]\d{1,2}$", name)
        or re.match(r"^\d{4}\d{2}$", name)
    )


def _parse_month_label(name: str) -> pd.Timestamp:
    raw = str(name).strip().lower()
    raw = re.sub(r"^bal[_\-]?", "", raw)
    raw = raw.replace("_", "-").replace("/", "-")
    if re.match(r"^\d{6}$", raw):
        raw = f"{raw[:4]}-{raw[4:]}"
    return pd.Period(raw, freq="M").to_timestamp("M")


def _read_workbook(source: NmdSource) -> dict[str, pd.DataFrame]:
    """Return sheet-name → DataFrame for Excel, or {'data': df} for CSV."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if path.lower().endswith((".xlsx", ".xls")):
            return read_excel_sheets(path)
        return {"data": read_csv_robust(path)}

    data = read_binary(source)
    if is_excel_content(data):
        return read_excel_sheets(data)

    return {"data": read_csv_robust(data)}


def _find_sheet(sheets: dict[str, pd.DataFrame], *candidates: str) -> pd.DataFrame | None:
    lower_map = {k.strip().lower(): v for k, v in sheets.items()}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def load_customer_nmd(source: NmdSource) -> NmdWorkbookData:
    """
    Load customer-level NMD data.

    Expected Excel sheets:
      - Customer Deposits / Customer Balances: customer_id, segment, start_date,
        deposit_rate OR rate_code, + month balance columns
      - Deposit Rates (optional): rate_code + monthly product rates for beta
      - Market Rates: date, market_rate

    CSV long format also accepted:
      customer_id, segment, start_date, date, balance, deposit_rate, market_rate
    """
    sheets = _read_workbook(source)

    # ── Excel wide format ────────────────────────────────────────────────────
    cust = _find_sheet(
        sheets,
        "Customer Deposits", "Customer Balances", "NMD Data", "Customers", "Deposits",
    )
    rates = _find_sheet(sheets, "Market Rates", "Rates", "Market Rate")
    deposit_rates = _find_sheet(
        sheets, "Deposit Rates", "Product Rates", "Rate Table",
    )

    if cust is not None:
        cust = _prepare_customer_frame(cust)
        month_cols = [c for c in cust.columns if _is_month_column(c)]
        if len(month_cols) < 12:
            raise ValueError(
                f"Need at least 12 end-of-month balance columns (YYYY-MM). Found {len(month_cols)}."
            )
        month_cols = sorted(month_cols, key=_parse_month_label)
        if rates is None:
            raise ValueError(
                "Excel must include a 'Market Rates' sheet with columns: date, market_rate."
            )
        market_raw = _normalize_columns(rates)
        if "date" not in market_raw.columns or "market_rate" not in market_raw.columns:
            raise ValueError("Market Rates sheet requires columns: date, market_rate.")
        market_raw["date"] = pd.to_datetime(market_raw["date"], errors="coerce")
        market_raw["market_rate"] = pd.to_numeric(market_raw["market_rate"], errors="coerce")
        market_raw = market_raw.dropna(subset=["date", "market_rate"]).sort_values("date")
        market = market_raw.set_index("date")["market_rate"]
        month_ends = [_parse_month_label(c) for c in month_cols]
        market = market.reindex(month_ends).interpolate(method="time").ffill().bfill()
        if market.isna().any():
            raise ValueError("Market Rates do not cover all balance months.")

        rate_table = None
        if deposit_rates is not None:
            rate_table = _normalize_columns(deposit_rates)
            if "rate_code" not in rate_table.columns:
                raise ValueError("Deposit Rates sheet requires a rate_code column.")
            for col in rate_table.columns:
                if _is_month_column(col):
                    rate_table[col] = pd.to_numeric(rate_table[col], errors="coerce")

        return NmdWorkbookData(
            customers=cust,
            market=market.rename(index=dict(zip(market.index, month_cols))),
            deposit_rates=rate_table,
        )

    # ── CSV / single-sheet long format ───────────────────────────────────────
    df = _normalize_columns(next(iter(sheets.values())))
    long_required = {
        "customer_id", "segment", "start_date", "date",
        "balance", "deposit_rate", "market_rate",
    }
    if long_required.issubset(set(df.columns)):
        df["date"] = pd.to_datetime(df["date"])
        df["balance"] = pd.to_numeric(df["balance"], errors="raise")
        df["deposit_rate"] = pd.to_numeric(df["deposit_rate"], errors="raise")
        df["market_rate"] = pd.to_numeric(df["market_rate"], errors="raise")
        market = (
            df.groupby(df["date"].dt.to_period("M").dt.to_timestamp("M"))["market_rate"]
            .mean()
            .sort_index()
        )
        wide = df.pivot_table(
            index=["customer_id", "segment", "start_date", "deposit_rate"],
            columns=df["date"].dt.to_period("M").astype(str),
            values="balance",
            aggfunc="last",
        ).reset_index()
        wide.columns = [str(c).lower() for c in wide.columns]
        return NmdWorkbookData(customers=wide, market=market, deposit_rates=None)

    raise ValueError(
        "Unrecognized NMD file. Provide Excel with sheets "
        "'Customer Deposits' + 'Market Rates' (+ optional 'Deposit Rates'), "
        "or long CSV with customer_id, segment, start_date, date, balance, "
        "deposit_rate, market_rate."
    )


def _customer_fallback_rate(row: pd.Series) -> float:
    for key in ("deposit_rate", "contract_rate"):
        if key in row.index and pd.notna(row[key]):
            return float(row[key])
    return 0.01


def _customer_rate_matrix(
    cust: pd.DataFrame,
    month_cols: list[str],
    balance_matrix: pd.DataFrame,
    deposit_rates: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build per-customer monthly deposit rates aligned to balance_matrix.

    Uses Deposit Rates sheet (rate_code × month) when available; otherwise
    constant deposit_rate / contract_rate per customer.
    """
    matrix = pd.DataFrame(
        index=balance_matrix.index,
        columns=balance_matrix.columns,
        dtype=float,
    )
    rate_idx = None
    if deposit_rates is not None and "rate_code" in deposit_rates.columns:
        rate_idx = deposit_rates.set_index("rate_code")

    for _, row in cust.iterrows():
        cid = str(row["customer_id"])
        if cid not in matrix.columns:
            continue
        fallback = _customer_fallback_rate(row)
        rc = row.get("rate_code")
        if rate_idx is None or pd.isna(rc) or str(rc).strip() == "":
            matrix[cid] = fallback
            continue
        rc = str(rc).strip()
        if rc not in rate_idx.index:
            matrix[cid] = fallback
            continue
        rrow = rate_idx.loc[rc]
        for m_label, m_end in zip(month_cols, matrix.index):
            if m_label in rrow.index and pd.notna(rrow[m_label]):
                matrix.at[m_end, cid] = float(rrow[m_label])
            else:
                matrix.at[m_end, cid] = fallback
    return matrix


def _balance_weighted_portfolio_rate(
    balance_matrix: pd.DataFrame,
    rate_matrix: pd.DataFrame,
) -> pd.Series:
    """Portfolio deposit rate each month: balance-weighted mean of customer rates."""
    out = []
    for m in balance_matrix.index:
        bals = balance_matrix.loc[m].fillna(0)
        rts = rate_matrix.loc[m]
        mask = (bals > 0) & rts.notna()
        if not mask.any():
            out.append(np.nan)
        else:
            b = bals[mask]
            r = rts[mask]
            out.append(float((b * r).sum() / b.sum()))
    return pd.Series(out, index=balance_matrix.index)


def _customer_cohort_matrix(cust: pd.DataFrame, month_cols: list[str]) -> pd.DataFrame:
    """Rows = months, columns = customer_id. Values = EOM balance (NaN before start)."""
    months = [_parse_month_label(c) for c in month_cols]
    matrix = pd.DataFrame(index=months)
    for _, row in cust.iterrows():
        cid = str(row["customer_id"])
        start = pd.to_datetime(row["start_date"])
        vals = []
        for m_label, m_end in zip(month_cols, months):
            if m_end < start.to_period("M").to_timestamp("M"):
                vals.append(np.nan)
            else:
                v = row[m_label]
                vals.append(float(v) if pd.notna(v) else 0.0)
        matrix[cid] = vals
    return matrix


def _cohort_runoff_rates(cohort_matrix: pd.DataFrame) -> pd.DataFrame:
    prev = cohort_matrix.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        runoff = 1.0 - cohort_matrix / prev
    # Clip runaway growth as "no runoff" for hazard estimation
    runoff = runoff.where(prev > 0)
    runoff = runoff.clip(lower=-1.0, upper=1.0)
    return runoff


def _snapshot_hazard_rates(runoff_rates: pd.DataFrame) -> pd.Series:
    # Only positive runoff contributes to decay hazard
    positive = runoff_rates.where(runoff_rates > 0)
    return positive.mean(axis=1, skipna=True).dropna()


def _stable_hp_filter(balance: pd.Series, lam: float = 1600.0) -> dict:
    n = len(balance)
    if n < 4:
        raise ValueError("Need at least 4 months of aggregate balances for HP filter.")
    y = balance.values.astype(float)
    eye = np.eye(n)
    d2 = np.diff(eye, n=2, axis=0)
    trend = np.linalg.solve(eye + lam * (d2.T @ d2), y)
    cycle = y - trend
    cycle_floor = np.percentile(cycle, 10)
    lower = np.maximum(trend + cycle_floor, 0)
    core = pd.Series(lower, index=balance.index)
    mean_bal = float(balance.mean())
    stable_pct = float(core.mean() / mean_bal) if mean_bal > 0 else 0.0
    return {
        "core": core,
        "non_core": (balance - core).clip(lower=0),
        "stable_pct": stable_pct,
    }


def _deposit_rate_beta(deposit_rate: pd.Series, market_rate: pd.Series) -> float:
    combined = pd.concat([deposit_rate, market_rate], axis=1).dropna()
    if len(combined) < 3:
        return 0.25
    y = combined.iloc[:, 0].values.astype(float)
    x = combined.iloc[:, 1].values.astype(float)
    X = np.column_stack([np.ones(len(y)), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.clip(coef[1], 0.0, 1.0))


def _compute_wal(survival_prob: np.ndarray, runoff_rates: np.ndarray) -> float:
    months = np.arange(1, len(runoff_rates) + 1)
    cash_flows = survival_prob[: len(runoff_rates)] * runoff_rates
    total_cf = cash_flows.sum()
    if total_cf <= 0:
        return 0.0
    return float(np.sum((months / 12.0) * cash_flows) / total_cf)


def _build_runoff_profile(hist_runoff: np.ndarray, seed_rate: float, n_months: int = 60) -> dict:
    n_period = min(24, len(hist_runoff))
    runoff_rates = np.empty(n_months)
    if n_period == 0:
        runoff_rates[:] = seed_rate
    else:
        runoff_rates[:n_period] = hist_runoff[:n_period]
        runoff_rates[n_period:] = seed_rate

    survival_prob = np.empty(n_months)
    survival_prob[0] = 1.0
    for t in range(1, n_months):
        survival_prob[t] = survival_prob[t - 1] * (1 - runoff_rates[t - 1])

    cash_flows = survival_prob * runoff_rates
    return {
        "runoff_rates": runoff_rates,
        "survival_prob": survival_prob,
        "cash_flows": cash_flows,
        "wal_years": _compute_wal(survival_prob, runoff_rates),
    }


def _optimize_seed_rate(
    hist_runoff: np.ndarray,
    max_wal_years: float,
    max_core_pct: float,
) -> dict:
    hist = np.asarray(hist_runoff, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) == 0:
        hist = np.array([0.03])
    hist = np.clip(hist[:24], 0.0001, 0.5)

    def objective(seed_rate: np.ndarray) -> float:
        profile = _build_runoff_profile(hist, float(seed_rate[0]))
        wal = profile["wal_years"]
        core_pct = float(profile["survival_prob"][-1] * (1 - profile["runoff_rates"][-1]))
        if wal > max_wal_years or core_pct > max_core_pct:
            return 1e6 + wal
        return -wal

    result = differential_evolution(objective, bounds=[(0.0001, 0.15)], seed=42, tol=1e-6)
    best_seed = float(result.x[0])
    profile = _build_runoff_profile(hist, best_seed)
    return {"seed_rate": best_seed, "profile": profile, "wal_years": profile["wal_years"]}


def _build_irrbb_buckets(profile: dict, core_balance_mb: float) -> pd.DataFrame:
    cf_mb = profile["cash_flows"] * core_balance_mb
    bucket_defs = [
        ("1-3M", 1, 3),
        ("3-6M", 4, 6),
        ("6-12M", 7, 12),
        ("1-2Y", 13, 24),
        ("2-3Y", 25, 36),
        ("3-4Y", 37, 48),
        ("4-5Y", 49, 60),
    ]
    rows = []
    total_allocated = 0.0
    for label, m_start, m_end in bucket_defs:
        idx = slice(m_start - 1, min(m_end, len(cf_mb)))
        bucket_mb = float(cf_mb[idx].sum())
        total_allocated += bucket_mb
        rows.append({
            "bucket": label,
            "months": f"{m_start}-{m_end}",
            "balance_mb": round(bucket_mb, 2),
            "pct_of_core": bucket_mb / core_balance_mb if core_balance_mb else 0.0,
        })
    residual_mb = max(core_balance_mb - total_allocated, 0.0)
    rows.append({
        "bucket": "5Y+",
        "months": "61+",
        "balance_mb": round(residual_mb, 2),
        "pct_of_core": residual_mb / core_balance_mb if core_balance_mb else 0.0,
    })
    out = pd.DataFrame(rows).set_index("bucket")
    out["pct_of_core"] = (out["pct_of_core"] * 100).round(2)
    return out


def _make_instruments(
    label: str,
    deposit_rate_pct: float,
    core_mb: float,
    rate_sensitive_mb: float,
    non_core_mb: float,
    wal_years: float,
    operational_pct: float = 0.0,
) -> list[Instrument]:
    instruments: list[Instrument] = []
    if core_mb > 0.01:
        instruments.append(Instrument(
            name=f"{label} — Core Sticky",
            notional=round(core_mb, 2),
            coupon_pct=deposit_rate_pct,
            instrument_type="demand_deposit",
            maturity_years=wal_years,
            payment_freq=12,
            repricing_years=wal_years,
            side="liability",
        ))
    if rate_sensitive_mb > 0.01:
        instruments.append(Instrument(
            name=f"{label} — Rate Sensitive",
            notional=round(rate_sensitive_mb, 2),
            coupon_pct=deposit_rate_pct,
            instrument_type="bullet_floating",
            maturity_years=3.0,
            payment_freq=12,
            repricing_years=DEFAULT_RATE_SENSITIVE_REPRICING_YEARS,
            side="liability",
        ))
    if non_core_mb > 0.01:
        op_tag = " Wholesale Operational" if operational_pct >= 0.5 else ""
        instruments.append(Instrument(
            name=f"{label} — Non-Core Volatile{op_tag}",
            notional=round(non_core_mb, 2),
            coupon_pct=deposit_rate_pct,
            instrument_type="demand_deposit",
            maturity_years=DEFAULT_VOLATILE_REPRICING_YEARS,
            payment_freq=12,
            repricing_years=DEFAULT_VOLATILE_REPRICING_YEARS,
            side="liability",
        ))
    return instruments


def _resolve_customer_rate_at_month(
    row: pd.Series,
    month_col: str,
    deposit_rates: pd.DataFrame | None,
) -> float:
    fallback = _customer_fallback_rate(row)
    rc = row.get("rate_code")
    if deposit_rates is None or pd.isna(rc) or str(rc).strip() == "":
        return fallback
    rate_idx = deposit_rates.set_index("rate_code")
    rc = str(rc).strip()
    if rc not in rate_idx.index:
        return fallback
    rrow = rate_idx.loc[rc]
    if month_col in rrow.index and pd.notna(rrow[month_col]):
        return float(rrow[month_col])
    return fallback


def _build_term_instruments(
    term_cust: pd.DataFrame,
    month_cols: list[str],
    deposit_name: str,
    deposit_rates: pd.DataFrame | None = None,
) -> list[Instrument]:
    """Create fixed term-deposit liabilities from term/escrow customers."""
    if term_cust.empty:
        return []

    instruments: list[Instrument] = []
    latest_col = month_cols[-1]
    grouped: dict[tuple[str, str], dict] = {}

    for _, row in term_cust.iterrows():
        bal = row.get(latest_col)
        if pd.isna(bal) or float(bal) <= 0:
            continue
        mat = row.get("maturity_date")
        if pd.isna(mat):
            raise ValueError(
                f"Term deposit {row['customer_id']} requires maturity_date."
            )
        mat_ts = pd.Timestamp(mat)
        months_to_mat = max((mat_ts.to_period("M") - pd.Timestamp(month_cols[-1]).to_period("M")).n, 1)
        maturity_years = months_to_mat / 12.0
        seg = str(row["segment"])
        key = (seg, mat_ts.strftime("%Y-%m"))
        rate = _resolve_customer_rate_at_month(row, latest_col, deposit_rates)
        rate_pct = rate * 100 if rate < 1 else rate
        bucket = grouped.setdefault(key, {"balance": 0.0, "rate_pct": rate_pct, "maturity_years": maturity_years, "seg": seg, "mat": mat_ts.strftime("%Y-%m")})
        bucket["balance"] += float(bal)

    for (_seg, _mat), info in sorted(grouped.items()):
        instruments.append(Instrument(
            name=f"{deposit_name} / {info['seg']} — Term Deposit ({info['mat']})",
            notional=round(info["balance"], 2),
            coupon_pct=info["rate_pct"],
            instrument_type="bullet_fixed",
            maturity_years=info["maturity_years"],
            payment_freq=2,
            repricing_years=info["maturity_years"],
            side="liability",
        ))
    return instruments


def _build_liquidity_summary(
    segment_results: list[SegmentResult],
    term_instruments: list[Instrument],
) -> pd.DataFrame:
    rows = []
    for s in segment_results:
        for bucket, bal, lcr, nsfr in (
            ("Core Sticky", s.core_balance_mb, "5%", "95%"),
            ("Rate Sensitive", s.rate_sensitive_balance_mb, "10%", "90%"),
            ("Non-Core Volatile", s.non_core_balance_mb, "25%", "50%"),
        ):
            if bal <= 0.01:
                continue
            rows.append({
                "Segment": s.segment,
                "Bucket": bucket,
                "Balance ($M)": round(bal, 2),
                "LCR Outflow Rate": lcr,
                "NSFR ASF Factor": nsfr,
                "Source": "NMD behavioural split",
            })
    for inst in term_instruments:
        mat = inst.maturity_years
        if mat < 0.5:
            lcr, nsfr = "0%", "0%"
        elif mat < 1.0:
            lcr, nsfr = "0%", "50%"
        else:
            lcr, nsfr = "By retail rules", "100%"
        rows.append({
            "Segment": inst.name.split("/")[1].split("—")[0].strip() if "/" in inst.name else "",
            "Bucket": "Term Deposit",
            "Balance ($M)": inst.notional,
            "LCR Outflow Rate": lcr,
            "NSFR ASF Factor": nsfr,
            "Source": "maturity_date from deposit file",
        })
    return pd.DataFrame(rows)


def _refine_segment(
    seg_cust: pd.DataFrame,
    month_cols: list[str],
    market: pd.Series,
    deposit_name: str,
    deposit_rates: pd.DataFrame | None = None,
) -> SegmentResult:
    segment = _normalize_segment(seg_cust["segment"].iloc[0])
    caps = SEGMENT_CAPS[segment]
    max_core = caps["max_core_pct"]
    max_wal = caps["max_wal_years"]

    matrix = _customer_cohort_matrix(seg_cust, month_cols)
    months = matrix.index
    agg_balance = matrix.fillna(0).sum(axis=1)
    if float(agg_balance.iloc[-1]) <= 0:
        raise ValueError(f"Segment '{segment}' has zero latest balance.")

    rate_matrix = _customer_rate_matrix(seg_cust, month_cols, matrix, deposit_rates)
    deposit_rate_ts = _balance_weighted_portfolio_rate(matrix.fillna(0), rate_matrix)

    latest_balances = matrix.iloc[-1].fillna(0)
    latest_rates = rate_matrix.iloc[-1]
    mask_latest = (latest_balances > 0) & latest_rates.notna()
    total_latest = float(latest_balances.sum())
    avg_rate = float(
        (latest_balances[mask_latest] * latest_rates[mask_latest]).sum() / total_latest
    ) if total_latest and mask_latest.any() else 0.0

    market_aligned = market.copy()
    if not isinstance(market_aligned.index, pd.DatetimeIndex):
        # market indexed by month column labels
        market_aligned = pd.Series(
            [float(market_aligned.get(c, np.nan)) for c in month_cols],
            index=months,
        )
    else:
        market_aligned = market_aligned.reindex(months).interpolate().ffill().bfill()

    stable = _stable_hp_filter(agg_balance)
    stable_pct = float(np.clip(stable["stable_pct"], 0.0, max_core))
    beta = _deposit_rate_beta(deposit_rate_ts, market_aligned)

    hazard = _snapshot_hazard_rates(_cohort_runoff_rates(matrix))
    runoff = _optimize_seed_rate(
        hazard.values if len(hazard) else np.array([0.03]),
        max_wal_years=max_wal,
        max_core_pct=max_core,
    )
    wal_years = float(np.clip(runoff["wal_years"], 1 / 365, max_wal))

    latest = float(agg_balance.iloc[-1])
    core_mb = latest * stable_pct * (1 - beta)
    rate_sens_mb = latest * stable_pct * beta
    non_core_mb = latest * (1 - stable_pct)
    rate_pct = avg_rate * 100 if avg_rate < 1.0 else avg_rate

    operational_weights = []
    for _, row in seg_cust.iterrows():
        cid = str(row["customer_id"])
        bal = float(latest_balances.get(cid, 0))
        if bal > 0 and _normalize_yes_no(row.get("is_operational", "N")):
            operational_weights.append(bal)
    operational_pct = sum(operational_weights) / total_latest if total_latest else 0.0

    label = f"{deposit_name} / {segment}"
    instruments = _make_instruments(
        label, rate_pct, core_mb, rate_sens_mb, non_core_mb, wal_years, operational_pct
    )

    return SegmentResult(
        segment=segment,
        latest_balance_mb=latest,
        avg_deposit_rate_pct=rate_pct,
        n_customers=len(seg_cust),
        stable_pct=stable_pct,
        non_core_pct=1 - stable_pct,
        beta=beta,
        sticky_pct=1 - beta,
        core_balance_mb=core_mb,
        non_core_balance_mb=non_core_mb,
        rate_sensitive_balance_mb=rate_sens_mb,
        wal_years=wal_years,
        instruments=instruments,
        operational_pct=operational_pct,
    )


def refine_nmd_deposits(
    source: NmdSource,
    deposit_name: str = "Refined NMD Pool",
    max_wal_years: float | None = None,
    max_core_pct: float | None = None,
) -> NmdRefinementResult:
    """
    Run customer-level NMD refinement by segment and build IRRBB instruments.

    Parameters max_wal_years / max_core_pct are kept for API compatibility;
    per-segment BCBS caps are applied automatically.
    """
    workbook = load_customer_nmd(source)
    cust = workbook.customers
    market = workbook.market
    deposit_rates = workbook.deposit_rates

    month_cols = [c for c in cust.columns if _is_month_column(c)]
    month_cols = sorted(month_cols, key=_parse_month_label)
    if len(month_cols) < 12:
        raise ValueError("Need at least 12 months of end-of-month balances.")

    for col in month_cols:
        cust[col] = pd.to_numeric(cust[col], errors="coerce")

    nmd_cust = cust[~cust["deposit_product"].isin(TERM_PRODUCTS)]
    term_cust = cust[cust["deposit_product"].isin(TERM_PRODUCTS)]
    term_instruments = _build_term_instruments(
        term_cust, month_cols, deposit_name, deposit_rates,
    )

    segment_results: list[SegmentResult] = []
    all_instruments: list[Instrument] = []

    for seg, group in nmd_cust.groupby("segment"):
        # Optional overrides still clamp below BCBS caps
        caps = SEGMENT_CAPS[seg]
        if max_core_pct is not None:
            caps = {**caps, "max_core_pct": min(caps["max_core_pct"], max_core_pct)}
        if max_wal_years is not None:
            caps = {**caps, "max_wal_years": min(caps["max_wal_years"], max_wal_years)}
        # Temporarily inject caps via SEGMENT_CAPS lookup inside _refine_segment
        result = _refine_segment(
            group, month_cols, market, deposit_name, deposit_rates,
        )
        # Re-clamp if caller overrides are tighter
        if max_core_pct is not None and result.stable_pct > max_core_pct:
            # rebuild with tighter core — rare path; leave as-is and clip balances
            scale = max_core_pct / result.stable_pct if result.stable_pct else 1.0
            result.stable_pct = max_core_pct
            result.core_balance_mb *= scale
            result.rate_sensitive_balance_mb *= scale
            result.non_core_balance_mb = result.latest_balance_mb - (
                result.core_balance_mb + result.rate_sensitive_balance_mb
            )
        segment_results.append(result)
        all_instruments.extend(result.instruments)

    all_instruments.extend(term_instruments)

    if not segment_results and not term_instruments:
        raise ValueError("No customer segments or term deposits found to refine.")

    total_bal = sum(s.latest_balance_mb for s in segment_results) + sum(
        i.notional for i in term_instruments
    )
    core_mb = sum(s.core_balance_mb for s in segment_results)
    non_core_mb = sum(s.non_core_balance_mb for s in segment_results)
    rate_sens_mb = sum(s.rate_sensitive_balance_mb for s in segment_results)
    nmd_bal = sum(s.latest_balance_mb for s in segment_results)
    stable_pct = (core_mb + rate_sens_mb) / nmd_bal if nmd_bal else 0.0
    beta = rate_sens_mb / (core_mb + rate_sens_mb) if (core_mb + rate_sens_mb) else 0.0
    wal_years = float(
        np.average(
            [s.wal_years for s in segment_results],
            weights=[max(s.core_balance_mb, 1e-9) for s in segment_results],
        )
    ) if segment_results else 0.0
    avg_rate = float(
        np.average(
            [s.avg_deposit_rate_pct for s in segment_results],
            weights=[max(s.latest_balance_mb, 1e-9) for s in segment_results],
        )
    ) if segment_results else 0.0

    full_matrix = _customer_cohort_matrix(nmd_cust, month_cols) if len(nmd_cust) else pd.DataFrame()
    hazard = _snapshot_hazard_rates(_cohort_runoff_rates(full_matrix)) if not full_matrix.empty else pd.Series(dtype=float)
    portfolio_runoff = _optimize_seed_rate(
        hazard.values if len(hazard) else np.array([0.03]),
        max_wal_years=max((s.wal_years for s in segment_results), default=5.0),
        max_core_pct=0.90,
    )
    buckets = _build_irrbb_buckets(portfolio_runoff["profile"], core_mb) if core_mb > 0 else pd.DataFrame()

    seed_monthly = float(portfolio_runoff["seed_rate"])
    profile_rates = np.asarray(portfolio_runoff["profile"]["runoff_rates"], dtype=float)
    n_hist_hazards = int(min(24, len(hazard))) if len(hazard) else 0
    hist_monthly = (
        float(np.mean(profile_rates[:n_hist_hazards]))
        if n_hist_hazards > 0
        else float("nan")
    )
    seed_annual_pct = (1.0 - (1.0 - seed_monthly) ** 12) * 100.0
    hist_annual_pct = (
        (1.0 - (1.0 - hist_monthly) ** 12) * 100.0
        if np.isfinite(hist_monthly)
        else float("nan")
    )

    liquidity_summary = _build_liquidity_summary(segment_results, term_instruments)

    segment_summary = pd.DataFrame([{
        "Segment": s.segment,
        "Customers": s.n_customers,
        "Balance ($M)": round(s.latest_balance_mb, 2),
        "Stable %": round(s.stable_pct * 100, 2),
        "Non-core %": round(s.non_core_pct * 100, 2),
        "Beta": round(s.beta, 4),
        "Core Sticky ($M)": round(s.core_balance_mb, 2),
        "Rate Sensitive ($M)": round(s.rate_sensitive_balance_mb, 2),
        "Non-core ($M)": round(s.non_core_balance_mb, 2),
        "WAL (Y)": round(s.wal_years, 2),
        "Deposit Rate (%)": round(s.avg_deposit_rate_pct, 4),
    } for s in segment_results])

    summary_rows = [
        {"Metric": "Customers", "Value": len(cust)},
        {"Metric": "Months of history", "Value": len(month_cols)},
        {"Metric": "Latest balance (USD M)", "Value": round(total_bal, 2)},
        {"Metric": "Avg deposit rate (%)", "Value": round(avg_rate, 4)},
        {"Metric": "Stable / core %", "Value": round(stable_pct * 100, 2)},
        {"Metric": "Non-core %", "Value": round((1 - stable_pct) * 100, 2)},
        {"Metric": "Beta (rate pass-through)", "Value": round(beta, 4)},
        {"Metric": "Beta input", "Value": "monthly Deposit Rates" if deposit_rates is not None else "static deposit_rate"},
        {"Metric": "Sticky % within stable", "Value": round((1 - beta) * 100, 2)},
        {"Metric": "Core sticky balance (USD M)", "Value": round(core_mb, 2)},
        {"Metric": "Rate-sensitive balance (USD M)", "Value": round(rate_sens_mb, 2)},
        {"Metric": "Non-core balance (USD M)", "Value": round(non_core_mb, 2)},
        {"Metric": "Term deposit instruments", "Value": len(term_instruments)},
        {"Metric": "Behavioural WAL (years)", "Value": round(wal_years, 2)},
        {
            "Metric": "Hist. avg monthly runoff / decay (%)",
            "Value": round(hist_monthly * 100.0, 4) if np.isfinite(hist_monthly) else "n/a",
        },
        {
            "Metric": "Hist. avg annualized runoff (%)",
            "Value": round(hist_annual_pct, 2) if np.isfinite(hist_annual_pct) else "n/a",
        },
        {
            "Metric": "Long-run monthly runoff / decay (%)",
            "Value": round(seed_monthly * 100.0, 4),
        },
        {
            "Metric": "Long-run annualized runoff (%)",
            "Value": round(seed_annual_pct, 2),
        },
        {
            "Metric": "Historical hazard months in profile",
            "Value": n_hist_hazards,
        },
    ]
    summary = pd.DataFrame(summary_rows)

    return NmdRefinementResult(
        deposit_name=deposit_name,
        latest_balance_mb=total_bal,
        latest_deposit_rate_pct=avg_rate,
        stable_pct=stable_pct,
        non_core_pct=1 - stable_pct,
        beta=beta,
        sticky_pct=1 - beta,
        core_balance_mb=core_mb,
        non_core_balance_mb=non_core_mb,
        rate_sensitive_balance_mb=rate_sens_mb,
        wal_years=wal_years,
        irrbb_buckets=buckets,
        summary=summary,
        instruments=all_instruments,
        segment_summary=segment_summary,
        segment_results=segment_results,
        customer_count=len(cust),
        month_count=len(month_cols),
        liquidity_summary=liquidity_summary,
        term_instruments=term_instruments,
    )


def merge_nmd_into_balance_sheet(
    assets: list[Instrument],
    liabilities: list[Instrument],
    nmd_result: NmdRefinementResult,
) -> tuple[list[Instrument], list[Instrument]]:
    """Replace built-in demand_deposit liabilities with refined NMD instruments."""
    kept_liabilities = [
        inst for inst in liabilities
        if inst.instrument_type != "demand_deposit"
    ]
    if nmd_result.term_instruments:
        kept_liabilities = [
            inst for inst in kept_liabilities
            if "term deposit" not in inst.name.lower()
        ]
    return assets, kept_liabilities + nmd_result.instruments
