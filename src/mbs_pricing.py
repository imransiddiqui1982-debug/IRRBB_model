"""
mbs_pricing.py
--------------
Option-adjusted MBS / whole-loan prepayment (spec Steps A → B → C).

Critical invariant: cash flows and price are ALWAYS re-derived from the
``curve`` argument. Never cache a schedule across curve bumps — that is the
#1 bug that silently overstates KR01 / duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .time_buckets import N_BUCKETS, years_to_bucket
from .yield_curve import YieldCurve


# Defaults from the implementation spec (illustrative, not portfolio-calibrated)
DEFAULT_SPREAD_TO_CURVE = 0.0175
DEFAULT_OAS = 0.005
DEFAULT_BASE_TURNOVER = 0.06
DEFAULT_MAX_REFI_CPR = 0.34
DEFAULT_LOGISTIC_K = 2.2
DEFAULT_LOGISTIC_MIDPOINT = 0.60
DEFAULT_SEASONING_RAMP_MONTHS = 30
DEFAULT_ANCHOR_TENOR = 10.0


@dataclass
class MbsTerms:
    """
    Homogeneous pool / loan-bucket terms for Steps A/B/C.

    ``is_whole_loan`` only affects static LCR/NSFR tags and credit_spread
    placeholder — pricing Steps A/B/C are identical to MBS.
    """
    notional: float
    wac: float                          # decimal, e.g. 0.055
    wam_months: int
    pool_age_months: int = 0
    anchor_tenor: float = DEFAULT_ANCHOR_TENOR
    spread_to_curve: float = DEFAULT_SPREAD_TO_CURVE
    oas: float = DEFAULT_OAS
    base_turnover: float = DEFAULT_BASE_TURNOVER
    max_refi_cpr: float = DEFAULT_MAX_REFI_CPR
    logistic_k: float = DEFAULT_LOGISTIC_K
    logistic_midpoint: float = DEFAULT_LOGISTIC_MIDPOINT
    seasoning_ramp_months: int = DEFAULT_SEASONING_RAMP_MONTHS
    is_whole_loan: bool = False
    hqla_level: str = "level_2a"        # level_1 | level_2a | not_eligible
    nsfr_rsf_factor: float | None = None  # informational override; None → infer
    credit_spread: float = 0.0          # whole-loan placeholder (not in IRRBB path)
    name: str = ""

    def __post_init__(self):
        if self.is_whole_loan:
            self.hqla_level = "not_eligible"
        if self.nsfr_rsf_factor is None:
            if self.is_whole_loan:
                self.nsfr_rsf_factor = 0.65
            elif self.hqla_level == "level_1":
                self.nsfr_rsf_factor = 0.05
            elif self.hqla_level == "level_2a":
                self.nsfr_rsf_factor = 0.15
            else:
                self.nsfr_rsf_factor = 0.85
        # Soft warning: anchor vs WAM consistency
        if self.anchor_tenor > max(self.wam_months / 12.0, 0.25) + 1e-9:
            # leave as-is; callers may log — do not raise
            pass


@dataclass
class MbsPricingResult:
    times: np.ndarray
    cashflows: np.ndarray
    price: float
    wal: float
    mortgage_rate: float
    refi_incentive_pp: float
    seasoned_cpr: float
    psa: float


def step_a_mortgage_rate(
    curve: YieldCurve,
    terms: MbsTerms,
) -> tuple[float, float]:
    """
    Derive mortgage rate from the (already shocked / floored) curve anchor.

    Returns (mortgage_rate, refi_incentive_pp) where incentive is in
    **percentage points** (WAC − mortgage_rate) × 100.
    """
    anchor_rate = float(curve.rate(terms.anchor_tenor))
    mortgage_rate = anchor_rate + float(terms.spread_to_curve)
    refi_incentive_pp = (float(terms.wac) - mortgage_rate) * 100.0
    return mortgage_rate, refi_incentive_pp


def step_b_cpr(
    terms: MbsTerms,
    refi_incentive_pp: float,
    age_months: float,
) -> float:
    """Logistic S-curve × seasoning ramp → annual CPR (fraction)."""
    k = float(terms.logistic_k)
    mid = float(terms.logistic_midpoint)
    refi_response = float(terms.max_refi_cpr) / (
        1.0 + np.exp(-k * (float(refi_incentive_pp) - mid))
    )
    ramp_m = max(int(terms.seasoning_ramp_months), 1)
    seasoning_ramp = min(float(age_months) / ramp_m, 1.0)
    cpr = (float(terms.base_turnover) + refi_response) * seasoning_ramp
    return float(np.clip(cpr, 0.0, 0.999))


def cpr_to_psa(cpr: float) -> float:
    """100 PSA ≡ 6% seasoned CPR."""
    return float(cpr / 0.06) * 100.0


def step_c_flows_and_price(
    curve: YieldCurve,
    terms: MbsTerms,
) -> MbsPricingResult:
    """
    Monthly amortisation with live CPR each month → CF vector → OAS-discounted price.
    """
    mortgage_rate, refi_incentive_pp = step_a_mortgage_rate(curve, terms)
    monthly_coupon = float(terms.wac) / 12.0
    n = max(int(terms.wam_months), 1)
    if monthly_coupon <= 0:
        payment_factor = 1.0 / n
    else:
        payment_factor = monthly_coupon / (1.0 - (1.0 + monthly_coupon) ** (-n))

    balance = float(terms.notional)
    times_list: list[float] = []
    cfs_list: list[float] = []
    eps = 1e-12

    for month in range(1, n + 1):
        if balance <= eps:
            break
        age = float(terms.pool_age_months) + month
        cpr = step_b_cpr(terms, refi_incentive_pp, age)
        smm = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)

        interest = balance * monthly_coupon
        scheduled_prin = balance * payment_factor - interest
        scheduled_prin = max(min(scheduled_prin, balance), 0.0)
        prepayment = (balance - scheduled_prin) * smm
        if month == n:
            prepayment = balance - scheduled_prin
        total_cf = interest + scheduled_prin + prepayment

        t = month / 12.0
        times_list.append(t)
        cfs_list.append(total_cf)
        balance = balance - scheduled_prin - prepayment

    times = np.asarray(times_list, dtype=float)
    cashflows = np.asarray(cfs_list, dtype=float)

    if len(times) == 0:
        return MbsPricingResult(
            times=times, cashflows=cashflows, price=0.0, wal=0.0,
            mortgage_rate=mortgage_rate, refi_incentive_pp=refi_incentive_pp,
            seasoned_cpr=step_b_cpr(terms, refi_incentive_pp, 60),
            psa=cpr_to_psa(step_b_cpr(terms, refi_incentive_pp, 60)),
        )

    disc = curve.rate(times) + float(terms.oas)
    disc = np.maximum(np.asarray(disc, dtype=float), -0.999)
    dfs = 1.0 / np.power(1.0 + disc, times)
    price = float(np.sum(cashflows * dfs))
    wal = float(np.sum(times * cashflows) / np.sum(cashflows)) if cashflows.sum() > 0 else 0.0

    seasoned = step_b_cpr(terms, refi_incentive_pp, 60.0)
    return MbsPricingResult(
        times=times,
        cashflows=cashflows,
        price=price,
        wal=wal,
        mortgage_rate=mortgage_rate,
        refi_incentive_pp=refi_incentive_pp,
        seasoned_cpr=seasoned,
        psa=cpr_to_psa(seasoned),
    )


def bucket_cashflows_from_terms(curve: YieldCurve, terms: MbsTerms) -> np.ndarray:
    """Map Step-C monthly CFs onto the 19 BCBS buckets."""
    res = step_c_flows_and_price(curve, terms)
    out = np.zeros(N_BUCKETS)
    for t, amt in zip(res.times, res.cashflows):
        out[years_to_bucket(float(t))] += float(amt)
    return out


def hqla_from_mbs_level(mbs_level: str, is_whole_loan: bool) -> str:
    """Static HQLA tag — never derived from CPR."""
    if is_whole_loan:
        return "not_eligible"
    lvl = (mbs_level or "").strip().lower()
    if lvl in ("ginnie", "level_1", "level1", "l1"):
        return "level_1"
    if lvl in ("agency", "level_2a", "level2a", "l2a", "fannie", "freddie"):
        return "level_2a"
    return "not_eligible"


def terms_from_instrument(inst, spread_override: float | None = None) -> MbsTerms:
    """Build MbsTerms from an Instrument (or duck-typed object)."""
    itype = getattr(inst, "instrument_type", "")
    is_wl = itype == "whole_loan" or (
        itype == "amortising" and bool(getattr(inst, "prepay_enabled", False))
        and "mbs" not in str(getattr(inst, "name", "")).lower()
    )
    wac = getattr(inst, "wac", None)
    if wac is None:
        wac = float(inst.coupon_pct) / 100.0
    elif float(wac) > 1.0:
        wac = float(wac) / 100.0
    else:
        wac = float(wac)

    wam = getattr(inst, "wam_months", None)
    if wam is None or int(wam) <= 0:
        wam = max(int(round(float(inst.maturity_years) * 12)), 1)
    else:
        wam = int(wam)

    age = int(getattr(inst, "pool_age_months", None) or getattr(inst, "age_months", 0) or 0)
    anchor = float(getattr(inst, "anchor_tenor", None) or DEFAULT_ANCHOR_TENOR)
    spread = spread_override
    if spread is None:
        spread = getattr(inst, "spread_to_curve", None)
    if spread is None:
        spread = DEFAULT_SPREAD_TO_CURVE
    else:
        spread = float(spread)
        if spread > 1.0:  # allow bp input
            spread = spread / 10_000.0

    oas = getattr(inst, "oas", None)
    oas = DEFAULT_OAS if oas is None else float(oas)
    if oas > 1.0:
        oas = oas / 10_000.0

    mbs_level = getattr(inst, "mbs_level", "") or ""
    hqla = getattr(inst, "hqla_level", None) or hqla_from_mbs_level(mbs_level, is_wl)

    return MbsTerms(
        notional=float(inst.notional),
        wac=wac,
        wam_months=wam,
        pool_age_months=age,
        anchor_tenor=anchor,
        spread_to_curve=spread,
        oas=oas,
        base_turnover=float(getattr(inst, "base_turnover", DEFAULT_BASE_TURNOVER) or DEFAULT_BASE_TURNOVER),
        max_refi_cpr=float(getattr(inst, "max_refi_cpr", DEFAULT_MAX_REFI_CPR) or DEFAULT_MAX_REFI_CPR),
        logistic_k=float(getattr(inst, "logistic_k", DEFAULT_LOGISTIC_K) or DEFAULT_LOGISTIC_K),
        logistic_midpoint=float(
            getattr(inst, "logistic_midpoint", DEFAULT_LOGISTIC_MIDPOINT) or DEFAULT_LOGISTIC_MIDPOINT
        ),
        seasoning_ramp_months=int(
            getattr(inst, "seasoning_ramp_months", DEFAULT_SEASONING_RAMP_MONTHS)
            or DEFAULT_SEASONING_RAMP_MONTHS
        ),
        is_whole_loan=is_wl,
        hqla_level=hqla,
        nsfr_rsf_factor=getattr(inst, "nsfr_rsf_factor", None),
        credit_spread=float(getattr(inst, "credit_spread", 0.0) or 0.0),
        name=str(getattr(inst, "name", "")),
    )


def is_option_adjusted_prepayable(inst) -> bool:
    """True when live Steps A/B/C should drive EVE / KR01."""
    itype = getattr(inst, "instrument_type", "")
    if itype in ("mbs", "whole_loan"):
        return True
    if itype == "amortising" and bool(getattr(inst, "prepay_enabled", False)):
        # Explicit opt-out via use_option_adjusted=False
        return bool(getattr(inst, "use_option_adjusted", True))
    return False


def prepayment_diagnostics(
    instruments: Sequence,
    curve: YieldCurve,
    scenario_name: str = "Base",
    spread_override: float | None = None,
) -> pd.DataFrame:
    """Per-position mortgage rate / incentive / seasoned CPR / WAL for reporting."""
    rows = []
    for inst in instruments:
        if not is_option_adjusted_prepayable(inst):
            continue
        terms = terms_from_instrument(inst, spread_override=spread_override)
        res = step_c_flows_and_price(curve, terms)
        sign = "+" if res.refi_incentive_pp > 0 else ""
        rows.append({
            "Scenario": scenario_name,
            "Position": terms.name or getattr(inst, "name", ""),
            "Type": "Whole loan" if terms.is_whole_loan else "MBS",
            "Mortgage rate %": round(res.mortgage_rate * 100.0, 3),
            "Refi incentive (pp)": f"{sign}{res.refi_incentive_pp:.2f}",
            "Seasoned CPR %": round(res.seasoned_cpr * 100.0, 2),
            "PSA": round(res.psa, 1),
            "WAL (years)": round(res.wal, 2),
            "Price ($M)": round(res.price, 2),
        })
    return pd.DataFrame(rows)
