"""
prepayment.py
-------------
Mortgage Constant Prepayment Rate (CPR) methodology for IRRBB.

Converts annual CPR to periodic Single Monthly Mortality (SMM) and builds
rate-incentive S-curves used by amortising instruments with prepayment.

CPR (annual) → SMM (periodic):
    SMM = 1 − (1 − CPR)^(1 / payment_freq)

Refinance incentive (decimal):
    incentive = coupon_pct/100 − market_mortgage_rate

S-curve CPR (industry standard ALM form):
    CPR = CPR_min + (CPR_max − CPR_min) / (1 + exp(−steepness × (incentive − midpoint)))

Optional Hugging Face Chronos (amazon/chronos-t5-tiny) can refine the base CPR
level from recent prepayment history when ``use_hf_chronos=True`` and
transformers/chronos-forecasting are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Typical US fixed-rate mortgage primary rate premium over the swap / treasury
# curve used as a refinance reference when only the IRRBB discount curve is known.
MORTGAGE_SPREAD_OVER_CURVE = 0.015  # 150 bp

DEFAULT_CPR_FLOOR = 0.02   # 2% CPR when deep out-of-the-money
DEFAULT_CPR_CEILING = 0.45  # 45% CPR when deeply in-the-money
DEFAULT_S_MIDPOINT = 0.005  # 50 bp incentive at curve midpoint
DEFAULT_S_STEEPNESS = 80.0  # dimensionless


@dataclass(frozen=True)
class PrepaymentParams:
    """S-curve CPR parameters (annual CPR fractions)."""
    cpr_floor: float = DEFAULT_CPR_FLOOR
    cpr_ceiling: float = DEFAULT_CPR_CEILING
    midpoint: float = DEFAULT_S_MIDPOINT
    steepness: float = DEFAULT_S_STEEPNESS
    psa_multiplier: float = 1.0  # scale relative to 100% PSA ramp if used
    use_psa_ramp: bool = True
    age_months_at_start: int = 0


def cpr_to_smm(cpr: float, payment_freq: int = 12) -> float:
    """Annual CPR → periodic SMM for ``payment_freq`` payments per year."""
    cpr = float(np.clip(cpr, 0.0, 0.999))
    freq = max(int(payment_freq), 1)
    return 1.0 - (1.0 - cpr) ** (1.0 / freq)


def smm_to_cpr(smm: float, payment_freq: int = 12) -> float:
    """Periodic SMM → annual CPR."""
    smm = float(np.clip(smm, 0.0, 0.999))
    freq = max(int(payment_freq), 1)
    return 1.0 - (1.0 - smm) ** freq


def psa_cpr(age_months: int, psa_percent: float = 100.0) -> float:
    """
    PSA standard ramp: 0.2% CPR per month to month 30, then 6% CPR.
    ``psa_percent`` = 100 → 100% PSA.
    """
    age = max(int(age_months), 0)
    base = min(age, 30) * 0.002
    if age > 30:
        base = 0.06
    return float(np.clip(base * (psa_percent / 100.0), 0.0, 0.999))


def s_curve_cpr(
    incentive: float,
    params: PrepaymentParams | None = None,
) -> float:
    """
    Logistic S-curve CPR from refinance incentive (coupon − market, decimal).

    Positive incentive (coupon > market) → higher CPR (borrowers refinance).
    """
    p = params or PrepaymentParams()
    x = float(incentive)
    denom = 1.0 + np.exp(-p.steepness * (x - p.midpoint))
    cpr = p.cpr_floor + (p.cpr_ceiling - p.cpr_floor) / denom
    return float(np.clip(cpr, 0.0, 0.999))


def mortgage_market_rate_from_curve(
    curve_rates: Sequence[float],
    maturity_years: float,
    bucket_midpoints: Sequence[float] | None = None,
    spread: float = MORTGAGE_SPREAD_OVER_CURVE,
) -> float:
    """
    Approximate primary mortgage rate as interpolated curve rate at
    ``maturity_years`` (capped at 10Y for 30Y book proxies) + mortgage spread.
    """
    from .time_buckets import BUCKET_MIDPOINTS

    mids = list(bucket_midpoints) if bucket_midpoints is not None else list(BUCKET_MIDPOINTS)
    rates = np.asarray(curve_rates, dtype=float)
    if len(rates) != len(mids):
        # Fall back to last available rate
        base = float(rates[-1]) if len(rates) else 0.05
    else:
        tenor = min(float(maturity_years), 10.0)
        base = float(np.interp(tenor, mids, rates))
    return max(base + spread, 0.0)


def incentive_cpr(
    coupon_pct: float,
    market_mortgage_rate: float,
    age_months: int = 0,
    params: PrepaymentParams | None = None,
) -> float:
    """
    Combined PSA seasoning × S-curve CPR.

    If ``use_psa_ramp``, blend PSA age ramp with S-curve so young loans
    do not jump to full S-curve CPR immediately.
    """
    p = params or PrepaymentParams()
    incentive = coupon_pct / 100.0 - float(market_mortgage_rate)
    sc = s_curve_cpr(incentive, p)
    if not p.use_psa_ramp:
        return sc
    psa = psa_cpr(age_months + p.age_months_at_start, psa_percent=100.0 * p.psa_multiplier)
    # Seasoning weight: reach full S-curve weight by month 30
    w = min(max(age_months + p.age_months_at_start, 0), 30) / 30.0
    return float(np.clip((1.0 - w) * psa + w * sc, 0.0, 0.999))


def scenario_market_mortgage_rate(
    base_curve_rates: Sequence[float],
    shocks_bp: Sequence[float] | None,
    maturity_years: float,
    spread: float = MORTGAGE_SPREAD_OVER_CURVE,
) -> float:
    """Market mortgage rate under optional BCBS bucket shocks."""
    rates = np.asarray(base_curve_rates, dtype=float)
    if shocks_bp is not None:
        shocked = np.maximum(rates + np.asarray(shocks_bp, dtype=float) / 10_000.0, 0.0)
    else:
        shocked = rates
    return mortgage_market_rate_from_curve(shocked, maturity_years, spread=spread)


def enhance_cpr_with_hf_chronos(
    historical_cpr: Sequence[float],
    base_cpr: float,
    prediction_length: int = 1,
) -> float:
    """
    Optional Hugging Face Chronos refinement of CPR.

    Uses ``amazon/chronos-t5-tiny`` when ``chronos-forecasting`` / transformers
    are installed. On any failure, returns ``base_cpr`` unchanged.

    Parameters
    ----------
    historical_cpr : recent monthly CPR observations (fractions)
    base_cpr       : S-curve CPR to blend with the forecast
    """
    hist = np.asarray(list(historical_cpr), dtype=float)
    if len(hist) < 8:
        return float(base_cpr)
    try:
        import torch
        from chronos import ChronosPipeline
    except Exception:
        return float(base_cpr)

    try:
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        context = torch.tensor(hist, dtype=torch.float32)
        forecast = pipeline.predict(context, prediction_length=prediction_length)
        # Median of samples
        pred = float(forecast[0].median(dim=0).values[-1].item())
        pred = float(np.clip(pred, 0.0, 0.999))
        # Blend 70% structural S-curve with 30% time-series forecast
        return float(0.7 * base_cpr + 0.3 * pred)
    except Exception:
        return float(base_cpr)


def effective_cpr(
    coupon_pct: float,
    market_mortgage_rate: float,
    age_months: int = 0,
    params: PrepaymentParams | None = None,
    historical_cpr: Sequence[float] | None = None,
    use_hf_chronos: bool = False,
) -> float:
    """Public entry: structural CPR, optionally refined by Chronos."""
    cpr = incentive_cpr(coupon_pct, market_mortgage_rate, age_months, params)
    if use_hf_chronos and historical_cpr is not None:
        cpr = enhance_cpr_with_hf_chronos(historical_cpr, cpr)
    return cpr
