"""
Basel SA-CCR (BCBS 279) — simplified IR interest-rate hedge-set for plain IRS.

EAD = α × (RC + PFE), α = 1.4
RC  = max(V − C, 0)   (unmargined prototype: C = 0 → max(V, 0))
PFE = multiplier × AddOn_aggregate

IR Add-on: supervisory factor 0.50%, duration ≈ (exp(−0.05×S) − exp(−0.05×E)) / 0.05
Maturity buckets: <1Y, 1–5Y, >5Y with full offset within bucket; 70% across.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .irs_pricing import IRSTrade


SF_IR = 0.005  # 0.50%
ALPHA = 1.4


@dataclass
class SACCRResult:
    replacement_cost_m: float
    addon_m: float
    multiplier: float
    pfe_m: float
    ead_m: float
    v_m: float


def _supervisory_duration(start: float, end: float) -> float:
    s = max(float(start), 0.0)
    e = max(float(end), s + 1e-6)
    return float((np.exp(-0.05 * s) - np.exp(-0.05 * e)) / 0.05)


def _bucket(end_years: float) -> int:
    if end_years < 1.0:
        return 0
    if end_years <= 5.0:
        return 1
    return 2


def saccr_irs_portfolio(
    trades: list[IRSTrade],
    mark_to_market_m: float,
    collateral_m: float = 0.0,
) -> SACCRResult:
    """SA-CCR EAD for a single IR hedge set of vanilla IRS (USD)."""
    v = float(mark_to_market_m)
    c = float(collateral_m)
    rc = max(v - c, 0.0)

    # Effective notional by maturity bucket (signed: pay-fixed positive)
    buckets = np.zeros(3, dtype=float)
    for tr in trades:
        if tr.notional_m <= 0 or tr.tenor_years <= 0:
            continue
        # Trade-level supervisory delta: +1 pay-fixed / −1 receive-fixed
        # (rates up → pay-fixed gains)
        delta = 1.0 if tr.pay_fixed else -1.0
        d = _supervisory_duration(tr.start_years, tr.start_years + tr.tenor_years)
        eff = delta * d * SF_IR * float(tr.notional_m)
        buckets[_bucket(tr.start_years + tr.tenor_years)] += eff

    # Aggregation: full netting in bucket; 70% across buckets (BCBS formula)
    d1, d2, d3 = buckets
    addon = np.sqrt(
        d1 ** 2 + d2 ** 2 + d3 ** 2 + 1.4 * (d1 * d2 + d2 * d3) + 0.6 * d1 * d3
    )
    addon = float(abs(addon))

    # Multiplier floor 0.05
    floor = 0.05
    if addon <= 1e-12:
        mult = 1.0
    else:
        mult = float(min(1.0, floor + (1.0 - floor) * np.exp((v - c) / (2.0 * addon))))
    pfe = mult * addon
    ead = ALPHA * (rc + pfe)
    return SACCRResult(
        replacement_cost_m=rc,
        addon_m=addon,
        multiplier=mult,
        pfe_m=pfe,
        ead_m=ead,
        v_m=v,
    )
