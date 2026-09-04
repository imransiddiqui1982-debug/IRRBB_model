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

User / UI overrides
-------------------
Banks often supply scenario CPR or PSA speeds directly (ALCO assumptions).
``DEFAULT_SHOCK_CPR_PCT`` and ``ShockCprTable`` hold base + six BCBS scenario
annual CPR fractions used for mortgages and MBS in EVE/NII.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# Typical US fixed-rate mortgage primary rate premium over the swap / treasury
# curve used as a refinance reference when only the IRRBB discount curve is known.
MORTGAGE_SPREAD_OVER_CURVE = 0.015  # 150 bp

DEFAULT_CPR_FLOOR = 0.02   # 2% CPR when deep out-of-the-money
DEFAULT_CPR_CEILING = 0.45  # 45% CPR when deeply in-the-money
DEFAULT_S_MIDPOINT = 0.005  # 50 bp incentive at curve midpoint
DEFAULT_S_STEEPNESS = 80.0  # dimensionless

# Default ALCO-style CPR (%) by environment — used when the UI supplies overrides.
# Rates up → slower prepay; rates down → faster prepay (negative convexity).
DEFAULT_SHOCK_CPR_PCT: dict[str, float] = {
    "BASE": 6.0,          # ~100% PSA terminal
    "PS_UP": 3.0,         # extension
    "PS_DOWN": 25.0,      # refinance wave
    "STEEPENER": 8.0,
    "FLATTENER": 5.0,
    "SHORT_UP": 4.0,
    "SHORT_DOWN": 18.0,
}


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


def psa_percent_to_cpr(psa_percent: float, age_months: int = 30) -> float:
    """Convert a PSA multiple (%) into an annual CPR at the given loan age."""
    return psa_cpr(age_months, psa_percent=float(psa_percent))


def cpr_pct_to_fraction(cpr_pct: float) -> float:
    """UI percent (e.g. 6.0) → annual CPR fraction (0.06)."""
    v = float(cpr_pct)
    if v > 1.0:
        v = v / 100.0
    return float(np.clip(v, 0.0, 0.999))


@dataclass
class ShockCprTable:
    """
    User-supplied annual CPR by environment for mortgages / MBS.

    Keys: ``BASE`` plus BCBS scenario ids (``PS_UP``, ``PS_DOWN``, …).
    Values are annual CPR **fractions** (0.06 = 6%).
    """
    cpr_by_key: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_pct_map(cls, pct_map: Mapping[str, float] | None = None) -> "ShockCprTable":
        src = dict(DEFAULT_SHOCK_CPR_PCT)
        if pct_map:
            src.update({str(k): float(v) for k, v in pct_map.items()})
        return cls(cpr_by_key={k: cpr_pct_to_fraction(v) for k, v in src.items()})

    @classmethod
    def from_psa_map(
        cls,
        psa_map: Mapping[str, float] | None = None,
        age_months: int = 30,
    ) -> "ShockCprTable":
        """Build from PSA % multiples (100 = 100% PSA → 6% CPR when seasoned)."""
        src = {
            "BASE": 100.0,
            "PS_UP": 50.0,
            "PS_DOWN": 400.0,
            "STEEPENER": 125.0,
            "FLATTENER": 80.0,
            "SHORT_UP": 60.0,
            "SHORT_DOWN": 300.0,
        }
        if psa_map:
            src.update({str(k): float(v) for k, v in psa_map.items()})
        return cls(
            cpr_by_key={
                k: psa_percent_to_cpr(v, age_months=age_months) for k, v in src.items()
            }
        )

    def base_cpr(self) -> float:
        return float(self.cpr_by_key.get("BASE", cpr_pct_to_fraction(DEFAULT_SHOCK_CPR_PCT["BASE"])))

    def cpr_for_scenario(self, scenario_id: str) -> float:
        if scenario_id in self.cpr_by_key:
            return float(self.cpr_by_key[scenario_id])
        return self.base_cpr()

    def as_pct_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        for key, frac in self.cpr_by_key.items():
            rows.append({
                "Environment": key,
                "CPR (%)": round(frac * 100.0, 2),
                "PSA (% of std)": round(frac / 0.06 * 100.0, 1) if frac > 0 else 0.0,
            })
        return pd.DataFrame(rows)


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
