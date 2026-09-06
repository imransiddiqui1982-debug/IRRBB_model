"""
cpr_calibration.py
------------------
Option B CPR calibration for community-bank IRRBB.

Maps portfolio WAC and current PMMS (primary mortgage rate) through an
agency-style refinance S-curve to produce BASE + six BCBS scenario CPR %
(and matching PSA %) assumptions.

Historical relevance
--------------------
BCBS does not publish official scenario CPRs. Calibration is defended by:
  1. An S-curve (CPR vs refinance incentive) in the spirit of Fannie/Freddie
     cohort S-curves / Clarity dashboards (illustrative defaults provided).
  2. Mapping each BCBS shock to a mortgage-tenor rate move (~7Y–10Y).
  3. Annotating analogous US rate regimes (e.g. 2020 refi boom, 2022–24 lock-in).

Replace the default S-curve points with your bank's or GSE cohort extract
for exam-ready documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .prepayment import (
    DEFAULT_SHOCK_CPR_PCT,
    psa_cpr,
    s_curve_cpr,
    PrepaymentParams,
)
from .scenarios import SCENARIOS, Scenario

# Mortgage pricing tenor for mapping BCBS shocks onto PMMS (years).
MORTGAGE_SHOCK_TENOR_YEARS = 7.0

# Illustrative agency-style S-curve: refinance incentive (bp) → annual CPR %.
# Incentive = (WAC − PMMS) × 10_000. Positive = in-the-money to refinance.
# Shape mirrors published GSE cohort S-curves (turnover floor → steep refi → saturation).
DEFAULT_AGENCY_SCURVE: tuple[tuple[float, float], ...] = (
    (-200.0, 3.0),   # deep OTM — housing turnover
    (-100.0, 4.5),
    (-50.0, 5.5),
    (0.0, 7.0),      # at-the-money
    (50.0, 14.0),
    (100.0, 25.0),
    (150.0, 35.0),
    (200.0, 42.0),
    (300.0, 48.0),   # deep ITM — saturates
)

# Annotated US regimes for ALCO "historical relevance" narrative.
HISTORICAL_REGIMES: tuple[dict, ...] = (
    {
        "period": "2019 (pre-COVID)",
        "pmms_pct": 3.9,
        "typical_cpr_pct": 12.0,
        "note": "Moderate refi; rates near multi-year lows.",
    },
    {
        "period": "2020–21 refi boom",
        "pmms_pct": 2.8,
        "typical_cpr_pct": 35.0,
        "note": "Parallel-Down analogue — deep ITM, very high CPR.",
    },
    {
        "period": "2022–24 lock-in",
        "pmms_pct": 6.8,
        "typical_cpr_pct": 4.0,
        "note": "Parallel-Up analogue — OTM, CPR near turnover floor.",
    },
    {
        "period": "2025 easing start",
        "pmms_pct": 6.2,
        "typical_cpr_pct": 7.0,
        "note": "Incentive rebuilding; raise CPR vs 2023–24 trough.",
    },
)


@dataclass(frozen=True)
class CprCalibrationInputs:
    """User inputs for Option B CPR calibration."""
    wac_pct: float                 # portfolio weighted-average coupon (%)
    pmms_pct: float                # current 30Y PMMS / offering rate (%)
    age_months: int = 30           # for PSA conversion
    scurve_points: tuple[tuple[float, float], ...] = DEFAULT_AGENCY_SCURVE
    use_logistic_fallback: bool = True  # if points sparse, blend with logistic


def cpr_from_scurve(
    incentive_bp: float,
    scurve_points: Sequence[tuple[float, float]] = DEFAULT_AGENCY_SCURVE,
) -> float:
    """
    Interpolate annual CPR % from incentive (bp) along an S-curve table.

    Extrapolates flat outside the table range.
    """
    pts = sorted((float(x), float(y)) for x, y in scurve_points)
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    x = float(incentive_bp)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(x, xs, ys))


def incentive_bp(wac_pct: float, market_mortgage_pct: float) -> float:
    """Refinance incentive in basis points (WAC − market)."""
    return (float(wac_pct) - float(market_mortgage_pct)) * 100.0


def mortgage_shock_bp(scenario: Scenario, tenor_years: float = MORTGAGE_SHOCK_TENOR_YEARS) -> float:
    """
    BCBS shock (bp) at the mortgage pricing tenor.

    Interpolates the scenario's reference shocks (O/N, 1Y, 2Y, 5Y, 10Y, 20Y).
    """
    from .scenarios import REF_TENORS

    return float(np.interp(tenor_years, REF_TENORS, scenario.ref_shocks_bp))


def cpr_pct_to_psa_pct(cpr_pct: float, age_months: int = 30) -> float:
    """
    Approximate PSA multiple that matches ``cpr_pct`` at the given age.

    At age ≥ 30, 100% PSA = 6% CPR → PSA% = CPR% / 6 × 100.
    """
    age = max(int(age_months), 0)
    base_cpr = psa_cpr(age, psa_percent=100.0) * 100.0  # percent
    if base_cpr <= 1e-9:
        return 0.0
    return float(np.clip(cpr_pct / base_cpr * 100.0, 0.0, 1000.0))


def nearest_historical_regime(implied_pmms_pct: float) -> dict:
    """Pick the annotated regime whose PMMS is closest to the implied rate."""
    best = min(
        HISTORICAL_REGIMES,
        key=lambda r: abs(float(r["pmms_pct"]) - float(implied_pmms_pct)),
    )
    return dict(best)


def calibrate_scenario_cprs(
    inputs: CprCalibrationInputs,
    scenarios: Sequence[Scenario] | None = None,
) -> pd.DataFrame:
    """
    Build BASE + scenario CPR/PSA table from WAC, PMMS, and S-curve.

    Returns columns:
      key, scenario, mortgage_shock_bp, implied_pmms_pct, incentive_bp,
      cpr_pct, psa_pct, historical_analogue, analogue_note
    """
    scenarios = list(scenarios) if scenarios is not None else list(SCENARIOS)
    rows = []

    # BASE
    base_inc = incentive_bp(inputs.wac_pct, inputs.pmms_pct)
    base_cpr = cpr_from_scurve(base_inc, inputs.scurve_points)
    if inputs.use_logistic_fallback:
        # Soft blend with parametric S-curve for smoothness between knots
        log_cpr = s_curve_cpr(base_inc / 10_000.0, PrepaymentParams()) * 100.0
        base_cpr = 0.7 * base_cpr + 0.3 * log_cpr
    base_analogue = nearest_historical_regime(inputs.pmms_pct)
    rows.append({
        "key": "BASE",
        "scenario": "Base (no shock)",
        "mortgage_shock_bp": 0.0,
        "implied_pmms_pct": round(inputs.pmms_pct, 3),
        "incentive_bp": round(base_inc, 1),
        "cpr_pct": round(base_cpr, 2),
        "psa_pct": round(cpr_pct_to_psa_pct(base_cpr, inputs.age_months), 1),
        "historical_analogue": base_analogue["period"],
        "analogue_note": base_analogue["note"],
    })

    for sc in scenarios:
        shock = mortgage_shock_bp(sc)
        implied_pmms = max(inputs.pmms_pct + shock / 100.0, 0.0)
        inc = incentive_bp(inputs.wac_pct, implied_pmms)
        cpr = cpr_from_scurve(inc, inputs.scurve_points)
        if inputs.use_logistic_fallback:
            log_cpr = s_curve_cpr(inc / 10_000.0, PrepaymentParams()) * 100.0
            cpr = 0.7 * cpr + 0.3 * log_cpr
        analogue = nearest_historical_regime(implied_pmms)
        rows.append({
            "key": sc.id,
            "scenario": sc.name,
            "mortgage_shock_bp": round(shock, 1),
            "implied_pmms_pct": round(implied_pmms, 3),
            "incentive_bp": round(inc, 1),
            "cpr_pct": round(float(cpr), 2),
            "psa_pct": round(cpr_pct_to_psa_pct(cpr, inputs.age_months), 1),
            "historical_analogue": analogue["period"],
            "analogue_note": analogue["note"],
        })

    return pd.DataFrame(rows)


def calibration_to_cpr_maps(calib_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Convert calibration DataFrame → session-store CPR % and PSA % maps."""
    cpr_map = {str(r["key"]): float(r["cpr_pct"]) for _, r in calib_df.iterrows()}
    psa_map = {str(r["key"]): float(r["psa_pct"]) for _, r in calib_df.iterrows()}
    # Ensure all default keys exist
    for k, v in DEFAULT_SHOCK_CPR_PCT.items():
        cpr_map.setdefault(k, v)
    from .prepayment import DEFAULT_SHOCK_PSA_PCT
    for k, v in DEFAULT_SHOCK_PSA_PCT.items():
        psa_map.setdefault(k, v)
    return cpr_map, psa_map


def scurve_dataframe(
    scurve_points: Sequence[tuple[float, float]] = DEFAULT_AGENCY_SCURVE,
) -> pd.DataFrame:
    """S-curve table for display / download."""
    return pd.DataFrame(
        [{"incentive_bp": x, "cpr_pct": y} for x, y in scurve_points]
    )


def historical_regimes_dataframe() -> pd.DataFrame:
    """Annotated PMMS / CPR regimes for ALCO documentation."""
    return pd.DataFrame(list(HISTORICAL_REGIMES))
