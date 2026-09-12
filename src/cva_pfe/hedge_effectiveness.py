"""
Prospective / progressive hedge-effectiveness testing (ASC 815 style prototype).

Compares fair-value changes of the **hedging instrument** (IRS) vs the
**hedged item** (balance-sheet instrument) under parallel curve shocks.

Regression (offset form for fair-value pay-fixed vs fixed asset):
    ΔHedge = a + b · (−ΔItem)

Pass criteria (configurable defaults aligned with common US practice):
    R² ≥ 0.80, slope b ∈ [0.80, 1.25]

Progressive testing: recompute R²/slope as the shock sample expands
(smallest |bp| first → full grid), so ALCO sees whether effectiveness
holds as the scenario set grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..cashflows import Instrument
from ..key_rate_duration import instrument_pv
from ..yield_curve import YieldCurve
from .irs_pricing import IRSTrade, swap_mtm_m


@dataclass
class EffectivenessResult:
    designation: str
    hedged_item: str
    hedge_label: str
    n_obs: int
    intercept: float
    slope: float
    r_squared: float
    dollar_offset_mean: float
    pass_r2: bool
    pass_slope: bool
    overall_pass: bool
    r2_min: float
    slope_lo: float
    slope_hi: float
    shocks_bp: list[float]
    delta_item_m: list[float]
    delta_hedge_m: list[float]
    progressive: list[dict]


def _bump_curve(curve: YieldCurve, shock_bp: float) -> YieldCurve:
    tenors = list(getattr(curve, "_ref_tenors", []) or [])
    rates = list(getattr(curve, "_ref_rates", []) or [])
    if len(tenors) >= 2:
        return YieldCurve(
            ref_tenors=tenors,
            ref_rates=[r + shock_bp / 10_000.0 for r in rates],
        )
    from copy import deepcopy
    out = deepcopy(curve)
    out.base_rates = np.asarray(curve.base_rates, dtype=float) + shock_bp / 10_000.0
    return out


def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS y = a + b x → intercept, slope, R²."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0, 0.0, 0.0
    b, a = np.polyfit(x, y, 1)
    y_hat = a + b * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    return float(a), float(b), float(max(0.0, min(r2, 1.0)))


def default_shock_grid_bp() -> list[float]:
    """Symmetric parallel shocks for prospective testing (excludes 0)."""
    return [-200.0, -150.0, -100.0, -50.0, -25.0, 25.0, 50.0, 100.0, 150.0, 200.0]


def eligible_hedged_items(
    assets: Sequence[Instrument],
    liabilities: Sequence[Instrument],
) -> list[Instrument]:
    """Instruments commonly eligible as FV / CF hedged items (prototype filter)."""
    out: list[Instrument] = []
    for i in assets:
        if i.instrument_type in (
            "bullet_fixed", "amortising", "mbs", "whole_loan", "bullet_floating",
        ):
            out.append(i)
    for i in liabilities:
        if i.instrument_type in ("bullet_fixed", "bullet_floating", "amortising"):
            out.append(i)
    return out


def prospective_effectiveness(
    curve: YieldCurve,
    hedged_item: Instrument,
    hedge_trades: Sequence[IRSTrade],
    *,
    designation: str = "fair_value",
    shocks_bp: list[float] | None = None,
    r2_min: float = 0.80,
    slope_lo: float = 0.80,
    slope_hi: float = 1.25,
    layer_notional_m: float | None = None,
) -> EffectivenessResult:
    """
    Prospective regression of hedge MtM changes vs hedged-item PV changes.

    For ``fair_value``: regress ΔHedge on (−ΔItem) so perfect offset → slope ≈ 1.
    For ``cash_flow``: same offset form (prototype); cash-flow hedges in practice
    often use hypothetical-derivative method — flagged in UI.

    If ``layer_notional_m`` is set and smaller than the item notional, ΔItem is
    scaled by layer / item notional (portfolio-layer style).
    """
    shocks = list(shocks_bp or default_shock_grid_bp())
    shocks = [s for s in shocks if abs(s) > 1e-9]
    # Progressive order: smallest absolute shocks first
    shocks_prog = sorted(shocks, key=lambda s: abs(s))

    pv0 = instrument_pv(hedged_item, curve)
    h0 = sum(swap_mtm_m(curve, tr) for tr in hedge_trades)
    scale = 1.0
    if layer_notional_m is not None and float(hedged_item.notional) > 1e-9:
        scale = float(np.clip(float(layer_notional_m) / float(hedged_item.notional), 0.0, 1.0))

    d_item: list[float] = []
    d_hedge: list[float] = []
    for bp in shocks_prog:
        c1 = _bump_curve(curve, bp)
        d_item.append((instrument_pv(hedged_item, c1) - pv0) * scale)
        d_hedge.append(sum(swap_mtm_m(c1, tr) for tr in hedge_trades) - h0)

    x_off = -np.asarray(d_item, dtype=float)  # offset form
    y = np.asarray(d_hedge, dtype=float)

    progressive: list[dict] = []
    a = b = r2 = 0.0
    for k in range(2, len(x_off) + 1):
        a, b, r2 = _linreg(x_off[:k], y[:k])
        ratio = []
        for i in range(k):
            if abs(d_item[i]) > 1e-9:
                ratio.append(-d_hedge[i] / d_item[i])
        do = float(np.mean(ratio)) if ratio else float("nan")
        pass_r2 = r2 >= r2_min
        pass_slope = slope_lo <= b <= slope_hi
        progressive.append({
            "n_obs": k,
            "max_|shock|_bp": abs(shocks_prog[k - 1]),
            "intercept": round(a, 6),
            "slope": round(b, 4),
            "R²": round(r2, 4),
            "dollar_offset": round(do, 4) if np.isfinite(do) else None,
            "Pass": "PASS" if (pass_r2 and pass_slope) else "FAIL",
        })

    ratio_all = [
        -d_hedge[i] / d_item[i]
        for i in range(len(d_item))
        if abs(d_item[i]) > 1e-9
    ]
    dollar_offset = float(np.mean(ratio_all)) if ratio_all else float("nan")
    pass_r2 = r2 >= r2_min
    pass_slope = slope_lo <= b <= slope_hi
    hedge_label = " + ".join(tr.label for tr in hedge_trades) if hedge_trades else "—"

    return EffectivenessResult(
        designation=designation,
        hedged_item=str(getattr(hedged_item, "name", "item")),
        hedge_label=hedge_label,
        n_obs=len(d_item),
        intercept=a,
        slope=b,
        r_squared=r2,
        dollar_offset_mean=dollar_offset,
        pass_r2=pass_r2,
        pass_slope=pass_slope,
        overall_pass=pass_r2 and pass_slope,
        r2_min=r2_min,
        slope_lo=slope_lo,
        slope_hi=slope_hi,
        shocks_bp=shocks_prog,
        delta_item_m=d_item,
        delta_hedge_m=d_hedge,
        progressive=progressive,
    )


def effectiveness_scatter_frame(result: EffectivenessResult):
    import pandas as pd

    return pd.DataFrame({
        "Shock (bp)": result.shocks_bp,
        "ΔItem ($M)": np.round(result.delta_item_m, 4),
        "ΔHedge ($M)": np.round(result.delta_hedge_m, 4),
        "−ΔItem ($M)": np.round([-x for x in result.delta_item_m], 4),
    })


def progressive_frame(result: EffectivenessResult):
    import pandas as pd

    return pd.DataFrame(result.progressive)
