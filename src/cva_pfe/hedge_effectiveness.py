"""
Prospective / progressive hedge-effectiveness testing (ASC 815 style prototype).

Fair value
----------
Regress ΔHedge MtM on (−ΔItem PV) under parallel curve shocks.
Typical: pay-fixed IRS vs fixed-rate asset.

Cash flow (hypothetical derivative)
-----------------------------------
ASC 815-style: build a **hypothetical derivative** that would perfectly hedge
the item's interest variability (matching notional, tenor, pay frequency, side),
then regress ΔActualHedge on ΔHypotheticalDerivative.

Typical: receive-fixed / pay-float IRS vs floating-rate **asset**
         (or pay-fixed vs floating-rate **liability**).

Matching critical terms → R²≈1, slope≈1 → PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..yield_curve import YieldCurve
from .bloomberg_ticket import par_swap_rate_freq, swap_mtm_freq_m
from .irs_pricing import IRSTrade, discount_factor

if TYPE_CHECKING:
    from ..cashflows import Instrument


def _instrument_pv(instrument: Instrument, curve: YieldCurve) -> float:
    from ..key_rate_duration import instrument_pv
    return float(instrument_pv(instrument, curve))


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
    method: str = "fair_value_pv"
    method_note: str = ""
    critical_terms: dict = field(default_factory=dict)


def _bump_curve(curve: YieldCurve, shock_bp: float) -> YieldCurve:
    tenors = list(getattr(curve, "_ref_tenors", []) or [])
    rates = list(getattr(curve, "_ref_rates", []) or [])
    if len(tenors) >= 2:
        out = YieldCurve(
            ref_tenors=tenors,
            ref_rates=[r + shock_bp / 10_000.0 for r in rates],
        )
        out._df_tenors = tenors
        out._dfs = [
            1.0 if t < 1e-12 else float(1.0 / (1.0 + z) ** t)
            for t, z in zip(tenors, out._ref_rates)
        ]
        return out
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


def _trade_pay_freq(tr: IRSTrade) -> int:
    return int(getattr(tr, "pay_freq", 1) or 0)


def _trade_float_freq(tr: IRSTrade) -> int:
    return int(getattr(tr, "float_pay_freq", 2) or 2)


def _hedge_mtm(curve: YieldCurve, trades: Sequence[IRSTrade]) -> float:
    total = 0.0
    for tr in trades:
        freq = _trade_pay_freq(tr)
        ffreq = _trade_float_freq(tr)
        k = float(tr.fixed_rate) if tr.fixed_rate is not None else par_swap_rate_freq(
            curve, tr.tenor_years, freq, tr.start_years,
            float_pay_freq=ffreq,
            float_spread_bp=float(getattr(tr, "float_spread_bp", 0.0) or 0.0),
        )
        total += swap_mtm_freq_m(
            curve,
            notional_m=tr.notional_m,
            tenor_years=tr.tenor_years,
            pay_fixed=tr.pay_fixed,
            fixed_rate=k,
            pay_freq=freq,
            start_years=tr.start_years,
            float_spread_bp=float(getattr(tr, "float_spread_bp", 0.0) or 0.0),
            float_pay_freq=ffreq,
        )
    return float(total)


def _simple_forward(curve: YieldCurve, t0: float, t1: float) -> float:
    """Simply compounded forward from zero curve."""
    t0 = max(float(t0), 0.0)
    t1 = max(float(t1), t0 + 1e-8)
    df0 = 1.0 if t0 <= 1e-10 else discount_factor(curve, t0)
    df1 = discount_factor(curve, t1)
    dt = t1 - t0
    return float((df0 / df1 - 1.0) / dt)


def interest_cashflow_pv(instrument: Instrument, curve: YieldCurve) -> float:
    """
    PV of remaining **interest** cash flows (no principal) — CF-hedge risk measure.

    Floating: each period uses simply compounded forward × notional × dt.
    Fixed / other: contractual coupon on notional until maturity.
    ``payment_freq <= 0``: single interest accrual at maturity.
    """
    from .irs_pricing import discount_factor as _df

    N = float(instrument.notional)
    T = float(instrument.maturity_years)
    freq = int(instrument.payment_freq or 0)
    if N <= 0 or T <= 0:
        return 0.0
    itype = str(instrument.instrument_type)
    if freq <= 0:
        if itype == "bullet_floating":
            rate = _simple_forward(curve, 0.0, T)
        else:
            rate = float(instrument.coupon_pct) / 100.0
        return float(N * rate * T * _df(curve, T))

    n = max(int(round(T * freq)), 1)
    dt = T / n
    total = 0.0
    for i in range(n):
        t0 = i * dt
        t1 = (i + 1) * dt
        if itype == "bullet_floating":
            rate = _simple_forward(curve, t0, t1)
        else:
            rate = float(instrument.coupon_pct) / 100.0
        cf = N * rate * dt
        total += cf * _df(curve, t1)
    return float(total)


def build_hypothetical_derivative(
    curve: YieldCurve,
    hedged_item: Instrument,
    *,
    layer_notional_m: float | None = None,
) -> IRSTrade:
    """
    Perfect CF hedge IRS for the item (critical-terms match).

    * Floating **asset** → receive-fixed / pay-float (lock asset yield)
    * Floating **liability** → pay-fixed / receive-float (lock funding cost)
    * Fixed asset/liability → pay-fixed / receive-fixed respectively as FV-style HD fallback

    Item ``payment_freq=0`` (at maturity) → HD fixed leg also at maturity;
    float leg defaults to semi.
    """
    N = float(layer_notional_m) if layer_notional_m and layer_notional_m > 0 else float(hedged_item.notional)
    T = float(hedged_item.maturity_years)
    freq = int(hedged_item.payment_freq or 0)
    side = str(hedged_item.side).lower()
    itype = str(hedged_item.instrument_type)

    if itype == "bullet_floating" and side == "asset":
        pay_fixed = False
    elif itype == "bullet_floating" and side == "liability":
        pay_fixed = True
    elif side == "asset":
        pay_fixed = True
    else:
        pay_fixed = False

    float_freq = 2
    par = par_swap_rate_freq(curve, T, freq, float_pay_freq=float_freq)
    return IRSTrade(
        tenor_years=T,
        notional_m=abs(N),
        pay_fixed=pay_fixed,
        fixed_rate=par,
        start_years=0.0,
        name=f"HD {('Pay' if pay_fixed else 'Recv')}-fixed {T:g}Y",
        pay_freq=freq,
        float_pay_freq=float_freq,
    )


def _critical_terms_check(
    hedged_item: Instrument,
    hedge_trades: Sequence[IRSTrade],
    hd: IRSTrade,
) -> dict:
    if not hedge_trades:
        return {"match": False, "notes": "No hedge trades"}
    # Aggregate hedge as primary ticket (first / largest)
    tr = max(hedge_trades, key=lambda t: t.notional_m)
    notes = []
    match = True
    if abs(tr.notional_m - hd.notional_m) > 0.05 * max(hd.notional_m, 1e-6):
        match = False
        notes.append(
            f"Notional mismatch: hedge {tr.notional_m:.1f} vs HD/item {hd.notional_m:.1f}"
        )
    if abs(tr.tenor_years - hd.tenor_years) > 0.05:
        match = False
        notes.append(f"Tenor mismatch: hedge {tr.tenor_years:g}Y vs item {hd.tenor_years:g}Y")
    if tr.pay_fixed != hd.pay_fixed:
        match = False
        notes.append(
            f"Side mismatch: hedge {'pay' if tr.pay_fixed else 'recv'}-fixed vs "
            f"HD {'pay' if hd.pay_fixed else 'recv'}-fixed"
        )
    hf = _trade_pay_freq(tr)
    if hf != hd.pay_freq:
        match = False
        from .irs_pricing import pay_freq_label
        notes.append(
            f"Pay frequency mismatch: hedge {pay_freq_label(hf)} vs item/HD {pay_freq_label(hd.pay_freq)}"
        )    if match:
        notes.append("Critical terms aligned with hypothetical derivative.")
    return {
        "match": match,
        "hedge_side": "pay_fixed" if tr.pay_fixed else "receive_fixed",
        "hd_side": "pay_fixed" if hd.pay_fixed else "receive_fixed",
        "hedge_notional": tr.notional_m,
        "hd_notional": hd.notional_m,
        "hedge_tenor": tr.tenor_years,
        "hd_tenor": hd.tenor_years,
        "hedge_freq": hf,
        "hd_freq": hd.pay_freq,
        "notes": "; ".join(notes),
    }


def _finalize(
    *,
    designation: str,
    method: str,
    method_note: str,
    hedged_item: Instrument,
    hedge_trades: Sequence[IRSTrade],
    shocks_prog: list[float],
    d_x: list[float],
    d_y: list[float],
    r2_min: float,
    slope_lo: float,
    slope_hi: float,
    critical_terms: dict | None = None,
    dollar_offset_from_neg_x: bool = True,
) -> EffectivenessResult:
    x = np.asarray(d_x, dtype=float)
    y = np.asarray(d_y, dtype=float)

    progressive: list[dict] = []
    a = b = r2 = 0.0
    for k in range(2, len(x) + 1):
        a, b, r2 = _linreg(x[:k], y[:k])
        ratio = []
        for i in range(k):
            if abs(d_x[i]) > 1e-9:
                ratio.append(d_y[i] / d_x[i])
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

    ratio_all = [d_y[i] / d_x[i] for i in range(len(d_x)) if abs(d_x[i]) > 1e-9]
    dollar_offset = float(np.mean(ratio_all)) if ratio_all else float("nan")
    # For FV offset form, historical dollar_offset used −ΔItem; keep similar display
    if dollar_offset_from_neg_x:
        pass
    pass_r2 = r2 >= r2_min
    pass_slope = slope_lo <= b <= slope_hi
    hedge_label = " + ".join(tr.label for tr in hedge_trades) if hedge_trades else "—"

    return EffectivenessResult(
        designation=designation,
        hedged_item=str(getattr(hedged_item, "name", "item")),
        hedge_label=hedge_label,
        n_obs=len(d_x),
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
        delta_item_m=list(d_x),
        delta_hedge_m=list(d_y),
        progressive=progressive,
        method=method,
        method_note=method_note,
        critical_terms=critical_terms or {},
    )


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
    Prospective effectiveness under parallel curve shocks.

    * ``fair_value``: ΔHedge vs (−ΔItem PV)
    * ``cash_flow``: ΔActualHedge vs ΔHypotheticalDerivative (ASC 815 HD method)
    """
    shocks = list(shocks_bp or default_shock_grid_bp())
    shocks = [s for s in shocks if abs(s) > 1e-9]
    shocks_prog = sorted(shocks, key=lambda s: abs(s))
    desig = (designation or "fair_value").lower().strip()

    if desig == "cash_flow":
        return _cash_flow_hd_effectiveness(
            curve, hedged_item, hedge_trades,
            shocks_prog=shocks_prog,
            r2_min=r2_min, slope_lo=slope_lo, slope_hi=slope_hi,
            layer_notional_m=layer_notional_m,
        )

    # ── Fair value (PV offset) ───────────────────────────────────────────────
    pv0 = _instrument_pv(hedged_item, curve)
    h0 = _hedge_mtm(curve, hedge_trades)
    scale = 1.0
    if layer_notional_m is not None and float(hedged_item.notional) > 1e-9:
        scale = float(np.clip(float(layer_notional_m) / float(hedged_item.notional), 0.0, 1.0))

    d_item: list[float] = []
    d_hedge: list[float] = []
    for bp in shocks_prog:
        c1 = _bump_curve(curve, bp)
        d_item.append((_instrument_pv(hedged_item, c1) - pv0) * scale)
        d_hedge.append(_hedge_mtm(c1, hedge_trades) - h0)

    x_off = [-x for x in d_item]
    return _finalize(
        designation="fair_value",
        method="fair_value_pv",
        method_note=(
            "FV: regress ΔHedge on (−ΔItem PV). Best for pay-fixed vs fixed-rate asset. "
            "Floating assets have near-zero PV duration — use cash_flow instead."
        ),
        hedged_item=hedged_item,
        hedge_trades=hedge_trades,
        shocks_prog=shocks_prog,
        d_x=x_off,
        d_y=d_hedge,
        r2_min=r2_min,
        slope_lo=slope_lo,
        slope_hi=slope_hi,
        dollar_offset_from_neg_x=True,
    )


def _cash_flow_hd_effectiveness(
    curve: YieldCurve,
    hedged_item: Instrument,
    hedge_trades: Sequence[IRSTrade],
    *,
    shocks_prog: list[float],
    r2_min: float,
    slope_lo: float,
    slope_hi: float,
    layer_notional_m: float | None,
) -> EffectivenessResult:
    hd = build_hypothetical_derivative(
        curve, hedged_item, layer_notional_m=layer_notional_m,
    )
    terms = _critical_terms_check(hedged_item, hedge_trades, hd)

    h0 = _hedge_mtm(curve, hedge_trades)
    hd0 = _hedge_mtm(curve, [hd])

    d_hd: list[float] = []
    d_act: list[float] = []
    for bp in shocks_prog:
        c1 = _bump_curve(curve, bp)
        d_hd.append(_hedge_mtm(c1, [hd]) - hd0)
        d_act.append(_hedge_mtm(c1, hedge_trades) - h0)

    note = (
        "CF / hypothetical-derivative: regress ΔActualHedge on ΔHD. "
        "HD matches item notional, tenor, pay frequency and hedge side "
        "(recv-fixed vs floating asset; pay-fixed vs floating liability). "
        + str(terms.get("notes", ""))
    )
    return _finalize(
        designation="cash_flow",
        method="cash_flow_hypothetical_derivative",
        method_note=note,
        hedged_item=hedged_item,
        hedge_trades=hedge_trades,
        shocks_prog=shocks_prog,
        d_x=d_hd,
        d_y=d_act,
        r2_min=r2_min,
        slope_lo=slope_lo,
        slope_hi=slope_hi,
        critical_terms=terms,
        dollar_offset_from_neg_x=False,
    )


def effectiveness_scatter_frame(result: EffectivenessResult):
    import pandas as pd

    if result.method == "cash_flow_hypothetical_derivative":
        return pd.DataFrame({
            "Shock (bp)": result.shocks_bp,
            "ΔHD ($M)": np.round(result.delta_item_m, 4),
            "ΔActual hedge ($M)": np.round(result.delta_hedge_m, 4),
        })
    # FV: regression X was (−ΔItem PV), stored in delta_item_m
    return pd.DataFrame({
        "Shock (bp)": result.shocks_bp,
        "−ΔItem PV ($M)": np.round(result.delta_item_m, 4),
        "ΔHedge ($M)": np.round(result.delta_hedge_m, 4),
    })


def progressive_frame(result: EffectivenessResult):
    import pandas as pd

    return pd.DataFrame(result.progressive)
