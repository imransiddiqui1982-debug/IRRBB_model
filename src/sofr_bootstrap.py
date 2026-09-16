"""
SOFR / USD IRS → zero-coupon discount curve (Bloomberg-style bootstrap).

Primary path: sequential numpy bootstrap from deposit + par-swap quotes.
Optional path: rateslib Solver when the library imports cleanly (Streamlit Cloud).

Quotes are interpreted as SOFR-style single-curve instruments (post-LIBOR):
the same DF curve discounts and forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .yield_curve import YieldCurve


@dataclass
class ZeroCurveResult:
    curve: YieldCurve
    method: str
    notes: list[str] = field(default_factory=list)
    zero_rates: dict[float, float] = field(default_factory=dict)
    discount_factors: dict[float, float] = field(default_factory=dict)


def _loglin_df(tenors: list[float], dfs: list[float], t: float) -> float:
    """Log-linear DF interpolation (constant continuous forward between pillars)."""
    t = max(float(t), 0.0)
    if t <= 1e-12:
        return 1.0
    x = np.asarray(tenors, dtype=float)
    y = np.log(np.maximum(np.asarray(dfs, dtype=float), 1e-18))
    if t <= x[0]:
        return float(np.exp(y[0]))
    if t >= x[-1]:
        # flat zero beyond last pillar
        z = -y[-1] / max(x[-1], 1e-12)
        return float(np.exp(-z * t))
    return float(np.exp(np.interp(t, x, y)))


def annually_compounded_zero(df: float, t: float) -> float:
    t = max(float(t), 1e-12)
    df = max(float(df), 1e-18)
    return float(df ** (-1.0 / t) - 1.0)


def attach_discount_curve(
    curve: YieldCurve,
    tenors: list[float],
    dfs: list[float],
) -> YieldCurve:
    """Attach DF pillars so ``curve.df(t)`` uses the bootstrapped zeros."""
    from .time_buckets import BUCKET_MIDPOINTS

    order = np.argsort(np.asarray(tenors, dtype=float))
    t = [0.0] + [float(tenors[i]) for i in order if float(tenors[i]) > 1e-12]
    d = [1.0] + [float(dfs[i]) for i in order if float(tenors[i]) > 1e-12]
    tt: list[float] = []
    dd: list[float] = []
    for ti, di in zip(t, d):
        if tt and abs(ti - tt[-1]) < 1e-12:
            tt[-1], dd[-1] = ti, di
        else:
            tt.append(ti)
            dd.append(di)
    curve._df_tenors = tt
    curve._dfs = dd
    zeros = [0.0 if ti < 1e-12 else annually_compounded_zero(di, ti) for ti, di in zip(tt, dd)]
    curve._ref_tenors = list(tt)
    curve._ref_rates = list(zeros)
    curve.base_rates = np.interp(BUCKET_MIDPOINTS, tt, zeros)
    return curve


def bootstrap_numpy_sofr(
    par_quotes: Mapping[float, float],
    *,
    swap_freq: int = 2,
    short_simple_cutoff: float = 1.0,
) -> ZeroCurveResult:
    """
    Sequential bootstrap.

    * Tenors ≤ ``short_simple_cutoff``: simple-money DF = 1 / (1 + r·T)
    * Longer tenors: par SOFR IRS with ``swap_freq`` payments/year
      ``S · Σ δ DF(t_i) = DF(0) − DF(T)`` → solve DF(T)
    """
    freq = max(int(swap_freq), 1)
    quotes = sorted((float(t), float(r)) for t, r in par_quotes.items() if float(t) > 1e-12)
    if not quotes:
        raise ValueError("Need at least one positive-tenor quote to bootstrap")

    notes = [
        f"Numpy SOFR-style bootstrap (swap freq={freq}x; "
        f"simple money ≤ {short_simple_cutoff:g}Y)"
    ]
    tenors = [0.0]
    dfs = [1.0]

    for T, S in quotes:
        S = max(S, -0.01)
        if T <= short_simple_cutoff + 1e-12:
            df_T = 1.0 / (1.0 + S * T)
            tenors.append(T)
            dfs.append(float(df_T))
            continue

        # Swap schedule on [0, T]
        n = max(int(round(T * freq)), 1)
        dt = T / n
        times = [(i + 1) * dt for i in range(n)]
        # Known DFs for times before T; last node is unknown DF(T)
        known_ann = 0.0
        for t in times[:-1]:
            known_ann += dt * _loglin_df(tenors, dfs, t)
        # Equation: S * (known_ann + dt * DF_T) = 1 - DF_T
        # → DF_T * (1 + S*dt) = 1 - S*known_ann
        denom = 1.0 + S * dt
        numer = 1.0 - S * known_ann
        if denom <= 1e-12:
            df_T = _loglin_df(tenors, dfs, T)
        else:
            df_T = max(numer / denom, 1e-12)
        tenors.append(T)
        dfs.append(float(df_T))

    zeros = {
        t: (0.0 if t < 1e-12 else annually_compounded_zero(d, t))
        for t, d in zip(tenors, dfs)
    }
    df_map = {t: d for t, d in zip(tenors, dfs)}
    curve = YieldCurve(ref_tenors=list(tenors), ref_rates=[zeros[t] for t in tenors])
    attach_discount_curve(curve, tenors, dfs)
    return ZeroCurveResult(
        curve=curve,
        method="numpy_bootstrap",
        notes=notes,
        zero_rates=zeros,
        discount_factors=df_map,
    )


def bootstrap_rateslib_sofr(
    par_quotes: Mapping[float, float],
    *,
    swap_freq: int = 1,
    effective=None,
) -> ZeroCurveResult:
    """
    Optional rateslib global solve. Raises if rateslib is unavailable / API mismatch.
    """
    from datetime import datetime as _dt

    from rateslib import Curve, IRS, Solver  # type: ignore
    try:
        from rateslib import dt as rl_dt  # type: ignore
        today = effective or rl_dt(2025, 1, 2)
    except Exception:
        today = effective or _dt(2025, 1, 2)

    quotes = sorted((float(t), float(r)) for t, r in par_quotes.items() if float(t) > 1e-12)
    freq_map = {1: "A", 2: "S", 4: "Q", 12: "M", 0: "Z"}
    freq = freq_map.get(int(swap_freq), "A")

    nodes: dict = {today: 1.0}
    instruments = []
    rates = []
    for T, S in quotes:
        # Prefer tenor string termination ("5Y") when rateslib supports it
        try:
            node_date = today.add(days=int(round(T * 365.25)))  # type: ignore[attr-defined]
        except Exception:
            from datetime import timedelta
            node_date = today + timedelta(days=int(round(T * 365.25)))
        nodes[node_date] = 1.0
        instruments.append(IRS(today, f"{T:.0f}Y" if abs(T - round(T)) < 1e-9 else f"{T}Y", freq, curves="sofr"))
        rates.append(float(S) * 100.0)

    sofr = Curve(nodes=nodes, id="sofr", convention="act365f")
    Solver(curves=[sofr], instruments=instruments, s=rates)

    tenors = [0.0]
    dfs = [1.0]
    for T, _ in quotes:
        try:
            node_date = today.add(days=int(round(T * 365.25)))  # type: ignore[attr-defined]
        except Exception:
            from datetime import timedelta
            node_date = today + timedelta(days=int(round(T * 365.25)))
        tenors.append(T)
        dfs.append(float(sofr[node_date]))

    zeros = {
        t: (0.0 if t < 1e-12 else annually_compounded_zero(d, t))
        for t, d in zip(tenors, dfs)
    }
    curve = YieldCurve(ref_tenors=list(tenors), ref_rates=[zeros[t] for t in tenors])
    attach_discount_curve(curve, tenors, dfs)
    return ZeroCurveResult(
        curve=curve,
        method="rateslib",
        notes=["rateslib Curve + IRS Solver (SOFR-style single curve)"],
        zero_rates=zeros,
        discount_factors={t: d for t, d in zip(tenors, dfs)},
    )


def bootstrap_zero_curve(
    par_quotes: Mapping[float, float],
    *,
    swap_freq: int = 2,
    prefer_rateslib: bool = True,
) -> ZeroCurveResult:
    """Bootstrap zeros; prefer rateslib when available, else numpy."""
    if prefer_rateslib:
        try:
            return bootstrap_rateslib_sofr(par_quotes, swap_freq=max(swap_freq, 1))
        except Exception as exc:
            result = bootstrap_numpy_sofr(par_quotes, swap_freq=swap_freq)
            result.notes.append(f"rateslib unavailable ({exc}); used numpy bootstrap")
            return result
    return bootstrap_numpy_sofr(par_quotes, swap_freq=swap_freq)


def bootstrap_from_market_snapshot(snap, *, swap_freq: int = 2) -> ZeroCurveResult:
    """Build zeros from ``MarketCurveSnapshot.points`` (live SOFR + IRS mids)."""
    points = dict(getattr(snap, "points", {}) or {})
    result = bootstrap_zero_curve(points, swap_freq=swap_freq, prefer_rateslib=True)
    as_of = getattr(snap, "as_of", "")
    if as_of:
        result.notes.insert(0, f"Market as-of {as_of}")
    for n in getattr(snap, "source_notes", []) or []:
        result.notes.append(str(n))
    return result
