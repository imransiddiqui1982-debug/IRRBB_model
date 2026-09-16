"""
Vanilla SOFR-style fixed-vs-float IRS pricing on a YieldCurve.

Discount factors come from bootstrapped zeros (``curve.df``) when available —
Bloomberg-style single-curve SOFR world.

``pay_freq`` / ``float_pay_freq``:
  0 = at maturity (one cashflow), 1/2/4/12 = annual/semi/quarterly/monthly.

Float leg on a single curve telescopes to DF(0)−DF(T) when un-spread;
with spread, add spread × float annuity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..yield_curve import YieldCurve

PAY_FREQ_LABELS = {
    0: "At maturity",
    1: "Annual",
    2: "Semi",
    4: "Quarterly",
    12: "Monthly",
}


def pay_freq_label(freq: int) -> str:
    return PAY_FREQ_LABELS.get(int(freq), f"{int(freq)}x")


@dataclass(frozen=True)
class IRSTrade:
    """One plain-vanilla IRS. Signed notional: + = pay fixed, − = receive fixed."""

    tenor_years: float
    notional_m: float  # absolute size in $M
    pay_fixed: bool
    fixed_rate: float | None = None  # None → price at par (MtM ≈ 0)
    start_years: float = 0.0
    name: str = ""
    pay_freq: int = 1  # fixed leg: 0 = at maturity
    float_spread_bp: float = 0.0
    float_pay_freq: int = 2  # floating leg frequency

    @property
    def signed_notional_m(self) -> float:
        return self.notional_m if self.pay_fixed else -self.notional_m

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        side = "Pay-fixed" if self.pay_fixed else "Receive-fixed"
        return (
            f"{side} {self.tenor_years:g}Y "
            f"fix={pay_freq_label(self.pay_freq)} "
            f"flt={pay_freq_label(self.float_pay_freq)}"
        )


def discount_factor(curve: YieldCurve, t: float) -> float:
    """DF(t) from bootstrapped zeros when available, else 1/(1+z)^t."""
    if hasattr(curve, "df"):
        return float(curve.df(t))
    t = max(float(t), 1e-8)
    r = float(curve.rate(t))
    return float(1.0 / (1.0 + r) ** t)


def payment_schedule(
    tenor_years: float,
    pay_freq: int,
    start_years: float = 0.0,
) -> tuple[list[float], list[float]]:
    """
    Return (payment_times, year_fractions).

    ``pay_freq == 0`` → single payment at maturity with δ = T (simple accrual).
    """
    T = float(tenor_years)
    t0 = float(start_years)
    freq = int(pay_freq)
    if T <= 0:
        return [], []
    if freq <= 0:
        return [t0 + T], [T]
    n = max(int(round(T * freq)), 1)
    dt = T / n
    times = [t0 + (i + 1) * dt for i in range(n)]
    deltas = [dt] * n
    return times, deltas


def level_annuity(
    curve: YieldCurve,
    tenor_years: float,
    pay_freq: int,
    start_years: float = 0.0,
) -> float:
    """Σ δ_i · DF(t_i) for a payment schedule."""
    times, deltas = payment_schedule(tenor_years, pay_freq, start_years)
    if not times:
        return 0.0
    return float(sum(d * discount_factor(curve, t) for t, d in zip(times, deltas)))


def float_leg_pv_unit(
    curve: YieldCurve,
    tenor_years: float,
    float_pay_freq: int = 2,
    start_years: float = 0.0,
    float_spread_bp: float = 0.0,
) -> float:
    """
    PV of pay-float (unit notional): ≈ DF(t0)−DF(T) + spread·A_float.

    Single-curve telescoping holds for any float schedule without convexity.
    """
    T = float(tenor_years)
    t0 = float(start_years)
    df0 = discount_factor(curve, t0) if t0 > 1e-8 else 1.0
    dfT = discount_factor(curve, t0 + T)
    spread = float(float_spread_bp) / 10_000.0
    a_flt = level_annuity(curve, T, float_pay_freq, t0)
    return float((df0 - dfT) + spread * a_flt)


def par_swap_rate(
    curve: YieldCurve,
    tenor_years: float,
    start_years: float = 0.0,
    pay_freq: int = 1,
    float_pay_freq: int = 2,
    float_spread_bp: float = 0.0,
) -> float:
    """ATM fixed rate so receive-fixed MtM = 0 vs the float leg."""
    T = float(tenor_years)
    t0 = float(start_years)
    if T <= 0:
        return float(curve.rate(t0))
    a_fix = level_annuity(curve, T, pay_freq, t0)
    if a_fix <= 1e-12:
        return float(curve.rate(t0 + T))
    pv_float = float_leg_pv_unit(
        curve, T, float_pay_freq, t0, float_spread_bp=float_spread_bp,
    )
    return float(pv_float / a_fix)


def swap_mtm_m(
    curve: YieldCurve,
    trade: IRSTrade,
    fixed_rate: float | None = None,
) -> float:
    """
    Mark-to-market in $M from the bank's perspective.

    Receive-fixed MtM = N · (K · A_fix − PV_float)
    """
    T = float(trade.tenor_years)
    t0 = float(trade.start_years)
    N = float(trade.notional_m)
    if N <= 0 or T <= 0:
        return 0.0
    freq = int(getattr(trade, "pay_freq", 1) or 0)
    ffreq = int(getattr(trade, "float_pay_freq", 2) or 2)
    spread = float(getattr(trade, "float_spread_bp", 0.0) or 0.0)
    a_fix = level_annuity(curve, T, freq, t0)
    pv_float = float_leg_pv_unit(curve, T, ffreq, t0, float_spread_bp=spread)
    k = float(fixed_rate if fixed_rate is not None else (
        trade.fixed_rate if trade.fixed_rate is not None else par_swap_rate(
            curve, T, t0, pay_freq=freq, float_pay_freq=ffreq, float_spread_bp=spread,
        )
    ))
    recv_fixed_unit = k * a_fix - pv_float
    mtm_unit = -recv_fixed_unit if trade.pay_fixed else recv_fixed_unit
    return float(mtm_unit * N)


def swap_dv01_k(curve: YieldCurve, trade: IRSTrade, bp: float = 1.0) -> float:
    """Parallel DV01 in $K per bp."""
    from copy import deepcopy

    base = swap_mtm_m(curve, trade)
    if hasattr(curve, "shocked_curve"):
        # Uniform parallel on bucket grid → shocked_curve
        from ..time_buckets import N_BUCKETS
        bumped = curve.shocked_curve([bp] * N_BUCKETS)
    else:
        tenors = list(getattr(curve, "_ref_tenors", []) or [])
        rates = list(getattr(curve, "_ref_rates", []) or [])
        if len(tenors) >= 2:
            bumped = YieldCurve(
                ref_tenors=tenors,
                ref_rates=[r + bp / 10_000.0 for r in rates],
            )
            # rebuild DF from zeros
            bumped._df_tenors = tenors
            bumped._dfs = [
                1.0 if t < 1e-12 else float(1.0 / (1.0 + z) ** t)
                for t, z in zip(tenors, bumped._ref_rates)
            ]
        else:
            bumped = deepcopy(curve)
            bumped.base_rates = np.asarray(curve.base_rates, dtype=float) + bp / 10_000.0
    shocked = swap_mtm_m(bumped, trade)
    return float((base - shocked) * 1_000.0)


def trades_from_signed_notionals(
    notionals: dict[float, float],
    curve: YieldCurve,
    use_par: bool = True,
) -> list[IRSTrade]:
    """Convert playbook signed notionals (+ pay-fixed) into IRSTrade list."""
    trades: list[IRSTrade] = []
    for T, signed in sorted(notionals.items()):
        n = float(signed)
        if abs(n) < 1e-9:
            continue
        pay_fixed = n > 0
        k = par_swap_rate(curve, float(T)) if use_par else None
        trades.append(
            IRSTrade(
                tenor_years=float(T),
                notional_m=abs(n),
                pay_fixed=pay_fixed,
                fixed_rate=k,
                name=f"{'Pay' if pay_fixed else 'Recv'}-fixed {float(T):g}Y",
            )
        )
    return trades
