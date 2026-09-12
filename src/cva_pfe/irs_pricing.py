"""
Vanilla SOFR-style fixed-vs-float IRS pricing on a YieldCurve.

Convention (prototype): annual payments, Act/365.25 simple compounding via
curve zero rates. Float leg ≈ par floater (PV ≈ 1 − DF(T) on unit notional).
Positive MtM = receive-fixed is in-the-money (bank receives fixed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..yield_curve import YieldCurve


@dataclass(frozen=True)
class IRSTrade:
    """One plain-vanilla IRS. Signed notional: + = pay fixed, − = receive fixed."""

    tenor_years: float
    notional_m: float  # absolute size in $M
    pay_fixed: bool
    fixed_rate: float | None = None  # None → price at par (MtM ≈ 0)
    start_years: float = 0.0
    name: str = ""

    @property
    def signed_notional_m(self) -> float:
        return self.notional_m if self.pay_fixed else -self.notional_m

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        side = "Pay-fixed" if self.pay_fixed else "Receive-fixed"
        return f"{side} {self.tenor_years:g}Y"


def discount_factor(curve: YieldCurve, t: float) -> float:
    t = max(float(t), 1e-8)
    r = float(curve.rate(t))
    return float(1.0 / (1.0 + r) ** t)


def par_swap_rate(curve: YieldCurve, tenor_years: float, start_years: float = 0.0) -> float:
    """ATM fixed rate for an annual-pay swap."""
    T = float(tenor_years)
    t0 = float(start_years)
    if T <= 0:
        return float(curve.rate(t0))
    n = max(int(round(T)), 1)
    payment_times = [t0 + (i + 1) * (T / n) for i in range(n)]
    dfs = np.array([discount_factor(curve, t) for t in payment_times], dtype=float)
    df0 = discount_factor(curve, t0) if t0 > 1e-8 else 1.0
    dfT = float(dfs[-1])
    annuity = float(np.sum(dfs) * (T / n))
    if annuity <= 1e-12:
        return float(curve.rate(t0 + T))
    # Par rate from (DF_start - DF_end) / annuity
    return float((df0 - dfT) / annuity)


def swap_mtm_m(
    curve: YieldCurve,
    trade: IRSTrade,
    fixed_rate: float | None = None,
) -> float:
    """
    Mark-to-market in $M from the bank's perspective.

    Receive-fixed MtM = N * (K * A - (DF0 - DFT))
    Pay-fixed MtM = − that amount.
    """
    T = float(trade.tenor_years)
    t0 = float(trade.start_years)
    N = float(trade.notional_m)
    if N <= 0 or T <= 0:
        return 0.0
    n = max(int(round(T)), 1)
    dt = T / n
    payment_times = [t0 + (i + 1) * dt for i in range(n)]
    dfs = np.array([discount_factor(curve, t) for t in payment_times], dtype=float)
    df0 = discount_factor(curve, t0) if t0 > 1e-8 else 1.0
    dfT = float(dfs[-1])
    annuity = float(np.sum(dfs) * dt)
    k = float(fixed_rate if fixed_rate is not None else (
        trade.fixed_rate if trade.fixed_rate is not None else par_swap_rate(curve, T, t0)
    ))
    # Value of receive-fixed vs pay-float on unit notional ($ of notional = 1)
    recv_fixed_unit = k * annuity - (df0 - dfT)
    mtm_unit = -recv_fixed_unit if trade.pay_fixed else recv_fixed_unit
    return float(mtm_unit * N)


def swap_dv01_k(curve: YieldCurve, trade: IRSTrade, bp: float = 1.0) -> float:
    """Parallel DV01 in $K per bp (loss if rates +1bp → positive when pay-fixed)."""
    from copy import deepcopy

    base = swap_mtm_m(curve, trade)
    # Bump continuous pillars if present
    tenors = list(getattr(curve, "_ref_tenors", []) or [])
    rates = list(getattr(curve, "_ref_rates", []) or [])
    if len(tenors) >= 2:
        bumped = YieldCurve(
            ref_tenors=tenors,
            ref_rates=[r + bp / 10_000.0 for r in rates],
        )
    else:
        bumped = deepcopy(curve)
        bumped.base_rates = np.asarray(curve.base_rates, dtype=float) + bp / 10_000.0
    shocked = swap_mtm_m(bumped, trade)
    # $M loss × 1000 → $K
    return float((base - shocked) * 1_000.0)


def trades_from_signed_notionals(
    notionals: dict[float, float],
    curve: YieldCurve,
    use_par: bool = True,
) -> list[IRSTrade]:
    """
    Convert playbook signed notionals (+ pay-fixed) into IRSTrade list.
    Fixed rate = par on current curve when use_par (ATM hedges → MtM ≈ 0).
    """
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
