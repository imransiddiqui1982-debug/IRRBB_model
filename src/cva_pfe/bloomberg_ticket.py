"""
Bloomberg-style manual IRS + hedged-item tickets for hedge accounting.

User enters notional, side, tenor, pay frequency, fixed rate (or Solve Par),
optional spread; engine prices NPV / DV01 and links to a manual hedged item
for prospective effectiveness testing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..yield_curve import YieldCurve
from .irs_pricing import (
    IRSTrade,
    discount_factor,
    par_swap_rate,
    swap_dv01_k,
    swap_mtm_m,
)


def par_swap_rate_freq(
    curve: YieldCurve,
    tenor_years: float,
    pay_freq: int = 2,
    start_years: float = 0.0,
) -> float:
    """ATM fixed rate with ``pay_freq`` payments per year."""
    T = float(tenor_years)
    t0 = float(start_years)
    freq = max(int(pay_freq), 1)
    if T <= 0:
        return float(curve.rate(t0))
    n = max(int(round(T * freq)), 1)
    dt = T / n
    payment_times = [t0 + (i + 1) * dt for i in range(n)]
    dfs = [discount_factor(curve, t) for t in payment_times]
    df0 = discount_factor(curve, t0) if t0 > 1e-8 else 1.0
    dfT = float(dfs[-1])
    annuity = float(sum(dfs) * dt)
    if annuity <= 1e-12:
        return float(curve.rate(t0 + T))
    return float((df0 - dfT) / annuity)


def swap_mtm_freq_m(
    curve: YieldCurve,
    *,
    notional_m: float,
    tenor_years: float,
    pay_fixed: bool,
    fixed_rate: float,
    pay_freq: int = 2,
    start_years: float = 0.0,
    float_spread_bp: float = 0.0,
) -> float:
    """
    MtM ($M). Float spread shifts effective fixed vs float: approx −spread×annuity
    on the receive-fixed unit value (bank pays float+spread).
    """
    T = float(tenor_years)
    t0 = float(start_years)
    N = float(notional_m)
    freq = max(int(pay_freq), 1)
    if N <= 0 or T <= 0:
        return 0.0
    n = max(int(round(T * freq)), 1)
    dt = T / n
    payment_times = [t0 + (i + 1) * dt for i in range(n)]
    dfs = [discount_factor(curve, t) for t in payment_times]
    df0 = discount_factor(curve, t0) if t0 > 1e-8 else 1.0
    dfT = float(dfs[-1])
    annuity = float(sum(dfs) * dt)
    k = float(fixed_rate)
    spread = float(float_spread_bp) / 10_000.0
    # Receive fixed K, pay float+spread ≈ (K - spread)*A - (DF0-DFT)
    recv_fixed_unit = (k - spread) * annuity - (df0 - dfT)
    mtm_unit = -recv_fixed_unit if pay_fixed else recv_fixed_unit
    return float(mtm_unit * N)


@dataclass
class IRSTicketResult:
    trade: IRSTrade
    par_rate: float
    fixed_rate_used: float
    mtm_m: float
    dv01_k: float
    pay_freq: int
    float_spread_bp: float
    npv_is_par: bool
    summary_row: dict


def price_irs_ticket(
    curve: YieldCurve,
    *,
    notional_m: float,
    tenor_years: float,
    pay_fixed: bool,
    pay_freq: int = 2,
    fixed_rate_pct: float | None = None,
    solve_par: bool = True,
    float_spread_bp: float = 0.0,
    start_years: float = 0.0,
    name: str = "",
) -> IRSTicketResult:
    """
    Price one IRS ticket. If ``solve_par`` (or fixed rate blank), set K = par
    so NPV ≈ 0 at inception (Bloomberg SWPM-style).
    """
    par = par_swap_rate_freq(curve, tenor_years, pay_freq, start_years)
    if solve_par or fixed_rate_pct is None:
        k = par
        at_par = True
    else:
        k = float(fixed_rate_pct) / 100.0
        at_par = abs(k - par) < 1e-8 and abs(float_spread_bp) < 1e-9

    mtm = swap_mtm_freq_m(
        curve,
        notional_m=notional_m,
        tenor_years=tenor_years,
        pay_fixed=pay_fixed,
        fixed_rate=k,
        pay_freq=pay_freq,
        start_years=start_years,
        float_spread_bp=float_spread_bp,
    )
    trade = IRSTrade(
        tenor_years=float(tenor_years),
        notional_m=float(notional_m),
        pay_fixed=bool(pay_fixed),
        fixed_rate=k,
        start_years=float(start_years),
        name=name or (
            f"{'Pay' if pay_fixed else 'Recv'}-fixed {tenor_years:g}Y "
            f"{pay_freq}x"
        ),
        pay_freq=int(pay_freq),
        float_spread_bp=float(float_spread_bp),
    )
    # DV01 via existing annual proxy trade (close enough for ticket screen)
    dv = swap_dv01_k(curve, trade)
    row = {
        "Trade": trade.label,
        "Side": "Pay-fixed" if pay_fixed else "Receive-fixed",
        "Notional ($M)": round(notional_m, 2),
        "Tenor (Y)": tenor_years,
        "Pay freq": pay_freq,
        "Par %": round(par * 100.0, 4),
        "Fixed % used": round(k * 100.0, 4),
        "Float spread (bp)": round(float_spread_bp, 2),
        "NPV / MtM ($M)": round(mtm, 6),
        "DV01 ($K/bp)": round(dv, 2),
        "At par (NPV≈0)": "Yes" if at_par and abs(mtm) < 1e-4 else "No",
    }
    return IRSTicketResult(
        trade=trade,
        par_rate=par,
        fixed_rate_used=k,
        mtm_m=mtm,
        dv01_k=dv,
        pay_freq=pay_freq,
        float_spread_bp=float_spread_bp,
        npv_is_par=bool(at_par and abs(mtm) < 1e-4),
        summary_row=row,
    )


def build_manual_hedged_item(
    *,
    name: str,
    notional_m: float,
    coupon_pct: float,
    maturity_years: float,
    side: str = "asset",
    instrument_type: str = "bullet_fixed",
    payment_freq: int = 2,
    repricing_years: float | None = None,
    credit_spread_bp: float = 0.0,
):
    """
    Build a synthetic BS instrument from manual ticket fields for effectiveness.

    ``credit_spread_bp`` is stored on the instrument for display; discounting
    still uses the shared YieldCurve in the effectiveness engine (prototype).
    """
    from ..cashflows import Instrument

    itype = instrument_type.lower().strip()
    if itype not in (
        "bullet_fixed", "bullet_floating", "amortising", "mbs", "whole_loan",
    ):
        itype = "bullet_fixed"
    rep = float(repricing_years) if repricing_years is not None else float(maturity_years)
    if itype == "bullet_floating":
        rep = float(repricing_years) if repricing_years is not None else min(0.25, float(maturity_years))
    inst = Instrument(
        name=name or "Manual hedged item",
        notional=float(notional_m),
        coupon_pct=float(coupon_pct),
        instrument_type=itype,  # type: ignore[arg-type]
        maturity_years=float(maturity_years),
        payment_freq=max(int(payment_freq), 1),
        repricing_years=rep,
        side="liability" if side.lower().startswith("liab") else "asset",
        credit_spread=float(credit_spread_bp) / 10_000.0,
        use_option_adjusted=False,
        prepay_enabled=False,
    )
    inst.generate_cashflows()
    return inst


def trades_from_tickets(tickets: list[IRSTicketResult]) -> list[IRSTrade]:
    return [t.trade for t in tickets if t.trade.notional_m > 0]
