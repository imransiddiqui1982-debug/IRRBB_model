"""
Bloomberg-style manual IRS + hedged-item tickets for hedge accounting.

Supports fixed/float payment frequencies including **at maturity** (one CF),
priced on a bootstrapped zero / DF curve.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..yield_curve import YieldCurve
from .irs_pricing import (
    IRSTrade,
    discount_factor,
    float_leg_pv_unit,
    level_annuity,
    par_swap_rate,
    pay_freq_label,
    swap_dv01_k,
    swap_mtm_m,
)


def par_swap_rate_freq(
    curve: YieldCurve,
    tenor_years: float,
    pay_freq: int = 2,
    start_years: float = 0.0,
    float_pay_freq: int = 2,
    float_spread_bp: float = 0.0,
) -> float:
    """ATM fixed rate; ``pay_freq=0`` = bullet / at-maturity fixed leg."""
    return par_swap_rate(
        curve,
        tenor_years,
        start_years,
        pay_freq=pay_freq,
        float_pay_freq=float_pay_freq,
        float_spread_bp=float_spread_bp,
    )


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
    float_pay_freq: int = 2,
) -> float:
    """MtM ($M) with independent fixed / float schedules."""
    trade = IRSTrade(
        tenor_years=float(tenor_years),
        notional_m=float(notional_m),
        pay_fixed=bool(pay_fixed),
        fixed_rate=float(fixed_rate),
        start_years=float(start_years),
        pay_freq=int(pay_freq),
        float_spread_bp=float(float_spread_bp),
        float_pay_freq=int(float_pay_freq),
    )
    return swap_mtm_m(curve, trade)


def swap_annuity(
    curve: YieldCurve,
    tenor_years: float,
    pay_freq: int = 2,
    start_years: float = 0.0,
) -> float:
    """Fixed-leg level annuity Σ δ·DF (for CVA bp conversion)."""
    return level_annuity(curve, tenor_years, pay_freq, start_years)


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
    cva_charge_bp: float = 0.0
    annuity: float = 0.0
    cva_adjusted_par: float | None = None
    float_pay_freq: int = 2
    curve_method: str = ""


def price_irs_ticket(
    curve: YieldCurve,
    *,
    notional_m: float,
    tenor_years: float,
    pay_fixed: bool,
    pay_freq: int = 2,
    float_pay_freq: int = 2,
    fixed_rate_pct: float | None = None,
    solve_par: bool = True,
    float_spread_bp: float = 0.0,
    cva_charge_bp: float = 0.0,
    start_years: float = 0.0,
    name: str = "",
    curve_method: str = "",
) -> IRSTicketResult:
    """
    Price one IRS ticket on the zero curve.

    Fixed ``pay_freq=0`` (at maturity) vs float ``float_pay_freq=2`` (semi) is the
    classic hedge of a bullet deposit: one fixed CF at T, floating semi resets.
    """
    from .cva import apply_cva_charge_to_fixed

    par = par_swap_rate_freq(
        curve, tenor_years, pay_freq, start_years,
        float_pay_freq=float_pay_freq, float_spread_bp=float_spread_bp,
    )
    annuity = swap_annuity(curve, tenor_years, pay_freq, start_years)
    cva_bp = float(cva_charge_bp or 0.0)
    cva_par = apply_cva_charge_to_fixed(par, pay_fixed=pay_fixed, cva_charge_bp=cva_bp)

    if solve_par or fixed_rate_pct is None:
        k = cva_par if abs(cva_bp) > 1e-12 else par
        at_par = abs(cva_bp) < 1e-12
    else:
        k = float(fixed_rate_pct) / 100.0
        at_par = abs(k - par) < 1e-8 and abs(float_spread_bp) < 1e-9 and abs(cva_bp) < 1e-12

    trade = IRSTrade(
        tenor_years=float(tenor_years),
        notional_m=float(notional_m),
        pay_fixed=bool(pay_fixed),
        fixed_rate=k,
        start_years=float(start_years),
        name=name or (
            f"{'Pay' if pay_fixed else 'Recv'}-fixed {tenor_years:g}Y "
            f"fix={pay_freq_label(pay_freq)} flt={pay_freq_label(float_pay_freq)}"
        ),
        pay_freq=int(pay_freq),
        float_spread_bp=float(float_spread_bp),
        float_pay_freq=int(float_pay_freq),
    )
    mtm = swap_mtm_m(curve, trade)
    dv = swap_dv01_k(curve, trade)
    # Diagnostic: float PV vs fixed annuity on zeros
    pv_flt = float_leg_pv_unit(
        curve, tenor_years, float_pay_freq, start_years, float_spread_bp,
    )
    row = {
        "Trade": trade.label,
        "Side": "Pay-fixed" if pay_fixed else "Receive-fixed",
        "Notional ($M)": round(notional_m, 2),
        "Tenor (Y)": tenor_years,
        "Fixed freq": pay_freq_label(pay_freq),
        "Float freq": pay_freq_label(float_pay_freq),
        "Par % (ex-CVA)": round(par * 100.0, 4),
        "CVA charge (bp)": round(cva_bp, 2),
        "CVA-adj par %": round(cva_par * 100.0, 4),
        "Fixed % used": round(k * 100.0, 4),
        "Float spread (bp)": round(float_spread_bp, 2),
        "Fixed annuity": round(annuity, 4),
        "Float PV (unit)": round(pv_flt, 6),
        "DF(T)": round(discount_factor(curve, float(start_years) + float(tenor_years)), 6),
        "NPV / MtM ($M)": round(mtm, 6),
        "DV01 ($K/bp)": round(dv, 2),
        "At mid (NPV≈0)": "Yes" if at_par and abs(mtm) < 1e-4 else "No",
        "Curve": curve_method or ("zero DF" if getattr(curve, "_dfs", None) else "interp zeros"),
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
        cva_charge_bp=cva_bp,
        annuity=annuity,
        cva_adjusted_par=cva_par,
        float_pay_freq=float_pay_freq,
        curve_method=curve_method,
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
    Build a synthetic BS instrument from manual ticket fields.

    ``payment_freq=0`` → interest + principal paid **at maturity** only (1 CF date).
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
    freq = int(payment_freq)
    if freq < 0:
        freq = 0
    if itype == "amortising" and freq <= 0:
        freq = 2  # amortising needs a period schedule
    inst = Instrument(
        name=name or "Manual hedged item",
        notional=float(notional_m),
        coupon_pct=float(coupon_pct),
        instrument_type=itype,  # type: ignore[arg-type]
        maturity_years=float(maturity_years),
        payment_freq=freq,
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
