"""
USD/EUR (and USDSAR scaffold) cross-currency float/float swap pricing.

Primary: rateslib multi-curve (SOFR, €STR, FXForwards, XCS basis).
Fallback: numpy DF + basis-adjusted EUR discount under USD CSA.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .market_data import XccyMarketSnapshot, fetch_xccy_market, tenor_to_years


@dataclass
class XccyTradeSpec:
    pair: str = "EURUSD"
    notional_ccy: str = "EUR"
    notional: float = 10_000_000.0
    tenor: str = "5Y"
    frequency: str = "Q"
    day_count: str = "Act360"
    effective: str | None = None
    mtm: bool = True
    float_spread_bp: float | None = None  # None → use fair basis


@dataclass
class XccyPriceResult:
    fair_basis_bp: float
    npv_usd: float
    npv_foreign: float
    spot: float
    notional_usd: float
    notional_foreign: float
    engine: str
    curves_used: dict[str, str] = field(default_factory=dict)
    cashflows: list[dict] = field(default_factory=list)
    market_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _freq_code(frequency: str) -> str:
    f = (frequency or "Q").strip().upper()
    return {"Q": "Q", "S": "S", "A": "A", "M": "M", "Z": "Z"}.get(f[:1], "Q")


def _dc_code(day_count: str) -> str:
    d = (day_count or "Act360").replace("/", "").replace("-", "").lower()
    if "365" in d:
        return "Act365F"
    if "30" in d:
        return "30E360"
    return "Act360"


def _parse_effective(effective: str | None) -> datetime:
    if effective:
        return datetime.fromisoformat(effective[:10])
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)


def _rateslib_available() -> bool:
    try:
        import rateslib  # noqa: F401
        return True
    except Exception:
        return False


def _rl_nodes(curve: dict[str, float], effective) -> dict:
    """Build rateslib Curve nodes: effective date → DF pillars.

    ``rateslib.dt`` is a plain ``datetime.datetime`` (no ``.add``); advance with
    ``timedelta`` or ``add_tenor`` when available.

    rateslib requires strictly increasing unique node dates — market curves often
    insert short tenors after long ones, so we always sort + dedupe.
    """
    try:
        from rateslib import add_tenor
    except Exception:
        add_tenor = None

    # (date, years, df) — keep years so collisions prefer the longer tenor DF
    pillars: list[tuple] = [(effective, 0.0, 1.0)]
    for lab, rate in curve.items():
        label = str(lab).upper().strip()
        try:
            t = 0.0 if label in ("ON", "O/N", "TN", "SN") else tenor_to_years(label)
        except ValueError:
            continue
        if t <= 1e-12:
            continue
        z = float(rate)
        df = 1.0 / ((1.0 + z) ** t) if t >= 1.0 else 1.0 / (1.0 + z * max(t, 1e-8))
        d = None
        if add_tenor is not None and re.fullmatch(r"\d+[YMW]", label):
            try:
                d = add_tenor(effective, label, "MF", "nyc")
            except Exception:
                d = None
        if d is None:
            d = effective + timedelta(days=int(round(t * 365.25)))
        # Normalise to midnight so date equality is stable across helpers
        if hasattr(d, "replace"):
            d = d.replace(hour=0, minute=0, second=0, microsecond=0)
        pillars.append((d, t, float(max(df, 1e-8))))

    pillars.sort(key=lambda p: (p[0], p[1]))
    nodes: dict = {}
    for d, _t, df in pillars:
        nodes[d] = df  # later/longer tenor wins on exact date collision
    return dict(sorted(nodes.items()))


def price_xccy_rateslib(
    snap: XccyMarketSnapshot,
    trade: XccyTradeSpec,
    *,
    basis_override_bp: dict[str, float] | None = None,
) -> XccyPriceResult:
    from rateslib import Curve, FXForwards, FXRates, Solver, XCS, dt

    warnings: list[str] = []
    eff = _parse_effective(trade.effective)
    effective = dt(eff.year, eff.month, eff.day)
    freq = _freq_code(trade.frequency)
    dc = _dc_code(trade.day_count)
    tenor = trade.tenor.upper()
    basis = dict(basis_override_bp or snap.xccy_basis_bp)

    sofr = Curve(
        nodes=_rl_nodes(snap.usd_curve, effective),
        id="sofr",
        convention=dc,
    )
    estr = Curve(
        nodes=_rl_nodes(snap.eur_curve, effective),
        id="estr",
        convention=dc,
    )
    eurusd = Curve(
        nodes=_rl_nodes(snap.eur_curve, effective),
        id="eurusd",
        convention=dc,
    )

    fxr = FXRates({"eurusd": float(snap.spot)}, settlement=effective)
    fxf = FXForwards(
        fx_rates=fxr,
        fx_curves={"eureur": estr, "usdusd": sofr, "eurusd": eurusd},
    )

    xcs_instruments, xcs_s = [], []
    for lab, bp in sorted(basis.items(), key=lambda kv: tenor_to_years(kv[0])):
        try:
            xcs_instruments.append(
                XCS(
                    effective,
                    lab,
                    spec="eurusd_xcs",
                    curves=["estr", "eurusd", "sofr", "sofr"],
                    frequency=freq,
                )
            )
            xcs_s.append(float(bp))
        except Exception as exc:
            warnings.append(f"Skip XCS {lab}: {exc}")

    if xcs_instruments:
        try:
            Solver(
                curves=[eurusd],
                instruments=xcs_instruments,
                s=xcs_s,
                fx=fxf,
                id="eurusd_xccy",
            )
        except Exception as exc:
            warnings.append(f"XCS basis solver: {exc}")

    n_ccy = (trade.notional_ccy or "EUR").upper()
    notional = float(trade.notional)
    if n_ccy == "USD":
        notional_usd = notional
        notional_eur = notional / float(snap.spot)
        curves = ["sofr", "sofr", "estr", "eurusd"]
        xcs_kwargs = dict(
            effective=effective,
            termination=tenor,
            frequency=freq,
            currency="usd",
            mtm=bool(trade.mtm),
            notional=notional_usd,
            pair="eurusd",
            convention=dc,
        )
    else:
        notional_eur = notional
        notional_usd = notional * float(snap.spot)
        curves = ["estr", "eurusd", "sofr", "sofr"]
        xcs_kwargs = dict(
            effective=effective,
            termination=tenor,
            frequency=freq,
            currency="eur",
            mtm=bool(trade.mtm),
            notional=notional_eur,
            pair="eurusd",
            convention=dc,
        )

    xcs_mid = XCS(**xcs_kwargs)
    fair_bp = float(xcs_mid.rate(curves=curves, fx=fxf))
    spread = trade.float_spread_bp if trade.float_spread_bp is not None else fair_bp
    xcs = XCS(**xcs_kwargs, float_spread=float(spread))
    npv = float(xcs.npv(curves=curves, fx=fxf, base="usd"))
    npv_eur = float(xcs.npv(curves=curves, fx=fxf, base="eur"))

    cfs: list[dict] = []
    try:
        df = xcs.cashflows(curves=curves, fx=fxf)
        cfs = df.reset_index().head(40).to_dict(orient="records")  # type: ignore[union-attr]
    except Exception as exc:
        warnings.append(f"cashflows unavailable: {exc}")

    return XccyPriceResult(
        fair_basis_bp=fair_bp,
        npv_usd=npv,
        npv_foreign=npv_eur,
        spot=float(snap.spot),
        notional_usd=notional_usd,
        notional_foreign=notional_eur,
        engine="rateslib",
        curves_used={
            "usd": "SOFR (live)",
            "eur": "€STR-anchored",
            "eurusd_csa": "calibrated to XCCY basis",
            "fx": "EURUSD spot + CIP FX",
        },
        cashflows=cfs,
        market_notes=list(snap.source_notes),
        warnings=warnings,
        summary={
            "pair": "EURUSD",
            "tenor": tenor,
            "frequency": freq,
            "day_count": dc,
            "mtm": trade.mtm,
            "spread_used_bp": spread,
            "basis_is_live": snap.basis_is_live,
            "forwards_are_cip": snap.forwards_are_cip,
            "as_of": snap.as_of,
        },
    )


def _df_curve(curve: dict[str, float], t: float) -> float:
    pts = []
    for k, v in curve.items():
        tt = 0.0 if k.upper() == "ON" else tenor_to_years(k)
        z = float(v)
        df = 1.0 if tt <= 1e-12 else (
            1.0 / ((1.0 + z) ** tt) if tt >= 1 else 1.0 / (1.0 + z * tt)
        )
        pts.append((tt, df))
    pts = sorted(pts)
    if not pts:
        return float(np.exp(-0.03 * t))
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.log(np.maximum([p[1] for p in pts], 1e-18))
    if t <= xs[0]:
        return float(np.exp(ys[0]))
    if t >= xs[-1]:
        z = -ys[-1] / max(xs[-1], 1e-12)
        return float(np.exp(-z * t))
    return float(np.exp(np.interp(t, xs, ys)))


def price_xccy_numpy(
    snap: XccyMarketSnapshot,
    trade: XccyTradeSpec,
    *,
    basis_override_bp: dict[str, float] | None = None,
) -> XccyPriceResult:
    """
    Float/float XCCY NPV (constant-notional approximation):

    * Forecast EUR from local €STR DFs; discount EUR on USD-CSA curve
      DF_csa(t)=DF_eur(t)·exp(−basis(t)·t)
    * USD leg forecast+discount on SOFR
    * Convert EUR PV with spot (MtM FX resets not fully simulated)
    """
    warnings = [
        "rateslib unavailable — numpy multi-curve XCCY approximation "
        "(forecast €STR / discount USD-CSA; constant notional)."
    ]
    basis = dict(basis_override_bp or snap.xccy_basis_bp)
    T = tenor_to_years(trade.tenor)
    n_per = {"Q": 4, "S": 2, "A": 1, "M": 12}.get(_freq_code(trade.frequency), 4)
    n = max(int(round(T * n_per)), 1)
    dt = T / n
    times = [(i + 1) * dt for i in range(n)]

    def basis_cont(t: float) -> float:
        pts = sorted((tenor_to_years(k), float(v) / 10_000.0) for k, v in basis.items())
        if not pts:
            return 0.0
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if t <= xs[0]:
            return ys[0]
        if t >= xs[-1]:
            return ys[-1]
        return float(np.interp(t, xs, ys))

    def df_usd(t: float) -> float:
        return _df_curve(snap.usd_curve, t)

    def df_eur(t: float) -> float:
        return _df_curve(snap.eur_curve, t)

    def df_csa(t: float) -> float:
        return float(df_eur(t) * np.exp(-basis_cont(t) * t))

    def fwd(df_fn, t0: float, t1: float) -> float:
        d0 = df_fn(t0) if t0 > 1e-12 else 1.0
        d1 = df_fn(t1)
        return float((d0 / d1 - 1.0) / max(t1 - t0, 1e-8))

    spot = float(snap.spot)
    if (trade.notional_ccy or "EUR").upper() == "USD":
        n_usd = float(trade.notional)
        n_eur = n_usd / spot
    else:
        n_eur = float(trade.notional)
        n_usd = n_eur * spot

    def npv_at(spread_bp: float) -> float:
        s = spread_bp / 10_000.0
        # USD pay float + principal
        pv_usd = 0.0
        t_prev = 0.0
        for t in times:
            fv = fwd(df_usd, t_prev, t)
            pv_usd -= n_usd * fv * (t - t_prev) * df_usd(t)
            t_prev = t
        pv_usd -= n_usd * df_usd(T)

        # EUR receive float (estr forecast) + spread + principal, CSA discount
        pv_eur = 0.0
        t_prev = 0.0
        ann = 0.0
        for t in times:
            fv = fwd(df_eur, t_prev, t)
            delta = t - t_prev
            d_c = df_csa(t)
            pv_eur += n_eur * fv * delta * d_c
            ann += delta * d_c
            t_prev = t
        pv_eur += n_eur * s * ann
        pv_eur += n_eur * df_csa(T)

        # Initial exchange marks to zero when N_usd = spot * N_eur
        return float(spot * pv_eur + pv_usd)

    lo, hi = -300.0, 300.0
    f_lo, f_hi = npv_at(lo), npv_at(hi)
    if f_lo * f_hi > 0:
        # No sign change — pick spread minimizing |NPV|
        grid = np.linspace(lo, hi, 61)
        fair_bp = float(min(grid, key=lambda x: abs(npv_at(float(x)))))
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if npv_at(lo) * npv_at(mid) <= 0:
                hi = mid
            else:
                lo = mid
        fair_bp = 0.5 * (lo + hi)

    spread = trade.float_spread_bp if trade.float_spread_bp is not None else fair_bp
    npv = npv_at(spread)

    return XccyPriceResult(
        fair_basis_bp=float(fair_bp),
        npv_usd=float(npv),
        npv_foreign=float(npv / spot),
        spot=spot,
        notional_usd=n_usd,
        notional_foreign=n_eur,
        engine="numpy_fallback",
        curves_used={
            "usd": "SOFR DFs (live)",
            "eur_forecast": "€STR DFs",
            "eurusd_csa": "EUR DF × exp(−basis·t)",
            "fx": "spot + CIP forwards",
        },
        cashflows=[],
        market_notes=list(snap.source_notes),
        warnings=warnings,
        summary={
            "pair": "EURUSD",
            "tenor": trade.tenor,
            "frequency": _freq_code(trade.frequency),
            "day_count": _dc_code(trade.day_count),
            "mtm": trade.mtm,
            "spread_used_bp": spread,
            "basis_is_live": snap.basis_is_live,
            "forwards_are_cip": snap.forwards_are_cip,
            "as_of": snap.as_of,
        },
    )


def price_cross_currency_swap(
    trade: XccyTradeSpec,
    *,
    snap: XccyMarketSnapshot | None = None,
    basis_override_bp: dict[str, float] | None = None,
) -> XccyPriceResult:
    pair = (trade.pair or "EURUSD").upper().replace("/", "")
    if pair in ("USDSAR", "SARUSD"):
        return XccyPriceResult(
            fair_basis_bp=0.0,
            npv_usd=0.0,
            npv_foreign=0.0,
            spot=0.0,
            notional_usd=0.0,
            notional_foreign=float(trade.notional),
            engine="unsupported",
            warnings=[
                "USD/SAR XCCY needs live SAIBOR/SARON + USD/SAR FX swaps + SAR basis. "
                "EURUSD is fully wired; SAR coming next."
            ],
            summary={"pair": pair, "status": "not_implemented"},
        )

    snap = snap or fetch_xccy_market("EURUSD")
    if _rateslib_available():
        try:
            return price_xccy_rateslib(snap, trade, basis_override_bp=basis_override_bp)
        except Exception as exc:
            res = price_xccy_numpy(snap, trade, basis_override_bp=basis_override_bp)
            res.warnings = [f"rateslib failed ({exc}); numpy fallback"] + list(res.warnings)
            return res
    return price_xccy_numpy(snap, trade, basis_override_bp=basis_override_bp)


def interpret_user_request(text: str) -> XccyTradeSpec:
    """Map free-text like '10m EUR 5Y quarterly act/360' → trade spec."""
    import re

    t = (text or "").lower()
    pair = "USDSAR" if "sar" in t else "EURUSD"
    notional = 10_000_000.0
    m = re.search(r"([\d_.]+)\s*(mm|m|million|bn|b)\b", t)
    if m:
        val = float(m.group(1).replace("_", ""))
        unit = m.group(2)
        notional = val * (1e9 if unit in ("bn", "b") else 1e6)
    tenor = "5Y"
    m = re.search(r"(\d+)\s*y(ear)?s?\b", t)
    if m:
        tenor = f"{m.group(1)}Y"
    freq = "Q"
    if "semi" in t:
        freq = "S"
    elif "annual" in t:
        freq = "A"
    elif "month" in t:
        freq = "M"
    dc = "Act365F" if "365" in t else "Act360"
    ccy = "SAR" if pair == "USDSAR" else ("USD" if ("notional" in t and "usd" in t) else "EUR")
    return XccyTradeSpec(
        pair=pair,
        notional_ccy=ccy,
        notional=notional,
        tenor=tenor,
        frequency=freq,
        day_count=dc,
    )
