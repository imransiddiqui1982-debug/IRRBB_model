"""
nii_engine.py
=============
Net Interest Income (NII) simulation for the IRRBB balance-sheet engine.

Distinct from EVE:

    EVE:  discount every CF at its own tenor (run-off PV). No horizon.
    NII:  accrue interest month-by-month over a 12-/24-month horizon.
          NO discounting. Constant balance sheet replaces runoff at the
          scenario market rate for remaining months in the horizon.

BCBS 368: parallel ±200 bp, instantaneous, 12m, constant BS (no mandated
NII threshold). US practice adds ±100/300/400 and 12m ramps.

Adapted from docs/incoming_nii/nii_engine.py to use ``Instrument`` +
``YieldCurve`` (not the standalone alm_engine classes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .cashflows import Instrument
from .mbs_pricing import step_c_flows_and_price, terms_from_instrument
from .scenarios import Scenario
from .yield_curve import YieldCurve

__all__ = [
    "BCBS_NII_SCENARIOS",
    "US_NII_SCENARIOS",
    "NIIScenario",
    "NIIResult",
    "NIIPortfolio",
    "project_instrument",
    "nii_scenario_from_eve",
    "parallel_curve",
]


@dataclass(frozen=True)
class NIIScenario:
    name: str
    shock_bp: float
    ramp: bool = False
    ramp_months: int = 12

    def shock_at_month(self, month: int) -> float:
        full = self.shock_bp / 1e4
        if not self.ramp:
            return full
        return full * min(month / max(self.ramp_months, 1), 1.0)


BCBS_NII_SCENARIOS: tuple[NIIScenario, ...] = (
    NIIScenario("par_up_200", +200.0),
    NIIScenario("par_down_200", -200.0),
)

US_NII_SCENARIOS: tuple[NIIScenario, ...] = (
    NIIScenario("shock_100_up", +100.0),
    NIIScenario("shock_100_down", -100.0),
    NIIScenario("shock_200_up", +200.0),
    NIIScenario("shock_200_down", -200.0),
    NIIScenario("shock_300_up", +300.0),
    NIIScenario("shock_300_down", -300.0),
    NIIScenario("shock_400_up", +400.0),
    NIIScenario("shock_400_down", -400.0),
    NIIScenario("ramp_200_up", +200.0, ramp=True),
    NIIScenario("ramp_200_down", -200.0, ramp=True),
)


@dataclass
class MonthlyProjection:
    interest: np.ndarray
    runoff: np.ndarray
    closing_balance: np.ndarray


def parallel_curve(base: YieldCurve, shock_decimal: float) -> YieldCurve:
    """Shift all continuous pillars by the same decimal shock."""
    tenors = list(getattr(base, "_ref_tenors", None) or [0.25, 1, 2, 5, 7, 10, 30])
    rates = list(getattr(base, "_ref_rates", None) or list(base.base_rates))
    if len(rates) != len(tenors):
        rates = [float(base.rate(t)) for t in tenors]
    bumped = [max(float(r) + float(shock_decimal), -0.999) for r in rates]
    return YieldCurve(ref_tenors=tenors, ref_rates=bumped)


def nii_scenario_from_eve(scenario: Scenario) -> NIIScenario:
    """
    Map an EVE ``Scenario`` onto a parallel NII shock.

    Uses the 1Y pillar shock (index 1 in REF_TENORS) so short-rate-heavy
    scenarios still move floaters/NMD funding cost appropriately.
    """
    pillars = list(getattr(scenario, "ref_shocks_bp", None) or [])
    if len(pillars) >= 2:
        bp = float(pillars[1])  # 1Y
    elif scenario.shocks_bp:
        bp = float(np.mean(scenario.shocks_bp))
    else:
        bp = 0.0
    return NIIScenario(name=scenario.id, shock_bp=bp)


def _sign(inst: Instrument) -> int:
    return 1 if str(inst.side).lower() == "asset" else -1


def _coupon_dec(inst: Instrument) -> float:
    return float(inst.coupon_pct) / 100.0


def _current_rate(inst: Instrument) -> float:
    raw = getattr(inst, "current_rate", None)
    if raw is not None and raw != "":
        r = float(raw)
        return r / 100.0 if r > 1.0 else r
    return _coupon_dec(inst)


def _reinvestment_rate(
    base_curve: YieldCurve,
    scenario: NIIScenario,
    month: int,
    anchor_tenor: float,
    spread: float,
) -> float:
    shock = scenario.shock_at_month(month)
    shocked = parallel_curve(base_curve, shock)
    return float(shocked.rate(anchor_tenor)) + float(spread)


def project_instrument(
    inst: Instrument,
    base_curve: YieldCurve,
    scenario: NIIScenario,
    horizon_months: int,
) -> MonthlyProjection:
    itype = inst.instrument_type
    if itype == "bullet_fixed":
        return _project_fixed_bullet(inst, scenario, horizon_months)
    if itype == "amortising" and not getattr(inst, "is_option_adjusted", False):
        return _project_amortising_static(inst, horizon_months)
    if itype in ("mbs", "whole_loan") or (
        itype == "amortising" and getattr(inst, "is_option_adjusted", False)
    ):
        return _project_mbs(inst, base_curve, scenario, horizon_months)
    if itype == "bullet_floating":
        return _project_floater(inst, base_curve, scenario, horizon_months)
    if itype == "demand_deposit":
        return _project_nmd(inst, scenario, horizon_months)
    # Fallback: treat as fixed bullet
    return _project_fixed_bullet(inst, scenario, horizon_months)


def _project_fixed_bullet(
    inst: Instrument,
    scenario: NIIScenario,
    horizon_months: int,
) -> MonthlyProjection:
    del scenario  # fixed coupon until maturity; reinvestment handled by portfolio
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.full(horizon_months, float(inst.notional))
    maturity_month = max(1, int(round(float(inst.maturity_years) * 12)))
    monthly_rate = _coupon_dec(inst) / 12.0
    sgn = _sign(inst)
    for m in range(1, horizon_months + 1):
        if m <= maturity_month:
            interest[m - 1] = sgn * float(inst.notional) * monthly_rate
            closing[m - 1] = float(inst.notional)
        else:
            interest[m - 1] = 0.0
            closing[m - 1] = 0.0
        if m == maturity_month:
            runoff[m - 1] = float(inst.notional)
    return MonthlyProjection(interest, runoff, closing)


def _project_amortising_static(
    inst: Instrument,
    horizon_months: int,
) -> MonthlyProjection:
    """Equal-principal amortising (legacy non-OA path)."""
    n = max(int(round(float(inst.maturity_years) * max(inst.payment_freq, 1))), 1)
    # Monthly view: approximate by spreading contractual schedule
    months_total = max(int(round(float(inst.maturity_years) * 12)), 1)
    sched_m = float(inst.notional) / months_total
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.zeros(horizon_months)
    bal = float(inst.notional)
    coupon = _coupon_dec(inst)
    sgn = _sign(inst)
    for m in range(1, horizon_months + 1):
        if bal <= 1e-12:
            break
        interest[m - 1] = sgn * bal * coupon / 12.0
        prin = min(sched_m, bal)
        runoff[m - 1] = prin
        bal -= prin
        closing[m - 1] = max(bal, 0.0)
    return MonthlyProjection(interest, runoff, closing)


def _project_mbs(
    inst: Instrument,
    base_curve: YieldCurve,
    scenario: NIIScenario,
    horizon_months: int,
) -> MonthlyProjection:
    """Live CPR under shocked curve; interest vs principal from Step C."""
    shocked = parallel_curve(base_curve, scenario.shock_at_month(horizon_months))
    terms = terms_from_instrument(inst)
    res = step_c_flows_and_price(shocked, terms)
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.zeros(horizon_months)
    bal = float(inst.notional)
    monthly_c = float(terms.wac) / 12.0
    sgn = _sign(inst)
    n = min(horizon_months, len(res.cashflows))
    for m in range(n):
        if bal <= 1e-12:
            break
        int_amt = bal * monthly_c
        prin = max(float(res.cashflows[m]) - int_amt, 0.0)
        interest[m] = sgn * int_amt
        runoff[m] = prin
        bal = max(bal - prin, 0.0)
        closing[m] = bal
    return MonthlyProjection(interest, runoff, closing)


def _project_floater(
    inst: Instrument,
    base_curve: YieldCurve,
    scenario: NIIScenario,
    horizon_months: int,
) -> MonthlyProjection:
    reset_years = float(inst.repricing_years or 0.25)
    reset_month = max(1, int(round(reset_years * 12)))
    old_rate = _current_rate(inst)
    if old_rate == 0.0:
        old_rate = float(base_curve.rate(reset_years))
    new_rate = _reinvestment_rate(base_curve, scenario, reset_month, reset_years, 0.0)
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.full(horizon_months, float(inst.notional))
    sgn = _sign(inst)
    for m in range(1, horizon_months + 1):
        rate = old_rate if m <= reset_month else new_rate
        interest[m - 1] = sgn * float(inst.notional) * rate / 12.0
    return MonthlyProjection(interest, runoff, closing)


def _project_nmd(
    inst: Instrument,
    scenario: NIIScenario,
    horizon_months: int,
) -> MonthlyProjection:
    """
    Demand deposits: funding rate moves 1:1 with the parallel shock
    (floored at zero). No beta calibration — keeps NII simple.
    EVE still uses behavioural WAL/slotting separately.
    """
    base_r = _current_rate(inst)
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.full(horizon_months, float(inst.notional))
    sgn = _sign(inst)
    for m in range(1, horizon_months + 1):
        shock = scenario.shock_at_month(m)
        new_rate = max(base_r + shock, 0.0)
        interest[m - 1] = sgn * float(inst.notional) * new_rate / 12.0
    return MonthlyProjection(interest, runoff, closing)


@dataclass
class NIIResult:
    scenario: str
    total_nii: float
    monthly_net_interest: np.ndarray
    by_instrument: dict[str, float]


@dataclass
class NIIPortfolio:
    positions: list[Instrument] = field(default_factory=list)
    base_curve: YieldCurve = field(default_factory=YieldCurve)
    horizon_months: int = 12
    constant_balance_sheet: bool = True
    reinvestment_spread: float = 0.0175
    tier1: float = 100.0

    def add(self, *inst: Instrument) -> "NIIPortfolio":
        self.positions.extend(inst)
        return self

    def run(self, scenario: NIIScenario) -> NIIResult:
        monthly_net = np.zeros(self.horizon_months)
        by_instrument: dict[str, float] = {}

        for inst in self.positions:
            proj = project_instrument(
                inst, self.base_curve, scenario, self.horizon_months
            )
            monthly = proj.interest.copy()

            if self.constant_balance_sheet:
                anchor = float(
                    getattr(inst, "anchor_tenor", None)
                    or getattr(inst, "maturity_years", None)
                    or 5.0
                )
                for m in range(1, self.horizon_months + 1):
                    ro = float(proj.runoff[m - 1])
                    if ro <= 0:
                        continue
                    new_rate = _reinvestment_rate(
                        self.base_curve,
                        scenario,
                        m,
                        anchor,
                        self.reinvestment_spread,
                    )
                    remaining = self.horizon_months - m
                    if remaining <= 0:
                        continue
                    monthly[m:] += _sign(inst) * ro * new_rate / 12.0

            monthly_net += monthly
            by_instrument[inst.name] = float(monthly.sum())

        return NIIResult(
            scenario.name,
            float(monthly_net.sum()),
            monthly_net,
            by_instrument,
        )

    def base_nii(self) -> float:
        return self.run(NIIScenario("base", 0.0)).total_nii

    def grid(
        self, scenarios: tuple[NIIScenario, ...] = BCBS_NII_SCENARIOS
    ) -> dict[str, NIIResult]:
        return {s.name: self.run(s) for s in scenarios}

    def delta_grid(
        self, scenarios: tuple[NIIScenario, ...] = BCBS_NII_SCENARIOS
    ) -> list[dict]:
        base = self.base_nii()
        rows = []
        for s in scenarios:
            r = self.run(s)
            d = r.total_nii - base
            rows.append(
                {
                    "scenario": s.name,
                    "shock_bp": s.shock_bp,
                    "ramp": s.ramp,
                    "nii": r.total_nii,
                    "d_nii": d,
                    "pct_tier1": (d / self.tier1 * 100.0) if self.tier1 else 0.0,
                    "pct_base_nii": (d / base * 100.0) if abs(base) > 1e-9 else 0.0,
                }
            )
        return rows

    def limit_status(
        self,
        scenarios: tuple[NIIScenario, ...] = BCBS_NII_SCENARIOS,
        limit_pct_of_tier1: float | None = 5.0,
        limit_pct_of_base_nii: float | None = None,
    ) -> list[dict]:
        base = self.base_nii()
        rows = []
        for s in scenarios:
            r = self.run(s)
            d = r.total_nii - base
            row: dict = {"scenario": s.name, "d_nii": d, "base_nii": base}
            if limit_pct_of_tier1 is not None and self.tier1:
                pct_t1 = d / self.tier1 * 100.0
                row["pct_tier1"] = pct_t1
                row["status_tier1"] = (
                    "BREACH" if pct_t1 < -limit_pct_of_tier1 else "ok"
                )
            if limit_pct_of_base_nii is not None and abs(base) > 1e-9:
                pct_nii = d / base * 100.0
                row["pct_base_nii"] = pct_nii
                row["status_base_nii"] = (
                    "BREACH" if pct_nii < -limit_pct_of_base_nii else "ok"
                )
            rows.append(row)
        return rows


def portfolio_nii_delta(
    assets: Sequence[Instrument],
    liabilities: Sequence[Instrument],
    curve: YieldCurve,
    scenario: Scenario | NIIScenario,
    *,
    horizon_months: int = 12,
    constant_balance_sheet: bool = True,
    tier1: float = 100.0,
) -> tuple[float, float, float]:
    """
    ΔNII vs base for one scenario, split into asset / liability contributions.

    Returns (delta_total, delta_assets, delta_liabilities).
    """
    if isinstance(scenario, Scenario):
        nii_sc = nii_scenario_from_eve(scenario)
    else:
        nii_sc = scenario

    port = NIIPortfolio(
        positions=list(assets) + list(liabilities),
        base_curve=curve,
        horizon_months=horizon_months,
        constant_balance_sheet=constant_balance_sheet,
        tier1=tier1,
    )
    base = port.run(NIIScenario("base", 0.0))
    shocked = port.run(nii_sc)

    asset_names = {i.name for i in assets}
    liab_names = {i.name for i in liabilities}
    d_a = sum(
        shocked.by_instrument.get(n, 0.0) - base.by_instrument.get(n, 0.0)
        for n in asset_names
    )
    d_l = sum(
        shocked.by_instrument.get(n, 0.0) - base.by_instrument.get(n, 0.0)
        for n in liab_names
    )
    return float(shocked.total_nii - base.total_nii), float(d_a), float(d_l)
