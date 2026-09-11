"""
nii_engine.py
=============
Net Interest Income (NII) simulation engine.

Distinct from EVE in mechanism, not just in scope:

    EVE:  discounts every cash flow at its own exact time -> one PV number.
          No horizon; no reinvestment; run-off basis.

    NII:  accrues interest income/expense MONTH BY MONTH over a 12- or
          24-month horizon under a CONSTANT (or dynamic) balance sheet.
          NO DISCOUNTING ANYWHERE. Principal runoff matters not because of
          when it's worth less today, but because it determines how many
          months of the horizon are spent earning the OLD rate versus the
          NEW (reinvestment) rate.

BCBS 368 requirement (the actual Basel standard):
    - Two scenarios only: parallel up, parallel down (both instantaneous)
    - 12-month horizon
    - Constant balance sheet (runoff replaced like-for-like at current
      market rates)
    - Basel itself sets NO mandated NII threshold. (The EU/EBA separately
      mandates 5% of Tier 1 as a "large decline" trigger under CRD/EBA
      guidelines -- that is an EU rule, not a Basel one, and is NOT applied
      here by default.)

US bank practice (NOT a regulatory mandate -- the US has no IRRBB rule at
all; see the 2010 Interagency Advisory and the FFIEC handbook, both
principles-based):
    - More shock magnitudes: +/-100/200/300/400bp
    - BOTH instantaneous shocks AND 12-month linear RAMPS
    - BOTH 12-month AND 24-month horizons
    - BOTH static (no reinvestment) and dynamic (growth) balance sheets
    - Board-set POLICY LIMITS rather than one fixed regulatory number

This module defaults to the BCBS-compliant configuration and exposes a US
preset that turns on the fuller scenario set. Both run on the same engine --
only the scenario list, horizon, and balance-sheet assumption change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from alm_engine import (Curve, Instrument, FixedBullet, FixedAmortising,
                        Floater, NMD, MBS, Swap, bcbs_shock, post_shock_floor)

__all__ = [
    "BCBS_NII_SCENARIOS", "US_NII_SCENARIOS", "NIIScenario",
    "NIIResult", "NIIPortfolio", "project_instrument",
]


# ----------------------------------------------------------- scenario defs --

@dataclass(frozen=True)
class NIIScenario:
    name: str
    shock_bp: float                  # parallel shock magnitude, signed
    ramp: bool = False               # instantaneous (False) vs 12m linear ramp
    ramp_months: int = 12

    def shock_at_month(self, month: int) -> float:
        """Fraction of the full shock in effect at this month (decimal, not bp)."""
        full = self.shock_bp / 1e4
        if not self.ramp:
            return full
        return full * min(month / self.ramp_months, 1.0)


# BCBS 368: exactly these two, instantaneous, 12-month horizon.
BCBS_NII_SCENARIOS: tuple[NIIScenario, ...] = (
    NIIScenario("par_up_200", +200.0),
    NIIScenario("par_down_200", -200.0),
)

# US practice: wider magnitude set, both instantaneous and ramped. Not a
# regulatory requirement -- a common supervisory-adjacent convention because
# an instantaneous 200bp jump has never actually happened; a ramp is the
# more realistic companion scenario US banks show alongside the shock.
US_NII_SCENARIOS: tuple[NIIScenario, ...] = (
    NIIScenario("shock_100_up", +100.0),   NIIScenario("shock_100_down", -100.0),
    NIIScenario("shock_200_up", +200.0),   NIIScenario("shock_200_down", -200.0),
    NIIScenario("shock_300_up", +300.0),   NIIScenario("shock_300_down", -300.0),
    NIIScenario("shock_400_up", +400.0),   NIIScenario("shock_400_down", -400.0),
    NIIScenario("ramp_200_up", +200.0, ramp=True),
    NIIScenario("ramp_200_down", -200.0, ramp=True),
)


# --------------------------------------------------- per-instrument projection --

@dataclass
class MonthlyProjection:
    interest: np.ndarray          # signed: +income (asset) or -expense (liability)
    runoff: np.ndarray            # principal leaving the OLD-rate book this month
    closing_balance: np.ndarray


def _parallel_curve(base_curve: Curve, shock_decimal: float) -> Curve:
    return base_curve.with_shift(lambda t: np.full_like(t, shock_decimal))


def project_instrument(inst: Instrument, base_curve: Curve, scenario: NIIScenario,
                       horizon_months: int) -> MonthlyProjection:
    """Dispatch by instrument type. Every branch returns month-by-month
    (interest, runoff, closing_balance) for months 1..horizon_months.

    Interest is signed from the BANK's perspective directly (positive for
    assets, negative for liabilities) so summing across the whole book
    gives net interest income with no further sign bookkeeping needed.
    """
    if isinstance(inst, (FixedBullet,)):
        return _project_fixed_bullet(inst, base_curve, scenario, horizon_months)
    if isinstance(inst, FixedAmortising):
        return _project_amortising(inst, base_curve, scenario, horizon_months)
    if isinstance(inst, MBS):
        return _project_mbs(inst, base_curve, scenario, horizon_months)
    if isinstance(inst, Floater):
        return _project_floater(inst, base_curve, scenario, horizon_months)
    if isinstance(inst, Swap):
        return _project_swap(inst, base_curve, scenario, horizon_months)
    if isinstance(inst, NMD):
        return _project_nmd(inst, base_curve, scenario, horizon_months)
    raise TypeError(f"no NII projection defined for {type(inst).__name__}")


def _reinvestment_rate(base_curve: Curve, scenario: NIIScenario, month: int,
                       anchor_tenor: float, spread: float) -> float:
    """Rate newly-originated/reinvested balance earns from this month
    onward, under the scenario in effect AT THIS MONTH (matters for ramps)."""
    shock = scenario.shock_at_month(month)
    shocked = _parallel_curve(base_curve, shock)
    return shocked.anchor(anchor_tenor) + spread


def _project_fixed_bullet(inst: FixedBullet, base_curve: Curve,
                          scenario: NIIScenario, horizon_months: int) -> MonthlyProjection:
    """Fixed rate for its whole life. If it matures inside the horizon,
    the notional runs off and (under constant balance sheet -- reinvestment
    is applied by the caller, not here) simply stops earning at the old rate.
    A bullet that does NOT mature inside the horizon contributes THE SAME
    income in every scenario -- this is the key sanity property tested in
    test_nii_engine.py."""
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.full(horizon_months, inst.notional)
    maturity_month = int(round(inst.years * 12))
    monthly_rate = inst.coupon / 12.0
    for m in range(1, horizon_months + 1):
        if m <= maturity_month:
            interest[m-1] = inst.side * inst.notional * monthly_rate
            closing[m-1] = inst.notional
        else:
            interest[m-1] = 0.0
            closing[m-1] = 0.0
        if m == maturity_month:
            runoff[m-1] = inst.notional
    return MonthlyProjection(interest, runoff, closing)


def _project_amortising(inst: FixedAmortising, base_curve: Curve,
                        scenario: NIIScenario, horizon_months: int) -> MonthlyProjection:
    t, interest_full, principal_full = inst.flows_split(base_curve)
    n = min(horizon_months, len(t))
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.zeros(horizon_months)
    bal = inst.notional
    for m in range(n):
        interest[m] = inst.side * interest_full[m]
        runoff[m] = principal_full[m]
        bal -= principal_full[m]
        closing[m] = max(bal, 0.0)
    return MonthlyProjection(interest, runoff, closing)


def _project_mbs(inst: MBS, base_curve: Curve, scenario: NIIScenario,
                 horizon_months: int) -> MonthlyProjection:
    """Uses the SHOCKED curve to regenerate the schedule -- prepayment speed
    itself responds to the scenario (same live-CPR principle as EVE), which
    affects how fast principal runs off and gets reinvested within the
    horizon."""
    shocked = _parallel_curve(base_curve, scenario.shock_at_month(horizon_months))
    t, interest_full, principal_full = inst.flows_split(shocked)
    n = min(horizon_months, len(t))
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.zeros(horizon_months)
    bal = inst.notional
    for m in range(n):
        interest[m] = inst.side * interest_full[m]
        runoff[m] = principal_full[m]
        bal -= principal_full[m]
        closing[m] = max(bal, 0.0)
    return MonthlyProjection(interest, runoff, closing)


def _project_floater(inst: Floater, base_curve: Curve, scenario: NIIScenario,
                     horizon_months: int) -> MonthlyProjection:
    """Earns `current_rate` (fixed at the last reset) until `next_reset`,
    then reprices ONCE to the scenario's rate at that tenor and holds for
    the remainder of the horizon. (The engine's Floater has a single
    next_reset field, not a repeating schedule -- this matches that design;
    a recurring-reset instrument would need a richer class.)"""
    reset_month = max(1, int(round(inst.next_reset * 12)))
    old_rate = inst.current_rate if inst.current_rate is not None \
        else base_curve.anchor(inst.next_reset)
    new_rate = _reinvestment_rate(base_curve, scenario, reset_month,
                                  inst.next_reset, 0.0)
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.full(horizon_months, inst.notional)
    for m in range(1, horizon_months + 1):
        rate = old_rate if m <= reset_month else new_rate
        interest[m-1] = inst.side * inst.notional * rate / 12.0
    return MonthlyProjection(interest, runoff, closing)


def _project_swap(inst: Swap, base_curve: Curve, scenario: NIIScenario,
                  horizon_months: int) -> MonthlyProjection:
    """Pay-fixed (side=+1): -fixed leg + floating leg. Receive-fixed
    (side=-1): the reverse. Floating leg reprices once at next_reset, same
    convention as Floater -- the reset month itself still accrues at the
    OLD rate; the new rate takes effect the month after."""
    reset_month = max(1, int(round(inst.next_reset * 12)))
    old_float = inst.current_rate if inst.current_rate is not None \
        else base_curve.anchor(inst.next_reset)
    new_float = _reinvestment_rate(base_curve, scenario, reset_month,
                                   inst.next_reset, 0.0)
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.full(horizon_months, inst.notional)
    for m in range(1, horizon_months + 1):
        flt = old_float if m <= reset_month else new_float
        fixed_leg = inst.notional * inst.fixed_rate / 12.0
        float_leg = inst.notional * flt / 12.0
        interest[m-1] = inst.side * (float_leg - fixed_leg)
    return MonthlyProjection(interest, runoff, closing)


def _project_nmd(inst: NMD, base_curve: Curve, scenario: NIIScenario,
                 horizon_months: int) -> MonthlyProjection:
    """This is where BETA belongs -- NOT decay/WAL (those drive the EVE
    slotting only). Deposit rate reprices toward
        base_rate + beta * market_shock
    using beta on the way up and beta_down on the way down (asymmetric;
    banks typically cut deposit rates faster/further than they raise them).
    Floored at zero -- a deposit rate cannot go negative in a down shock,
    which is exactly what makes down-scenarios asymmetric and NII-painful
    in practice (funding cost can't fall as fast as asset yield does)."""
    beta_dn = inst.beta_down if inst.beta_down is not None else inst.beta
    interest = np.zeros(horizon_months)
    runoff = np.zeros(horizon_months)
    closing = np.full(horizon_months, inst.notional)
    for m in range(1, horizon_months + 1):
        shock = scenario.shock_at_month(m)
        beta = inst.beta if shock >= 0 else beta_dn
        new_rate = max(inst.current_rate + beta * shock, 0.0)
        interest[m-1] = inst.side * inst.notional * new_rate / 12.0
    return MonthlyProjection(interest, runoff, closing)


# ------------------------------------------------------------------ portfolio --

@dataclass
class NIIResult:
    scenario: str
    total_nii: float
    monthly_net_interest: np.ndarray
    by_instrument: dict[str, float]


@dataclass
class NIIPortfolio:
    positions: list[Instrument] = field(default_factory=list)
    base_curve: Curve = field(default_factory=lambda: Curve.flat(0.04))
    horizon_months: int = 12                 # 12 = BCBS; US banks also run 24
    constant_balance_sheet: bool = True       # BCBS requirement; False = static
    reinvestment_spread: float = 0.0175       # spread over the curve anchor
                                                # used to price REPLACEMENT
                                                # balance under constant BS
    tier1: float = 100.0

    def add(self, *inst: Instrument) -> "NIIPortfolio":
        self.positions.extend(inst)
        return self

    def run(self, scenario: NIIScenario) -> NIIResult:
        monthly_net = np.zeros(self.horizon_months)
        by_instrument: dict[str, float] = {}

        for inst in self.positions:
            proj = project_instrument(inst, self.base_curve, scenario,
                                      self.horizon_months)
            monthly = proj.interest.copy()

            if self.constant_balance_sheet:
                # Reinvest each month's runoff at the CURRENT (scenario, as
                # of that month -- matters for ramps) rate, for whatever
                # months remain in the horizon. This is the mechanism that
                # makes NII sensitive to WHEN principal comes back, without
                # any discounting: later runoff means fewer months at the
                # new rate before the horizon ends.
                anchor = getattr(inst, "anchor_tenor", None) or \
                         getattr(inst, "years", None) or 5.0
                for m in range(1, self.horizon_months + 1):
                    ro = proj.runoff[m-1]
                    if ro <= 0:
                        continue
                    new_rate = _reinvestment_rate(self.base_curve, scenario, m,
                                                  anchor, self.reinvestment_spread)
                    remaining_months = self.horizon_months - m
                    if remaining_months <= 0:
                        continue
                    sign = inst.side
                    monthly[m:] += sign * ro * new_rate / 12.0

            monthly_net += monthly
            by_instrument[inst.name] = float(monthly.sum())

        return NIIResult(scenario.name, float(monthly_net.sum()), monthly_net,
                         by_instrument)

    def base_nii(self) -> float:
        base_scenario = NIIScenario("base", 0.0)
        return self.run(base_scenario).total_nii

    def grid(self, scenarios: tuple[NIIScenario, ...] = BCBS_NII_SCENARIOS
            ) -> dict[str, NIIResult]:
        return {s.name: self.run(s) for s in scenarios}

    def limit_status(self, scenarios: tuple[NIIScenario, ...] = BCBS_NII_SCENARIOS,
                     limit_pct_of_tier1: float | None = 5.0,
                     limit_pct_of_base_nii: float | None = None) -> list[dict]:
        """Reports delta against whichever denominator(s) are supplied.

        Basel sets no NII threshold. The EU/EBA's 5% of Tier 1 is shown as
        the default ONLY because it is the most commonly cited external
        reference point, not because it is a Basel requirement -- state
        this explicitly in any report this feeds. `limit_pct_of_base_nii`
        is the more honest earnings-at-risk view and is often materially
        more binding than the Tier-1-relative figure, exactly as demonstrated
        by the worked ±200bp example earlier in this build (−17.4% of base
        NII vs only −2.0% of Tier 1 for the identical dollar move)."""
        base = self.base_nii()
        rows = []
        for s in scenarios:
            r = self.run(s)
            d = r.total_nii - base
            row = {"scenario": s.name, "d_nii": d, "base_nii": base}
            if limit_pct_of_tier1 is not None:
                pct_t1 = d / self.tier1 * 100.0
                row["pct_tier1"] = pct_t1
                row["status_tier1"] = "BREACH" if pct_t1 < -limit_pct_of_tier1 else "ok"
            if limit_pct_of_base_nii is not None and abs(base) > 1e-9:
                pct_nii = d / base * 100.0
                row["pct_base_nii"] = pct_nii
                row["status_base_nii"] = "BREACH" if pct_nii < -limit_pct_of_base_nii else "ok"
            rows.append(row)
        return rows
