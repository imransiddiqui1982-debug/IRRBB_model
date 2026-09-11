"""
alm_engine
==========
Banking-book interest rate risk engine.

Pipeline:

    balance sheet  ->  fine monthly cash flows
                   ->  EVE under six BCBS scenarios
                   ->  KR01 aggregated to a TRADEABLE key rate grid
                   ->  hedge solved against that grid
                   ->  each trade tagged with its hedge-accounting designation

Design rules baked in:
  * Cash flows are generated monthly. The key rate grid is separate and coarse,
    set to standard swap tenors, because you can only hedge what you can trade.
  * Behavioural models (prepayment, NMD decay) re-run INSIDE every curve bump.
    Freezing them overstates KR01 by roughly 25% on a mortgage book.
  * Fair value hedges -> P&L with an offset.  Cash flow hedges -> OCI.
    The engine tags which applies and flags items that cannot be designated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "Curve", "SCENARIOS", "bcbs_shock", "post_shock_floor",
    "Instrument", "FixedBullet", "FixedAmortising", "Floater", "NMD", "MBS",
    "Swap", "Portfolio", "KeyRateGrid",
    "HedgeSolution", "solve_hedge", "designate",
]

# ---------------------------------------------------------------- curves ---

SCENARIOS = ("par_up", "par_down", "short_up", "short_down",
             "steepener", "flattener")


def post_shock_floor(rate: np.ndarray, t: np.ndarray) -> np.ndarray:
    """BCBS floor: -100bp at the overnight point, rising 5bp/yr to 0% at 50y."""
    return np.maximum(rate, np.minimum(-0.01 + 0.0005 * t, 0.0))


def bcbs_shock(t: np.ndarray, scenario: str, *, r_par: float = 0.02,
               r_short: float = 0.03, r_long: float = 0.015) -> np.ndarray:
    """BCBS 368 formula-based shocks.  Defaults are the USD calibration.

    Scenario magnitudes are ARGUMENTS, never constants: the Committee
    recalibrates, and other currencies differ.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    s = np.exp(-t / 4.0)
    table = {
        "par_up":     np.full_like(t, r_par),
        "par_down":   np.full_like(t, -r_par),
        "short_up":   r_short * s,
        "short_down": -r_short * s,
        "steepener":  -0.65 * r_short * s + 0.90 * r_long * (1.0 - s),
        "flattener":  0.80 * r_short * s - 0.60 * r_long * (1.0 - s),
    }
    if scenario not in table:
        raise KeyError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    return table[scenario]


@dataclass(frozen=True)
class Curve:
    """Zero curve defined at pillars, log-linear in discount factor between."""
    pillars: np.ndarray
    zeros: np.ndarray
    shift: Callable[[np.ndarray], np.ndarray] | None = None

    @classmethod
    def flat(cls, level: float, out_to: float = 40.0) -> "Curve":
        p = np.array([0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, out_to])
        return cls(p, np.full_like(p, float(level)))

    def rate(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        r = np.interp(t, self.pillars, self.zeros)
        if self.shift is not None:
            r = r + np.atleast_1d(self.shift(t))
        return post_shock_floor(r, t)

    def df(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return 1.0 / (1.0 + self.rate(t)) ** t

    def with_shift(self, fn: Callable[[np.ndarray], np.ndarray]) -> "Curve":
        """Compose an additional shift on top of any existing one."""
        base = self.shift

        def combined(t):
            t = np.atleast_1d(np.asarray(t, dtype=float))
            out = np.atleast_1d(fn(t)).astype(float)
            if base is not None:
                out = out + np.atleast_1d(base(t))
            return out

        return Curve(self.pillars, self.zeros, combined)

    def anchor(self, tenor: float) -> float:
        """Rate at a single tenor -- used to drive behavioural models."""
        return float(self.rate(np.array([tenor]))[0])


# ----------------------------------------------------------- instruments ---

@dataclass
class Instrument:
    """Base class. `flows(curve)` returns (times in years, cash amounts).

    Subclasses whose schedule depends on rates read what they need off the
    curve, which is how behavioural models stay live inside a bump.
    """
    name: str
    notional: float
    side: int = 1                    # +1 asset, -1 liability
    rate_type: str = "fixed"         # "fixed" | "floating" -- drives designation

    def flows(self, curve: Curve) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def pv(self, curve: Curve) -> float:
        t, c = self.flows(curve)
        if len(t) == 0:
            return 0.0
        return float(self.side * np.sum(c * curve.df(t)))


@dataclass
class FixedBullet(Instrument):
    """Fixed coupon, principal at maturity."""
    coupon: float = 0.0
    years: float = 1.0
    freq: int = 2

    def flows(self, curve):
        n = int(round(self.years * self.freq))
        t = np.arange(1, n + 1) / self.freq
        c = np.full(n, self.notional * self.coupon / self.freq)
        c[-1] += self.notional
        return t, c


@dataclass
class FixedAmortising(Instrument):
    """Level-payment amortising loan."""
    coupon: float = 0.0
    years: float = 1.0
    freq: int = 12

    def flows(self, curve):
        n = int(round(self.years * self.freq))
        r = self.coupon / self.freq
        pmt = self.notional * (r / (1 - (1 + r) ** -n)) if r else self.notional / n
        t = np.arange(1, n + 1) / self.freq
        return t, np.full(n, pmt)

    def flows_split(self, curve):
        """Same schedule as flows(), but interest and scheduled principal
        kept separate -- needed for NII, which accrues interest on the
        OUTSTANDING BALANCE rather than discounting a combined cash flow."""
        n = int(round(self.years * self.freq))
        r = self.coupon / self.freq
        pmt = self.notional * (r / (1 - (1 + r) ** -n)) if r else self.notional / n
        t = np.arange(1, n + 1) / self.freq
        bal = self.notional
        interest, principal = [], []
        for _ in range(n):
            i_ = bal * r
            p_ = min(pmt - i_, bal) if r else pmt
            interest.append(i_)
            principal.append(p_)
            bal -= p_
        return t, np.array(interest), np.array(principal)


@dataclass
class Floater(Instrument):
    """Reprices to par at the next reset -- risk stops there, not at maturity.

    The coupon for the current period was fixed at the PRIOR reset, so the
    cash flow is known: notional grossed up at `current_rate`.  That is what
    gives a floater a duration equal to the time to its next reset rather than
    zero.  `current_rate` is snapshotted from the base curve by Portfolio and
    must NOT move when the curve is bumped.
    """
    next_reset: float = 0.25
    current_rate: float | None = None
    rate_type: str = "floating"

    def _fixing(self, curve: Curve) -> float:
        return self.current_rate if self.current_rate is not None \
            else curve.anchor(self.next_reset)

    def flows(self, curve):
        r = self._fixing(curve)
        cf = self.notional * (1.0 + r) ** self.next_reset
        return np.array([self.next_reset]), np.array([cf])


@dataclass
class NMD(Instrument):
    """Non-maturity deposit.

    Only the CORE portion is slotted beyond overnight; non-core and non-stable
    sit at the first bucket.  Core is spread by exponential decay, truncated
    and rescaled at the BCBS segment cap.

        core = balance x stable_pct x (1 - beta),  capped at core_cap

    `beta` (the up-cycle pass-through) drives the CORE/EVE split above.
    `current_rate` and `beta_down` exist for NII purposes only (see
    nii_engine.py) -- NII cares about how fast the deposit RATE reprices,
    which is a different question from how the BALANCE is slotted for EVE.
    Using decay/WAL for NII, or beta for EVE slotting, is the most common
    modelling error in NMD work; keeping the fields separate is deliberate.
    """
    stable_pct: float = 0.9
    beta: float = 0.35
    core_cap: float = 0.70           # BCBS segment maximum
    wal_cap: float = 4.5             # BCBS maximum average maturity, years
    decay: float = 0.30              # lambda; WAL = 1/lambda
    months: int = 120
    side: int = -1
    current_rate: float = 0.02       # NII only: today's deposit rate
    beta_down: float | None = None   # NII only: down-cycle beta; defaults to
                                      # `beta` if not set explicitly (banks
                                      # typically cut faster than they raise,
                                      # so beta_down is usually > beta)

    @property
    def core(self) -> float:
        raw = self.notional * self.stable_pct * (1.0 - self.beta)
        return min(raw, self.notional * self.core_cap)

    @property
    def wal(self) -> float:
        return min(1.0 / self.decay, self.wal_cap)

    def flows(self, curve):
        t = np.arange(1, self.months + 1) / 12.0
        lam = 1.0 / self.wal                      # respect the cap
        w = np.exp(-lam * (t - 1 / 12)) - np.exp(-lam * t)
        w = w / w.sum()
        c = self.core * w
        c[0] += self.notional - self.core         # non-core to the front bucket
        return t, c


@dataclass
class MBS(Instrument):
    """Mortgage pool with a prepayment S-curve.

    CPR is recomputed from the curve on every call, so the option stays live
    inside a KR01 bump.  The mortgage rate is read at `anchor_tenor`, which is
    why a bump at 3m barely moves prepayments and a bump at 10y moves them a lot.
    """
    wac: float = 0.055
    months: int = 360
    spread_to_curve: float = 0.0175   # mortgage rate = curve(anchor) + spread
    anchor_tenor: float = 7.0
    oas: float = 0.005
    base_cpr: float = 0.06
    max_refi: float = 0.34
    steepness: float = 2.2
    threshold: float = 0.60           # refi incentive, percentage points

    def cpr(self, incentive_pp: float, age_m: int) -> float:
        refi = self.max_refi / (1 + np.exp(-self.steepness * (incentive_pp - self.threshold)))
        return (self.base_cpr + refi) * min(age_m / 30.0, 1.0)

    def flows(self, curve):
        mtg = curve.anchor(self.anchor_tenor) + self.spread_to_curve
        inc = (self.wac - mtg) * 100.0
        bal, r = float(self.notional), self.wac / 12.0
        # BUGFIX: the dollar payment must be FIXED at the start, computed
        # once from the starting balance and remaining term, then held
        # constant every month thereafter. Recomputing `bal * factor` each
        # month (against the DECLINING balance) produces a geometric decay
        # that asymptotically approaches zero but never actually reaches
        # it -- a $300m/5.5%/324-months-remaining pool computed the wrong
        # way still had ~$153m outstanding after 500 months of "payments".
        # Real mortgage payments are constant in dollar terms; only the
        # interest/principal SPLIT changes month to month.
        payment_dollar = bal * r / (1 - (1 + r) ** -self.months)
        ts, cs = [], []
        for m in range(1, self.months + 1):
            if bal <= 1e-12:
                break
            smm = 1 - (1 - self.cpr(inc, m)) ** (1 / 12)
            interest = bal * r
            sched = min(payment_dollar - interest, bal)     # fixed payment, clipped at payoff
            prepay = max((bal - sched) * smm, 0.0)
            ts.append(m / 12.0)
            cs.append(interest + sched + prepay)
            bal -= sched + prepay
        return np.array(ts), np.array(cs)

    def flows_split(self, curve):
        """Same schedule and same corrected payment mechanics as flows(),
        but interest kept separate from total principal (scheduled +
        prepaid). Needed for NII, which accrues interest on the OUTSTANDING
        BALANCE rather than discounting a single combined cash flow.
        Deliberately duplicated rather than refactored out of flows() to
        avoid touching the method 39 existing tests depend on."""
        mtg = curve.anchor(self.anchor_tenor) + self.spread_to_curve
        inc = (self.wac - mtg) * 100.0
        bal, r = float(self.notional), self.wac / 12.0
        payment_dollar = bal * r / (1 - (1 + r) ** -self.months)
        ts, interest_l, principal_l = [], [], []
        for m in range(1, self.months + 1):
            if bal <= 1e-12:
                break
            smm = 1 - (1 - self.cpr(inc, m)) ** (1 / 12)
            interest = bal * r
            sched = min(payment_dollar - interest, bal)
            prepay = max((bal - sched) * smm, 0.0)
            ts.append(m / 12.0)
            interest_l.append(interest)
            principal_l.append(sched + prepay)
            bal -= sched + prepay
        return np.array(ts), np.array(interest_l), np.array(principal_l)

    def pv(self, curve: Curve) -> float:
        t, c = self.flows(curve)
        r = curve.rate(t) + self.oas
        return float(self.side * np.sum(c / (1 + r) ** t))


@dataclass
class Swap(Instrument):
    """Par interest rate swap = short a fixed bond + long a floater.

    `side = +1` pays fixed (gains as rates rise); `side = -1` receives fixed.
    """
    years: float = 10.0
    fixed_rate: float = 0.04
    freq: int = 2
    next_reset: float = 0.25
    current_rate: float | None = None
    rate_type: str = "swap"

    def _fixing(self, curve: Curve) -> float:
        return self.current_rate if self.current_rate is not None \
            else curve.anchor(self.next_reset)

    def flows(self, curve):
        n = int(round(self.years * self.freq))
        tf = np.arange(1, n + 1) / self.freq
        cf = np.full(n, -self.notional * self.fixed_rate / self.freq)
        cf[-1] -= self.notional
        flt_cf = self.notional * (1.0 + self._fixing(curve)) ** self.next_reset
        return (np.concatenate([tf, [self.next_reset]]),
                np.concatenate([cf, [flt_cf]]))


# --------------------------------------------------------- key rate grid ---

# Standard swap tenors.  Key rates match TRADEABLE instruments, not coupon
# dates: a key you cannot trade is a row in the hedge matrix with no column.
TRADEABLE_TENORS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)


@dataclass(frozen=True)
class KeyRateGrid:
    tenors: tuple[float, ...] = TRADEABLE_TENORS

    def tent(self, t: np.ndarray, i: int) -> np.ndarray:
        """Triangular weight, 1.0 at key i, decaying to zero at its neighbours.

        Weights sum to exactly 1.0 at every maturity (partition of unity),
        which is what makes the KR01s reconcile to total DV01.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        k = self.tenors[i]
        lo = self.tenors[i - 1] if i > 0 else 0.0
        hi = self.tenors[i + 1] if i < len(self.tenors) - 1 else np.inf
        w = np.zeros_like(t)
        left = (t > lo) & (t <= k)
        w[left] = (t[left] - lo) / (k - lo)
        if np.isfinite(hi):
            right = (t > k) & (t < hi)
            w[right] = (hi - t[right]) / (hi - k)
        if i == 0:
            w[t <= k] = 1.0
        if i == len(self.tenors) - 1:
            w[t >= k] = 1.0
        return w

    def check_partition(self, t: np.ndarray, tol: float = 1e-12) -> bool:
        total = sum(self.tent(t, i) for i in range(len(self.tenors)))
        return bool(np.all(np.abs(total - 1.0) < tol))


# ------------------------------------------------------------- portfolio ---

@dataclass
class Portfolio:
    positions: list[Instrument] = field(default_factory=list)
    curve: Curve = field(default_factory=lambda: Curve.flat(0.04))
    grid: KeyRateGrid = field(default_factory=KeyRateGrid)
    tier1: float = 100.0

    def __post_init__(self):
        self._snapshot_fixings()

    def _snapshot_fixings(self) -> None:
        """Freeze the current index fixing on every floating leg.

        Without this the coupon would move with the curve inside a bump and
        floaters would show zero DV01 instead of duration-to-next-reset.
        """
        for p in self.positions:
            if isinstance(p, (Floater, Swap)) and p.current_rate is None:
                p.current_rate = self.curve.anchor(p.next_reset)

    def add(self, *inst: Instrument) -> "Portfolio":
        self.positions.extend(inst)
        self._snapshot_fixings()
        return self

    # -- valuation ----------------------------------------------------------
    def pv(self, curve: Curve | None = None) -> float:
        c = curve or self.curve
        return float(sum(p.pv(c) for p in self.positions))

    def eve(self, scenario: str | None = None, **kw) -> float:
        if scenario is None:
            return self.pv()
        shocked = self.curve.with_shift(lambda t: bcbs_shock(t, scenario, **kw))
        return self.pv(shocked)

    def eve_grid(self, **kw) -> dict[str, float]:
        base = self.pv()
        return {s: self.eve(s, **kw) - base for s in SCENARIOS}

    # -- sensitivity --------------------------------------------------------
    def dv01(self, curve: Curve | None = None) -> float:
        c = curve or self.curve
        return (self.pv(c) - self.pv(c.with_shift(lambda t: np.full_like(t, 1e-4)))) * 1e6

    def kr01(self, curve: Curve | None = None) -> np.ndarray:
        """Key rate DV01, $ per bp, one entry per grid tenor.

        Full revaluation per bump, so behavioural models re-run and optionality
        is captured.  Sums to total DV01 up to second-order convexity.
        """
        c = curve or self.curve
        base = self.pv(c)
        out = []
        for i in range(len(self.grid.tenors)):
            bumped = c.with_shift(lambda t, i=i: 1e-4 * self.grid.tent(t, i))
            out.append((base - self.pv(bumped)) * 1e6)
        return np.array(out)

    def kr01_in_scenario(self, scenario: str, **kw) -> np.ndarray:
        """KR01 recomputed inside a shocked state -- reveals duration drift."""
        shocked = self.curve.with_shift(lambda t: bcbs_shock(t, scenario, **kw))
        return self.kr01(shocked)

    # -- attribution --------------------------------------------------------
    def shock_matrix(self, **kw) -> np.ndarray:
        """(n_keys x n_scenarios) shock in basis points at each key."""
        return np.array([[bcbs_shock(np.array([k]), s, **kw)[0] * 1e4
                          for s in SCENARIOS] for k in self.grid.tenors])

    def attribution(self, **kw) -> dict:
        """Decompose each scenario's dEVE by key rate.

        Returns predicted (linear), actual (full reval) and the difference,
        which IS the convexity.  Positive => bullet-like.  Negative => embedded
        optionality is biting.
        """
        k = self.kr01()
        sh = self.shock_matrix(**kw)
        contrib = -k[:, None] * sh / 1e6
        actual = np.array([self.eve_grid(**kw)[s] for s in SCENARIOS])
        return {
            "keys": np.array(self.grid.tenors),
            "scenarios": SCENARIOS,
            "kr01": k,
            "shocks_bp": sh,
            "contribution": contrib,
            "predicted": contrib.sum(axis=0),
            "actual": actual,
            "convexity": actual - contrib.sum(axis=0),
        }

    def limit_status(self, limit_pct: float = 15.0, amber_pct: float = 12.0,
                     **kw) -> list[dict]:
        rows = []
        for s, d in self.eve_grid(**kw).items():
            pct = d / self.tier1 * 100.0
            rows.append({
                "scenario": s, "d_eve": d, "pct_tier1": pct,
                "status": "BREACH" if pct < -limit_pct
                          else ("AMBER" if pct < -amber_pct else "ok"),
            })
        return rows


# --------------------------------------------------------------- hedging ---

@dataclass
class HedgeSolution:
    swaps: list[Swap]
    notionals: np.ndarray
    residual_kr01: np.ndarray
    pre_kr01: np.ndarray
    grid: KeyRateGrid

    @property
    def reduction_pct(self) -> float:
        pre = np.abs(self.pre_kr01).sum()
        return 0.0 if pre == 0 else (1 - np.abs(self.residual_kr01).sum() / pre) * 100


def solve_hedge(portfolio: Portfolio,
                tenors: Sequence[float] | None = None,
                target: np.ndarray | None = None,
                max_notional: float | None = None) -> HedgeSolution:
    """Solve for pay-fixed swap notionals that drive KR01 to `target`.

    `target=None` means neutralise.  Pass a non-zero target to hedge *to a
    limit* rather than to zero -- usually the right call, since hedging to zero
    throws away the return the balance sheet exists to earn.
    """
    grid = portfolio.grid
    tenors = tuple(tenors) if tenors else grid.tenors
    k = portfolio.kr01()
    goal = np.zeros_like(k) if target is None else np.asarray(target, dtype=float)

    unit = 100.0
    cols = []
    for T in tenors:
        probe = Portfolio([Swap(f"swap_{T:g}y", unit, side=1, years=T,
                                fixed_rate=portfolio.curve.anchor(T))],
                          portfolio.curve, grid)
        cols.append(probe.kr01())
    S = np.column_stack(cols)

    w, *_ = np.linalg.lstsq(S, goal - k, rcond=None)
    if max_notional is not None:
        w = np.clip(w, -max_notional / unit, max_notional / unit)

    swaps = [Swap(f"{T:g}y pay-fixed" if x > 0 else f"{T:g}y receive-fixed",
                  abs(x) * unit, side=int(np.sign(x)) or 1, years=T,
                  fixed_rate=portfolio.curve.anchor(T))
             for T, x in zip(tenors, w) if abs(x * unit) > 1e-6]
    return HedgeSolution(swaps, w * unit, k + S @ w, k, grid)


# --------------------------------------------------- hedge accounting tag ---

_DESIGNATION = {
    # (hedged item rate type) -> (designation, MTM destination, note)
    "fixed": ("fair_value", "P&L (offset by hedged item basis adjustment)",
              "ASU 2017-12; portfolio layer method (ASU 2022-01) for prepayable "
              "pools. Monitor the breach test if prepayments run fast."),
    "floating": ("cash_flow", "OCI -> AOCI, reclassified as flows occur",
                 "Hedge of variability in forecasted interest cash flows."),
}


def designate(hedge: HedgeSolution, portfolio: Portfolio) -> list[dict]:
    """Tag each proposed swap with its likely designation and hedged item.

    Not accounting advice -- confirm designations with your auditor.  The point
    is that the treasurer sees the accounting consequence next to the risk
    reduction, which is where hedging programmes usually get stuck.
    """
    fixed_assets = [p for p in portfolio.positions
                    if p.side > 0 and p.rate_type == "fixed"]
    float_liabs = [p for p in portfolio.positions
                   if p.side < 0 and p.rate_type == "floating"]
    nmds = [p for p in portfolio.positions if isinstance(p, NMD)]

    rows = []
    for sw, notional in zip(hedge.swaps, hedge.notionals):
        pays_fixed = sw.side > 0
        if pays_fixed and fixed_assets:
            item = max(fixed_assets, key=lambda p: p.notional)
            kind, dest, note = _DESIGNATION["fixed"]
        elif not pays_fixed and float_liabs:
            item = max(float_liabs, key=lambda p: p.notional)
            kind, dest, note = _DESIGNATION["floating"]
        elif float_liabs:
            item = max(float_liabs, key=lambda p: p.notional)
            kind, dest, note = _DESIGNATION["floating"]
        elif fixed_assets:
            item = max(fixed_assets, key=lambda p: p.notional)
            kind, dest, note = _DESIGNATION["fixed"]
        else:
            item, kind, dest, note = None, "undesignated", "P&L (full MTM)", \
                "No eligible hedged item identified."

        warn = ""
        if nmds and item is not None and not isinstance(item, NMD):
            warn = (f"NMD book of {sum(p.notional for p in nmds):,.0f} drives much of "
                    "this exposure but demand deposits generally cannot be the hedged "
                    "item in a fair value hedge. Designated against assets instead.")
        rows.append({
            "swap": sw.name,
            "notional": abs(notional),
            "designation": kind,
            "hedged_item": item.name if item else None,
            "mtm_to": dest,
            "note": note,
            "warning": warn,
        })
    return rows
