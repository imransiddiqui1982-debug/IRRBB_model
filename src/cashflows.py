"""
cashflows.py
------------
Cash flow generation engine for IRRBB.

Replaces the single-notional-per-instrument approximation with a proper
scheduled cash flow decomposition. Each instrument's principal and coupon
cash flows are generated at their contractual payment dates and slotted
into the correct BCBS 368 time bucket.

This is the foundation of a correct EVE calculation:

    EVE = Σ_k  CF_k / (1 + r_k)^t_k            [baseline]

    ΔEVE = Σ_k  CF_k / (1 + r_k + Δr_k)^t_k
           - Σ_k  CF_k / (1 + r_k)^t_k          [shocked minus baseline]

    where:
        CF_k   = cash flow amount in bucket k  (USD millions)
        r_k    = base discount rate for tenor k (decimal)
        Δr_k   = interest rate shock for bucket k (decimal)
        t_k    = bucket midpoint in years

Instrument types supported
--------------------------
    bullet_fixed    : fixed-rate bullet bond / term loan
                      coupons every period + principal at maturity
    bullet_floating : floating-rate bullet loan
                      single repricing cash flow at next reset date
    amortising      : fixed-rate amortising loan (equal principal)
                      principal paid evenly each period + declining coupons;
                      optional CPR prepayment accelerates principal
    mbs             : mortgage-backed security (pass-through) — amortising
                      schedule with CPR always enabled (user or S-curve)
    demand_deposit  : non-maturity deposit (NMD) — modelled as single
                      cash flow at behavioural repricing tenor
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd
from .time_buckets import years_to_bucket, N_BUCKETS, BUCKET_MIDPOINTS
from .prepayment import (
    PrepaymentParams,
    cpr_to_smm,
    effective_cpr,
)


InstrumentType = Literal[
    "bullet_fixed",
    "bullet_floating",
    "amortising",
    "mbs",
    "demand_deposit",
]

PREPAYABLE_TYPES = frozenset({"amortising", "mbs"})


@dataclass
class CashFlow:
    """A single scheduled cash flow."""
    time_years: float      # when it occurs (from today)
    amount:     float      # USD millions (positive = inflow for assets)
    cf_type:    str        # 'coupon' | 'principal' | 'prepayment' | 'repricing'
    bucket:     int        # BCBS 368 bucket index (derived)

    def __post_init__(self):
        self.bucket = years_to_bucket(self.time_years)


@dataclass
class Instrument:
    """
    A single balance sheet instrument with full cash flow schedule.

    Parameters
    ----------
    name            : instrument label
    notional        : face value (USD millions)
    coupon_pct      : contractual annual rate (%)
    instrument_type : see InstrumentType
    maturity_years  : contractual maturity in years from today
    payment_freq    : coupon/principal payments per year
                      (1=annual, 2=semi, 4=quarterly, 12=monthly)
    repricing_years : for floating instruments — years to next rate reset
    side            : 'asset' or 'liability'
    prepay_enabled  : if True (amortising), apply CPR / SMM schedule
    base_cpr        : optional fixed annual CPR override (fraction); if None,
                      CPR is derived from refinance incentive vs market rate
    market_mortgage_rate : primary mortgage rate (decimal) for incentive CPR
    age_months      : loan age at t=0 for PSA seasoning
    use_hf_chronos  : optional Chronos blend when historical_cpr provided
    historical_cpr  : optional recent monthly CPR series for HF refinement
    cashflows       : populated by generate_cashflows()
    """
    name:             str
    notional:         float
    coupon_pct:       float
    instrument_type:  InstrumentType
    maturity_years:   float
    payment_freq:     int = 2          # semi-annual default
    repricing_years:  float = None       # floating instruments only
    side:             str = "asset"
    prepay_enabled:   bool = False
    base_cpr:         float | None = None
    market_mortgage_rate: float | None = None
    age_months:       int = 0
    use_hf_chronos:   bool = False
    historical_cpr:   tuple[float, ...] | None = None
    prepay_params:    PrepaymentParams | None = field(default=None, repr=False)
    cashflows:        list[CashFlow] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.repricing_years is None:
            self.repricing_years = self.maturity_years
        name_l = self.name.lower()
        # MBS always prepays; mortgages auto-enable CPR by name
        if self.instrument_type == "mbs":
            self.prepay_enabled = True
        elif (
            self.instrument_type == "amortising"
            and not self.prepay_enabled
            and ("mortgage" in name_l or "mbs" in name_l or "mortgage-backed" in name_l)
        ):
            self.prepay_enabled = True
        if (
            self.prepay_enabled
            and self.instrument_type in PREPAYABLE_TYPES
            and self.market_mortgage_rate is None
        ):
            from .prepayment import scenario_market_mortgage_rate
            from .yield_curve import BASE_CURVE
            self.market_mortgage_rate = scenario_market_mortgage_rate(
                BASE_CURVE.base_rates, None, self.maturity_years,
            )
        self.cashflows = self.generate_cashflows()

    @property
    def is_prepayable(self) -> bool:
        """True for MBS and amortising mortgages with CPR enabled."""
        return self.instrument_type in PREPAYABLE_TYPES and bool(self.prepay_enabled)

    # ── Cash flow generators ──────────────────────────────────────────────────

    def generate_cashflows(self) -> list[CashFlow]:
        if self.instrument_type == "bullet_fixed":
            return self._bullet_fixed()
        elif self.instrument_type == "bullet_floating":
            return self._bullet_floating()
        elif self.instrument_type in ("amortising", "mbs"):
            return self._amortising()
        elif self.instrument_type == "demand_deposit":
            return self._demand_deposit()
        else:
            raise ValueError(f"Unknown instrument type: {self.instrument_type}")

    def _bullet_fixed(self) -> list[CashFlow]:
        """
        Fixed-rate bullet: coupon payments at each period + principal at maturity.
        Coupon per period = notional × (coupon_pct/100) / payment_freq
        """
        cfs = []
        period = 1.0 / self.payment_freq
        coupon_amount = self.notional * (self.coupon_pct / 100) / self.payment_freq
        n_periods = round(self.maturity_years * self.payment_freq)

        for i in range(1, n_periods + 1):
            t = i * period
            # coupon at every period
            cfs.append(CashFlow(t, coupon_amount, "coupon", years_to_bucket(t)))
            # principal only at maturity
            if i == n_periods:
                cfs.append(CashFlow(t, self.notional, "principal", years_to_bucket(t)))

        return cfs

    def _bullet_floating(self) -> list[CashFlow]:
        """
        Floating-rate bullet: only the next repricing cash flow matters for EVE/NII.
        After repricing, the instrument is assumed to reprice at par → no residual
        EVE sensitivity beyond the reset date. This is the standard BCBS approximation.
        """
        return [
            CashFlow(
                self.repricing_years,
                self.notional,   # full notional reprices
                "repricing",
                years_to_bucket(self.repricing_years),
            )
        ]

    def resolve_cpr(
        self,
        market_mortgage_rate: float | None = None,
        age_months: int | None = None,
    ) -> float:
        """Annual CPR for this instrument under a given market mortgage rate."""
        if self.base_cpr is not None:
            return float(np.clip(self.base_cpr, 0.0, 0.999))
        mkt = (
            market_mortgage_rate
            if market_mortgage_rate is not None
            else self.market_mortgage_rate
        )
        if mkt is None:
            # Neutral incentive fallback (~PSA 100% terminal)
            mkt = self.coupon_pct / 100.0
        age = self.age_months if age_months is None else age_months
        return effective_cpr(
            self.coupon_pct,
            mkt,
            age_months=age,
            params=self.prepay_params,
            historical_cpr=self.historical_cpr,
            use_hf_chronos=self.use_hf_chronos,
        )

    def _amortising(self) -> list[CashFlow]:
        """
        Fixed-rate amortising loan: equal scheduled principal each period.

        When ``prepay_enabled``, each period also applies SMM prepayment on the
        remaining balance after the scheduled principal payment (standard CPR
        convention for ALM cash-flow engines).
        """
        if self.prepay_enabled:
            return self._amortising_with_cpr()
        return self._amortising_static()

    def _amortising_static(self) -> list[CashFlow]:
        cfs = []
        period = 1.0 / self.payment_freq
        n_periods = max(round(self.maturity_years * self.payment_freq), 1)
        principal_per_period = self.notional / n_periods

        outstanding = self.notional
        for i in range(1, n_periods + 1):
            t = i * period
            coupon = outstanding * (self.coupon_pct / 100) / self.payment_freq
            cfs.append(CashFlow(t, coupon,               "coupon",    years_to_bucket(t)))
            cfs.append(CashFlow(t, principal_per_period, "principal", years_to_bucket(t)))
            outstanding -= principal_per_period

        return cfs

    def _amortising_with_cpr(
        self,
        market_mortgage_rate: float | None = None,
        cpr_override: float | None = None,
    ) -> list[CashFlow]:
        """
        Scheduled amortisation + CPR prepayments.

        Period i:
          coupon on opening balance
          scheduled principal = original_notional / n_periods (capped at balance)
          prepayment = SMM × (balance − scheduled)
          closing balance = balance − scheduled − prepayment
        """
        cfs: list[CashFlow] = []
        period = 1.0 / self.payment_freq
        n_periods = max(round(self.maturity_years * self.payment_freq), 1)
        scheduled = self.notional / n_periods
        outstanding = float(self.notional)

        for i in range(1, n_periods + 1):
            if outstanding <= 1e-12:
                break
            t = i * period
            age = self.age_months + i - 1
            if cpr_override is not None:
                cpr = float(np.clip(cpr_override, 0.0, 0.999))
            else:
                cpr = self.resolve_cpr(market_mortgage_rate, age_months=age)
            smm = cpr_to_smm(cpr, self.payment_freq)

            coupon = outstanding * (self.coupon_pct / 100) / self.payment_freq
            cfs.append(CashFlow(t, coupon, "coupon", years_to_bucket(t)))

            sched_prin = min(scheduled, outstanding)
            remaining_after_sched = outstanding - sched_prin
            prep = remaining_after_sched * smm if i < n_periods else remaining_after_sched
            # Final period: clear residual (balloon of unscheduled remainder)
            if i == n_periods:
                prep = remaining_after_sched
                remaining_after_sched = 0.0

            if sched_prin > 1e-12:
                cfs.append(CashFlow(t, sched_prin, "principal", years_to_bucket(t)))
            if prep > 1e-12:
                cfs.append(CashFlow(t, prep, "prepayment", years_to_bucket(t)))

            outstanding = max(remaining_after_sched - prep, 0.0)

        return cfs

    def cashflows_under_market_rate(
        self,
        market_mortgage_rate: float,
        cpr_override: float | None = None,
    ) -> list[CashFlow]:
        """Regenerate cash flows under a shocked primary mortgage rate (EVE)."""
        if not self.is_prepayable:
            return list(self.cashflows)
        return self._amortising_with_cpr(
            market_mortgage_rate=market_mortgage_rate,
            cpr_override=cpr_override,
        )

    def bucket_cashflows_under_market_rate(
        self,
        market_mortgage_rate: float,
        cpr_override: float | None = None,
    ) -> np.ndarray:
        result = np.zeros(N_BUCKETS)
        for cf in self.cashflows_under_market_rate(market_mortgage_rate, cpr_override):
            result[cf.bucket] += cf.amount
        return result

    def principal_within_years(self, horizon_years: float) -> float:
        """
        Scheduled + prepaid principal returned within ``horizon_years``.
        Used for LCR 30-day inflows and NSFR residual maturity proxies.
        """
        total = 0.0
        for cf in self.cashflows:
            if cf.cf_type in ("principal", "prepayment") and cf.time_years <= horizon_years + 1e-9:
                total += cf.amount
        return total

    def residual_balance_after_years(self, horizon_years: float) -> float:
        """Outstanding principal after cash flows up to ``horizon_years``."""
        returned = self.principal_within_years(horizon_years)
        return max(self.notional - returned, 0.0)

    def wal_years(self) -> float:
        """Principal-weighted average life (years) including prepayments."""
        prin = [
            (cf.time_years, cf.amount)
            for cf in self.cashflows
            if cf.cf_type in ("principal", "prepayment") and cf.amount > 0
        ]
        total = sum(a for _, a in prin)
        if total <= 0:
            return float(self.maturity_years)
        return sum(t * a for t, a in prin) / total

    def _demand_deposit(self) -> list[CashFlow]:
        """
        Non-maturity deposit (NMD): treated as a single repricing cash flow
        at the behavioural repricing tenor (repricing_years).
        A full NMD model would apply a decay/run-off profile, but that requires
        customer behaviour data. This is the standard BCBS 368 simplified treatment.
        """
        return [
            CashFlow(
                self.repricing_years,
                self.notional,
                "repricing",
                years_to_bucket(self.repricing_years),
            )
        ]

    # ── Aggregation helpers ───────────────────────────────────────────────────

    def cashflows_df(self) -> pd.DataFrame:
        """Returns all cash flows as a DataFrame."""
        return pd.DataFrame([{
            "time_years": cf.time_years,
            "amount":     cf.amount,
            "cf_type":    cf.cf_type,
            "bucket":     cf.bucket,
        } for cf in self.cashflows])

    def bucket_cashflows(self) -> np.ndarray:
        """
        Returns array of shape (N_BUCKETS,) with total cash flow per bucket.
        Used directly in EVE calculation.
        """
        result = np.zeros(N_BUCKETS)
        for cf in self.cashflows:
            result[cf.bucket] += cf.amount
        return result

    @property
    def effective_duration(self) -> float:
        """
        Cash-flow weighted average maturity (Macaulay duration proxy).
        Uses bucket midpoints as representative tenors.
        For EVE sensitivity reporting only.
        """
        total_cf = sum(abs(cf.amount) for cf in self.cashflows)
        if total_cf == 0:
            return 0.0
        return sum(
            abs(cf.amount) * BUCKET_MIDPOINTS[cf.bucket]
            for cf in self.cashflows
        ) / total_cf
