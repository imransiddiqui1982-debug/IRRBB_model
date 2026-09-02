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
                      principal paid evenly each period + declining coupons
    demand_deposit  : non-maturity deposit (NMD) — modelled as single
                      cash flow at behavioural repricing tenor
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd
from .time_buckets import years_to_bucket, N_BUCKETS, BUCKET_MIDPOINTS


InstrumentType = Literal[
    "bullet_fixed",
    "bullet_floating",
    "amortising",
    "demand_deposit",
]


@dataclass
class CashFlow:
    """A single scheduled cash flow."""
    time_years: float      # when it occurs (from today)
    amount:     float      # USD millions (positive = inflow for assets)
    cf_type:    str        # 'coupon' | 'principal' | 'repricing'
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
    cashflows:        list[CashFlow] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.repricing_years is None:
            self.repricing_years = self.maturity_years
        self.cashflows = self.generate_cashflows()

    # ── Cash flow generators ──────────────────────────────────────────────────

    def generate_cashflows(self) -> list[CashFlow]:
        if self.instrument_type == "bullet_fixed":
            return self._bullet_fixed()
        elif self.instrument_type == "bullet_floating":
            return self._bullet_floating()
        elif self.instrument_type == "amortising":
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

    def _amortising(self) -> list[CashFlow]:
        """
        Fixed-rate amortising loan: equal principal repaid each period.
        Outstanding balance declines linearly → declining coupon payments.
        """
        cfs = []
        period = 1.0 / self.payment_freq
        n_periods = round(self.maturity_years * self.payment_freq)
        principal_per_period = self.notional / n_periods

        outstanding = self.notional
        for i in range(1, n_periods + 1):
            t = i * period
            coupon = outstanding * (self.coupon_pct / 100) / self.payment_freq
            cfs.append(CashFlow(t, coupon,               "coupon",    years_to_bucket(t)))
            cfs.append(CashFlow(t, principal_per_period, "principal", years_to_bucket(t)))
            outstanding -= principal_per_period

        return cfs

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
