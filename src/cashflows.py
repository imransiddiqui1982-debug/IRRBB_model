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
    mbs             : mortgage-backed security (pass-through) — option-adjusted
                      CPR from curve anchor (live Steps A/B/C)
    whole_loan      : residential whole-loan pool — same prepay math as MBS,
                      but never HQLA; carries credit_spread placeholder
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
    "whole_loan",
    "demand_deposit",
]

PREPAYABLE_TYPES = frozenset({"amortising", "mbs", "whole_loan"})


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
    use_option_adjusted : live curve→CPR→CF path (default True for mbs /
                      whole_loan / prepayable amortising)
    wac / wam_months / anchor_tenor / spread_to_curve / oas : MBS Step A–C
    hqla_level / nsfr_rsf_factor / credit_spread : static LCR/NSFR tags
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
    mbs_level:        str = ""   # ginnie | agency | private (HQLA / NSFR)
    encumbered:       bool = False  # NSFR: encumbered >1Y → 100% RSF
    # Option-adjusted MBS / whole-loan fields (spec)
    use_option_adjusted: bool = True
    wac:              float | None = None   # decimal; default coupon_pct/100
    wam_months:       int | None = None
    pool_age_months:  int | None = None
    anchor_tenor:     float = 10.0
    spread_to_curve:  float = 0.0175
    oas:              float = 0.005
    # Step-B S-curve; None → data/calibrated_prepayment_params.json or illustrative
    base_turnover:    float | None = None
    max_refi_cpr:     float | None = None
    logistic_k:       float | None = None
    logistic_midpoint: float | None = None
    seasoning_ramp_months: int | None = None
    hqla_level:       str = ""   # level_1 | level_2a | not_eligible
    nsfr_rsf_factor:   float | None = None
    credit_spread:    float = 0.0  # whole-loan placeholder only
    # NII accrual: optional explicit rate (decimal); else coupon_pct/100
    current_rate:     float | None = None
    cashflows:        list[CashFlow] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.repricing_years is None:
            self.repricing_years = self.maturity_years
        # Resolve Step-B defaults from calibration JSON when blank
        try:
            from .calibrate_prepayment import get_engine_prepay_defaults
            _pp = get_engine_prepay_defaults()
        except Exception:
            _pp = {
                "base_turnover": 0.06,
                "max_refi_cpr": 0.34,
                "logistic_k": 2.2,
                "logistic_midpoint": 0.60,
                "seasoning_ramp_months": 30,
            }
        if self.base_turnover is None:
            self.base_turnover = float(_pp["base_turnover"])
        if self.max_refi_cpr is None:
            self.max_refi_cpr = float(_pp["max_refi_cpr"])
        if self.logistic_k is None:
            self.logistic_k = float(_pp["logistic_k"])
        if self.logistic_midpoint is None:
            self.logistic_midpoint = float(_pp["logistic_midpoint"])
        if self.seasoning_ramp_months is None:
            self.seasoning_ramp_months = int(_pp["seasoning_ramp_months"])
        name_l = self.name.lower()
        # MBS / whole loans always prepays; mortgages auto-enable CPR by name
        if self.instrument_type in ("mbs", "whole_loan"):
            self.prepay_enabled = True
        elif (
            self.instrument_type == "amortising"
            and not self.prepay_enabled
            and ("mortgage" in name_l or "mbs" in name_l or "mortgage-backed" in name_l)
        ):
            self.prepay_enabled = True
        if self.instrument_type == "mbs" or "mbs" in name_l:
            from .prepayment import infer_mbs_level_from_name, normalize_mbs_level
            self.mbs_level = normalize_mbs_level(self.mbs_level) or infer_mbs_level_from_name(self.name)
        if self.pool_age_months is None:
            self.pool_age_months = int(self.age_months or 0)
        if self.wam_months is None:
            self.wam_months = max(int(round(float(self.maturity_years) * 12)), 1)
        if self.wac is None:
            self.wac = float(self.coupon_pct) / 100.0
        elif float(self.wac) > 1.0:
            self.wac = float(self.wac) / 100.0
        # Static HQLA / RSF tags (never from CPR)
        from .mbs_pricing import hqla_from_mbs_level
        is_wl = self.instrument_type == "whole_loan" or (
            self.instrument_type == "amortising" and self.prepay_enabled
            and "mbs" not in name_l
        )
        if not self.hqla_level:
            self.hqla_level = hqla_from_mbs_level(self.mbs_level, is_wl)
        if is_wl:
            self.hqla_level = "not_eligible"
            if self.nsfr_rsf_factor is None:
                self.nsfr_rsf_factor = 0.65
        elif self.nsfr_rsf_factor is None and self.instrument_type == "mbs":
            if self.hqla_level == "level_1":
                self.nsfr_rsf_factor = 0.05
            elif self.hqla_level == "level_2a":
                self.nsfr_rsf_factor = 0.15
            else:
                self.nsfr_rsf_factor = 0.85
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
        """True for MBS, whole loans, and amortising mortgages with CPR enabled."""
        return self.instrument_type in PREPAYABLE_TYPES and bool(self.prepay_enabled)

    @property
    def is_option_adjusted(self) -> bool:
        from .mbs_pricing import is_option_adjusted_prepayable
        return is_option_adjusted_prepayable(self)

    # ── Cash flow generators ──────────────────────────────────────────────────

    def generate_cashflows(self) -> list[CashFlow]:
        if self.instrument_type == "bullet_fixed":
            return self._bullet_fixed()
        elif self.instrument_type == "bullet_floating":
            return self._bullet_floating()
        elif self.instrument_type in ("amortising", "mbs", "whole_loan"):
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
        """Regenerate cash flows under a shocked primary mortgage rate (legacy path)."""
        if not self.is_prepayable:
            return list(self.cashflows)
        return self._amortising_with_cpr(
            market_mortgage_rate=market_mortgage_rate,
            cpr_override=cpr_override,
        )

    def cashflows_under_curve(self, curve) -> list[CashFlow]:
        """
        Live option-adjusted schedule from ``curve`` (Steps A/B/C).
        Must be called fresh on every curve bump — never cache across curves.
        """
        if not self.is_option_adjusted:
            return list(self.cashflows)
        from .mbs_pricing import step_c_flows_and_price, terms_from_instrument
        res = step_c_flows_and_price(curve, terms_from_instrument(self))
        cfs: list[CashFlow] = []
        # Split total CF into coupon vs principal+prepay for reporting
        monthly_coupon = float(self.wac or self.coupon_pct / 100.0) / 12.0
        balance = float(self.notional)
        for t, total in zip(res.times, res.cashflows):
            interest = balance * monthly_coupon
            # Approximate split for bucket reporting
            prin_like = max(float(total) - interest, 0.0)
            if interest > 1e-12:
                cfs.append(CashFlow(float(t), interest, "coupon", years_to_bucket(float(t))))
            if prin_like > 1e-12:
                cfs.append(CashFlow(float(t), prin_like, "prepayment", years_to_bucket(float(t))))
            balance = max(balance - prin_like, 0.0)
        return cfs

    def bucket_cashflows_under_curve(self, curve) -> np.ndarray:
        if not self.is_option_adjusted:
            return self.bucket_cashflows()
        from .mbs_pricing import bucket_cashflows_from_terms, terms_from_instrument
        return bucket_cashflows_from_terms(curve, terms_from_instrument(self))

    def pv_under_curve(self, curve) -> float:
        """Option-adjusted PV (includes OAS on monthly CFs)."""
        if not self.is_option_adjusted:
            return float(curve.pv_cashflows(self.bucket_cashflows()))
        from .mbs_pricing import step_c_flows_and_price, terms_from_instrument
        return float(step_c_flows_and_price(curve, terms_from_instrument(self)).price)

    def bucket_cashflows_under_market_rate(
        self,
        market_mortgage_rate: float,
        cpr_override: float | None = None,
    ) -> np.ndarray:
        result = np.zeros(N_BUCKETS)
        for cf in self.cashflows_under_market_rate(market_mortgage_rate, cpr_override):
            result[cf.bucket] += cf.amount
        return result

    def contractual_principal_within_years(self, horizon_years: float) -> float:
        """
        Contractual scheduled principal only (no CPR) — for LCR/NSFR.
        """
        if self.instrument_type not in ("amortising", "mbs", "whole_loan"):
            # bullets: full notional if maturity within horizon
            if self.maturity_years <= horizon_years + 1e-9:
                return float(self.notional)
            return 0.0
        period = 1.0 / max(self.payment_freq, 1)
        n_periods = max(round(self.maturity_years * self.payment_freq), 1)
        scheduled = self.notional / n_periods
        total = 0.0
        for i in range(1, n_periods + 1):
            t = i * period
            if t > horizon_years + 1e-9:
                break
            total += scheduled
        return float(min(total, self.notional))

    def principal_within_years(self, horizon_years: float) -> float:
        """
        Scheduled + prepaid principal returned within ``horizon_years``.
        Used for behavioural analytics; LCR uses contractual_principal_within_years.
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

    def contractual_residual_maturity_years(self) -> float:
        """Contractual residual maturity for LCR/NSFR (never CPR-adjusted)."""
        return max(float(self.maturity_years), float(self.repricing_years or 0.0))

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
