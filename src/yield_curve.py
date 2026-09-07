"""
yield_curve.py
--------------
Base yield curve for EVE discounting.

A proper BCBS 368 EVE calculation requires discounting each cash flow at the
market rate corresponding to its tenor. This module provides:

1. A YieldCurve class that holds base rates per BCBS 368 bucket.
2. A shocked version of the curve: r_shocked[k] = r_base[k] + shock_bp[k]/10000

EVE calculation:

    PV_base(CF)    = Σ_k  CF_k / (1 + r_base[k])^t_k
    PV_shocked(CF) = Σ_k  CF_k / (1 + r_base[k] + Δr[k])^t_k
    ΔEVE           = PV_shocked(assets) - PV_shocked(liabilities)
                   - PV_base(assets)   + PV_base(liabilities)

The base curve here defaults to a stylised USD curve. Production runs should
use ``src.market_curve.get_live_yield_curve()`` which builds:

  - 0–12M from live SOFR (NY Fed overnight + SOFR swap tenors)
  - 1Y–10Y from USD SOFR IRS mid rates

All rates stored as decimals (e.g. 0.045 = 4.5%).
"""

from __future__ import annotations
import numpy as np
from .time_buckets import BUCKET_MIDPOINTS


# ── Stylised USD base curve (approximate, late 2024) ─────────────────────────
# Reference tenors (years) and corresponding rates (decimal)
_REF_TENORS = [0.0,   0.25,  0.5,   1.0,   2.0,   3.0,
               5.0,   7.0,  10.0,  15.0,  20.0,  30.0]

_REF_RATES = [0.053, 0.053, 0.052, 0.050, 0.047, 0.046,
              0.045, 0.045, 0.044, 0.044, 0.044, 0.043]


class YieldCurve:
    """
    Yield curve with one base rate per BCBS 368 time bucket.

    Continuous pillar rates (``_ref_tenors`` / ``_ref_rates``) drive
    ``rate(t)`` for MBS Step A/C; bucket ``base_rates`` drive EVE discounting
    of non-OA instruments.
    """

    def __init__(
        self,
        ref_tenors: list[float] = _REF_TENORS,
        ref_rates:  list[float] = _REF_RATES,
    ):
        self._ref_tenors = list(ref_tenors)
        self._ref_rates = list(ref_rates)
        self.base_rates: np.ndarray = np.interp(
            BUCKET_MIDPOINTS, self._ref_tenors, self._ref_rates
        )

    def rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Continuously interpolated rate at tenor ``t`` (years).

        Prefers the continuous pillar grid so anchors like 10Y match scenario
        pillars exactly; falls back to bucket midpoints for KR01-cloned curves.
        """
        t_arr = np.asarray(t, dtype=float)
        if getattr(self, "_ref_tenors", None) and len(self._ref_tenors) >= 2:
            out = np.interp(t_arr, self._ref_tenors, self._ref_rates)
        else:
            out = np.interp(t_arr, BUCKET_MIDPOINTS, self.base_rates)
        if np.ndim(t) == 0:
            return float(out)
        return out

    def shocked_rates(self, shocks_bp: list[float]) -> np.ndarray:
        """
        Returns shocked rates: base + shock (bp converted to decimal).
        Rates are floored at 0 (no negative rates in this model).
        """
        shocks_dec = np.array(shocks_bp) / 10_000
        return np.maximum(self.base_rates + shocks_dec, 0.0)

    def with_rates(self, rates: np.ndarray) -> "YieldCurve":
        """Clone with an explicit 19-bucket rate vector (already floored)."""
        yc = YieldCurve.__new__(YieldCurve)
        yc.base_rates = np.asarray(rates, dtype=float).copy()
        yc._ref_tenors = list(BUCKET_MIDPOINTS)
        yc._ref_rates = list(yc.base_rates)
        return yc

    def shocked_curve(self, shocks_bp: list[float]) -> "YieldCurve":
        """
        Full shocked curve: continuous pillars + bucket rates, floored at 0.
        """
        shocks = np.asarray(shocks_bp, dtype=float)
        pillar_shocks = np.interp(
            np.asarray(self._ref_tenors, dtype=float),
            np.asarray(BUCKET_MIDPOINTS, dtype=float),
            shocks,
        )
        new_ref = np.maximum(
            np.asarray(self._ref_rates, dtype=float) + pillar_shocks / 10_000.0,
            0.0,
        )
        return YieldCurve(list(self._ref_tenors), list(new_ref))

    def discount_factors(self, rates: np.ndarray) -> np.ndarray:
        """
        Returns discount factor array: DF[k] = 1 / (1 + rates[k])^t_k
        using bucket midpoint t_k as the representative tenor.
        """
        t = np.array(BUCKET_MIDPOINTS)
        return 1.0 / np.power(1.0 + rates, t)

    def pv_cashflows(
        self,
        bucket_cashflows: np.ndarray,
        shocks_bp: list[float] | None = None,
    ) -> float:
        """
        Present value of a cash flow array (one value per bucket).

        Parameters
        ----------
        bucket_cashflows : array of shape (N_BUCKETS,)
        shocks_bp        : if None, uses base curve; otherwise shocks first
        """
        rates = self.shocked_rates(shocks_bp) if shocks_bp is not None \
            else self.base_rates
        dfs = self.discount_factors(rates)
        return float(np.dot(bucket_cashflows, dfs))


# Module-level default curve (shared across the model)
BASE_CURVE = YieldCurve()
