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

    Parameters
    ----------
    ref_tenors : list of reference tenor points (years)
    ref_rates  : list of corresponding rates (decimal)
    """

    def __init__(
        self,
        ref_tenors: list[float] = _REF_TENORS,
        ref_rates:  list[float] = _REF_RATES,
    ):
        # Interpolate base rates at each of the 19 bucket midpoints
        self.base_rates: np.ndarray = np.interp(
            BUCKET_MIDPOINTS, ref_tenors, ref_rates
        )

    def shocked_rates(self, shocks_bp: list[float]) -> np.ndarray:
        """
        Returns shocked rates: base + shock (bp converted to decimal).
        Rates are floored at 0 (no negative rates in this model).
        """
        shocks_dec = np.array(shocks_bp) / 10_000
        return np.maximum(self.base_rates + shocks_dec, 0.0)

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
