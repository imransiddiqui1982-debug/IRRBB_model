"""
Economic exposure profiles for a portfolio of IRS.

Prototype: parallel yield-curve Monte Carlo (normal rate shocks) — not a full
Hull–White multi-factor model. Good enough for Streamlit EE / PFE charts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..yield_curve import YieldCurve
from .irs_pricing import IRSTrade, swap_mtm_m


@dataclass
class ExposureProfile:
    times: np.ndarray          # years
    ee_m: np.ndarray           # expected exposure $M
    pfe_m: np.ndarray          # potential future exposure $M
    ene_m: np.ndarray          # expected negative exposure $M
    mean_mtm_m: np.ndarray


def _bump_curve(curve: YieldCurve, shock_bp: float) -> YieldCurve:
    tenors = list(getattr(curve, "_ref_tenors", []) or [])
    rates = list(getattr(curve, "_ref_rates", []) or [])
    if len(tenors) >= 2:
        return YieldCurve(
            ref_tenors=tenors,
            ref_rates=[r + shock_bp / 10_000.0 for r in rates],
        )
    from copy import deepcopy
    out = deepcopy(curve)
    out.base_rates = np.asarray(curve.base_rates, dtype=float) + shock_bp / 10_000.0
    return out


def simulate_exposure(
    curve: YieldCurve,
    trades: list[IRSTrade],
    horizon_years: float = 10.0,
    n_steps: int = 20,
    n_paths: int = 500,
    rate_vol_bp: float = 80.0,
    pfe_percentile: float = 95.0,
    seed: int = 42,
) -> ExposureProfile:
    """
    Monte Carlo EE / PFE under Brownian parallel rate shocks.

    At horizon t, shock ~ N(0, vol_bp * sqrt(t)) in bp; reprice remaining
    swap life with start rolled forward (simplified: shorten tenor by t).
    """
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, float(horizon_years), int(n_steps) + 1)
    n_t = len(times)
    n_p = int(n_paths)
    mtm = np.zeros((n_p, n_t), dtype=float)

    for j, t in enumerate(times):
        if t <= 1e-12:
            # t=0: deterministic MtM
            v0 = sum(swap_mtm_m(curve, tr) for tr in trades)
            mtm[:, j] = v0
            continue
        shocks = rng.normal(0.0, rate_vol_bp * np.sqrt(t), size=n_p)
        for i, sh in enumerate(shocks):
            c_i = _bump_curve(curve, float(sh))
            total = 0.0
            for tr in trades:
                rem = float(tr.tenor_years) - float(t)
                if rem <= 0.05:
                    continue
                aged = IRSTrade(
                    tenor_years=rem,
                    notional_m=tr.notional_m,
                    pay_fixed=tr.pay_fixed,
                    fixed_rate=tr.fixed_rate,
                    start_years=0.0,
                    name=tr.name,
                )
                total += swap_mtm_m(c_i, aged)
            mtm[i, j] = total

    pos = np.maximum(mtm, 0.0)
    neg = np.minimum(mtm, 0.0)
    ee = pos.mean(axis=0)
    ene = neg.mean(axis=0)
    pfe = np.percentile(pos, pfe_percentile, axis=0)
    return ExposureProfile(
        times=times,
        ee_m=ee,
        pfe_m=pfe,
        ene_m=ene,
        mean_mtm_m=mtm.mean(axis=0),
    )
