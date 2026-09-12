"""Unilateral CVA from EE profile and constant hazard (CDS / LGD proxy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..yield_curve import YieldCurve
from .exposure import ExposureProfile
from .irs_pricing import discount_factor


@dataclass
class CVAResult:
    cva_m: float
    hazard_rate: float
    recovery: float
    cds_spread_bp: float
    epe_m: float  # average positive EE


def hazard_from_cds(cds_spread_bp: float, recovery: float = 0.40) -> float:
    """Rough credit triangle: λ ≈ s / LGD."""
    s = max(float(cds_spread_bp), 0.0) / 10_000.0
    lgd = max(1.0 - float(recovery), 1e-6)
    return float(s / lgd)


def compute_cva(
    curve: YieldCurve,
    profile: ExposureProfile,
    cds_spread_bp: float = 100.0,
    recovery: float = 0.40,
) -> CVAResult:
    """
    CVA ≈ (1−R) Σ EE(t_i) * DF(t_i) * (PD(t_i) − PD(t_{i−1}))

    Survival: S(t) = exp(−λ t), PD(t) = 1 − S(t).
    """
    lam = hazard_from_cds(cds_spread_bp, recovery)
    lgd = 1.0 - float(recovery)
    times = np.asarray(profile.times, dtype=float)
    ee = np.asarray(profile.ee_m, dtype=float)
    cva = 0.0
    pd_prev = 0.0
    for i in range(1, len(times)):
        t = float(times[i])
        pd_t = 1.0 - np.exp(-lam * t)
        dpd = max(pd_t - pd_prev, 0.0)
        df = discount_factor(curve, t)
        # mid-point EE
        ee_mid = 0.5 * (float(ee[i]) + float(ee[i - 1]))
        cva += lgd * ee_mid * df * dpd
        pd_prev = pd_t
    epe = float(np.mean(ee[1:])) if len(ee) > 1 else float(ee[0]) if len(ee) else 0.0
    return CVAResult(
        cva_m=float(cva),
        hazard_rate=lam,
        recovery=float(recovery),
        cds_spread_bp=float(cds_spread_bp),
        epe_m=epe,
    )
