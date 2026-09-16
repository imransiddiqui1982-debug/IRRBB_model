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
    cva_bp: float = 0.0  # running spread vs annuity / DV01 (IRS charge)


def hazard_from_cds(cds_spread_bp: float, recovery: float = 0.40) -> float:
    """Rough credit triangle: λ ≈ s / LGD."""
    s = max(float(cds_spread_bp), 0.0) / 10_000.0
    lgd = max(1.0 - float(recovery), 1e-6)
    return float(s / lgd)


def cva_running_spread_bp(
    cva_m: float,
    *,
    annuity: float,
    notional_m: float,
) -> float:
    """
    Convert CVA ($M) into a running fixed-rate charge (bp).

    PV of 1 bp on notional N with annuity A ≈ N · A · 1e−4 ($M), so
    ``cva_bp = cva_m / (N · A · 1e−4)``.
    """
    denom = abs(float(notional_m)) * max(float(annuity), 0.0) * 1e-4
    if denom <= 1e-18 or float(cva_m) <= 0.0:
        return 0.0
    return float(cva_m / denom)


def cva_running_spread_bp_from_dv01(cva_m: float, dv01_k: float) -> float:
    """
    Convert CVA ($M) to bp using portfolio DV01 ($K / bp).

    ``cva_bp = (cva_m · 1e6) / (|dv01_k| · 1e3) = cva_m · 1000 / |dv01_k|``.
    """
    d = abs(float(dv01_k))
    if d <= 1e-12 or float(cva_m) <= 0.0:
        return 0.0
    return float(cva_m * 1000.0 / d)


def apply_cva_charge_to_fixed(
    par_rate: float,
    *,
    pay_fixed: bool,
    cva_charge_bp: float,
) -> float:
    """
    Adjust par fixed rate so risk-free MtM offsets unilateral CVA cost.

    * Receive-fixed → receive higher fixed (+bp)
    * Pay-fixed → pay lower fixed (−bp)
    """
    adj = float(cva_charge_bp) / 10_000.0
    if abs(adj) < 1e-18:
        return float(par_rate)
    return float(par_rate) + adj if not pay_fixed else float(par_rate) - adj


def compute_cva(
    curve: YieldCurve,
    profile: ExposureProfile,
    cds_spread_bp: float = 100.0,
    recovery: float = 0.40,
    *,
    annuity: float | None = None,
    notional_m: float | None = None,
    dv01_k: float | None = None,
) -> CVAResult:
    """
    CVA ≈ (1−R) Σ EE(t_i) * DF(t_i) * (PD(t_i) − PD(t_{i−1}))

    Survival: S(t) = exp(−λ t), PD(t) = 1 − S(t).

    Optional annuity / DV01 converts the dollar CVA into a running IRS charge (bp).
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

    cva_bp = 0.0
    if annuity is not None and notional_m is not None:
        cva_bp = cva_running_spread_bp(cva, annuity=float(annuity), notional_m=float(notional_m))
    elif dv01_k is not None:
        cva_bp = cva_running_spread_bp_from_dv01(cva, float(dv01_k))

    return CVAResult(
        cva_m=float(cva),
        hazard_rate=lam,
        recovery=float(recovery),
        cds_spread_bp=float(cds_spread_bp),
        epe_m=epe,
        cva_bp=float(cva_bp),
    )
