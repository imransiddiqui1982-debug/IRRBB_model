"""ALM hedge derivative pricing + economic CVA/PFE + SA-CCR (prototype)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cva import CVAResult, compute_cva, hazard_from_cds
from .exposure import ExposureProfile, simulate_exposure
from .irs_pricing import (
    IRSTrade,
    par_swap_rate,
    swap_dv01_k,
    swap_mtm_m,
    trades_from_signed_notionals,
)
from .saccr import SACCRResult, saccr_irs_portfolio

if TYPE_CHECKING:
    from .engine import HedgeCCRReport

__all__ = [
    "IRSTrade",
    "par_swap_rate",
    "swap_mtm_m",
    "swap_dv01_k",
    "trades_from_signed_notionals",
    "ExposureProfile",
    "simulate_exposure",
    "CVAResult",
    "compute_cva",
    "hazard_from_cds",
    "SACCRResult",
    "saccr_irs_portfolio",
    "HedgeCCRReport",
    "run_hedge_ccr",
    "exposure_frame",
    "saccr_frame",
]


def __getattr__(name: str):
    if name in ("HedgeCCRReport", "run_hedge_ccr", "exposure_frame", "saccr_frame"):
        from . import engine as m
        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
