"""ALM hedge derivative pricing + economic CVA/PFE + SA-CCR (prototype)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .bloomberg_ticket import (
    IRSTicketResult,
    build_manual_hedged_item,
    par_swap_rate_freq,
    price_irs_ticket,
    swap_mtm_freq_m,
    trades_from_tickets,
)
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
    from .hedge_effectiveness import EffectivenessResult

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
    "EffectivenessResult",
    "prospective_effectiveness",
    "build_hypothetical_derivative",
    "eligible_hedged_items",
    "effectiveness_scatter_frame",
    "progressive_frame",
    "default_shock_grid_bp",
    "IRSTicketResult",
    "price_irs_ticket",
    "build_manual_hedged_item",
    "par_swap_rate_freq",
    "swap_mtm_freq_m",
    "trades_from_tickets",
]

_HE_NAMES = {
    "EffectivenessResult",
    "prospective_effectiveness",
    "build_hypothetical_derivative",
    "eligible_hedged_items",
    "effectiveness_scatter_frame",
    "progressive_frame",
    "default_shock_grid_bp",
}
_ENGINE_NAMES = {
    "HedgeCCRReport",
    "run_hedge_ccr",
    "exposure_frame",
    "saccr_frame",
}


def __getattr__(name: str):
    if name in _ENGINE_NAMES:
        from . import engine as m
        return getattr(m, name)
    if name in _HE_NAMES:
        from . import hedge_effectiveness as m
        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
