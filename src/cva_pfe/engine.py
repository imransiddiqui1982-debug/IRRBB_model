"""
Orchestrate hedge IRS pricing + economic EE/PFE/CVA + SA-CCR EAD.

Designed for Streamlit: returns plain DataFrames / dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..yield_curve import YieldCurve
from .cva import CVAResult, compute_cva
from .exposure import ExposureProfile, simulate_exposure
from .irs_pricing import (
    par_swap_rate,
    swap_dv01_k,
    swap_mtm_m,
    trades_from_signed_notionals,
)
from .saccr import SACCRResult, saccr_irs_portfolio


@dataclass
class HedgeCCRReport:
    trades_df: object  # pd.DataFrame
    portfolio_mtm_m: float
    portfolio_dv01_k: float
    exposure: ExposureProfile
    cva: CVAResult
    saccr: SACCRResult
    summary: dict


def run_hedge_ccr(
    curve: YieldCurve,
    signed_notionals: dict[float, float],
    *,
    cds_spread_bp: float = 100.0,
    recovery: float = 0.40,
    rate_vol_bp: float = 80.0,
    n_paths: int = 400,
    n_steps: int = 16,
    horizon_years: float | None = None,
    pfe_percentile: float = 95.0,
    collateral_m: float = 0.0,
    seed: int = 42,
) -> HedgeCCRReport:
    import pandas as pd

    trades = trades_from_signed_notionals(signed_notionals, curve, use_par=True)
    if not trades:
        empty_exp = simulate_exposure(curve, [], horizon_years=1.0, n_steps=2, n_paths=10)
        empty_cva = compute_cva(curve, empty_exp, cds_spread_bp, recovery)
        empty_saccr = saccr_irs_portfolio([], 0.0, collateral_m)
        return HedgeCCRReport(
            trades_df=pd.DataFrame(),
            portfolio_mtm_m=0.0,
            portfolio_dv01_k=0.0,
            exposure=empty_exp,
            cva=empty_cva,
            saccr=empty_saccr,
            summary={"note": "No non-zero hedge notionals"},
        )

    rows = []
    mtm_tot = 0.0
    dv01_tot = 0.0
    for tr in trades:
        mtm = swap_mtm_m(curve, tr)
        dv = swap_dv01_k(curve, tr)
        par = par_swap_rate(curve, tr.tenor_years, tr.start_years)
        mtm_tot += mtm
        dv01_tot += dv
        rows.append({
            "Trade": tr.label,
            "Tenor (Y)": tr.tenor_years,
            "Notional ($M)": round(tr.notional_m, 2),
            "Side": "Pay-fixed" if tr.pay_fixed else "Receive-fixed",
            "Fixed rate %": round(par * 100.0, 3),
            "MtM ($M)": round(mtm, 4),
            "DV01 ($K/bp)": round(dv, 2),
        })

    horiz = float(horizon_years) if horizon_years else max(tr.tenor_years for tr in trades)
    profile = simulate_exposure(
        curve,
        trades,
        horizon_years=horiz,
        n_steps=n_steps,
        n_paths=n_paths,
        rate_vol_bp=rate_vol_bp,
        pfe_percentile=pfe_percentile,
        seed=seed,
    )
    cva = compute_cva(curve, profile, cds_spread_bp=cds_spread_bp, recovery=recovery)
    saccr = saccr_irs_portfolio(trades, mtm_tot, collateral_m=collateral_m)

    summary = {
        "n_trades": len(trades),
        "portfolio_mtm_m": round(mtm_tot, 4),
        "portfolio_dv01_k": round(dv01_tot, 2),
        "cva_m": round(cva.cva_m, 4),
        "epe_m": round(cva.epe_m, 4),
        "pfe_peak_m": round(float(profile.pfe_m.max()), 4),
        "saccr_ead_m": round(saccr.ead_m, 4),
        "saccr_pfe_m": round(saccr.pfe_m, 4),
        "cds_spread_bp": cds_spread_bp,
        "rate_vol_bp": rate_vol_bp,
        "pfe_percentile": pfe_percentile,
    }
    return HedgeCCRReport(
        trades_df=pd.DataFrame(rows),
        portfolio_mtm_m=mtm_tot,
        portfolio_dv01_k=dv01_tot,
        exposure=profile,
        cva=cva,
        saccr=saccr,
        summary=summary,
    )


def exposure_frame(profile: ExposureProfile):
    import numpy as np
    import pandas as pd

    return pd.DataFrame({
        "Time (Y)": np.round(profile.times, 3),
        "EE ($M)": np.round(profile.ee_m, 4),
        "PFE ($M)": np.round(profile.pfe_m, 4),
        "ENE ($M)": np.round(profile.ene_m, 4),
        "Mean MtM ($M)": np.round(profile.mean_mtm_m, 4),
    })


def saccr_frame(s: SACCRResult):
    import pandas as pd

    return pd.DataFrame([{
        "V MtM ($M)": round(s.v_m, 4),
        "RC ($M)": round(s.replacement_cost_m, 4),
        "Add-on ($M)": round(s.addon_m, 4),
        "Multiplier": round(s.multiplier, 4),
        "SA-CCR PFE ($M)": round(s.pfe_m, 4),
        "EAD = 1.4×(RC+PFE) ($M)": round(s.ead_m, 4),
    }])
