"""IRRBB engine package.

Keep this module lightweight so ``from src.prepayment import ...`` does not
pull calculator / Streamlit / matplotlib into a circular import on Cloud.
Import submodules directly, e.g. ``from src.calculator import IRRBBCalculator``.
"""

__all__ = [
    "BCBS_BUCKETS",
    "BUCKET_LABELS",
    "N_BUCKETS",
    "years_to_bucket",
    "Instrument",
    "CashFlow",
    "PrepaymentParams",
    "ShockCprTable",
    "DEFAULT_SHOCK_CPR_PCT",
    "DEFAULT_SHOCK_PSA_PCT",
    "YieldCurve",
    "BASE_CURVE",
    "get_instruments",
    "load_instruments_from_csv",
    "SCENARIOS",
    "SCENARIO_MAP",
    "IRRBBCalculator",
]


def __getattr__(name: str):
    """Lazy attribute access for backwards-compatible ``from src import X``."""
    if name in ("BCBS_BUCKETS", "BUCKET_LABELS", "N_BUCKETS", "years_to_bucket"):
        from . import time_buckets as m
        return getattr(m, name)
    if name in ("Instrument", "CashFlow"):
        from . import cashflows as m
        return getattr(m, name)
    if name in (
        "PrepaymentParams",
        "ShockCprTable",
        "DEFAULT_SHOCK_CPR_PCT",
        "DEFAULT_SHOCK_PSA_PCT",
        "cpr_to_smm",
        "effective_cpr",
        "incentive_cpr",
        "s_curve_cpr",
    ):
        from . import prepayment as m
        return getattr(m, name)
    if name in ("YieldCurve", "BASE_CURVE"):
        from . import yield_curve as m
        return getattr(m, name)
    if name == "get_instruments":
        from .balance_sheet import get_instruments
        return get_instruments
    if name == "load_instruments_from_csv":
        from .load_balance_sheet import load_instruments_from_csv
        return load_instruments_from_csv
    if name in ("SCENARIOS", "SCENARIO_MAP", "Scenario"):
        from . import scenarios as m
        return getattr(m, name)
    if name in ("IRRBBCalculator", "ScenarioResult", "suggest_irs_hedges"):
        from . import calculator as m
        return getattr(m, name)
    if name in ("compute_lcr", "LcrResult"):
        from . import lcr_calculator as m
        return getattr(m, name)
    if name in ("compute_nsfr", "NsfrResult"):
        from . import nsfr_calculator as m
        return getattr(m, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
