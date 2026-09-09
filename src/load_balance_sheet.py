"""
load_balance_sheet.py
---------------------
Load bank balance sheet instruments from a CSV file.

Each row becomes one Instrument; cash flows are generated automatically
by the existing cashflows engine.
"""

from __future__ import annotations

import io
from typing import BinaryIO, TextIO, Union

import pandas as pd

from .cashflows import Instrument, InstrumentType
from .file_io import read_csv_robust

try:
    from .calibrate_prepayment import get_engine_prepay_defaults
except Exception:  # pragma: no cover
    def get_engine_prepay_defaults():  # type: ignore
        return {
            "base_turnover": 0.06,
            "max_refi_cpr": 0.34,
            "logistic_k": 2.2,
            "logistic_midpoint": 0.60,
            "seasoning_ramp_months": 30,
        }

CsvSource = Union[str, TextIO, BinaryIO, io.BytesIO]

REQUIRED_COLUMNS = [
    "name",
    "side",
    "notional",
    "coupon_pct",
    "instrument_type",
    "maturity_years",
]
OPTIONAL_COLUMNS = [
    "payment_freq",
    "repricing_years",
    "prepay_enabled",
    "base_cpr",
    "age_months",
    "market_mortgage_rate",
    "mbs_level",
    "encumbered",
    # Option-adjusted MBS / whole-loan fields
    "wac",
    "wam_months",
    "pool_age_months",
    "anchor_tenor",
    "spread_to_curve",
    "oas",
    "base_turnover",
    "max_refi_cpr",
    "logistic_k",
    "logistic_midpoint",
    "seasoning_ramp_months",
    "hqla_level",
    "nsfr_rsf_factor",
    "credit_spread",
    "use_option_adjusted",
]
VALID_TYPES = {
    "bullet_fixed", "bullet_floating", "amortising",
    "mbs", "whole_loan", "demand_deposit",
}
VALID_SIDES = {"asset", "liability"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _validate_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required column(s): "
            + ", ".join(missing)
            + f". Required: {', '.join(REQUIRED_COLUMNS)}"
        )

    if df.empty:
        raise ValueError("CSV contains no instrument rows.")

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains empty values.")

    invalid_sides = set(df["side"].astype(str).str.lower().unique()) - VALID_SIDES
    if invalid_sides:
        raise ValueError(
            f"Invalid side value(s): {', '.join(sorted(invalid_sides))}. "
            f"Use: asset, liability"
        )

    invalid_types = set(df["instrument_type"].astype(str).str.lower().unique()) - VALID_TYPES
    if invalid_types:
        raise ValueError(
            f"Invalid instrument_type value(s): {', '.join(sorted(invalid_types))}. "
            f"Use: {', '.join(sorted(VALID_TYPES))}"
        )


def _opt_float(row: pd.Series, col: str, default=None):
    if col not in row.index or pd.isna(row[col]):
        return default
    return float(row[col])


def _opt_int(row: pd.Series, col: str, default=None):
    if col not in row.index or pd.isna(row[col]):
        return default
    return int(row[col])


def _row_to_instrument(row: pd.Series) -> Instrument:
    side = str(row["side"]).strip().lower()
    instrument_type = str(row["instrument_type"]).strip().lower()
    _pp = get_engine_prepay_defaults()

    payment_freq = 2
    if "payment_freq" in row.index and pd.notna(row["payment_freq"]):
        payment_freq = int(row["payment_freq"])

    repricing_years = None
    if "repricing_years" in row.index and pd.notna(row["repricing_years"]):
        repricing_years = float(row["repricing_years"])

    prepay_enabled = False
    if "prepay_enabled" in row.index and pd.notna(row["prepay_enabled"]):
        val = str(row["prepay_enabled"]).strip().lower()
        prepay_enabled = val in ("1", "true", "yes", "y", "t")

    base_cpr = None
    if "base_cpr" in row.index and pd.notna(row["base_cpr"]):
        base_cpr = float(row["base_cpr"])
        if base_cpr > 1.0:
            base_cpr = base_cpr / 100.0

    age_months = 0
    if "age_months" in row.index and pd.notna(row["age_months"]):
        age_months = int(row["age_months"])

    market_mortgage_rate = None
    if "market_mortgage_rate" in row.index and pd.notna(row["market_mortgage_rate"]):
        market_mortgage_rate = float(row["market_mortgage_rate"])
        if market_mortgage_rate > 1.0:
            market_mortgage_rate = market_mortgage_rate / 100.0

    mbs_level = ""
    if "mbs_level" in row.index and pd.notna(row["mbs_level"]):
        from .prepayment import normalize_mbs_level
        mbs_level = normalize_mbs_level(str(row["mbs_level"]))

    encumbered = False
    if "encumbered" in row.index and pd.notna(row["encumbered"]):
        encumbered = str(row["encumbered"]).strip().lower() in ("1", "true", "yes", "y", "t")

    use_oa = True
    if "use_option_adjusted" in row.index and pd.notna(row["use_option_adjusted"]):
        use_oa = str(row["use_option_adjusted"]).strip().lower() in (
            "1", "true", "yes", "y", "t",
        )

    wac = _opt_float(row, "wac")
    if wac is not None and wac > 1.0:
        wac = wac / 100.0

    spread = _opt_float(row, "spread_to_curve", 0.0175)
    if spread is not None and spread > 1.0:
        spread = spread / 10_000.0

    oas = _opt_float(row, "oas", 0.005)
    if oas is not None and oas > 1.0:
        oas = oas / 10_000.0

    hqla_level = ""
    if "hqla_level" in row.index and pd.notna(row["hqla_level"]):
        hqla_level = str(row["hqla_level"]).strip().lower()

    return Instrument(
        name=str(row["name"]).strip(),
        notional=float(row["notional"]),
        coupon_pct=float(row["coupon_pct"]),
        instrument_type=instrument_type,  # type: ignore[arg-type]
        maturity_years=float(row["maturity_years"]),
        payment_freq=payment_freq,
        repricing_years=repricing_years,
        side=side,
        prepay_enabled=prepay_enabled,
        base_cpr=base_cpr,
        age_months=age_months,
        market_mortgage_rate=market_mortgage_rate,
        mbs_level=mbs_level,
        encumbered=encumbered,
        use_option_adjusted=use_oa,
        wac=wac,
        wam_months=_opt_int(row, "wam_months"),
        pool_age_months=_opt_int(row, "pool_age_months"),
        anchor_tenor=_opt_float(row, "anchor_tenor", 10.0) or 10.0,
        spread_to_curve=spread if spread is not None else 0.0175,
        oas=oas if oas is not None else 0.005,
        base_turnover=_opt_float(row, "base_turnover", _pp["base_turnover"]) or _pp["base_turnover"],
        max_refi_cpr=_opt_float(row, "max_refi_cpr", _pp["max_refi_cpr"]) or _pp["max_refi_cpr"],
        logistic_k=_opt_float(row, "logistic_k", _pp["logistic_k"]) or _pp["logistic_k"],
        logistic_midpoint=_opt_float(row, "logistic_midpoint", _pp["logistic_midpoint"])
        or _pp["logistic_midpoint"],
        seasoning_ramp_months=_opt_int(row, "seasoning_ramp_months", _pp["seasoning_ramp_months"])
        or _pp["seasoning_ramp_months"],
        hqla_level=hqla_level,
        nsfr_rsf_factor=_opt_float(row, "nsfr_rsf_factor"),
        credit_spread=_opt_float(row, "credit_spread", 0.0) or 0.0,
    )


def load_instruments_from_csv(source: CsvSource) -> tuple[list[Instrument], list[Instrument]]:
    """
    Load instruments from a CSV file.

    Returns (assets, liabilities).
    """
    df = _normalize_columns(read_csv_robust(source))
    _validate_dataframe(df)

    assets: list[Instrument] = []
    liabilities: list[Instrument] = []

    for _, row in df.iterrows():
        inst = _row_to_instrument(row)
        if inst.side == "asset":
            assets.append(inst)
        else:
            liabilities.append(inst)

    if not assets:
        raise ValueError("CSV must contain at least one asset.")
    if not liabilities:
        raise ValueError("CSV must contain at least one liability.")

    return assets, liabilities


def instruments_to_dataframe(
    assets: list[Instrument],
    liabilities: list[Instrument],
) -> pd.DataFrame:
    """Serialize instruments to a CSV-friendly DataFrame."""
    rows = []
    for inst in assets + liabilities:
        rows.append({
            "name": inst.name,
            "side": inst.side,
            "notional": inst.notional,
            "coupon_pct": inst.coupon_pct,
            "instrument_type": inst.instrument_type,
            "maturity_years": inst.maturity_years,
            "payment_freq": inst.payment_freq,
            "repricing_years": inst.repricing_years,
            "prepay_enabled": int(bool(inst.prepay_enabled)),
            "base_cpr": "" if inst.base_cpr is None else inst.base_cpr,
            "age_months": inst.age_months,
            "market_mortgage_rate": (
                "" if inst.market_mortgage_rate is None else inst.market_mortgage_rate
            ),
            "mbs_level": getattr(inst, "mbs_level", "") or "",
            "encumbered": int(bool(getattr(inst, "encumbered", False))),
            "wac": "" if getattr(inst, "wac", None) is None else inst.wac,
            "wam_months": getattr(inst, "wam_months", "") or "",
            "pool_age_months": getattr(inst, "pool_age_months", "") or "",
            "anchor_tenor": getattr(inst, "anchor_tenor", 10.0),
            "spread_to_curve": getattr(inst, "spread_to_curve", 0.0175),
            "oas": getattr(inst, "oas", 0.005),
            "hqla_level": getattr(inst, "hqla_level", "") or "",
            "nsfr_rsf_factor": (
                "" if getattr(inst, "nsfr_rsf_factor", None) is None
                else inst.nsfr_rsf_factor
            ),
            "credit_spread": getattr(inst, "credit_spread", 0.0) or 0.0,
            "use_option_adjusted": int(bool(getattr(inst, "use_option_adjusted", True))),
        })
    return pd.DataFrame(rows)
