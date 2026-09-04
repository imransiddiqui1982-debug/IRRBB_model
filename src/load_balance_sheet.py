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
]
VALID_TYPES = {"bullet_fixed", "bullet_floating", "amortising", "mbs", "demand_deposit"}
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


def _row_to_instrument(row: pd.Series) -> Instrument:
    side = str(row["side"]).strip().lower()
    instrument_type = str(row["instrument_type"]).strip().lower()

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
            base_cpr = base_cpr / 100.0  # allow percent input

    age_months = 0
    if "age_months" in row.index and pd.notna(row["age_months"]):
        age_months = int(row["age_months"])

    market_mortgage_rate = None
    if "market_mortgage_rate" in row.index and pd.notna(row["market_mortgage_rate"]):
        market_mortgage_rate = float(row["market_mortgage_rate"])
        if market_mortgage_rate > 1.0:
            market_mortgage_rate = market_mortgage_rate / 100.0

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
        })
    return pd.DataFrame(rows)
