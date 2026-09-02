"""
balance_sheet.py
----------------
Bank balance sheet defined as a list of Instrument objects.
Each instrument generates its full scheduled cash flow set on construction.

Instrument types used:
    bullet_fixed    : fixed-rate bond or term loan (coupon + bullet principal)
    bullet_floating : floating-rate loan (single repricing CF at next reset)
    amortising      : fixed-rate amortising loan (declining balance)
    demand_deposit  : NMD modelled at behavioural repricing tenor
"""

from .cashflows import Instrument


def _m(months: float) -> float:
    return months / 12.0


def get_instruments() -> tuple[list[Instrument], list[Instrument]]:
    """
    Returns (assets, liabilities) as lists of Instrument objects.
    """

    assets = [
        # ── Money market & short-term ─────────────────────────────────────────
        Instrument(
            name="Cash & Central Bank Reserves",
            notional=150, coupon_pct=0.50,
            instrument_type="bullet_floating",
            maturity_years=_m(1), repricing_years=_m(1),
            payment_freq=12, side="asset",
        ),
        Instrument(
            name="T-Bills (3M)",
            notional=100, coupon_pct=4.90,
            instrument_type="bullet_fixed",
            maturity_years=_m(3),
            payment_freq=1, side="asset",
        ),
        Instrument(
            name="Floating Rate Notes (6M reset)",
            notional=200, coupon_pct=5.80,
            instrument_type="bullet_floating",
            maturity_years=3.0, repricing_years=_m(6),
            payment_freq=2, side="asset",
        ),
        # ── Medium-term ───────────────────────────────────────────────────────
        Instrument(
            name="Fixed Gov Bonds (2Y)",
            notional=200, coupon_pct=4.10,
            instrument_type="bullet_fixed",
            maturity_years=2.0,
            payment_freq=2, side="asset",
        ),
        Instrument(
            name="Floating Corp Loans (3Y, qtrly reset)",
            notional=600, coupon_pct=6.50,
            instrument_type="bullet_floating",
            maturity_years=3.0, repricing_years=_m(3),
            payment_freq=4, side="asset",
        ),
        Instrument(
            name="Fixed Retail Loans (4Y, amortising)",
            notional=250, coupon_pct=5.80,
            instrument_type="amortising",
            maturity_years=4.0,
            payment_freq=12, side="asset",
        ),
        Instrument(
            name="Fixed Gov Bonds (5Y)",
            notional=300, coupon_pct=3.95,
            instrument_type="bullet_fixed",
            maturity_years=5.0,
            payment_freq=2, side="asset",
        ),
        # ── Long-term ─────────────────────────────────────────────────────────
        Instrument(
            name="Fixed-Rate Mortgages (10Y, amortising)",
            notional=800, coupon_pct=5.20,
            instrument_type="amortising",
            maturity_years=10.0,
            payment_freq=12, side="asset",
        ),
        Instrument(
            name="Fixed Gov Bonds (15Y)",
            notional=200, coupon_pct=3.50,
            instrument_type="bullet_fixed",
            maturity_years=15.0,
            payment_freq=2, side="asset",
        ),
        Instrument(
            name="Fixed Infrastructure Bonds (20Y)",
            notional=100, coupon_pct=4.20,
            instrument_type="bullet_fixed",
            maturity_years=20.0,
            payment_freq=2, side="asset",
        ),
    ]

    liabilities = [
        # ── On-demand / NMD ───────────────────────────────────────────────────
        Instrument(
            name="Demand Deposits (Core NMD)",
            notional=700, coupon_pct=0.50,
            instrument_type="demand_deposit",
            maturity_years=_m(1), repricing_years=_m(1),
            payment_freq=12, side="liability",
        ),
        Instrument(
            name="Retail Savings (Sticky NMD)",
            notional=250, coupon_pct=1.50,
            instrument_type="demand_deposit",
            maturity_years=_m(3), repricing_years=_m(3),
            payment_freq=4, side="liability",
        ),
        Instrument(
            name="Overnight Repo",
            notional=100, coupon_pct=5.10,
            instrument_type="bullet_floating",
            maturity_years=1/365, repricing_years=1/365,
            payment_freq=365, side="liability",
        ),
        # ── Short-term fixed ──────────────────────────────────────────────────
        Instrument(
            name="Term Deposits (6M)",
            notional=200, coupon_pct=3.00,
            instrument_type="bullet_fixed",
            maturity_years=_m(6),
            payment_freq=2, side="liability",
        ),
        Instrument(
            name="Term Deposits (1Y)",
            notional=350, coupon_pct=3.20,
            instrument_type="bullet_fixed",
            maturity_years=1.0,
            payment_freq=2, side="liability",
        ),
        Instrument(
            name="Floating Senior Debt (3Y, semi reset)",
            notional=300, coupon_pct=5.80,
            instrument_type="bullet_floating",
            maturity_years=3.0, repricing_years=_m(6),
            payment_freq=2, side="liability",
        ),
        # ── Medium / long-term ────────────────────────────────────────────────
        Instrument(
            name="Fixed-Rate Covered Bonds (5Y)",
            notional=300, coupon_pct=4.30,
            instrument_type="bullet_fixed",
            maturity_years=5.0,
            payment_freq=2, side="liability",
        ),
        Instrument(
            name="Fixed-Rate Borrowings (7Y)",
            notional=200, coupon_pct=4.50,
            instrument_type="bullet_fixed",
            maturity_years=7.0,
            payment_freq=2, side="liability",
        ),
        Instrument(
            name="Subordinated Debt (10Y)",
            notional=200, coupon_pct=5.00,
            instrument_type="bullet_fixed",
            maturity_years=10.0,
            payment_freq=2, side="liability",
        ),
        Instrument(
            name="Tier 2 Capital Notes (15Y)",
            notional=100, coupon_pct=5.50,
            instrument_type="bullet_fixed",
            maturity_years=15.0,
            payment_freq=2, side="liability",
        ),
    ]

    return assets, liabilities
