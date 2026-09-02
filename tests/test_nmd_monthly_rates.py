"""Test monthly deposit rates from Deposit Rates sheet for beta regression."""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nmd_refinement import refine_nmd_deposits, load_customer_nmd  # noqa: E402

V2_PATH = r"c:\Users\acer\OneDrive\Desktop\comprehensive_deposit_template_v2.xlsx"


@pytest.mark.skipif(not os.path.exists(V2_PATH), reason="v2 file not on desktop")
def test_v2_monthly_rates_produce_material_beta():
    result = refine_nmd_deposits(V2_PATH, deposit_name="Beta Test")
    assert result.beta > 0.1
    assert result.rate_sensitive_balance_mb > result.core_balance_mb * 0.05


def test_monthly_rates_change_beta_vs_constant():
    """Synthetic: rising product rates should yield beta > 0 with monthly lookup."""
    months = pd.date_range("2022-01-31", periods=24, freq="ME")
    month_cols = [d.strftime("%Y-%m") for d in months]
    cust = pd.DataFrame({
        "customer_id": ["C1"],
        "segment": ["wholesale"],
        "deposit_product": ["demand"],
        "rate_code": ["WS-FIN"],
        "currency": ["USD"],
        "start_date": ["2022-01-31"],
        "maturity_date": [""],
        "is_insured": ["N"],
        "is_operational": ["N"],
        "lcr_bucket": ["auto"],
        "nsfr_bucket": ["auto"],
    })
    for i, m in enumerate(month_cols):
        cust[m] = 10.0 + i * 0.1

    deposit_rates = pd.DataFrame({
        "rate_code": ["WS-FIN"],
        **{m: 0.01 + i * 0.002 for i, m in enumerate(month_cols)},
    })
    market = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in months],
        "market_rate": [0.01 + i * 0.002 for i in range(len(months))],
    })

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        cust.to_excel(writer, sheet_name="Customer Deposits", index=False)
        deposit_rates.to_excel(writer, sheet_name="Deposit Rates", index=False)
        market.to_excel(writer, sheet_name="Market Rates", index=False)
    bio.seek(0)

    workbook = load_customer_nmd(bio)
    assert workbook.deposit_rates is not None

    result = refine_nmd_deposits(bio, deposit_name="Synth")
    assert result.beta > 0.5
