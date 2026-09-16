"""Zero-curve bootstrap and at-maturity IRS schedule tests (no pandas)."""

from __future__ import annotations

from src.cva_pfe.bloomberg_ticket import price_irs_ticket
from src.cva_pfe.irs_pricing import (
    discount_factor,
    level_annuity,
    par_swap_rate,
    payment_schedule,
)
from src.sofr_bootstrap import bootstrap_numpy_sofr


def _quotes():
    return {
        0.25: 0.05,
        0.5: 0.049,
        1.0: 0.047,
        2.0: 0.045,
        3.0: 0.044,
        5.0: 0.043,
        7.0: 0.0425,
        10.0: 0.042,
    }


def test_numpy_bootstrap_attaches_dfs():
    z = bootstrap_numpy_sofr(_quotes(), swap_freq=2)
    assert z.method == "numpy_bootstrap"
    assert abs(z.curve.df(0.0) - 1.0) < 1e-12
    assert 0.0 < z.curve.df(5.0) < 1.0
    # Calibrating 5Y semi par should reprice near the input quote
    k = par_swap_rate(z.curve, 5.0, pay_freq=2, float_pay_freq=2)
    assert abs(k - 0.043) < 1e-3


def test_at_maturity_schedule_and_par_mtm():
    times, deltas = payment_schedule(5.0, 0)
    assert times == [5.0]
    assert deltas == [5.0]

    z = bootstrap_numpy_sofr(_quotes(), swap_freq=2)
    curve = z.curve
    ticket = price_irs_ticket(
        curve,
        notional_m=100.0,
        tenor_years=5.0,
        pay_fixed=False,
        pay_freq=0,
        float_pay_freq=2,
        solve_par=True,
    )
    assert abs(ticket.mtm_m) < 1e-6
    assert ticket.trade.pay_freq == 0
    assert ticket.trade.float_pay_freq == 2
    # Bullet fixed annuity = T * DF(T)
    df5 = discount_factor(curve, 5.0)
    assert abs(level_annuity(curve, 5.0, 0) - 5.0 * df5) < 1e-9
    # Par ≈ (1 - DF) / (T * DF)
    expected = (1.0 - df5) / (5.0 * df5)
    assert abs(ticket.par_rate - expected) < 1e-9
