"""PMMS anchor + Freddie calibrator smoke tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cashflows import Instrument  # noqa: E402
from src.mbs_pricing import apply_pmms_anchor, step_a_mortgage_rate, terms_from_instrument  # noqa: E402
from src.yield_curve import YieldCurve  # noqa: E402


def test_apply_pmms_anchor_sets_mortgage_near_pmms():
    curve = YieldCurve()
    inst = Instrument(
        name="Test MBS",
        side="asset",
        notional=100.0,
        coupon_pct=5.5,
        instrument_type="mbs",
        maturity_years=25.0,
        wac=0.055,
        wam_months=300,
        pool_age_months=36,
        anchor_tenor=10.0,
        spread_to_curve=0.0175,
    )
    pmms = 0.067
    meta = apply_pmms_anchor([inst], curve, pmms_rate=pmms)
    assert meta["applied"] == 1
    terms = terms_from_instrument(inst)
    mtg, _ = step_a_mortgage_rate(curve, terms)
    assert abs(mtg - pmms) < 1e-9


def test_pmms_module_history_lookup_offline(tmp_path, monkeypatch):
    import pandas as pd
    from src import pmms as pm

    hist = pd.Series(
        [0.06, 0.065],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        name="pmms",
    )
    path = tmp_path / "pmms_history.csv"
    hist.to_csv(path, header=["pmms"])
    monkeypatch.setattr(pm, "HISTORY_CACHE", path)
    assert abs(pm.pmms_for_yyyymm("202002", hist) - 0.065) < 1e-12
    assert abs(pm.pmms_for_yyyymm("202001", hist) - 0.06) < 1e-12
