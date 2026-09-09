"""
tests/test_calibrate_prepayment.py
----------------------------------
Calibration pipeline + engine default wiring.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibrate_prepayment import (  # noqa: E402
    build_observation_table,
    build_synthetic_dataset,
    clear_calibrated_params_cache,
    fit_s_curve,
    fit_seasoning_ramp,
    get_engine_prepay_defaults,
    load_calibrated_params,
)
from src.mbs_pricing import terms_from_instrument  # noqa: E402
from src.cashflows import Instrument  # noqa: E402


def test_synthetic_fit_recovers_near_truth():
    truth = dict(
        base_turnover=0.06,
        max_refi_cpr=0.34,
        logistic_k=2.2,
        logistic_midpoint=0.60,
        ramp_months=30,
    )
    # Smaller panel than CLI validate — still enough for a stable fit
    loans = build_synthetic_dataset(truth, n_loans=2500, months=72, seed=11)
    obs = build_observation_table(loans, wac_col="orig_rate")
    assert len(obs) >= 6
    fitted = fit_s_curve(obs, source_label="unit_test")
    assert fitted.r_squared > 0.85
    assert abs(fitted.base_turnover - truth["base_turnover"]) < 0.03
    assert abs(fitted.max_refi_cpr - truth["max_refi_cpr"]) < 0.08
    ramp = fit_seasoning_ramp(
        loans.assign(
            incentive_pp=(loans["orig_rate"] - loans["prevailing_mtg_rate"]) * 100,
            prepaid=(loans["zero_balance_code"] == "01").astype(int),
        )
    )
    assert 15 <= int(ramp["ramp_months"]) <= 45


def test_engine_defaults_load_shipped_json():
    clear_calibrated_params_cache()
    cal = load_calibrated_params()
    assert cal is not None
    d = get_engine_prepay_defaults()
    assert d["calibration_source"] != "illustrative_defaults"
    assert 0.02 <= d["base_turnover"] <= 0.12
    assert 0.10 <= d["max_refi_cpr"] <= 0.60


def test_terms_from_instrument_uses_calibrated_defaults(tmp_path, monkeypatch):
    payload = {
        "base_turnover": 0.07,
        "max_refi_cpr": 0.40,
        "logistic_k": 2.5,
        "logistic_midpoint": 0.55,
        "seasoning_ramp_months": 24,
        "source": "unit_test_override",
        "r_squared": 0.99,
    }
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    from src import calibrate_prepayment as cp
    from src import mbs_pricing as mp

    clear_calibrated_params_cache()
    monkeypatch.setattr(cp, "DEFAULT_PARAMS_PATH", path)
    clear_calibrated_params_cache()

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
        # Step-B left blank → Instrument.__post_init__ + terms use calibration
    )
    assert abs(inst.base_turnover - 0.07) < 1e-12
    assert inst.seasoning_ramp_months == 24
    terms = mp.terms_from_instrument(inst)
    assert abs(terms.base_turnover - 0.07) < 1e-12
    assert terms.seasoning_ramp_months == 24
    clear_calibrated_params_cache()
