"""
tests/test_cpr_calibration.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cpr_calibration import (  # noqa: E402
    CprCalibrationInputs,
    calibrate_scenario_cprs,
    calibration_to_cpr_maps,
    cpr_from_scurve,
    incentive_bp,
    mortgage_shock_bp,
    nearest_historical_regime,
)
from src.scenarios import SCENARIO_MAP  # noqa: E402


def test_incentive_bp():
    assert incentive_bp(5.5, 6.5) == pytest.approx(-100.0)


def test_refi_incentive_signed():
    from src.cpr_calibration import format_refi_incentive_bp
    assert format_refi_incentive_bp(-100) == "-100 bp"
    assert format_refi_incentive_bp(150) == "+150 bp"
    assert format_refi_incentive_bp(0) == "0 bp"


def test_calib_has_refi_incentive_column():
    inputs = CprCalibrationInputs(wac_pct=5.5, pmms_pct=6.5, age_months=30)
    df = calibrate_scenario_cprs(inputs)
    assert "refi_incentive" in df.columns
    assert df.loc[df["key"] == "BASE", "refi_incentive"].iloc[0] == "-100 bp"
    from src.cpr_calibration import calibration_display_frame
    disp = calibration_display_frame(df)
    assert "Refi incentive" in disp.columns
    assert "WAC %" in disp.columns
    assert "PMMS %" in disp.columns


def test_scurve_monotonic():
    low = cpr_from_scurve(-100)
    mid = cpr_from_scurve(0)
    high = cpr_from_scurve(150)
    assert low < mid < high


def test_parallel_up_slower_than_down():
    inputs = CprCalibrationInputs(wac_pct=5.5, pmms_pct=6.5, age_months=30)
    df = calibrate_scenario_cprs(inputs)
    by_key = {r["key"]: r for _, r in df.iterrows()}
    assert by_key["PS_UP"]["cpr_pct"] < by_key["BASE"]["cpr_pct"]
    assert by_key["PS_DOWN"]["cpr_pct"] > by_key["BASE"]["cpr_pct"]


def test_mortgage_shock_parallel():
    assert mortgage_shock_bp(SCENARIO_MAP["PS_UP"]) == pytest.approx(200.0)
    assert mortgage_shock_bp(SCENARIO_MAP["PS_DOWN"]) == pytest.approx(-200.0)


def test_maps_cover_all_keys():
    inputs = CprCalibrationInputs(wac_pct=5.0, pmms_pct=6.0)
    df = calibrate_scenario_cprs(inputs)
    cpr_map, psa_map = calibration_to_cpr_maps(df)
    for k in ("BASE", "PS_UP", "PS_DOWN", "STEEPENER", "FLATTENER", "SHORT_UP", "SHORT_DOWN"):
        assert k in cpr_map and k in psa_map


def test_historical_analogue_lock_in():
    r = nearest_historical_regime(6.9)
    assert "lock-in" in r["note"].lower() or "2022" in r["period"]
