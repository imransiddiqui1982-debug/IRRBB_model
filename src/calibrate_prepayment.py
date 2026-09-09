"""
calibrate_prepayment.py
========================
Fits the S-curve prepayment model parameters (base_turnover, max_refi_cpr,
logistic_k, logistic_midpoint) to observed loan-level prepayment history,
instead of using illustrative defaults.

WHERE THE DATA COMES FROM
--------------------------
This script expects a CSV shaped like the Freddie Mac Single-Family
Loan-Level Dataset (monthly performance file), which is free but requires a
one-time account registration and manual download -- it's not reachable
from this sandboxed environment, so it cannot be fetched automatically here.

    https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

Expected columns after joining the origination file to the monthly
performance file (rename to match if your source differs):

    loan_id            str    unique loan identifier
    period             str    YYYY-MM, observation month
    orig_upb           float  original unpaid principal balance
    orig_rate          float  original note rate (decimal, e.g. 0.055)
    loan_age_months    int    months since origination as of `period`
    current_upb        float  scheduled UPB at start of `period`
    zero_balance_code  str    '01'=prepaid, '03'=short sale/REO, '' = active
                              (Freddie Mac coding; adjust if using another
                              servicer's convention)
    prevailing_mtg_rate float current market mortgage rate as of `period`
                              (join in from a PMMS/FRED series by month; see
                              `build_synthetic_dataset()` below for the shape)

If you don't have loan-level data yet, this script also runs end-to-end on
a SYNTHETIC dataset generated from KNOWN parameters, purely to validate that
the fitting machinery recovers the true values -- this is the standard way
to sanity-check a calibration pipeline before trusting it on real data.

USAGE
-----
    # validate the fitting code against synthetic data with known truth:
    python -m src.calibrate_prepayment --validate

    # calibrate against your real loan-level extract:
    python -m src.calibrate_prepayment --data path/to/loan_level_extract.csv \\
                                       --wac-col orig_rate \\
                                       --out data/calibrated_prepayment_params.json

The IRRBB engine loads ``data/calibrated_prepayment_params.json`` (if present)
as default ``base_turnover`` / ``max_refi_cpr`` / ``logistic_k`` /
``logistic_midpoint`` / ``seasoning_ramp_months`` for option-adjusted MBS
and whole loans when those fields are blank on the balance sheet.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "calibrated_prepayment_params.json"
)

# Illustrative engine defaults (used when no calibration JSON is present)
ILLUSTRATIVE_DEFAULTS = dict(
    base_turnover=0.06,
    max_refi_cpr=0.34,
    logistic_k=2.2,
    logistic_midpoint=0.60,
    seasoning_ramp_months=30,
)


# ----------------------------------------------------------- the S-curve --

def s_curve_cpr(incentive_pp: np.ndarray, base_turnover: float,
                max_refi_cpr: float, k: float, midpoint: float) -> np.ndarray:
    """Same functional form as MBS.step_b_cpr in the engine spec, vectorised
    over an array of incentive observations. No seasoning ramp here -- fit
    on SEASONED loans only (age >= ramp period) so the ramp doesn't
    contaminate the refi-response estimate. Seasoning is a separate,
    much simpler fit (see `fit_seasoning_ramp` below)."""
    refi = max_refi_cpr / (1 + np.exp(-k * (incentive_pp - midpoint)))
    return base_turnover + refi


DEFAULT_BOUNDS = dict(
    # (lower, upper) per parameter -- keeps the optimizer in economically
    # sensible territory and prevents degenerate fits on noisy data
    base_turnover=(0.02, 0.12),
    max_refi_cpr=(0.10, 0.60),
    k=(0.5, 6.0),
    midpoint=(0.10, 1.50),
)


# ------------------------------------------------------- observed -> fit --

def build_observation_table(loans: pd.DataFrame, *, wac_col: str = "orig_rate",
                            seasoning_floor_months: int = 30) -> pd.DataFrame:
    """Collapse loan-month records into (incentive_bucket -> observed SMM)
    pairs, the standard input shape for fitting a prepayment curve.

    Steps:
      1. Keep only seasoned loan-months (age >= seasoning_floor_months) so
         the ramp doesn't distort the refi-response estimate.
      2. Compute refi incentive per loan-month: WAC - prevailing market rate,
         in percentage points -- same sign convention as Step A in the
         engine (positive = borrower benefits from refinancing).
      3. Bucket incentive into 25bp bins.
      4. Within each bucket, observed CPR = 1 - (survival rate)^12, i.e.
         convert the realized monthly attrition into an annualized CPR
         using the SAME geometric relationship the engine uses going the
         other way (CPR -> SMM). This is the standard survival-based CPR
         estimator: prepaid balance / beginning balance -> SMM -> CPR.
    """
    df = loans.copy()
    df = df[df["loan_age_months"] >= seasoning_floor_months].copy()

    df["incentive_pp"] = (df[wac_col] - df["prevailing_mtg_rate"]) * 100.0
    df["prepaid"] = (df["zero_balance_code"] == "01").astype(int)

    # SMM per loan-month = prepaid balance / beginning-of-month scheduled UPB
    # (approximated here as: prepaid flag weighted by the loan's UPB, since
    # we're aggregating across many loans rather than tracking one pool's
    # balance run-off -- this is the loan-level equivalent of pool SMM)
    df["bucket"] = (np.round(df["incentive_pp"] / 0.25) * 0.25).astype(float)

    grouped = df.groupby("bucket").apply(
        lambda g: pd.Series({
            "n_loan_months": len(g),
            "n_prepaid": g["prepaid"].sum(),
            "upb_weighted_smm": np.average(g["prepaid"],
                                           weights=g["current_upb"]),
        }), include_groups=False
    ).reset_index()

    grouped["smm"] = grouped["upb_weighted_smm"]
    grouped["cpr_observed"] = 1 - (1 - grouped["smm"]) ** 12
    # drop buckets with too few observations to be reliable
    grouped = grouped[grouped["n_loan_months"] >= 30].reset_index(drop=True)
    return grouped[["bucket", "n_loan_months", "cpr_observed"]]


@dataclass
class CalibratedSCurve:
    base_turnover: float
    max_refi_cpr: float
    logistic_k: float
    logistic_midpoint: float
    n_observations: int
    n_buckets: int
    rmse: float
    r_squared: float
    source: str

    def to_engine_kwargs(self) -> dict:
        """Ready to unpack into MbsTerms / Instrument prepay fields."""
        return dict(
            base_turnover=self.base_turnover,
            max_refi_cpr=self.max_refi_cpr,
            logistic_k=self.logistic_k,
            logistic_midpoint=self.logistic_midpoint,
        )


@lru_cache(maxsize=4)
def load_calibrated_params(path: str | None = None) -> dict | None:
    """
    Load ``calibrated_prepayment_params.json`` for engine defaults.

    Returns None if the file is missing or incomplete.
    """
    p = Path(path) if path else DEFAULT_PARAMS_PATH
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = ("base_turnover", "max_refi_cpr", "logistic_k", "logistic_midpoint")
    if not all(k in data for k in required):
        return None
    return data


def clear_calibrated_params_cache() -> None:
    load_calibrated_params.cache_clear()


def get_engine_prepay_defaults(path: str | None = None) -> dict:
    """
    Defaults for Step B: prefer on-disk calibration, else illustrative values.
    """
    cal = load_calibrated_params(path)
    if not cal:
        out = dict(ILLUSTRATIVE_DEFAULTS)
        out["calibration_source"] = "illustrative_defaults"
        out["r_squared"] = None
        out["rmse"] = None
        return out
    return {
        "base_turnover": float(cal["base_turnover"]),
        "max_refi_cpr": float(cal["max_refi_cpr"]),
        "logistic_k": float(cal["logistic_k"]),
        "logistic_midpoint": float(cal["logistic_midpoint"]),
        "seasoning_ramp_months": int(
            cal.get("seasoning_ramp_months", ILLUSTRATIVE_DEFAULTS["seasoning_ramp_months"])
        ),
        "calibration_source": str(cal.get("source", "calibrated_file")),
        "r_squared": cal.get("r_squared"),
        "rmse": cal.get("rmse"),
        "n_observations": cal.get("n_observations"),
        "n_buckets": cal.get("n_buckets"),
    }


def fit_s_curve(obs: pd.DataFrame, *, source_label: str) -> CalibratedSCurve:
    """Nonlinear least squares fit of the logistic S-curve to observed
    (incentive_bucket, CPR) pairs, weighted by observation count so
    sparsely-populated incentive buckets don't dominate the fit."""
    x = obs["bucket"].to_numpy()
    y = obs["cpr_observed"].to_numpy()
    w = obs["n_loan_months"].to_numpy()

    lo = [DEFAULT_BOUNDS[k][0] for k in ("base_turnover", "max_refi_cpr", "k", "midpoint")]
    hi = [DEFAULT_BOUNDS[k][1] for k in ("base_turnover", "max_refi_cpr", "k", "midpoint")]
    p0 = [0.06, 0.34, 2.2, 0.60]           # start from the engine's illustrative defaults

    popt, pcov = curve_fit(
        s_curve_cpr, x, y, p0=p0, bounds=(lo, hi),
        sigma=1.0 / np.sqrt(w),              # heavier weight on well-observed buckets
        absolute_sigma=False, maxfev=20000,
    )
    base_turnover, max_refi_cpr, k, midpoint = popt

    y_pred = s_curve_cpr(x, *popt)
    resid = y - y_pred
    rmse = float(np.sqrt(np.average(resid ** 2, weights=w)))
    ss_res = float(np.sum(w * resid ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return CalibratedSCurve(
        base_turnover=float(base_turnover), max_refi_cpr=float(max_refi_cpr),
        logistic_k=float(k), logistic_midpoint=float(midpoint),
        n_observations=int(obs["n_loan_months"].sum()), n_buckets=len(obs),
        rmse=rmse, r_squared=float(r2), source=source_label,
    )


def fit_seasoning_ramp(loans: pd.DataFrame, *, max_months: int = 60,
                       min_obs_per_month: int = 100) -> dict:
    """Separate, simpler fit: at ~zero incentive (background turnover only,
    isolating the ramp from the refi response), how does observed SMM scale
    with loan age? Fits `ramp_months` such that
    seasoning_ramp(age) = min(age/ramp_months, 1).

    Restrict to a narrow incentive band around zero so refi behaviour
    doesn't contaminate the estimate.

    Per-month prepayment flags are Bernoulli with a small success
    probability, so the raw month-by-month mean is noisy even with a few
    hundred observations per month (see the module's own diagnostic: at
    ~1% true monthly rate, a single-month sample routinely reads 0% or
    1.5%+ by chance). A month-by-month threshold crossing on that raw
    series is unreliable -- it can trigger on an early noise spike well
    before the ramp has actually completed.

    Fix: fit `ramp_months` directly via least squares against the ramp's
    own functional form (linear-then-flat), using count-weighted regression
    over ALL months at once rather than hunting for a single crossing point.
    This is far more robust to the Bernoulli noise in any individual month.
    """
    band = loans[(loans["incentive_pp"].abs() < 0.15)
                & (loans["loan_age_months"] <= max_months)].copy()
    if len(band) < 50:
        return {"ramp_months": 30, "note": "insufficient near-zero-incentive "
                "observations; using engine default"}

    by_age = band.groupby("loan_age_months")["prepaid"].agg(["mean", "count"])
    by_age = by_age[by_age["count"] >= min_obs_per_month]
    if len(by_age) < 10:
        return {"ramp_months": 30, "note": "too few well-populated ages; "
                "using engine default"}

    # plateau = weighted average of the back half of the observed window,
    # where the ramp is presumed complete for ANY reasonable ramp_months
    tail = by_age.tail(max(6, len(by_age) // 3))
    plateau = float(np.average(tail["mean"], weights=tail["count"]))
    if plateau <= 0:
        return {"ramp_months": 30, "note": "degenerate plateau; using default"}

    ages = by_age.index.to_numpy(dtype=float)
    observed = by_age["mean"].to_numpy()
    weights = by_age["count"].to_numpy()

    def ramp_model(age, ramp_months):
        return plateau * np.minimum(age / ramp_months, 1.0)

    try:
        popt, _ = curve_fit(ramp_model, ages, observed, p0=[24.0],
                            bounds=([1.0], [float(max_months)]),
                            sigma=1.0 / np.sqrt(weights), absolute_sigma=False)
        ramp_months = float(popt[0])
    except RuntimeError:
        return {"ramp_months": 30, "note": "ramp fit did not converge; "
                "using engine default", "plateau_smm": plateau}

    return {"ramp_months": round(ramp_months), "plateau_smm": plateau}


# ------------------------------------------------- synthetic validation ---

def build_synthetic_dataset(true_params: dict, n_loans: int = 8000,
                            months: int = 84, seed: int = 7) -> pd.DataFrame:
    """Generates a loan-level panel from KNOWN parameters, so the fitting
    pipeline can be validated end-to-end before trusting it on real data.
    This mimics the Freddie Mac LLD shape closely enough that swapping in
    a real extract is a column-rename exercise, not a rewrite."""
    rng = np.random.default_rng(seed)

    orig_rate = rng.normal(0.055, 0.006, n_loans).clip(0.03, 0.08)
    orig_month = rng.integers(0, 24, n_loans)          # staggered origination

    # a simple mean-reverting synthetic path for the market mortgage rate,
    # standing in for a real PMMS/FRED series
    mkt = [0.058]
    for _ in range(months + 24):
        mkt.append(mkt[-1] + 0.35 * (0.045 - mkt[-1]) + rng.normal(0, 0.0035))
    mkt = np.array(mkt)

    rows = []
    for i in range(n_loans):
        upb = float(rng.lognormal(12.5, 0.4))
        alive = True
        for age in range(1, months + 1):
            cal_month = orig_month[i] + age
            if cal_month >= len(mkt) or not alive:
                break
            incentive = (orig_rate[i] - mkt[cal_month]) * 100.0
            cpr = s_curve_cpr(np.array([incentive]), true_params["base_turnover"],
                              true_params["max_refi_cpr"], true_params["logistic_k"],
                              true_params["logistic_midpoint"])[0]
            ramp = min(age / true_params["ramp_months"], 1.0)
            cpr *= ramp
            smm = 1 - (1 - cpr) ** (1 / 12)
            prepaid = rng.random() < smm

            rows.append(dict(
                loan_id=i, period=f"m{cal_month:04d}", orig_rate=orig_rate[i],
                loan_age_months=age, current_upb=upb,
                zero_balance_code="01" if prepaid else "",
                prevailing_mtg_rate=mkt[cal_month],
            ))
            upb *= 0.998
            if prepaid:
                alive = False
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="path to a loan-level CSV (see module docstring for schema)")
    ap.add_argument("--wac-col", type=str, default="orig_rate")
    ap.add_argument("--out", type=str, default="calibrated_params.json")
    ap.add_argument("--validate", action="store_true",
                    help="run the synthetic-data recovery check instead of "
                         "fitting real data")
    args = ap.parse_args()

    if args.validate or args.data is None:
        print("=" * 70)
        print("VALIDATION MODE: fitting synthetic data generated from KNOWN")
        print("parameters, to confirm the pipeline recovers them.")
        print("=" * 70)
        truth = dict(base_turnover=0.06, max_refi_cpr=0.34,
                    logistic_k=2.2, logistic_midpoint=0.60, ramp_months=30)
        print(f"\nTrue parameters used to generate the data:\n  {truth}\n")
        print("Generating synthetic loan-level panel (this stands in for a "
              "real Freddie Mac LLD extract)...")
        loans = build_synthetic_dataset(truth)
        print(f"  {loans.loan_id.nunique():,} loans, {len(loans):,} loan-months, "
              f"{(loans.zero_balance_code=='01').sum():,} prepayment events\n")

        obs = build_observation_table(loans, wac_col="orig_rate")
        print(f"Aggregated to {len(obs)} incentive buckets "
              f"({obs.n_loan_months.sum():,} seasoned loan-months)\n")

        fitted = fit_s_curve(obs, source_label="synthetic_validation")
        ramp = fit_seasoning_ramp(loans.assign(
            incentive_pp=(loans["orig_rate"] - loans["prevailing_mtg_rate"]) * 100,
            prepaid=(loans["zero_balance_code"] == "01").astype(int)))

        print("Fitted vs true:")
        print(f"  {'param':<20}{'true':>10}{'fitted':>10}{'diff':>10}")
        for k in ("base_turnover", "max_refi_cpr", "logistic_k", "logistic_midpoint"):
            fk = {"logistic_k": "logistic_k", "logistic_midpoint": "logistic_midpoint",
                 "base_turnover": "base_turnover", "max_refi_cpr": "max_refi_cpr"}[k]
            t = truth[k]; f = getattr(fitted, fk)
            print(f"  {k:<20}{t:>10.3f}{f:>10.3f}{f-t:>+10.3f}")
        print(f"  {'ramp_months':<20}{truth['ramp_months']:>10d}{ramp['ramp_months']:>10d}")
        print(f"\n  R-squared: {fitted.r_squared:.4f}   RMSE: {fitted.rmse:.4f}")
        print("\nClose recovery confirms the fitting machinery is correct. "
              "Point this script at a real loan-level extract with --data "
              "to get production parameters.")

        out = asdict(fitted)
        out["seasoning_ramp_months"] = ramp["ramp_months"]
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")
        return

    print(f"Loading {args.data} ...")
    loans = pd.read_csv(args.data)
    required = {"loan_age_months", "current_upb", "zero_balance_code",
               "prevailing_mtg_rate", args.wac_col}
    missing = required - set(loans.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. "
                         f"See the module docstring for the expected schema.")

    obs = build_observation_table(loans, wac_col=args.wac_col)
    print(f"Aggregated to {len(obs)} incentive buckets "
          f"({obs.n_loan_months.sum():,} seasoned loan-months)")
    if len(obs) < 6:
        raise SystemExit("Too few populated incentive buckets to fit reliably "
                         "(need a rate cycle with real dispersion in incentive -- "
                         "see the earlier discussion of the '36-month window' "
                         "problem: a flat-rate sample can't identify this curve).")

    fitted = fit_s_curve(obs, source_label=args.data)
    ramp = fit_seasoning_ramp(loans.assign(
        incentive_pp=(loans[args.wac_col] - loans["prevailing_mtg_rate"]) * 100,
        prepaid=(loans["zero_balance_code"] == "01").astype(int)))

    print(f"\nCalibrated parameters (R2={fitted.r_squared:.4f}, "
          f"RMSE={fitted.rmse:.4f}, n={fitted.n_observations:,}):")
    for k, v in fitted.to_engine_kwargs().items():
        print(f"  {k:<20} {v:.4f}")
    print(f"  {'seasoning_ramp_months':<20} {ramp['ramp_months']}")

    out = asdict(fitted)
    out["seasoning_ramp_months"] = ramp["ramp_months"]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}  ->  unpack into MBS(**json.load(open('{args.out}')))")


if __name__ == "__main__":
    main()
