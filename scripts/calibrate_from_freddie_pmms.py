"""
Calibrate Step-B prepay params from Freddie Mac SFLLD samples + FRED PMMS.

1. Register / download sample ZIPs from Freddie Clarity SFLLD:
   https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
2. Unzip into data/freddie/ so you have pairs like:
     data/freddie/sample_orig_2020.txt
     data/freddie/sample_svcg_2020.txt
3. Run:
     python scripts/calibrate_from_freddie_pmms.py
   → writes data/calibrated_prepayment_params.json

Also refreshes data/pmms_history.csv from FRED MORTGAGE30US.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrate_prepayment import (  # noqa: E402
    DEFAULT_PARAMS_PATH,
    build_observation_table,
    clear_calibrated_params_cache,
    fit_s_curve,
    fit_seasoning_ramp,
)
from src.freddie_lld import build_calibration_frame, find_freddie_sample_pairs  # noqa: E402
from src.pmms import fetch_pmms_history, get_latest_pmms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--freddie-dir",
        type=str,
        default=str(ROOT / "data" / "freddie"),
        help="Folder containing sample_orig_*/sample_svcg_* files",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_PARAMS_PATH),
        help="Output calibrated_prepayment_params.json",
    )
    ap.add_argument("--max-rows", type=int, default=1_500_000)
    ap.add_argument(
        "--pmms-only",
        action="store_true",
        help="Only refresh PMMS caches; skip Freddie fit",
    )
    args = ap.parse_args()

    print("Refreshing FRED PMMS (MORTGAGE30US)...")
    snap = get_latest_pmms(use_cache_on_failure=False)
    print(f"  Latest PMMS: {snap['rate'] * 100:.3f}% as of {snap['as_of']}")
    hist = fetch_pmms_history()
    print(f"  History months: {len(hist)} -> data/pmms_history.csv")

    if args.pmms_only:
        return 0

    freddie_dir = Path(args.freddie_dir)
    pairs = find_freddie_sample_pairs(freddie_dir)
    if not pairs:
        msg = (
            "\nNo Freddie sample pairs found under data/freddie/.\n"
            "Download sample_YYYY.zip from Freddie Clarity SFLLD, unzip so that\n"
            "  sample_orig_YYYY.txt + sample_svcg_YYYY.txt (or sample_perf_YYYY.txt)\n"
            "are visible under data/freddie/, then re-run this script.\n"
            "PMMS caches were updated; Step A PMMS anchor already works without Freddie.\n"
        )
        try:
            print(msg)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(msg.encode("utf-8", errors="replace"))
        return 2

    print(f"\nFound {len(pairs)} Freddie vintage pair(s). Building loan-month panel…")
    loans = build_calibration_frame(freddie_dir, max_rows=args.max_rows)
    print(
        f"  {loans['loan_id'].nunique():,} loans · {len(loans):,} loan-months · "
        f"prepays={(loans['zero_balance_code'] == '01').sum():,}"
    )

    obs = build_observation_table(loans, wac_col="orig_rate")
    print(f"  Incentive buckets: {len(obs)} (seasoned loan-months={obs['n_loan_months'].sum():,.0f})")
    if len(obs) < 6:
        print("Too few incentive buckets to fit reliably.")
        return 3

    fitted = fit_s_curve(obs, source_label=f"freddie_lld:{freddie_dir}")
    ramp = fit_seasoning_ramp(
        loans.assign(
            incentive_pp=(loans["orig_rate"] - loans["prevailing_mtg_rate"]) * 100.0,
            prepaid=(loans["zero_balance_code"] == "01").astype(int),
        )
    )

    out = asdict(fitted)
    out["seasoning_ramp_months"] = int(ramp.get("ramp_months", 30))
    out["pmms_latest"] = snap["rate"]
    out["pmms_as_of"] = snap["as_of"]
    out["note"] = (
        "Step-B params fit on Freddie SFLLD samples with FRED PMMS as prevailing "
        "mortgage rate. Step A anchors to live PMMS via apply_pmms_anchor."
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    clear_calibrated_params_cache()

    print("\nFitted Step-B parameters:")
    for k, v in fitted.to_engine_kwargs().items():
        print(f"  {k:<22} {v:.6f}")
    print(f"  {'seasoning_ramp_months':<22} {out['seasoning_ramp_months']}")
    print(f"  R²={fitted.r_squared:.4f}  RMSE={fitted.rmse:.4f}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
