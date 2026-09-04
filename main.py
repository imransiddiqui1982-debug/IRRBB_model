"""
main.py — IRRBB Model v3
Basel III / BCBS 368 · 19 Buckets · Full Cash Flow Discounting
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.balance_sheet import get_instruments  # noqa: E402
from src.scenarios import SCENARIOS  # noqa: E402
from src.calculator import IRRBBCalculator  # noqa: E402
from src.yield_curve import BASE_CURVE  # noqa: E402
from src.plots import (  # noqa: E402
    plot_nii, plot_eve, plot_repricing_gap, plot_shock_curves,
    plot_nii_decomposition, plot_instrument_eve_waterfall, plot_yield_curve,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def print_balance_sheet(assets, liabilities):
    for side, instruments in (("ASSETS", assets), ("LIABILITIES", liabilities)):
        print(f"\n  {side}")
        header = (
            f"  {'Instrument':<40} {'Notional':>9} {'Rate':>6}"
            f"  {'Type':<16}  {'Eff Dur':>7}  {'CFs':>5}"
        )
        print(header)
        print("  " + "─" * 84)
        total = 0
        for inst in instruments:
            line = (
                f"  {inst.name:<40} ${inst.notional:>7.0f}M {inst.coupon_pct:>5.2f}%"
                f"  {inst.instrument_type:<16}  {inst.effective_duration:>6.2f}Y"
                f"  {len(inst.cashflows):>5}"
            )
            print(line)
            total += inst.notional
        print(f"  {'TOTAL':<40} ${total:>7.0f}M")


def print_summary(results, tier1):
    sep = "─" * 86
    outlier_limit = tier1 * 0.15
    print(f"\nBCBS 368 RESULTS  |  Tier 1: ${tier1:.0f}M  |  Outlier: ${outlier_limit:.0f}M (15%)")
    print(sep)
    print(
        f"  {'Scenario':<24} {'ΔNII ($M)':>10}"
        f" {'ΔEVE ($M)':>10} {'|ΔEVE|/T1':>10}  Status"
    )
    print(sep)
    for r in results:
        flag = " ⚠ OUTLIER" if r.is_outlier else (" ~ WATCH" if r.is_watch else "")
        print(
            f"  {r.scenario.name:<24} {r.delta_nii:>+9.2f}M"
            f" {r.delta_eve:>+9.2f}M {r.delta_eve_pct:>9.1f}%  {r.status}{flag}"
        )
    print(sep)
    outliers = [r for r in results if r.is_outlier]
    watches = [r for r in results if r.is_watch]
    if outliers:
        n = len(outliers)
        print(f"\n  ⚠  {n} outlier breach(es). BCBS 368 §99: supervisor notification required.")
    elif watches:
        print(f"\n  ~  {len(watches)} scenario(s) approaching outlier threshold.")
    else:
        print("\n  All scenarios within outlier threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1", type=float, default=500.0)
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to balance sheet CSV (default: built-in sample balance sheet)",
    )
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 86)
    print("  IRRBB MODEL v3 — Basel III / BCBS 368 (April 2016)")
    print("  19 Repricing Buckets · 6 Shock Scenarios · Full Cash Flow Discounting")
    print("=" * 86)

    if args.csv:
        from src.load_balance_sheet import load_instruments_from_csv
        assets, liabilities = load_instruments_from_csv(args.csv)
        print(f"\n  Loaded balance sheet from: {args.csv}")
    else:
        assets, liabilities = get_instruments()
    total_cfs = sum(len(i.cashflows) for i in assets + liabilities)
    print(
        f"\n  Balance sheet: {len(assets)} assets,"
        f" {len(liabilities)} liabilities, {total_cfs} scheduled cash flows"
    )
    print_balance_sheet(assets, liabilities)

    calc = IRRBBCalculator(assets, liabilities, tier1_capital=args.tier1)
    results = calc.run_all(SCENARIOS)
    print_summary(results, args.tier1)

    from src.liquidity_ratios import compute_liquidity_ratios
    liq = compute_liquidity_ratios(assets, liabilities, tier1_capital=args.tier1)
    print(f"\n  LCR:  {liq.lcr.lcr_pct:.1f}%  ({'PASS' if liq.lcr.lcr_pass else 'FAIL'})")
    print(f"        HQLA ${liq.lcr.hqla_stock:.1f}M / Net outflows ${liq.lcr.net_cash_outflows:.1f}M")
    print(f"  NSFR: {liq.nsfr.nsfr_pct:.1f}%  ({'PASS' if liq.nsfr.nsfr_pass else 'FAIL'})")
    print(f"        ASF ${liq.nsfr.asf_total:.1f}M / RSF ${liq.nsfr.rsf_total:.1f}M")

    out = OUTPUT_DIR
    calc.summary_table(results).to_csv(os.path.join(out, "irrbb_summary.csv"), index=False)
    ps_up = next(r for r in results if r.scenario.id == "PS_UP")
    detail = calc.instrument_eve_detail(ps_up.scenario)
    detail.to_csv(os.path.join(out, "eve_attribution_ps_up.csv"), index=False)
    calc.repricing_gap().to_csv(os.path.join(out, "repricing_gap.csv"))

    print("\n  Generating charts...")
    gap = calc.repricing_gap()
    plot_shock_curves(SCENARIOS, save_path=os.path.join(out, "shock_curves.png"))
    plot_yield_curve(BASE_CURVE, SCENARIOS, save_path=os.path.join(out, "yield_curve.png"))
    plot_nii(results, save_path=os.path.join(out, "nii_sensitivity.png"))
    plot_eve(results, save_path=os.path.join(out, "eve_sensitivity.png"))
    plot_repricing_gap(gap, save_path=os.path.join(out, "repricing_gap.png"))
    plot_nii_decomposition(results, save_path=os.path.join(out, "nii_decomposition.png"))
    plot_instrument_eve_waterfall(
        detail, ps_up.scenario.name,
        save_path=os.path.join(out, "eve_waterfall.png"),
    )
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
