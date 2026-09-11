"""End-to-end NII demo: same balance sheet as alm_engine's demo.py, run
through both the EVE engine and the new NII engine, side by side."""
import numpy as np

from alm_engine import (Curve, KeyRateGrid, Portfolio, FixedBullet, Floater,
                        NMD, MBS, SCENARIOS)
from nii_engine import NIIPortfolio, NIIScenario, BCBS_NII_SCENARIOS, US_NII_SCENARIOS

BAR = "=" * 78


def main():
    curve = Curve.flat(0.04)

    # Same instruments as the EVE demo, PLUS current_rate/beta_down set on
    # the NMD (needed for NII, not used by EVE at all).
    def make_book():
        return [
            Floater("floating loans", 200.0, next_reset=0.25),
            FixedBullet("2y auto loans", 150.0, coupon=0.045, years=2),
            FixedBullet("5y CRE", 250.0, coupon=0.055, years=5),
            MBS("30y agency MBS", 200.0, wac=0.055, months=324,
               anchor_tenor=10.0, spread_to_curve=0.0175, oas=0.005),
            NMD("core deposits", 300.0, stable_pct=0.88, beta=0.35,
               beta_down=0.55, current_rate=0.010,     # NII fields
               core_cap=0.70, wal_cap=4.5, decay=0.33, side=-1),
            FixedBullet("6m CDs", 250.0, coupon=0.040, years=0.5, side=-1),
            FixedBullet("5y FHLB advances", 150.0, coupon=0.042, years=5, side=-1),
        ]

    eve_book = Portfolio(make_book(), curve, KeyRateGrid(), tier1=100.0)
    nii_book = NIIPortfolio(make_book(), curve, horizon_months=12,
                            constant_balance_sheet=True, tier1=100.0)

    print(BAR); print("SAME BALANCE SHEET -- EVE view vs NII view"); print(BAR)
    for p in eve_book.positions:
        tag = "asset " if p.side > 0 else "liab  "
        print(f"  {tag} {p.name:<22} {p.notional:>8,.0f}")

    # ------------------------------------------------------- EVE recap ---
    print(f"\n{BAR}"); print("EVE (recap) -- discounting-based, run-off, no horizon"); print(BAR)
    for row in eve_book.limit_status(limit_pct=15.0, amber_pct=12.0):
        print(f"  {row['scenario']:<12}{row['d_eve']:>10.2f}m{row['pct_tier1']:>9.1f}%{row['status']:>9}")

    # --------------------------------------------------- NII: BCBS mode ---
    print(f"\n{BAR}"); print("NII -- BCBS 368 COMPLIANT MODE"); print(BAR)
    print("  2 scenarios, 12-month horizon, constant balance sheet")
    base_nii = nii_book.base_nii()
    print(f"\n  Base 12-month NII: ${base_nii:.3f}m\n")
    rows = nii_book.limit_status(BCBS_NII_SCENARIOS, limit_pct_of_tier1=5.0,
                                 limit_pct_of_base_nii=None)
    print(f"  {'scenario':<16}{'dNII':>10}{'% Tier1':>10}{'status (5% T1)':>16}")
    for r in rows:
        print(f"  {r['scenario']:<16}{r['d_nii']:>9.3f}m{r['pct_tier1']:>9.1f}%{r['status_tier1']:>16}")
    print("\n  Note: Basel sets NO mandated NII threshold. The 5%-of-Tier-1 line")
    print("  shown is the EU/EBA convention, included only as a common external")
    print("  reference point -- not applied here as if it were a Basel rule.")

    # ---------------------------------------------------- NII: US mode ---
    print(f"\n{BAR}"); print("NII -- US BANK PRACTICE MODE"); print(BAR)
    print("  8 instantaneous shocks (+/-100/200/300/400bp) + 2 twelve-month ramps")
    print("  Shown at BOTH 12-month and 24-month horizons, both denominators\n")

    for horizon in (12, 24):
        nb = NIIPortfolio(make_book(), curve, horizon_months=horizon,
                          constant_balance_sheet=True, tier1=100.0)
        base = nb.base_nii()
        print(f"  --- {horizon}-month horizon (base NII ${base:.3f}m) ---")
        print(f"  {'scenario':<16}{'dNII':>10}{'% Tier1':>10}{'% base NII':>12}")
        rows = nb.limit_status(US_NII_SCENARIOS, limit_pct_of_tier1=5.0,
                               limit_pct_of_base_nii=10.0)
        for r in rows:
            print(f"  {r['scenario']:<16}{r['d_nii']:>9.3f}m{r['pct_tier1']:>9.1f}%"
                  f"{r.get('pct_base_nii', float('nan')):>11.1f}%")
        print()

    # --------------------------------------- static vs constant balance sheet ---
    print(f"{BAR}"); print("STATIC vs CONSTANT BALANCE SHEET -- same shock, different assumption")
    print(BAR)
    static = NIIPortfolio(make_book(), curve, horizon_months=12,
                          constant_balance_sheet=False, tier1=100.0)
    constant = NIIPortfolio(make_book(), curve, horizon_months=12,
                            constant_balance_sheet=True, tier1=100.0)
    for label, port in [("static (no reinvestment)", static),
                        ("constant (reinvest at scenario rate)", constant)]:
        b = port.base_nii()
        up = port.run(NIIScenario("up", 200)).total_nii
        print(f"  {label:<40} base ${b:.3f}m   +200bp ${up:.3f}m   "
              f"delta {up-b:+.3f}m")
    print("\n  Static understates sensitivity -- maturing/prepaying balance just")
    print("  disappears rather than repricing, so repricing risk goes unmeasured.")

    # -------------------------------------------------------- by instrument ---
    print(f"\n{BAR}"); print("NII CONTRIBUTION BY INSTRUMENT (12m, +200bp)"); print(BAR)
    r = nii_book.run(NIIScenario("up", 200.0))
    r0 = nii_book.run(NIIScenario("base", 0.0))
    print(f"  {'instrument':<22}{'base':>10}{'+200bp':>10}{'delta':>10}")
    for name in r.by_instrument:
        b, u = r0.by_instrument[name], r.by_instrument[name]
        print(f"  {name:<22}{b:>9.3f}m{u:>9.3f}m{u-b:>+9.3f}m")
    print(f"  {'TOTAL':<22}{r0.total_nii:>9.3f}m{r.total_nii:>9.3f}m"
          f"{r.total_nii-r0.total_nii:>+9.3f}m")


if __name__ == "__main__":
    main()
