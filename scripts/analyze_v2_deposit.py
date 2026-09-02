"""Analyze deposit Excel using NMD engine (supports v2 with Deposit Rates sheet)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.nmd_refinement import refine_nmd_deposits  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\acer\OneDrive\Desktop\comprehensive_deposit_template_v2.xlsx"

result = refine_nmd_deposits(path, deposit_name="v2 Upload Pool")
term_total = sum(i.notional for i in result.term_instruments)
nmd_total = result.latest_balance_mb - term_total

print("=" * 64)
print("NMD ENGINE — monthly deposit rates for beta")
print("=" * 64)
print(f"File: {path}")
print(f"Customers: {result.customer_count} | Months: {result.month_count}")
print(f"Total balance: {result.latest_balance_mb:,.2f} M | NMD pool: {nmd_total:,.2f} M")
print()
pct = lambda x: 100 * x / nmd_total if nmd_total else 0
print(f"Core sticky:       {result.core_balance_mb:>10,.2f} M  ({pct(result.core_balance_mb):.1f}%)")
print(f"Rate sensitive:    {result.rate_sensitive_balance_mb:>10,.2f} M  ({pct(result.rate_sensitive_balance_mb):.1f}%)")
print(f"Non-core volatile: {result.non_core_balance_mb:>10,.2f} M  ({pct(result.non_core_balance_mb):.1f}%)")
print()
print(f"Stable %: {result.stable_pct*100:.2f} | Beta: {result.beta:.4f} | Sticky %: {result.sticky_pct*100:.2f}")
print()
for _, row in result.segment_summary.iterrows():
    print(
        f"{row['Segment']}: beta={row['Beta']:.4f} | "
        f"core sticky {row['Core Sticky ($M)']:,.2f} | "
        f"rate sens {row['Rate Sensitive ($M)']:,.2f} | "
        f"non-core {row['Non-core ($M)']:,.2f}"
    )
