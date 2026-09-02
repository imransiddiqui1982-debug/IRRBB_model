"""
time_buckets.py
---------------
BCBS 368 (April 2016) — Annex 2, Table 2
The 19 prescribed repricing time buckets for notional cash flow slotting.

Each bucket is defined by:
  - label       : human-readable name
  - lower_years : lower bound in years (inclusive)
  - upper_years : upper bound in years (exclusive; None = open-ended)
  - midpoint    : representative tenor used for shock interpolation and
                  duration approximation (years)

Reference: BCBS 368, Annex 2, Table 2 — "Maturity schedule with 19 time
buckets for notional repricing cash flows".
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimeBucket:
    index:       int             # 0-based
    label:       str
    lower_years: float
    upper_years: Optional[float]  # None = open-ended (20Y+)
    midpoint:    float           # representative tenor in years


# ── 19 BCBS 368 time buckets ──────────────────────────────────────────────────
BCBS_BUCKETS: list[TimeBucket] = [
    TimeBucket(0, "Overnight",       0.000,  1/365,   1/730),
    TimeBucket(1, "O/N – 1M",        1/365,  1/12,    1/24),
    TimeBucket(2, "1M – 3M",         1/12,   3/12,    2/12),
    TimeBucket(3, "3M – 6M",         3/12,   6/12,    4.5/12),
    TimeBucket(4, "6M – 9M",         6/12,   9/12,    7.5/12),
    TimeBucket(5, "9M – 12M",        9/12,   1.0,     10.5/12),
    TimeBucket(6, "1Y – 1.5Y",       1.0,    1.5,     1.25),
    TimeBucket(7, "1.5Y – 2Y",       1.5,    2.0,     1.75),
    TimeBucket(8, "2Y – 3Y",         2.0,    3.0,     2.5),
    TimeBucket(9, "3Y – 4Y",         3.0,    4.0,     3.5),
    TimeBucket(10, "4Y – 5Y",         4.0,    5.0,     4.5),
    TimeBucket(11, "5Y – 6Y",         5.0,    6.0,     5.5),
    TimeBucket(12, "6Y – 7Y",         6.0,    7.0,     6.5),
    TimeBucket(13, "7Y – 8Y",         7.0,    8.0,     7.5),
    TimeBucket(14, "8Y – 9Y",         8.0,    9.0,     8.5),
    TimeBucket(15, "9Y – 10Y",        9.0,   10.0,     9.5),
    TimeBucket(16, "10Y – 15Y",      10.0,   15.0,    12.5),
    TimeBucket(17, "15Y – 20Y",      15.0,   20.0,    17.5),
    TimeBucket(18, "20Y+",           20.0,   None,    25.0),
]

N_BUCKETS = len(BCBS_BUCKETS)               # 19
BUCKET_LABELS = [b.label for b in BCBS_BUCKETS]
BUCKET_MIDPOINTS = [b.midpoint for b in BCBS_BUCKETS]


def years_to_bucket(years: float) -> int:
    """
    Map a tenor in years to the correct BCBS 368 bucket index (0-based).
    Uses the bucket's [lower, upper) half-open interval.
    The last bucket (20Y+) captures everything beyond 20 years.
    """
    for bucket in BCBS_BUCKETS:
        upper = bucket.upper_years if bucket.upper_years is not None else float("inf")
        if bucket.lower_years <= years < upper:
            return bucket.index
    # Fallback: slot into last bucket (should not normally reach here)
    return N_BUCKETS - 1


def get_bucket(index: int) -> TimeBucket:
    return BCBS_BUCKETS[index]
