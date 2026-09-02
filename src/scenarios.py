"""
scenarios.py
------------
BCBS 368 (April 2016) — Annex 2
Six prescribed interest rate shock scenarios.

The standard defines shocks at key reference tenors (overnight, 1Y, 2Y, 5Y,
10Y, 20Y) and requires interpolation for intermediate tenors. This module
implements linear interpolation between reference points to produce a
shock value for each of the 19 time bucket midpoints.

Reference tenors and shocks (basis points) per scenario:
  Tenors: [O/N, 1Y, 2Y, 5Y, 10Y, 20Y] (in years)

Source: BCBS 368, Annex 2, Table 1.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .time_buckets import BUCKET_LABELS, BUCKET_MIDPOINTS


# Reference tenors from BCBS 368 Annex 2 Table 1 (years)
REF_TENORS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]


def _interpolate_shocks(ref_shocks_bp: list[int]) -> list[float]:
    """
    Linearly interpolate reference-tenor shocks to each of the 19 bucket
    midpoints. Values beyond the last reference tenor are held flat.
    """
    return list(np.interp(BUCKET_MIDPOINTS, REF_TENORS, ref_shocks_bp))


@dataclass
class Scenario:
    id:             str
    name:           str
    description:    str
    ref_shocks_bp:  list[int]
    shocks_bp:      list[float] = field(init=False)

    def __post_init__(self):
        self.shocks_bp = _interpolate_shocks(self.ref_shocks_bp)

    def shock_series(self) -> pd.Series:
        return pd.Series(self.shocks_bp, index=BUCKET_LABELS, name=self.id)

    def shock_at_bucket(self, bucket_index: int) -> float:
        return self.shocks_bp[bucket_index]


# Six BCBS 368 scenarios — ref shocks at [O/N, 1Y, 2Y, 5Y, 10Y, 20Y] in bps
SCENARIOS: list[Scenario] = [
    Scenario(
        id="PS_UP",   name="Parallel Shift Up",
        description="Uniform +200bp across all tenors",
        ref_shocks_bp=[200, 200, 200, 200, 200, 200],
    ),
    Scenario(
        id="PS_DOWN", name="Parallel Shift Down",
        description="Uniform -200bp across all tenors",
        ref_shocks_bp=[-200, -200, -200, -200, -200, -200],
    ),
    Scenario(
        id="STEEPENER", name="Steepener",
        description="Short rates down / long rates up",
        ref_shocks_bp=[-100, -75, -50, 0, +100, +150],
    ),
    Scenario(
        id="FLATTENER", name="Flattener",
        description="Short rates up / long rates down",
        ref_shocks_bp=[+100, +75, +50, 0, -100, -150],
    ),
    Scenario(
        id="SHORT_UP", name="Short Rates Up",
        description="Short-end shock up, long end unchanged",
        ref_shocks_bp=[+250, +200, +150, +75, 0, 0],
    ),
    Scenario(
        id="SHORT_DOWN", name="Short Rates Down",
        description="Short-end shock down, long end unchanged",
        ref_shocks_bp=[-250, -200, -150, -75, 0, 0],
    ),
]

SCENARIO_MAP: dict[str, Scenario] = {s.id: s for s in SCENARIOS}
