"""
scenarios.py
------------
Interest rate shock scenarios for EVE / NII.

Pillar shocks are defined on the tradeable / key-rate grid
(0.25Y, 1Y, 2Y, 5Y, 7Y, 10Y) and linearly interpolated to the 19 BCBS
bucket midpoints. Parallel Up / Down remain uniform ±200 bp.

Short Up / Short Down / Steepener / Flattener use the calibrated key-tenor
shock vector supplied for this engine (not the raw BCBS Annex 2 Table 1
magnitudes at O/N–20Y).
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .time_buckets import BUCKET_LABELS, BUCKET_MIDPOINTS


# Key-rate / mortgage-pricing pillars (years) for scenario shock definition
REF_TENORS = [0.25, 1.0, 2.0, 5.0, 7.0, 10.0]
REF_LABELS = ["0.25Y", "1Y", "2Y", "5Y", "7Y", "10Y"]


def _interpolate_shocks(ref_shocks_bp: list[int]) -> list[float]:
    """
    Linearly interpolate reference-tenor shocks to each of the 19 bucket
    midpoints. Values outside the pillar range are held at the nearest edge.
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
        if len(self.ref_shocks_bp) != len(REF_TENORS):
            raise ValueError(
                f"{self.id}: expected {len(REF_TENORS)} pillar shocks, "
                f"got {len(self.ref_shocks_bp)}"
            )
        self.shocks_bp = _interpolate_shocks(self.ref_shocks_bp)

    def shock_series(self) -> pd.Series:
        return pd.Series(self.shocks_bp, index=BUCKET_LABELS, name=self.id)

    def shock_at_bucket(self, bucket_index: int) -> float:
        return self.shocks_bp[bucket_index]


# Pillar order: 0.25Y, 1Y, 2Y, 5Y, 7Y, 10Y (bp)
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
        description="Short rates down / long rates up (key-tenor calibrated)",
        ref_shocks_bp=[-175, -122, -65, +40, +76, +108],
    ),
    Scenario(
        id="FLATTENER", name="Flattener",
        description="Short rates up / long rates down (key-tenor calibrated)",
        ref_shocks_bp=[+220, +167, +110, +5, -29, -63],
    ),
    Scenario(
        id="SHORT_UP", name="Short Rates Up",
        description="Short-end shock up, decaying to +25bp at 10Y",
        ref_shocks_bp=[+282, +234, +182, +86, +49, +25],
    ),
    Scenario(
        id="SHORT_DOWN", name="Short Rates Down",
        description="Short-end shock down, decaying to -25bp at 10Y",
        ref_shocks_bp=[-282, -234, -182, -86, -49, -25],
    ),
]

SCENARIO_MAP: dict[str, Scenario] = {s.id: s for s in SCENARIOS}
