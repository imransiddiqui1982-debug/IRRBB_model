"""Fetch live SOFR (0-12M) and USD IRS mid (1Y-10Y+) and build IRRBB YieldCurve."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .yield_curve import YieldCurve

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "live_curve_cache.json")
NYFED_LATEST = "https://markets.newyorkfed.org/api/rates/all/latest.json"
BLUEGAMMA_SWAPS = "https://www.bluegamma.io/usd-swap-rates"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

UA = {"User-Agent": "Mozilla/5.0 IRRBB-Model/1.0 (research; contact: local)"}


@dataclass
class MarketCurveSnapshot:
    as_of: str
    source_notes: list[str] = field(default_factory=list)
    points: dict[float, float] = field(default_factory=dict)  # tenor years -> rate decimal
    short_end: dict[str, float] = field(default_factory=dict)
    irs_mid: dict[str, float] = field(default_factory=dict)

    def to_yield_curve(self) -> YieldCurve:
        tenors = sorted(self.points.keys())
        rates = [self.points[t] for t in tenors]
        if not tenors:
            return YieldCurve()
        # Extend beyond 10Y with flat 10Y so 15Y/20Y/30Y buckets still discount
        if tenors[-1] < 30:
            tenors = tenors + [15.0, 20.0, 30.0]
            rates = rates + [rates[-1], rates[-1], rates[-1]]
        return YieldCurve(ref_tenors=tenors, ref_rates=rates)


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _fred_latest(series: str) -> tuple[str, float] | None:
    text = _http_get(FRED_CSV.format(series=series))
    for line in reversed(text.strip().splitlines()[1:]):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date, val = parts[0].strip(), parts[1].strip()
        if val in ("", ".", "NA"):
            continue
        return date, float(val) / 100.0
    return None


def fetch_sofr_short_end() -> tuple[dict[str, float], list[str], str]:
    """
    0–12M SOFR curve points from NY Fed (overnight + compounded averages).

    Mapping:
      O/N  -> published SOFR
      1M   -> 30-day SOFR average
      3M   -> 90-day SOFR average
      6M   -> 180-day SOFR average
      12M  -> interpolated later vs 1Y IRS mid
    """
    notes = ["Short end: NY Fed Markets API (SOFR + 30/90/180-day averages)"]
    raw = json.loads(_http_get(NYFED_LATEST))
    rates: dict[str, float] = {}
    as_of = ""
    for item in raw.get("refRates", []):
        typ = item.get("type")
        if typ == "SOFR" and "percentRate" in item:
            rates["ON"] = float(item["percentRate"]) / 100.0
            as_of = item.get("effectiveDate", as_of)
        elif typ == "SOFRAI":
            if item.get("average30day") is not None:
                rates["1M"] = float(item["average30day"]) / 100.0
            if item.get("average90day") is not None:
                rates["3M"] = float(item["average90day"]) / 100.0
            if item.get("average180day") is not None:
                rates["6M"] = float(item["average180day"]) / 100.0
            as_of = item.get("effectiveDate", as_of) or as_of
    if "ON" not in rates:
        raise RuntimeError("NY Fed response missing overnight SOFR")
    # Fill missing averages from O/N if needed
    for k in ("1M", "3M", "6M"):
        rates.setdefault(k, rates["ON"])
    return rates, notes, as_of


def fetch_usd_irs_mids() -> tuple[dict[str, float], list[str], str]:
    """
    USD SOFR IRS mid rates for 1Y–10Y from BlueGamma public HTML (Yesterday column).
    Fallback: FRED Treasury CMT (DGS) as last-resort proxy.
    """
    notes: list[str] = []
    as_of = ""
    try:
        html = _http_get(BLUEGAMMA_SWAPS)
        # Meta description often embeds key tenors
        meta = re.search(
            r'content="[^"]*?2Y\s+([0-9.]+)%.+?5Y\s+([0-9.]+)%.+?10Y\s+([0-9.]+)%',
            html,
            re.I,
        )
        # Parse tenor rows: after each tenor label, first numeric % is Live/Unlock,
        # second is Yesterday mid (public).
        tenor_map = {
            "1 Month": "1M", "3 Month": "3M", "1 Year": "1Y", "2 Year": "2Y",
            "3 Year": "3Y", "4 Year": "4Y", "5 Year": "5Y", "7 Year": "7Y",
            "8 Year": "8Y", "10 Year": "10Y", "15 Year": "15Y", "20 Year": "20Y",
            "30 Year": "30Y",
        }
        rates: dict[str, float] = {}
        for label, key in tenor_map.items():
            # Tenor is in <th>…</th>; Live cell is locked; next <td>N.NN%</td> is Yesterday mid.
            pat = (
                rf">{re.escape(label)}</a></th>"
                rf".{{0,800}}?"
                rf"<td[^>]*>\s*([0-9]+\.[0-9]+)%\s*</td>"
            )
            m = re.search(pat, html, re.I | re.S)
            if m:
                rates[key] = float(m.group(1)) / 100.0

        m_date = re.search(
            r"Last update:\s*([A-Za-z]+ \d+, \d{4})", html
        ) or re.search(r'"dateModified":"([0-9-]+)"', html)
        if m_date:
            as_of = m_date.group(1)

        # Ensure core IRS tenors from meta if row parse missed them
        if meta:
            rates.setdefault("2Y", float(meta.group(1)) / 100.0)
            rates.setdefault("5Y", float(meta.group(2)) / 100.0)
            rates.setdefault("10Y", float(meta.group(3)) / 100.0)

        need = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
        if sum(1 for k in need if k in rates) >= 4:
            notes.append(
                "IRS mid 1Y–10Y: BlueGamma SOFR swap mids (public Yesterday column)"
            )
            return rates, notes, as_of
        raise RuntimeError(f"Insufficient BlueGamma IRS tenors: {list(rates)}")
    except Exception as exc:
        notes.append(f"BlueGamma IRS fetch failed ({exc}); falling back to FRED DGS")

    # Fallback Treasury CMT
    rates = {}
    for key, series in [
        ("1Y", "DGS1"), ("2Y", "DGS2"), ("3Y", "DGS3"),
        ("5Y", "DGS5"), ("7Y", "DGS7"), ("10Y", "DGS10"),
        ("20Y", "DGS20"), ("30Y", "DGS30"),
    ]:
        got = _fred_latest(series)
        if got:
            as_of = got[0]
            rates[key] = got[1]
    notes.append("IRS mid proxy: FRED Treasury constant-maturity yields (DGS)")
    if len(rates) < 3:
        raise RuntimeError("Unable to fetch USD IRS / Treasury long-end rates")
    return rates, notes, as_of


def build_live_curve_snapshot() -> MarketCurveSnapshot:
    sofr, n1, d1 = fetch_sofr_short_end()
    irs, n2, d2 = fetch_usd_irs_mids()

    # Prefer forward-looking SOFR swap tenors for 1M/3M when published alongside IRS
    if "1M" in irs:
        sofr["1M"] = irs["1M"]
        n1.append("1M SOFR: BlueGamma SOFR swap mid (replaces 30d average)")
    if "3M" in irs:
        sofr["3M"] = irs["3M"]
        n1.append("3M SOFR: BlueGamma SOFR swap mid (replaces 90d average)")

    # 12M SOFR: 1Y SOFR IRS mid (par swap expectation of compounded SOFR)
    sofr_12m = irs.get("1Y", sofr["6M"])
    sofr["12M"] = sofr_12m

    points: dict[float, float] = {
        0.0: sofr["ON"],
        1 / 12: sofr["1M"],
        0.25: sofr["3M"],
        0.5: sofr.get("6M", 0.5 * (sofr["3M"] + sofr_12m)),
        1.0: sofr_12m,
    }
    # Linear 6M if only averages: blend 3M and 1Y when BlueGamma has no 6M
    if "6M" not in irs:
        points[0.5] = 0.5 * (sofr["3M"] + sofr_12m)
        sofr["6M"] = points[0.5]
        n1.append("6M SOFR: interpolated between 3M and 1Y SOFR swap mids")

    for key, years in [
        ("2Y", 2.0), ("3Y", 3.0), ("4Y", 4.0), ("5Y", 5.0),
        ("7Y", 7.0), ("8Y", 8.0), ("10Y", 10.0),
        ("15Y", 15.0), ("20Y", 20.0), ("30Y", 30.0),
    ]:
        if key in irs:
            points[years] = irs[key]

    as_of = d2 or d1 or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap = MarketCurveSnapshot(
        as_of=as_of,
        source_notes=n1 + n2,
        points=points,
        short_end=sofr,
        irs_mid={k: v for k, v in irs.items()},
    )
    _save_cache(snap)
    return snap


def _save_cache(snap: MarketCurveSnapshot) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    payload = {
        "as_of": snap.as_of,
        "source_notes": snap.source_notes,
        "points": {str(k): v for k, v in snap.points.items()},
        "short_end": snap.short_end,
        "irs_mid": snap.irs_mid,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_cached_snapshot() -> MarketCurveSnapshot | None:
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return MarketCurveSnapshot(
        as_of=payload.get("as_of", ""),
        source_notes=payload.get("source_notes", []),
        points={float(k): float(v) for k, v in payload.get("points", {}).items()},
        short_end=payload.get("short_end", {}),
        irs_mid=payload.get("irs_mid", {}),
    )


def get_live_yield_curve(use_cache_on_failure: bool = True) -> tuple[YieldCurve, MarketCurveSnapshot]:
    """Fetch live market curve; fall back to cache then stylised curve."""
    try:
        snap = build_live_curve_snapshot()
        return snap.to_yield_curve(), snap
    except Exception as exc:
        if use_cache_on_failure:
            cached = load_cached_snapshot()
            if cached and cached.points:
                cached.source_notes = list(cached.source_notes) + [
                    f"Live fetch failed ({exc}); using cached curve"
                ]
                return cached.to_yield_curve(), cached
        raise
