"""
pmms.py
-------
Freddie Mac Primary Mortgage Market Survey (PMMS) 30Y via FRED ``MORTGAGE30US``.

Used to anchor Step A mortgage rate so incentive = WAC − PMMS (at the
base curve), while BCBS / KR01 curve bumps still move the mortgage rate
one-for-one with the pool anchor tenor.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .market_curve import FRED_CSV, _fred_latest, _http_get

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "pmms_cache.json"
HISTORY_CACHE = Path(__file__).resolve().parent.parent / "data" / "pmms_history.csv"


def fetch_latest_pmms() -> tuple[str, float]:
    """
    Latest PMMS 30Y conventional rate from FRED.

    Returns (as_of_date ISO, rate as decimal e.g. 0.0671).
    """
    hit = _fred_latest("MORTGAGE30US")
    if hit is None:
        raise RuntimeError("FRED MORTGAGE30US returned no observations")
    return hit[0], float(hit[1])


def fetch_pmms_history() -> pd.Series:
    """
    Full FRED MORTGAGE30US history as monthly Series (period end, decimal).

    Cached to ``data/pmms_history.csv``.
    """
    text = _http_get(FRED_CSV.format(series="MORTGAGE30US"), timeout=60)
    rows = []
    for line in text.strip().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date, val = parts[0].strip(), parts[1].strip()
        if val in ("", ".", "NA"):
            continue
        rows.append((date, float(val) / 100.0))
    if not rows:
        raise RuntimeError("Empty FRED MORTGAGE30US history")
    df = pd.DataFrame(rows, columns=["date", "pmms"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Month-end series for joining Freddie YYYYMM periods
    monthly = df["pmms"].resample("ME").last().dropna()
    HISTORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(HISTORY_CACHE, header=["pmms"])
    return monthly


@lru_cache(maxsize=1)
def get_latest_pmms(use_cache_on_failure: bool = True) -> dict:
    """
    Live PMMS with disk cache fallback.

    Returns dict: rate (decimal), as_of, source, fetched_at.
    """
    try:
        as_of, rate = fetch_latest_pmms()
        payload = {
            "rate": rate,
            "as_of": as_of,
            "source": "FRED:MORTGAGE30US",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    except Exception as exc:
        if use_cache_on_failure and CACHE_PATH.is_file():
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            data["source"] = data.get("source", "cache") + f" (live failed: {exc})"
            return data
        raise


def clear_pmms_cache() -> None:
    get_latest_pmms.cache_clear()


def load_pmms_history(refresh: bool = False) -> pd.Series:
    """Load monthly PMMS; refresh from FRED if missing or ``refresh``."""
    if refresh or not HISTORY_CACHE.is_file():
        return fetch_pmms_history()
    s = pd.read_csv(HISTORY_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = "pmms"
    return s


def pmms_for_yyyymm(period_yyyymm: str | int, history: pd.Series | None = None) -> float | None:
    """Look up PMMS for a Freddie monthly reporting period YYYYMM."""
    hist = history if history is not None else load_pmms_history()
    key = str(period_yyyymm).strip()[:6]
    if len(key) != 6 or not key.isdigit():
        return None
    ts = pd.Timestamp(year=int(key[:4]), month=int(key[4:6]), day=1) + pd.offsets.MonthEnd(0)
    # exact or backward as-of
    if ts in hist.index:
        return float(hist.loc[ts])
    prior = hist.loc[:ts]
    if prior.empty:
        return None
    return float(prior.iloc[-1])
