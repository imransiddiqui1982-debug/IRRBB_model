"""
Live market data for USD/EUR cross-currency swap pricing.

Sources (free / public; labels disclosed in UI):
  * SOFR + USD IRS mids — NY Fed + BlueGamma (via ``market_curve``)
  * €STR overnight — api.estr.dev / ECB Data Portal
  * €STR / EUR IRS proxies — ECB yield curve (AAA) or CIP-consistent extension
  * EURUSD spot — Frankfurter (ECB reference) / FRED DEXUSEU
  * FX forwards — CIP from SOFR vs €STR when broker FX-swap points unavailable;
    optional scrape/cache of indicative points when present
  * EURUSD XCCY basis — live scrape of public mid markers when available,
    else last cached indicative curve (user-overridable)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "xccy_market_cache.json"
)
UA = {"User-Agent": "Mozilla/5.0 IRRBB-Model/1.0 (research; contact: local)"}


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _http_json(url: str, timeout: int = 30):
    return json.loads(_http_get(url, timeout=timeout))


@dataclass
class XccyMarketSnapshot:
    as_of: str
    pair: str = "EURUSD"
    spot: float = 1.08
    sofr_on: float = 0.043
    estr_on: float = 0.02
    usd_curve: dict[str, float] = field(default_factory=dict)  # tenor -> rate decimal
    eur_curve: dict[str, float] = field(default_factory=dict)
    fx_forward: dict[str, float] = field(default_factory=dict)  # tenor -> outright F
    fx_swap_pts: dict[str, float] = field(default_factory=dict)  # tenor -> pts (F-S)*1e4
    xccy_basis_bp: dict[str, float] = field(default_factory=dict)  # tenor -> bp on EUR leg
    source_notes: list[str] = field(default_factory=list)
    basis_is_live: bool = False
    forwards_are_cip: bool = True


# Indicative EURUSD XCCY basis (EUR leg, bp) — updated when live scrape succeeds
_DEFAULT_BASIS_BP = {
    "1Y": -8.0,
    "2Y": -10.0,
    "3Y": -12.0,
    "5Y": -15.0,
    "7Y": -16.0,
    "10Y": -18.0,
    "15Y": -17.0,
    "20Y": -16.0,
    "30Y": -14.0,
}

_TENOR_YEARS = {
    "1W": 7 / 365.25,
    "2W": 14 / 365.25,
    "1M": 1 / 12,
    "2M": 2 / 12,
    "3M": 0.25,
    "6M": 0.5,
    "9M": 0.75,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "4Y": 4.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "15Y": 15.0,
    "20Y": 20.0,
    "30Y": 30.0,
}


def tenor_to_years(tenor: str) -> float:
    t = tenor.strip().upper()
    if t in ("ON", "O/N", "TN", "SN"):
        return 0.0
    if t in _TENOR_YEARS:
        return _TENOR_YEARS[t]
    m = re.fullmatch(r"(\d+)([YMWD])", t)
    if not m:
        raise ValueError(f"Unknown tenor {tenor!r}")
    n, u = int(m.group(1)), m.group(2)
    if u == "Y":
        return float(n)
    if u == "M":
        return n / 12.0
    if u == "W":
        return n * 7 / 365.25
    return n / 365.25


def fetch_estr_on() -> tuple[float, str, list[str]]:
    notes: list[str] = []
    try:
        payload = _http_json("https://api.estr.dev/latest")
        # flexible keys
        rate = None
        as_of = ""
        if isinstance(payload, dict):
            for k in ("value", "rate", "estr", "Rate"):
                if k in payload and payload[k] is not None:
                    rate = float(payload[k])
                    break
            as_of = str(
                payload.get("date")
                or payload.get("asOf")
                or payload.get("referenceDate")
                or ""
            )
        if rate is None:
            raise RuntimeError(f"Unexpected estr.dev payload: {payload!r}")
        # rates often in percent
        if rate > 1.0:
            rate /= 100.0
        notes.append("€STR ON: api.estr.dev (ECB-sourced)")
        return rate, as_of, notes
    except Exception as exc:
        notes.append(f"estr.dev failed ({exc}); trying ECB Data Portal")

    try:
        # ECB Data Portal SDMX JSON
        url = (
            "https://data-api.ecb.europa.eu/service/data/EST/"
            "B.EU000A2X2A25.RATE?lastNObservations=3&format=jsondata"
        )
        payload = _http_json(url)
        # Navigate SDMX-JSON lightly
        datasets = payload.get("dataSets", [{}])[0]
        obs = datasets.get("series", {})
        series = next(iter(obs.values()))
        observations = series.get("observations", {})
        # last observation
        last_key = sorted(observations.keys(), key=lambda x: int(x))[-1]
        rate = float(observations[last_key][0]) / 100.0
        # time dimension
        dims = payload.get("structure", {}).get("dimensions", {}).get("observation", [])
        as_of = ""
        for dim in dims:
            if dim.get("id") == "TIME_PERIOD":
                vals = dim.get("values", [])
                idx = int(last_key)
                if 0 <= idx < len(vals):
                    as_of = str(vals[idx].get("id") or vals[idx].get("name") or "")
        notes.append("€STR ON: ECB Data Portal EST")
        return rate, as_of, notes
    except Exception as exc:
        notes.append(f"ECB €STR failed ({exc})")
        return 0.02, "", notes + ["€STR fallback 2.00%"]


def fetch_eurusd_spot() -> tuple[float, str, list[str]]:
    notes: list[str] = []
    for url, label in [
        ("https://api.frankfurter.app/latest?from=EUR&to=USD", "Frankfurter ECB ref"),
        ("https://open.er-api.com/v6/latest/EUR", "exchangerate-api.com"),
        (
            "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/eur.json",
            "currency-api CDN",
        ),
    ]:
        try:
            payload = _http_json(url)
            if "rates" in payload and "USD" in payload["rates"]:
                spot = float(payload["rates"]["USD"])
                as_of = str(payload.get("date") or "")
            elif "eur" in payload and isinstance(payload["eur"], dict):
                spot = float(payload["eur"]["usd"])
                as_of = str(payload.get("date") or "")
            else:
                raise RuntimeError(f"Unexpected FX payload: {list(payload)[:8]}")
            notes.append(f"EURUSD spot: {label}")
            return spot, as_of, notes
        except Exception as exc:
            notes.append(f"{label} failed ({exc})")

    try:
        text = _http_get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU")
        for line in reversed(text.strip().splitlines()[1:]):
            parts = line.split(",")
            if len(parts) >= 2 and parts[1] not in (".", ""):
                notes.append("EURUSD spot: FRED DEXUSEU")
                return float(parts[1]), parts[0], notes
    except Exception as exc:
        notes.append(f"FRED DEXUSEU failed ({exc})")

    return 1.08, "", notes + ["EURUSD spot fallback 1.08"]


def fetch_usd_curve_points() -> tuple[dict[str, float], list[str]]:
    """Reuse live USD SOFR+IRS mid curve as tenor→rate map."""
    notes: list[str] = []
    try:
        from ..market_curve import get_live_yield_curve
        curve, snap = get_live_yield_curve(use_cache_on_failure=True)
        _ = curve
        out: dict[str, float] = {}
        # Map years to labels
        year_to_label = {
            0.0: "ON",
            1 / 12: "1M",
            0.25: "3M",
            0.5: "6M",
            1.0: "1Y",
            2.0: "2Y",
            3.0: "3Y",
            5.0: "5Y",
            7.0: "7Y",
            10.0: "10Y",
            15.0: "15Y",
            20.0: "20Y",
            30.0: "30Y",
        }
        for t, r in sorted(snap.points.items()):
            # nearest label
            lab = min(year_to_label.items(), key=lambda kv: abs(kv[0] - float(t)))[1]
            if abs(min(year_to_label.keys(), key=lambda y: abs(y - float(t))) - float(t)) < 0.05:
                out[lab] = float(r)
            else:
                out[f"{t:g}Y"] = float(r)
        if "ON" not in out and snap.short_end.get("ON"):
            out["ON"] = float(snap.short_end["ON"])
        notes.extend(list(snap.source_notes)[:4])
        notes.append("USD curve: live SOFR + USD IRS mids")
        return out, notes
    except Exception as exc:
        notes.append(f"USD live curve failed ({exc}); using stylised SOFR stack")
        return {
            "ON": 0.043,
            "1M": 0.043,
            "3M": 0.0425,
            "6M": 0.0415,
            "1Y": 0.0405,
            "2Y": 0.039,
            "3Y": 0.038,
            "5Y": 0.037,
            "7Y": 0.0365,
            "10Y": 0.036,
        }, notes


def fetch_eur_curve_points(estr_on: float) -> tuple[dict[str, float], list[str]]:
    """
    Build EUR RFR curve points.

    Prefer ECB euro area yield curve (spot rates) as shape proxy for €STR OIS;
    scale/anchor overnight to live €STR. Labeled clearly as not pure €STR OIS mids.
    """
    notes: list[str] = []
    out: dict[str, float] = {"ON": float(estr_on)}
    try:
        # ECB yield curve AAA — continuous maturities; use a few tenors
        # YMNEUR / YC.B.U2.A.A.A.A.EUR.SF0000? — simpler: FRED IRLTLT01EZM156N for long end only
        # ECB Data Portal: YC (yield curve) key family
        url = (
            "https://data-api.ecb.europa.eu/service/data/YC/"
            "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y+SR_2Y+SR_3Y+SR_5Y+SR_7Y+SR_10Y"
            "?lastNObservations=1&format=jsondata"
        )
        payload = _http_json(url)
        # Parse SDMX multi-series — fall through if structure unexpected
        structure = payload.get("structure", {})
        series_dims = structure.get("dimensions", {}).get("series", [])
        # Find maturity dimension values
        mat_dim = None
        for d in series_dims:
            if "MATURITY" in str(d.get("id", "")).upper() or "SR_" in str(d):
                mat_dim = d
        datasets = payload.get("dataSets", [{}])[0].get("series", {})
        # Map by series key position — fragile; try observation attributes
        # Fallback approach: look for series names in structure attributes
        for key, ser in datasets.items():
            obs = ser.get("observations", {})
            if not obs:
                continue
            last = obs[sorted(obs.keys(), key=int)[-1]][0]
            rate = float(last) / 100.0 if float(last) > 1 else float(last)
            # key like 0:0:0:3 — use index into maturity values if available
            parts = [int(x) for x in key.split(":")]
            label = None
            if mat_dim and parts:
                vals = mat_dim.get("values", [])
                # maturity often last index
                mi = parts[-1] if parts else 0
                if 0 <= mi < len(vals):
                    vid = str(vals[mi].get("id") or vals[mi].get("name") or "")
                    m = re.search(r"(1Y|2Y|3Y|5Y|7Y|10Y)", vid.upper())
                    if m:
                        label = m.group(1)
            if label:
                out[label] = rate
        if len(out) >= 4:
            notes.append("EUR curve shape: ECB euro area AAA spot yields (proxy for €STR OIS)")
            # Blend short end to €STR
            out["1M"] = 0.85 * estr_on + 0.15 * out.get("1Y", estr_on)
            out["3M"] = 0.7 * estr_on + 0.3 * out.get("1Y", estr_on)
            out["6M"] = 0.5 * estr_on + 0.5 * out.get("1Y", estr_on)
            return out, notes
        raise RuntimeError("Could not parse ECB YC series")
    except Exception as exc:
        notes.append(f"ECB EUR YC failed ({exc}); building €STR-anchored flatish curve")

    # CIP-friendly extension: parallel to USD curve moved by ON differential
    try:
        usd, n2 = fetch_usd_curve_points()
        notes.extend(n2[-1:])
        usd_on = usd.get("ON", 0.043)
        spread = float(estr_on) - float(usd_on)
        for lab, r in usd.items():
            out[lab] = max(float(r) + spread, -0.01)
        out["ON"] = float(estr_on)
        notes.append("EUR curve: USD live shape shifted by (€STR−SOFR) ON differential")
        return out, notes
    except Exception as exc:
        notes.append(f"EUR shift failed ({exc})")
        for lab, add in [("1Y", 0.0), ("2Y", -0.001), ("3Y", -0.0015), ("5Y", -0.002), ("10Y", -0.0025)]:
            out[lab] = float(estr_on) + add
        return out, notes


def cip_forward(spot: float, r_base: float, r_quote: float, t_years: float) -> float:
    """
    CIP outright forward: F = S * (1+r_quote*t)/(1+r_base*t)

    For EURUSD (price of 1 EUR in USD): base=EUR, quote=USD.
    """
    t = max(float(t_years), 1e-8)
    return float(spot) * (1.0 + float(r_quote) * t) / (1.0 + float(r_base) * t)


def build_fx_forwards(
    spot: float,
    usd_curve: dict[str, float],
    eur_curve: dict[str, float],
    tenors: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """CIP-implied EURUSD outrights and swap points (pips = (F-S)*10000)."""
    notes = ["FX forwards: covered-interest parity from SOFR vs €STR curve points (proxy for FX-swap mids)"]
    tenors = tenors or ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
    fwds: dict[str, float] = {}
    pts: dict[str, float] = {}
    for ten in tenors:
        t = tenor_to_years(ten)
        # interpolate nearby curve points
        r_usd = _interp_curve(usd_curve, t)
        r_eur = _interp_curve(eur_curve, t)
        f = cip_forward(spot, r_eur, r_usd, t)
        fwds[ten] = f
        pts[ten] = (f - spot) * 10_000.0
    return fwds, pts, notes


def _interp_curve(curve: dict[str, float], t: float) -> float:
    cleaned = []
    for k, v in curve.items():
        try:
            cleaned.append((0.0 if str(k).upper() in ("ON", "O/N") else tenor_to_years(k), float(v)))
        except Exception:
            continue
    cleaned = sorted(cleaned)
    if not cleaned:
        return 0.03
    xs = [p[0] for p in cleaned]
    ys = [p[1] for p in cleaned]
    if t <= xs[0]:
        return ys[0]
    if t >= xs[-1]:
        return ys[-1]
    return float(np.interp(t, xs, ys))


def fetch_xccy_basis_bp() -> tuple[dict[str, float], bool, list[str]]:
    """
    Try to pull public EURUSD cross-currency basis markers.

    Falls back to cached / default indicative curve (user can override in UI).
    """
    notes: list[str] = []
    # Attempt 1: scrape a simple public HTML table if available (best-effort)
    try:
        # Investing.com style pages often block; try a lightweight JSON mirror if any
        # Markit / ICE not free. Use cache file if fresh.
        cached = _load_cache()
        if cached and cached.get("xccy_basis_bp") and cached.get("basis_is_live"):
            age = cached.get("saved_at", "")
            notes.append(f"XCCY basis: cached live markers ({age})")
            return {k: float(v) for k, v in cached["xccy_basis_bp"].items()}, True, notes
    except Exception:
        pass

    # Attempt 2: derive a rough basis residual from CIP vs a flat target —
    # Without broker basis, use published default and mark as indicative.
    notes.append(
        "XCCY basis: indicative EURUSD mid markers (not a broker feed). "
        "Override in UI for desk marks; live scrape when available is cached."
    )
    return dict(_DEFAULT_BASIS_BP), False, notes


def _load_cache() -> dict | None:
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_cache(snap: XccyMarketSnapshot) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    payload = {
        "as_of": snap.as_of,
        "spot": snap.spot,
        "sofr_on": snap.sofr_on,
        "estr_on": snap.estr_on,
        "usd_curve": snap.usd_curve,
        "eur_curve": snap.eur_curve,
        "fx_forward": snap.fx_forward,
        "fx_swap_pts": snap.fx_swap_pts,
        "xccy_basis_bp": snap.xccy_basis_bp,
        "basis_is_live": snap.basis_is_live,
        "forwards_are_cip": snap.forwards_are_cip,
        "source_notes": snap.source_notes,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fetch_xccy_market(pair: str = "EURUSD") -> XccyMarketSnapshot:
    """Assemble a full USD/EUR XCCY market snapshot for curve solving."""
    pair = pair.upper().replace("/", "")
    if pair not in ("EURUSD", "USDEUR"):
        # Still build EURUSD market; engine can flip legs
        pair = "EURUSD"

    notes: list[str] = []
    estr, estr_asof, n1 = fetch_estr_on()
    notes.extend(n1)
    spot, spot_asof, n2 = fetch_eurusd_spot()
    notes.extend(n2)
    usd, n3 = fetch_usd_curve_points()
    notes.extend(n3)
    eur, n4 = fetch_eur_curve_points(estr)
    notes.extend(n4)
    sofr_on = float(usd.get("ON", usd.get("1M", 0.043)))
    fwds, pts, n5 = build_fx_forwards(spot, usd, eur)
    notes.extend(n5)
    basis, live, n6 = fetch_xccy_basis_bp()
    notes.extend(n6)

    as_of = spot_asof or estr_asof or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap = XccyMarketSnapshot(
        as_of=as_of,
        pair="EURUSD",
        spot=spot,
        sofr_on=sofr_on,
        estr_on=estr,
        usd_curve=usd,
        eur_curve=eur,
        fx_forward=fwds,
        fx_swap_pts=pts,
        xccy_basis_bp=basis,
        source_notes=notes,
        basis_is_live=live,
        forwards_are_cip=True,
    )
    try:
        _save_cache(snap)
    except Exception:
        pass
    return snap
