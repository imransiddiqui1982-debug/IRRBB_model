"""
key_rate_duration.py
--------------------
Key-rate DV01 (KR01) engine for treasury hedging.

Adapted from the standalone alm_engine design rules:
  * Cash flows stay on the BCBS 19-bucket schedule.
  * Key rates sit on a *tradeable* swap grid (1/2/3/5/7/10Y).
  * Tent weights form a partition of unity so KR01 sums to parallel DV01.
  * Prepayable mortgages / MBS regenerate cash flows under each key bump
    so optionality is live inside the sensitivity.

Output is used for:
  * Balance-sheet key-rate gap charts (assets vs liabilities)
  * Indicative pay/receive-fixed swap hedges by tenor
  * High-level hedge-accounting designation tags
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .cashflows import Instrument
from .prepayment import scenario_market_mortgage_rate
from .time_buckets import BUCKET_MIDPOINTS
from .yield_curve import YieldCurve

# Standard USD SOFR / IRS mid tenors — keys you can actually trade.
TRADEABLE_TENORS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)


@dataclass(frozen=True)
class KeyRateGrid:
    """Coarse key-rate grid with triangular (tent) bump weights."""

    tenors: tuple[float, ...] = TRADEABLE_TENORS

    def tent(self, t: np.ndarray, i: int) -> np.ndarray:
        """
        Weight = 1 at key i, linearly decays to 0 at neighbouring keys.

        Partition of unity: sum_i tent(t, i) == 1 for every maturity t.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        k = self.tenors[i]
        lo = self.tenors[i - 1] if i > 0 else 0.0
        hi = self.tenors[i + 1] if i < len(self.tenors) - 1 else np.inf
        w = np.zeros_like(t, dtype=float)
        left = (t > lo) & (t <= k)
        w[left] = (t[left] - lo) / (k - lo)
        if np.isfinite(hi):
            right = (t > k) & (t < hi)
            w[right] = (hi - t[right]) / (hi - k)
        if i == 0:
            w[t <= k] = 1.0
        if i == len(self.tenors) - 1:
            w[t >= k] = 1.0
        return w

    def check_partition(self, t: np.ndarray, tol: float = 1e-9) -> bool:
        total = sum(self.tent(t, i) for i in range(len(self.tenors)))
        return bool(np.all(np.abs(total - 1.0) < tol))


def _clone_curve(base_rates: np.ndarray) -> YieldCurve:
    """Build a YieldCurve whose bucket rates equal ``base_rates`` exactly."""
    curve = YieldCurve.__new__(YieldCurve)
    curve.base_rates = np.asarray(base_rates, dtype=float).copy()
    return curve


def _bump_curve(
    curve: YieldCurve,
    grid: KeyRateGrid,
    key_index: int,
    bp: float = 1.0,
) -> YieldCurve:
    """Return a YieldCurve with a +bp triangular bump at ``key_index``."""
    mids = np.asarray(BUCKET_MIDPOINTS, dtype=float)
    weights = grid.tent(mids, key_index)
    bumped = np.maximum(curve.base_rates + weights * (bp / 10_000.0), 0.0)
    return _clone_curve(bumped)


def _instrument_cf_vector(
    inst: Instrument,
    curve: YieldCurve,
    cpr_override: float | None = None,
) -> np.ndarray:
    """Bucket cash flows; prepayable instruments use curve-implied mortgage rate."""
    if getattr(inst, "is_prepayable", False):
        mkt = scenario_market_mortgage_rate(
            curve.base_rates, None, inst.maturity_years,
        )
        cpr = cpr_override
        if cpr is None and getattr(inst, "base_cpr", None) is not None:
            cpr = inst.base_cpr
        return inst.bucket_cashflows_under_market_rate(mkt, cpr_override=cpr)
    return inst.bucket_cashflows()


def instrument_pv(
    inst: Instrument,
    curve: YieldCurve,
    cpr_override: float | None = None,
) -> float:
    """Present value of one instrument under ``curve`` ($M)."""
    cfs = _instrument_cf_vector(inst, curve, cpr_override=cpr_override)
    return float(curve.pv_cashflows(cfs))


def portfolio_pv(
    assets: Sequence[Instrument],
    liabilities: Sequence[Instrument],
    curve: YieldCurve,
    cpr_override: float | None = None,
) -> float:
    """Economic value of equity proxy: PV(assets) − PV(liabilities)."""
    a = sum(instrument_pv(i, curve, cpr_override) for i in assets)
    l = sum(instrument_pv(i, curve, cpr_override) for i in liabilities)
    return float(a - l)


def parallel_dv01(
    assets: Sequence[Instrument],
    liabilities: Sequence[Instrument],
    curve: YieldCurve,
    cpr_override: float | None = None,
) -> float:
    """
    Parallel DV01 of net EVE ($ thousands per +1 bp).

    Positive => asset-sensitive (EVE falls when rates rise).
    """
    bumped = _clone_curve(np.maximum(curve.base_rates + 1e-4, 0.0))
    # $M PV change × 1000 = $K per bp; sign: loss when rates up => positive DV01
    return float(
        (
            portfolio_pv(assets, liabilities, curve, cpr_override)
            - portfolio_pv(assets, liabilities, bumped, cpr_override)
        )
        * 1_000.0
    )


def compute_kr01(
    assets: Sequence[Instrument],
    liabilities: Sequence[Instrument],
    curve: YieldCurve,
    grid: KeyRateGrid | None = None,
    cpr_override: float | None = None,
) -> pd.DataFrame:
    """
    Key-rate DV01 gap by tradeable tenor.

    Returns columns:
      tenor_years, label,
      asset_kr01_k, liability_kr01_k, net_kr01_k   ($ thousands per bp)
    Asset KR01 > 0 means assets lose value when that key rate rises.
    Liability KR01 > 0 means PV(liabilities) falls when that key rises.
    net = asset − liability.
    """
    grid = grid or KeyRateGrid()
    base_a = np.array(
        [instrument_pv(i, curve, cpr_override) for i in assets], dtype=float,
    )
    base_l = np.array(
        [instrument_pv(i, curve, cpr_override) for i in liabilities], dtype=float,
    )

    rows = []
    for i, tenor in enumerate(grid.tenors):
        bumped = _bump_curve(curve, grid, i, bp=1.0)
        a_bump = np.array(
            [instrument_pv(inst, bumped, cpr_override) for inst in assets],
            dtype=float,
        )
        l_bump = np.array(
            [instrument_pv(inst, bumped, cpr_override) for inst in liabilities],
            dtype=float,
        )
        # PV drop under +1bp × 1000 => $K / bp
        asset_k = float(np.sum(base_a - a_bump) * 1_000.0)
        liab_k = float(np.sum(base_l - l_bump) * 1_000.0)
        rows.append({
            "tenor_years": tenor,
            "label": f"{tenor:g}Y",
            "asset_kr01_k": asset_k,
            "liability_kr01_k": liab_k,
            "net_kr01_k": asset_k - liab_k,
        })
    return pd.DataFrame(rows)


def suggest_key_rate_hedges(
    kr01_df: pd.DataFrame,
    hedge_ratio: float = 0.80,
    min_abs_kr01_k: float = 5.0,
) -> pd.DataFrame:
    """
    Indicative pay-fixed / receive-fixed IRS notionals by key tenor.

    DV01 ($M/bp) ≈ notional($M) × tenor × 1e-4
    => notional = |KR01_$M| × ratio / (tenor × 1e-4)
                = |KR01_$K| × ratio / (tenor × 0.1)
    """
    rows = []
    for _, row in kr01_df.iterrows():
        net = float(row["net_kr01_k"])
        tenor = float(row["tenor_years"])
        if abs(net) < min_abs_kr01_k or tenor <= 0:
            continue
        # net > 0: asset-heavy at this key → pay fixed to shorten
        if net > 0:
            structure = "Pay-fixed / receive-floating IRS"
            action = "Enter payer swap"
        else:
            structure = "Receive-fixed / pay-floating IRS"
            action = "Enter receiver swap"
        notional_m = abs(net) * hedge_ratio / (tenor * 0.1)
        rows.append({
            "Tenor": row["label"],
            "Net KR01 ($K/bp)": round(net, 1),
            "Hedge action": action,
            "IRS structure": structure,
            "Indicative notional ($M)": round(notional_m, 1),
            "Target hedge (%)": int(hedge_ratio * 100),
        })
    return pd.DataFrame(rows)


def designate_key_rate_hedges(
    hedge_df: pd.DataFrame,
    assets: Sequence[Instrument],
    liabilities: Sequence[Instrument],
) -> pd.DataFrame:
    """
    Tag proposed key-rate swaps with indicative hedge-accounting notes.

    Not accounting advice — confirms the treasurer sees accounting consequence
    next to risk reduction (fair value vs cash-flow hedge).
    """
    if hedge_df is None or hedge_df.empty:
        return pd.DataFrame()

    fixed_assets = [
        i for i in assets
        if i.side == "asset"
        and i.instrument_type in ("bullet_fixed", "amortising", "mbs")
    ]
    float_liabs = [
        i for i in liabilities
        if i.side == "liability" and i.instrument_type == "bullet_floating"
    ]
    nmds = [i for i in liabilities if i.instrument_type == "demand_deposit"]

    rows = []
    for _, h in hedge_df.iterrows():
        pays_fixed = "Pay-fixed" in str(h.get("IRS structure", ""))
        warn = ""
        if pays_fixed and fixed_assets:
            item = max(fixed_assets, key=lambda p: p.notional)
            designation = "fair_value"
            mtm_to = "P&L (offset by hedged-item basis adjustment)"
            note = (
                "ASU 2017-12; portfolio layer method (ASU 2022-01) for prepayable pools. "
                "Monitor breach tests if prepayments run fast."
            )
        elif (not pays_fixed) and float_liabs:
            item = max(float_liabs, key=lambda p: p.notional)
            designation = "cash_flow"
            mtm_to = "OCI → AOCI, reclassified as flows occur"
            note = "Hedge of variability in forecasted interest cash flows."
        elif float_liabs:
            item = max(float_liabs, key=lambda p: p.notional)
            designation = "cash_flow"
            mtm_to = "OCI → AOCI, reclassified as flows occur"
            note = "Hedge of variability in forecasted interest cash flows."
        elif fixed_assets:
            item = max(fixed_assets, key=lambda p: p.notional)
            designation = "fair_value"
            mtm_to = "P&L (offset by hedged-item basis adjustment)"
            note = "Confirm designation with auditor before booking."
        else:
            item = None
            designation = "undesignated"
            mtm_to = "P&L (full MTM)"
            note = "No eligible hedged item identified."

        if nmds and item is not None and item.instrument_type != "demand_deposit":
            warn = (
                f"NMD book of ${sum(p.notional for p in nmds):,.0f}M drives much of "
                "this exposure but demand deposits generally cannot be the hedged "
                "item in a fair-value hedge. Designated against assets / floaters instead."
            )

        rows.append({
            "Tenor": h["Tenor"],
            "Notional ($M)": h["Indicative notional ($M)"],
            "Designation": designation,
            "Hedged item": item.name if item else "—",
            "MTM to": mtm_to,
            "Note": note if not warn else f"{note} {warn}",
        })
    return pd.DataFrame(rows)


def instrument_kr01_attribution(
    instruments: Sequence[Instrument],
    curve: YieldCurve,
    grid: KeyRateGrid | None = None,
    cpr_override: float | None = None,
) -> pd.DataFrame:
    """Per-instrument KR01 ($K/bp) across the key grid — for drill-down tables."""
    grid = grid or KeyRateGrid()
    base = {id(i): instrument_pv(i, curve, cpr_override) for i in instruments}
    data = {
        "instrument": [i.name for i in instruments],
        "side": [i.side for i in instruments],
    }
    for ki, tenor in enumerate(grid.tenors):
        bumped = _bump_curve(curve, grid, ki, bp=1.0)
        col = []
        for inst in instruments:
            pv0 = base[id(inst)]
            pv1 = instrument_pv(inst, bumped, cpr_override)
            col.append(float((pv0 - pv1) * 1_000.0))
        data[f"{tenor:g}Y"] = col
    return pd.DataFrame(data)


# ── KR01 ↔ EVE bridge (ALCO / Board / Treasury) ───────────────────────────────

def shocked_yield_curve(curve: YieldCurve, shocks_bp: Sequence[float]) -> YieldCurve:
    """Apply BCBS bucket shocks (bp) to produce a new YieldCurve."""
    return _clone_curve(curve.shocked_rates(list(shocks_bp)))


def shocks_at_key_tenors(
    scenario,
    grid: KeyRateGrid | None = None,
) -> np.ndarray:
    """
    Interpolate BCBS scenario shocks (bp) onto the tradeable key grid.

    Uses the scenario's 19-bucket shock vector and bucket midpoints.
    """
    from .time_buckets import BUCKET_MIDPOINTS as mids

    grid = grid or KeyRateGrid()
    bucket_shocks = np.asarray(scenario.shocks_bp, dtype=float)
    return np.interp(np.asarray(grid.tenors, dtype=float), mids, bucket_shocks)


def scenario_shock_matrix(
    scenarios: Sequence,
    grid: KeyRateGrid | None = None,
) -> pd.DataFrame:
    """Rows = key tenors, columns = scenario names, values = shock bp."""
    grid = grid or KeyRateGrid()
    data = {"Key": [f"{t:g}Y" for t in grid.tenors]}
    for sc in scenarios:
        data[sc.name] = list(shocks_at_key_tenors(sc, grid))
    return pd.DataFrame(data).set_index("Key")


def apply_pay_fixed_to_kr01(
    kr01_df: pd.DataFrame,
    tenor_years: float,
    notional_m: float,
) -> pd.DataFrame:
    """
    Pay-fixed IRS of ``notional_m`` at ``tenor_years`` reduces net KR01
    by ≈ notional × tenor × 0.1 ($K/bp).
    """
    out = kr01_df.copy()
    delta_k = -float(notional_m) * float(tenor_years) * 0.1
    mask = np.isclose(out["tenor_years"].astype(float), float(tenor_years))
    if mask.any():
        out.loc[mask, "net_kr01_k"] = out.loc[mask, "net_kr01_k"] + delta_k
    return out


def apply_hedge_notionals_to_kr01(
    kr01_df: pd.DataFrame,
    notionals: dict[float, float],
) -> pd.DataFrame:
    """``notionals`` maps tenor_years → pay-fixed notional ($M); negative = receive-fixed."""
    out = kr01_df.copy()
    for tenor, n in notionals.items():
        out = apply_pay_fixed_to_kr01(out, float(tenor), float(n))
    return out


def eve_attribution_from_kr01(
    kr01_df: pd.DataFrame,
    scenarios: Sequence,
    actual_delta_eve: dict[str, float] | None = None,
    grid: KeyRateGrid | None = None,
) -> dict:
    """
    Bridge: ΔEVE contribution from key i = −KR01ᵢ($K/bp) × shock_bpᵢ / 1000  → $M.

    Returns contribution matrix, predicted totals, actual, convexity, driver notes.
    """
    grid = grid or KeyRateGrid()
    keys = list(kr01_df["label"])
    kr = kr01_df["net_kr01_k"].astype(float).values

    contrib = {}
    predicted = {}
    for sc in scenarios:
        shocks = shocks_at_key_tenors(sc, grid)
        # $M = − ($K/bp) × bp / 1000
        c = -kr * shocks / 1_000.0
        contrib[sc.name] = c
        predicted[sc.name] = float(c.sum())

    contrib_df = pd.DataFrame(contrib, index=keys)
    contrib_df.index.name = "Key"
    pred_row = pd.DataFrame([predicted], index=["Predicted ΔEVE ($M)"])
    actual = actual_delta_eve or {}
    actual_aligned = {sc.name: float(actual.get(sc.id, actual.get(sc.name, np.nan)))
                      for sc in scenarios}
    act_row = pd.DataFrame([actual_aligned], index=["Actual ΔEVE ($M)"])
    convex = {
        sc.name: (
            actual_aligned[sc.name] - predicted[sc.name]
            if np.isfinite(actual_aligned[sc.name]) else np.nan
        )
        for sc in scenarios
    }
    conv_row = pd.DataFrame([convex], index=["Convexity ($M)"])

    return {
        "contribution_m": contrib_df,
        "predicted_m": predicted,
        "actual_m": actual_aligned,
        "convexity_m": convex,
        "summary": pd.concat([contrib_df, pred_row, act_row, conv_row]),
        "shock_bp": scenario_shock_matrix(scenarios, grid),
    }


def limit_dashboard(
    actual_delta_eve: dict[str, float],
    contribution_m: pd.DataFrame,
    tier1_m: float,
    scenarios: Sequence,
    breach_pct: float = 15.0,
    amber_pct: float = 12.0,
) -> pd.DataFrame:
    """Board / ALCO limit view with key-rate driver of each breach."""
    rows = []
    for sc in scenarios:
        name = sc.name
        d_eve = float(actual_delta_eve.get(sc.id, actual_delta_eve.get(name, np.nan)))
        pct = d_eve / tier1_m * 100.0 if tier1_m else 0.0
        if pct <= -breach_pct:
            status = "BREACH"
        elif pct <= -amber_pct:
            status = "AMBER"
        else:
            status = "ok"

        driver = "—"
        if name in contribution_m.columns and d_eve < 0:
            col = contribution_m[name]
            # Most negative contribution = main loss driver
            k = col.idxmin()
            loss = float(col.sum())
            share = float(col[k] / loss * 100.0) if loss < 0 else 0.0
            driver = f"{k} — {share:.0f}% of predicted loss"

        rows.append({
            "Scenario": name,
            "ΔEVE ($M)": round(d_eve, 2),
            "% Tier 1": round(pct, 1),
            "Status": status,
            "Driver": driver,
        })
    return pd.DataFrame(rows)


def swap_unit_kr01_k(tenor_years: float, notional_m: float = 100.0) -> float:
    """Pay-fixed swap KR01 in $K/bp (negative = shortens asset-heavy book)."""
    return -float(notional_m) * float(tenor_years) * 0.1


def hedge_efficiency_table(
    scenarios: Sequence,
    grid: KeyRateGrid | None = None,
    tenors: tuple[float, ...] = (2.0, 5.0, 10.0),
    notional_m: float = 100.0,
) -> pd.DataFrame:
    """
    EVE relief ($M) from $notional_m pay-fixed at each tenor under each scenario.

    Relief = −swap_KR01 × shock_bp / 1000.
    """
    grid = grid or KeyRateGrid()
    rows = []
    for T in tenors:
        kr = swap_unit_kr01_k(T, notional_m)
        row = {
            "Instrument": f"{T:g}Y pay-fixed ${notional_m:g}m",
            "KR01 ($K/bp)": round(kr, 1),
        }
        for sc in scenarios:
            shocks = shocks_at_key_tenors(sc, grid)
            # Map tenor to nearest grid key for shock
            idx = int(np.argmin(np.abs(np.asarray(grid.tenors) - T)))
            shock = float(shocks[idx])
            row[sc.name] = round(-kr * shock / 1_000.0, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def proposed_swap_notionals(
    kr01_df: pd.DataFrame,
    tenors: tuple[float, ...] = (2.0, 5.0, 10.0),
    hedge_ratio: float = 0.80,
    min_abs_kr01_k: float = 1.0,
) -> dict[float, float]:
    """
    Indicative pay-fixed notionals ($M) at selected tenors to close net KR01.

    Positive notional = pay-fixed; negative = receive-fixed.
    """
    out: dict[float, float] = {}
    for T in tenors:
        rows = kr01_df.loc[np.isclose(kr01_df["tenor_years"].astype(float), float(T))]
        if rows.empty:
            # nearest key on the grid
            idx = int(np.argmin(np.abs(kr01_df["tenor_years"].astype(float) - float(T))))
            row = kr01_df.iloc[idx]
            T = float(row["tenor_years"])
            net = float(row["net_kr01_k"])
        else:
            net = float(rows.iloc[0]["net_kr01_k"])
        if abs(net) < min_abs_kr01_k or T <= 0:
            out[float(T)] = 0.0
            continue
        out[float(T)] = net * hedge_ratio / (T * 0.1)
    return out


def post_swap_eve_impact(
    kr01_df: pd.DataFrame,
    scenarios: Sequence,
    tier1_m: float,
    actual_delta_eve: dict[str, float],
    notionals: dict[float, float],
    grid: KeyRateGrid | None = None,
) -> dict:
    """
    Key-wise (bucket) predicted ΔEVE before vs after executing proposed IRS.

    Returns:
      notionals_df, contrib_before, contrib_after, contrib_delta,
      summary (ΔEVE $M and % Tier 1 before/after by scenario)
    """
    grid = grid or KeyRateGrid()
    attr0 = eve_attribution_from_kr01(kr01_df, scenarios, actual_delta_eve, grid)
    kr_h = apply_hedge_notionals_to_kr01(kr01_df, notionals)
    attr1 = eve_attribution_from_kr01(kr_h, scenarios, None, grid)

    before = attr0["contribution_m"]
    after = attr1["contribution_m"]
    # Align columns
    delta = after - before

    notional_rows = []
    for T, n in sorted(notionals.items()):
        structure = (
            "Pay-fixed / receive-float" if n > 0
            else ("Receive-fixed / pay-float" if n < 0 else "—")
        )
        notional_rows.append({
            "Tenor": f"{T:g}Y",
            "Notional ($M)": round(abs(n), 1),
            "Structure": structure,
            "Signed notional ($M)": round(n, 1),
            "Swap KR01 ($K/bp)": round(swap_unit_kr01_k(T, n), 1),
        })
    notionals_df = pd.DataFrame(notional_rows)

    summary_rows = []
    for sc in scenarios:
        name = sc.name
        pred0 = float(attr0["predicted_m"][name])
        pred1 = float(attr1["predicted_m"][name])
        conv = attr0["convexity_m"].get(name, 0.0)
        if not np.isfinite(conv):
            conv = 0.0
        # Unhedged: prefer actual EVE; hedged: predicted + same convexity approx
        eve0 = float(actual_delta_eve.get(sc.id, actual_delta_eve.get(name, pred0)))
        eve1 = pred1 + conv
        pct0 = eve0 / tier1_m * 100.0 if tier1_m else 0.0
        pct1 = eve1 / tier1_m * 100.0 if tier1_m else 0.0
        summary_rows.append({
            "Scenario": name,
            "ΔEVE before ($M)": round(eve0, 2),
            "ΔEVE after ($M)": round(eve1, 2),
            "ΔEVE change ($M)": round(eve1 - eve0, 2),
            "% Tier 1 before": round(pct0, 1),
            "% Tier 1 after": round(pct1, 1),
            "pp change": round(pct1 - pct0, 1),
        })

    return {
        "notionals": notionals_df,
        "contrib_before": before.round(2),
        "contrib_after": after.round(2),
        "contrib_delta": delta.round(2),
        "summary": pd.DataFrame(summary_rows),
        "kr01_after": kr_h,
    }


def why_hedge_rationale(
    limit_df: pd.DataFrame,
    contribution_m: pd.DataFrame,
    efficiency_df: pd.DataFrame,
) -> list[str]:
    """Plain-language why a given IRS tenor is the right bridge for breaches."""
    notes: list[str] = []
    breaches = limit_df[limit_df["Status"].isin(["BREACH", "AMBER"])]
    if breaches.empty:
        notes.append(
            "No amber/breach scenarios — maintain a light KR01 overlay; "
            "do not over-hedge and give up NII unnecessarily."
        )
        return notes

    # Collect loss-driving keys
    driver_keys: list[str] = []
    for _, row in breaches.iterrows():
        notes.append(
            f"**{row['Scenario']}** is {row['Status']} at {row['% Tier 1']}% of Tier 1 "
            f"(ΔEVE ${row['ΔEVE ($M)']}M). Driver: {row['Driver']}."
        )
        if "—" in str(row["Driver"]):
            continue
        driver_keys.append(str(row["Driver"]).split("—")[0].strip())

    # Prefer longest common driver tenor among efficiency instruments
    if driver_keys:
        # Pick the tenor that appears most / is longest (e.g. 10Y)
        key = max(set(driver_keys), key=lambda k: (driver_keys.count(k), _tenor_sort(k)))
        notes.append(
            f"Both policy pressure and attribution concentrate at **{key}**, so a "
            f"**{key} pay-fixed IRS** is the primary bridge: it buys the most Parallel-Up "
            f"and Steepener relief per dollar of notional among the liquid tenors."
        )
        # Warn if short swap hurts steepener
        if "2Y" in efficiency_df["Instrument"].astype(str).str.cat(sep=" "):
            notes.append(
                "A **2Y pay-fixed** is inefficient (or harmful) in a Steepener: short rates "
                "fall while you pay fixed — MTM / EVE relief goes the wrong way. Use 2Y only "
                "to trim short-key KR01, not to fix a 10Y-driven breach."
            )
    return notes


def _tenor_sort(label: str) -> float:
    try:
        return float(str(label).upper().replace("Y", ""))
    except ValueError:
        return 0.0


def hedge_package_comparison(
    kr01_df: pd.DataFrame,
    scenarios: Sequence,
    tier1_m: float,
    actual_delta_eve: dict[str, float],
    grid: KeyRateGrid | None = None,
    hedge_ratio: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare unhedged vs packaged IRS overlays using the linear KR01→EVE bridge.

    Packages:
      A — dominant-key pay-fixed sized to neutralise that key's Parallel-Up loss
      B — full multi-tenor KR01 hedge at ``hedge_ratio``
      C — 60% of package A notional (partial)
    """
    grid = grid or KeyRateGrid()
    attr0 = eve_attribution_from_kr01(kr01_df, scenarios, actual_delta_eve, grid)
    contrib = attr0["contribution_m"]

    # Dominant key = largest |predicted loss contribution| under Parallel Up if present
    par_cols = [c for c in contrib.columns if "Parallel" in c and "Up" in c]
    if not par_cols:
        par_cols = [contrib.columns[0]]
    par_col = par_cols[0]
    # Most negative contribution
    dom_key = contrib[par_col].idxmin()
    dom_tenor = float(str(dom_key).upper().replace("Y", ""))
    dom_kr = float(kr01_df.loc[kr01_df["label"] == dom_key, "net_kr01_k"].iloc[0])

    # Size pay-fixed to zero that key's KR01 (full)
    notional_a = (dom_kr / (dom_tenor * 0.1)) if dom_tenor > 0 else 0.0
    # Only pay-fixed if asset-heavy (positive KR01)
    if notional_a < 0:
        notional_a = 0.0  # would need receive-fixed; handle via full package

    # Package notionals: tenor -> pay-fixed $M
    pkg_a = {dom_tenor: notional_a} if notional_a > 0 else {}
    pkg_b: dict[float, float] = {}
    for _, row in kr01_df.iterrows():
        net = float(row["net_kr01_k"])
        t = float(row["tenor_years"])
        if t <= 0 or abs(net) < 1.0:
            continue
        # positive KR01 → pay fixed; negative → receive fixed (negative pay notional)
        pkg_b[t] = net * hedge_ratio / (t * 0.1)
    pkg_c = {dom_tenor: 0.6 * notional_a} if notional_a > 0 else {}

    packages = {
        "Unhedged": {},
        f"A: {notional_a:.1f}m {dom_key} only": pkg_a,
        "B: full KR01 hedge": pkg_b,
        f"C: {0.6 * notional_a:.1f}m {dom_key} only": pkg_c,
    }

    # Build % Tier 1 table using predicted + convexity (keep base convexity)
    pct_rows = {}
    detail_rows = []
    for pname, notionals in packages.items():
        kr_h = apply_hedge_notionals_to_kr01(kr01_df, notionals) if notionals else kr01_df
        attr = eve_attribution_from_kr01(kr_h, scenarios, None, grid)
        row_pct = {}
        for sc in scenarios:
            pred = attr["predicted_m"][sc.name]
            conv = attr0["convexity_m"].get(sc.name, 0.0)
            if not np.isfinite(conv):
                conv = 0.0
            # For unhedged use actual; for hedged use predicted+convexity approx
            if not notionals:
                d_eve = float(actual_delta_eve.get(sc.id, actual_delta_eve.get(sc.name, pred)))
            else:
                d_eve = pred + conv
            row_pct[sc.name] = round(d_eve / tier1_m * 100.0, 1) if tier1_m else 0.0
        pct_rows[pname] = row_pct
        detail_rows.append({
            "Package": pname,
            "Notionals": (
                ", ".join(f"{n:.1f}m@{t:g}Y" for t, n in sorted(notionals.items()))
                if notionals else "—"
            ),
            "Primary tool": (
                f"{'Pay' if notional_a > 0 else 'Receive'}-fixed {dom_key}"
                if pname.startswith("A") or pname.startswith("C")
                else ("Multi-tenor IRS ladder" if pname.startswith("B") else "None")
            ),
        })

    pct_df = pd.DataFrame(pct_rows).T
    pct_df.index.name = "Package"
    detail_df = pd.DataFrame(detail_rows)
    return pct_df, detail_df


def compute_kr01_under_scenarios(
    assets: Sequence[Instrument],
    liabilities: Sequence[Instrument],
    curve: YieldCurve,
    scenarios: Sequence,
    cpr_for_scenario=None,
    grid: KeyRateGrid | None = None,
) -> pd.DataFrame:
    """
    Dynamic KR01: recompute key-rate gaps on each shocked curve.

    ``cpr_for_scenario(scenario_id) -> float|None`` optional CPR override.
    Returns wide DataFrame: Key × scenario net KR01 ($K/bp), plus Base.
    """
    grid = grid or KeyRateGrid()
    base = compute_kr01(assets, liabilities, curve, grid, cpr_override=(
        cpr_for_scenario("BASE") if cpr_for_scenario else None
    ))
    out = pd.DataFrame({"Key": base["label"], "Base": base["net_kr01_k"].values})
    for sc in scenarios:
        cpr = cpr_for_scenario(sc.id) if cpr_for_scenario else None
        shocked = shocked_yield_curve(curve, sc.shocks_bp)
        kr = compute_kr01(assets, liabilities, shocked, grid, cpr_override=cpr)
        out[sc.name] = kr["net_kr01_k"].values
    return out.set_index("Key")


def build_treasury_alco_pack(
    assets: Sequence[Instrument],
    liabilities: Sequence[Instrument],
    curve: YieldCurve,
    scenarios: Sequence,
    tier1_m: float,
    actual_delta_eve: dict[str, float],
    cpr_for_scenario=None,
    hedge_ratio: float = 0.80,
    breach_pct: float = 15.0,
    amber_pct: float = 12.0,
) -> dict:
    """One-shot pack for Board / ALCO / Treasury dashboards."""
    grid = KeyRateGrid()
    cpr_base = cpr_for_scenario("BASE") if cpr_for_scenario else None
    kr01 = compute_kr01(assets, liabilities, curve, grid, cpr_override=cpr_base)
    attr = eve_attribution_from_kr01(kr01, scenarios, actual_delta_eve, grid)
    limits = limit_dashboard(
        actual_delta_eve, attr["contribution_m"], tier1_m, scenarios,
        breach_pct=breach_pct, amber_pct=amber_pct,
    )
    efficiency = hedge_efficiency_table(scenarios, grid)
    rationale = why_hedge_rationale(limits, attr["contribution_m"], efficiency)
    packages_pct, packages_detail = hedge_package_comparison(
        kr01, scenarios, tier1_m, actual_delta_eve, grid, hedge_ratio=hedge_ratio,
    )
    scenario_kr01 = compute_kr01_under_scenarios(
        assets, liabilities, curve, scenarios, cpr_for_scenario, grid,
    )
    hedges = suggest_key_rate_hedges(kr01, hedge_ratio=hedge_ratio)
    ladder_notionals = proposed_swap_notionals(
        kr01, tenors=(2.0, 5.0, 10.0), hedge_ratio=hedge_ratio,
    )
    post_swap = post_swap_eve_impact(
        kr01, scenarios, tier1_m, actual_delta_eve, ladder_notionals, grid,
    )
    return {
        "kr01": kr01,
        "shock_bp": attr["shock_bp"],
        "attribution": attr["summary"],
        "contribution_m": attr["contribution_m"],
        "limits": limits,
        "efficiency": efficiency,
        "rationale": rationale,
        "packages_pct": packages_pct,
        "packages_detail": packages_detail,
        "scenario_kr01": scenario_kr01,
        "hedges": hedges,
        "ladder_notionals": ladder_notionals,
        "post_swap": post_swap,
        "parallel_dv01_k": parallel_dv01(assets, liabilities, curve, cpr_base),
    }
