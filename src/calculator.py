"""
calculator.py
-------------
IRRBB calculation engine — proper cash flow discounting, 19 BCBS 368 buckets.

EVE methodology (BCBS 368 §119–121)
-------------------------------------
For each instrument, cash flows are pre-bucketed into an array of shape
(N_BUCKETS,). EVE sensitivity is computed by discounting those cash flows
at the base curve and again at the shocked curve, then taking the difference:

    PV_base(i)    = Σ_k  CF_i[k]  × DF_base[k]
    PV_shocked(i) = Σ_k  CF_i[k]  × DF_shocked[k]
    ΔPVE(i)       = PV_shocked(i) - PV_base(i)

    ΔEVE = Σ_assets ΔPVE(i)  -  Σ_liabilities ΔPVE(i)

This correctly accounts for:
  - Coupons and principal arriving at different times (full schedule)
  - Different shock magnitudes per bucket (non-parallel scenarios)
  - Discount rate flooring at 0% (no negative rates)

NII methodology (BCBS 368 §109–112)
--------------------------------------
NII is computed over a 1-year horizon. Floating-rate / NMD notionals that
reprice within the horizon contribute:

    ΔNII(i) = repricing_notional(i) × shock(bucket_i) / 10_000

Mortgage CPR: prepaid principal returned within 1Y is assumed to reinvest
at the shocked short rate (O/N bucket), so higher CPR under rate-down
scenarios reduces asset NII (lost coupon on prepaid balances).

Mortgage prepayment (CPR)
-------------------------
For amortising instruments with ``prepay_enabled``, cash-flow schedules are
regenerated under each scenario's implied primary mortgage rate so EVE
captures negative convexity (duration shortens when rates fall).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .cashflows import Instrument
from .prepayment import ShockCprTable, scenario_market_mortgage_rate
from .scenarios import Scenario
from .yield_curve import YieldCurve, BASE_CURVE
from .time_buckets import BCBS_BUCKETS, BUCKET_MIDPOINTS, N_BUCKETS


@dataclass
class ScenarioResult:
    scenario:       Scenario
    delta_nii:      float
    delta_eve:      float
    delta_eve_pct:  float
    nii_asset:      float
    nii_liability:  float
    eve_asset:      float
    eve_liability:  float
    tier1:          float
    is_outlier:     bool
    is_watch:       bool

    @property
    def status(self) -> str:
        if self.is_outlier:
            return "OUTLIER"
        if self.is_watch:
            return "WATCH"
        return "PASS"


class IRRBBCalculator:
    """
    IRRBB calculator with proper cash flow discounting.

    Parameters
    ----------
    assets         : list of asset Instrument objects
    liabilities    : list of liability Instrument objects
    tier1_capital  : Tier 1 capital (USD millions)
    yield_curve    : YieldCurve instance (defaults to module-level BASE_CURVE)
    outlier_threshold : 0.15 (15%) per BCBS 368 §99
    watch_threshold   : 0.10 (10%) internal warning
    cpr_table      : optional user CPR by BASE + scenario id (mortgages / MBS)
    """

    def __init__(
        self,
        assets:            list[Instrument],
        liabilities:       list[Instrument],
        tier1_capital:     float = 500.0,
        yield_curve:       YieldCurve = None,
        outlier_threshold: float = 0.15,
        watch_threshold:   float = 0.10,
        cpr_table:         ShockCprTable | None = None,
    ):
        self.assets = assets
        self.liabilities = liabilities
        self.tier1 = tier1_capital
        self.curve = yield_curve or BASE_CURVE
        self.outlier_thr = outlier_threshold
        self.watch_thr = watch_threshold
        self.cpr_table = cpr_table

        # Base (no-shock) bucketed CFs — mortgages/MBS use base CPR
        self._asset_cfs = np.array([
            self._instrument_base_cfs(i) for i in assets
        ])
        self._liab_cfs = np.array([
            self._instrument_base_cfs(i) for i in liabilities
        ])

    def _cpr_override_for(self, scenario_id: str | None) -> float | None:
        if self.cpr_table is None:
            return None
        if scenario_id is None or scenario_id == "BASE":
            return self.cpr_table.base_cpr()
        return self.cpr_table.cpr_for_scenario(scenario_id)

    def _instrument_base_cfs(self, inst: Instrument) -> np.ndarray:
        """Bucket CFs under the base yield curve (incentive or user CPR)."""
        if inst.is_prepayable:
            cpr = self._cpr_override_for("BASE")
            mkt = scenario_market_mortgage_rate(
                self.curve.base_rates, None, inst.maturity_years,
            )
            return inst.bucket_cashflows_under_market_rate(mkt, cpr_override=cpr)
        return inst.bucket_cashflows()

    def _instrument_shocked_cfs(
        self,
        inst: Instrument,
        scenario: Scenario,
    ) -> np.ndarray:
        """Bucket CFs under scenario discount + CPR (negative convexity)."""
        if inst.is_prepayable:
            cpr = self._cpr_override_for(scenario.id)
            mkt = scenario_market_mortgage_rate(
                self.curve.base_rates,
                scenario.shocks_bp,
                inst.maturity_years,
            )
            return inst.bucket_cashflows_under_market_rate(mkt, cpr_override=cpr)
        return inst.bucket_cashflows()

    # ── EVE ───────────────────────────────────────────────────────────────────

    def _pv_matrix(
        self,
        cf_matrix: np.ndarray,        # shape (n_instruments, N_BUCKETS)
        shocks_bp: list[float] | None,
    ) -> np.ndarray:
        """
        Returns PV array of shape (n_instruments,).
        cf_matrix @ discount_factors  =  vector dot product per instrument.
        """
        rates = self.curve.shocked_rates(shocks_bp) if shocks_bp is not None \
            else self.curve.base_rates
        dfs = self.curve.discount_factors(rates)  # shape (N_BUCKETS,)
        return cf_matrix @ dfs                       # shape (n_instruments,)

    def calc_eve(self, scenario: Scenario) -> tuple[float, float, float]:
        """
        Returns (delta_eve, asset_contribution, liability_contribution).
        ΔEVE = [PV_shocked(assets) - PV_base(assets)]
               - [PV_shocked(liabs) - PV_base(liabs)]

        Prepayable mortgages regenerate CF schedules under the shocked
        primary mortgage rate before discounting (behavioural optionality).
        """
        asset_shocked = np.array([
            self._instrument_shocked_cfs(i, scenario) for i in self.assets
        ])
        liab_shocked = np.array([
            self._instrument_shocked_cfs(i, scenario) for i in self.liabilities
        ])

        pv_base_a = self._pv_matrix(self._asset_cfs, None)
        pv_shocked_a = self._pv_matrix(asset_shocked, scenario.shocks_bp)
        pv_base_l = self._pv_matrix(self._liab_cfs, None)
        pv_shocked_l = self._pv_matrix(liab_shocked, scenario.shocks_bp)

        eve_asset = float(np.sum(pv_shocked_a - pv_base_a))
        eve_liab = float(np.sum(pv_shocked_l - pv_base_l))
        return eve_asset - eve_liab, eve_asset, eve_liab

    # ── NII ───────────────────────────────────────────────────────────────────

    def _nii_side(
        self,
        instruments: list[Instrument],
        scenario:    Scenario,
        sign:        int,
    ) -> float:
        total = 0.0
        short_shock = scenario.shock_at_bucket(0) / 10_000
        for inst in instruments:
            if inst.instrument_type in ("bullet_floating", "demand_deposit"):
                bucket = inst.cashflows[0].bucket   # single repricing CF
                shock_dec = scenario.shock_at_bucket(bucket) / 10_000
                total += sign * inst.notional * shock_dec
            elif inst.is_prepayable and inst.side == "asset":
                # Lost coupon on balances prepaid within 1Y under the shock,
                # partially offset by reinvestment at the shocked short rate.
                cpr_base = self._cpr_override_for("BASE")
                cpr_shock = self._cpr_override_for(scenario.id)
                mkt_base = scenario_market_mortgage_rate(
                    self.curve.base_rates, None, inst.maturity_years,
                )
                mkt_shock = scenario_market_mortgage_rate(
                    self.curve.base_rates, scenario.shocks_bp, inst.maturity_years,
                )
                base_cfs = inst.cashflows_under_market_rate(
                    mkt_base, cpr_override=cpr_base,
                )
                shock_cfs = inst.cashflows_under_market_rate(
                    mkt_shock, cpr_override=cpr_shock,
                )

                def _prep_1y(cfs):
                    return sum(
                        cf.amount for cf in cfs
                        if cf.cf_type == "prepayment" and cf.time_years <= 1.0 + 1e-9
                    )

                prep_base = _prep_1y(base_cfs)
                prep_shock = _prep_1y(shock_cfs)
                extra_prep = prep_shock - prep_base
                coupon_dec = inst.coupon_pct / 100.0
                reinvest = float(self.curve.base_rates[0]) + short_shock
                total += sign * extra_prep * (reinvest - coupon_dec) * 0.5
        return total

    def calc_nii(self, scenario: Scenario) -> tuple[float, float, float]:
        nii_a = self._nii_side(self.assets,      scenario, +1)
        nii_l = self._nii_side(self.liabilities, scenario, -1)
        return nii_a + nii_l, nii_a, nii_l

    # ── Full scenario ─────────────────────────────────────────────────────────

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        delta_nii, nii_a, nii_l = self.calc_nii(scenario)
        delta_eve, eve_a, eve_l = self.calc_eve(scenario)
        eve_pct = abs(delta_eve) / self.tier1 * 100

        return ScenarioResult(
            scenario=scenario,
            delta_nii=delta_nii,
            delta_eve=delta_eve,
            delta_eve_pct=eve_pct,
            nii_asset=nii_a,
            nii_liability=nii_l,
            eve_asset=eve_a,
            eve_liability=eve_l,
            tier1=self.tier1,
            is_outlier=eve_pct > self.outlier_thr * 100,
            is_watch=(eve_pct > self.watch_thr * 100)
            and (eve_pct <= self.outlier_thr * 100),
        )

    def run_all(self, scenarios: list[Scenario]) -> list[ScenarioResult]:
        return [self.run_scenario(s) for s in scenarios]

    # ── Repricing gap ─────────────────────────────────────────────────────────

    _MATURING_CF_TYPES = frozenset({"principal", "prepayment", "repricing"})

    def _cf_dv01(self, amount: float, bucket_idx: int) -> float:
        """DV01 ($M per +1bp) for a maturing / repricing cash flow in one bucket."""
        t = BUCKET_MIDPOINTS[bucket_idx]
        r = float(self.curve.base_rates[bucket_idx])
        df = (1.0 + r) ** (-t)
        return amount * t * df * 1e-4

    def bucket_dv01_gap(self) -> pd.DataFrame:
        """
        Bucket-wise DV01 from maturing / repricing cash flows.

        Uses principal and repricing legs only (not coupon income). Asset DV01 is
        positive, liability DV01 is stored as a positive magnitude; net = assets − liabs.
        """
        asset_dv01 = np.zeros(N_BUCKETS)
        liab_dv01 = np.zeros(N_BUCKETS)

        for inst in self.assets:
            for cf in inst.cashflows:
                if cf.cf_type not in self._MATURING_CF_TYPES:
                    continue
                asset_dv01[cf.bucket] += self._cf_dv01(cf.amount, cf.bucket)

        for inst in self.liabilities:
            for cf in inst.cashflows:
                if cf.cf_type not in self._MATURING_CF_TYPES:
                    continue
                liab_dv01[cf.bucket] += self._cf_dv01(cf.amount, cf.bucket)

        rows = []
        for b in BCBS_BUCKETS:
            idx = b.index
            a = float(asset_dv01[idx])
            l = float(liab_dv01[idx])
            rows.append({
                "bucket": b.label,
                "asset_dv01": a,
                "liability_dv01": l,
                "net_dv01": a - l,
            })
        return pd.DataFrame(rows).set_index("bucket")

    def bucket_maturity_gap(self) -> pd.DataFrame:
        """
        Notional gap using only principal / repricing cash flows per bucket.

        Aligns with the DV01 bucket view (unlike ``repricing_gap()`` which counts
        full instrument notional when any coupon lands in the bucket).
        """
        asset_n = np.zeros(N_BUCKETS)
        liab_n = np.zeros(N_BUCKETS)
        for inst in self.assets:
            for cf in inst.cashflows:
                if cf.cf_type in self._MATURING_CF_TYPES:
                    asset_n[cf.bucket] += cf.amount
        for inst in self.liabilities:
            for cf in inst.cashflows:
                if cf.cf_type in self._MATURING_CF_TYPES:
                    liab_n[cf.bucket] += cf.amount
        rows = []
        for b in BCBS_BUCKETS:
            idx = b.index
            a, l = float(asset_n[idx]), float(liab_n[idx])
            rows.append({
                "bucket": b.label,
                "assets": a,
                "liabilities": l,
                "net_gap": a - l,
            })
        return pd.DataFrame(rows).set_index("bucket")

    def irs_hedge_suggestions(
        self,
        dv01_gap: pd.DataFrame,
        hedge_ratio: float = 0.80,
        min_net_dv01_k: float = 1.0,
    ) -> pd.DataFrame:
        return suggest_irs_hedges(dv01_gap, hedge_ratio, min_net_dv01_k)

    def repricing_gap(self) -> pd.DataFrame:
        """Notional repricing gap per BCBS 368 bucket (all 19)."""
        rows = []
        for b in BCBS_BUCKETS:
            a = sum(i.notional for i in self.assets
                    if any(cf.bucket == b.index for cf in i.cashflows))
            liab_total = sum(
                i.notional for i in self.liabilities
                if any(cf.bucket == b.index for cf in i.cashflows)
            )
            rows.append({"bucket": b.label, "assets": a,
                         "liabilities": liab_total, "net_gap": a - liab_total})
        return pd.DataFrame(rows).set_index("bucket")

    # ── Instrument-level EVE detail ───────────────────────────────────────────

    def instrument_eve_detail(self, scenario: Scenario) -> pd.DataFrame:
        """Per-instrument ΔEVE breakdown — useful for attribution."""
        rows = []
        for sign, instruments, base_matrix in (
            (+1, self.assets, self._asset_cfs),
            (-1, self.liabilities, self._liab_cfs),
        ):
            shocked_matrix = np.array([
                self._instrument_shocked_cfs(i, scenario) for i in instruments
            ])
            pv_base = self._pv_matrix(base_matrix, None)
            pv_shocked = self._pv_matrix(shocked_matrix, scenario.shocks_bp)
            for inst, pv_b, pv_s in zip(instruments, pv_base, pv_shocked):
                delta_pv = sign * (pv_s - pv_b)
                cpr_note = ""
                if inst.is_prepayable:
                    cpr = self._cpr_override_for(scenario.id)
                    if cpr is None:
                        mkt = scenario_market_mortgage_rate(
                            self.curve.base_rates, scenario.shocks_bp, inst.maturity_years,
                        )
                        cpr = inst.resolve_cpr(mkt)
                    cpr_note = f"{cpr * 100:.1f}%"
                rows.append({
                    "side":          inst.side.upper(),
                    "instrument":    inst.name,
                    "notional":      inst.notional,
                    "type":          inst.instrument_type,
                    "eff_duration":  round(inst.effective_duration, 2),
                    "pv_base":       round(sign * pv_b, 2),
                    "pv_shocked":    round(sign * pv_s, 2),
                    "delta_eve":     round(delta_pv, 2),
                    "cpr_shocked":   cpr_note,
                })
        return pd.DataFrame(rows)

    # ── Summary table ─────────────────────────────────────────────────────────

    def summary_table(self, results: list[ScenarioResult]) -> pd.DataFrame:
        return pd.DataFrame([{
            "Scenario":      r.scenario.name,
            "Description":   r.scenario.description,
            "ΔNII ($M)":     round(r.delta_nii,     2),
            "ΔEVE ($M)":     round(r.delta_eve,     2),
            "|ΔEVE|/T1 (%)": round(r.delta_eve_pct, 1),
            "Status":        r.status,
        } for r in results])


def suggest_irs_hedges(
    dv01_gap: pd.DataFrame,
    hedge_ratio: float = 0.80,
    min_net_dv01_k: float = 1.0,
) -> pd.DataFrame:
    """
    Suggest pay-fixed / receive-fixed IRS hedges to offset bucket DV01 gaps.

    Indicative swap notional neutralises ``hedge_ratio`` of |net DV01| using
    DV01 ≈ notional × bucket_midpoint × 0.0001 ($M per bp).
    """
    label_to_idx = {b.label: b.index for b in BCBS_BUCKETS}
    rows = []
    for bucket, row in dv01_gap.iterrows():
        net_m = float(row["net_dv01"])
        net_k = net_m * 1000.0
        if abs(net_k) < min_net_dv01_k:
            continue
        idx = label_to_idx.get(str(bucket), 0)
        duration = max(BUCKET_MIDPOINTS[idx], 1 / 365)
        notional_m = abs(net_m) * hedge_ratio / (duration * 1e-4)

        if net_m > 0:
            structure = "Pay-fixed / receive-floating IRS"
            action = "Enter payer swap"
            rationale = (
                "Asset-heavy bucket — paying fixed converts floating asset "
                "exposure and reduces NII/EVE loss if rates rise."
            )
        else:
            structure = "Receive-fixed / pay-floating IRS"
            action = "Enter receiver swap"
            rationale = (
                "Liability-heavy bucket — receiving fixed adds duration and "
                "offsets deposit / funding repricing when rates rise."
            )

        rows.append({
            "Bucket": bucket,
            "Net DV01 ($K/bp)": round(net_k, 1),
            "Hedge action": action,
            "IRS structure": structure,
            "Indicative tenor (Y)": round(duration, 2),
            "Indicative notional ($M)": round(notional_m, 1),
            "Target hedge (%)": int(hedge_ratio * 100),
            "Rationale": rationale,
        })
    return pd.DataFrame(rows)
