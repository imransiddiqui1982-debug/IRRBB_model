"""
app.py
------
IRRBB Model — Interactive Streamlit Dashboard
Basel III / BCBS 368 · 19 Buckets · Full Cash Flow Discounting

Run:  streamlit run app.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.balance_sheet import get_instruments  # noqa: E402
from src.load_balance_sheet import load_instruments_from_csv  # noqa: E402
from src.nmd_refinement import refine_nmd_deposits, merge_nmd_into_balance_sheet, load_customer_nmd  # noqa: E402
from src.prepayment import (  # noqa: E402
    DEFAULT_SHOCK_CPR_PCT,
    DEFAULT_SHOCK_PSA_PCT,
    ShockCprTable,
)
from src.calculator import IRRBBCalculator, suggest_irs_hedges  # noqa: E402
from src.key_rate_duration import (  # noqa: E402
    build_treasury_alco_pack,
    designate_key_rate_hedges,
    instrument_kr01_attribution,
)
from src.liquidity_ratios import compute_liquidity_ratios_with_workbook  # noqa: E402
from src.market_curve import get_live_yield_curve  # noqa: E402
from src.scenarios import SCENARIOS, REF_LABELS  # noqa: E402
from src.time_buckets import BUCKET_LABELS, N_BUCKETS  # noqa: E402
from src.yield_curve import YieldCurve  # noqa: E402

# Import after core modules; capture full traceback for Streamlit Cloud logs/UI
_CPR_IMPORT_ERROR: str | None = None
try:
    from src.cpr_calibration import (  # noqa: E402
        CprCalibrationInputs,
        calibrate_scenario_cprs,
        calibration_display_frame,
        calibration_to_cpr_maps,
        format_refi_incentive_bp,
        incentive_bp,
        scurve_dataframe,
        scurve_display_frame,
        historical_regimes_dataframe,
    )
except Exception:  # pragma: no cover - Cloud diagnostics
    import traceback
    _CPR_IMPORT_ERROR = traceback.format_exc()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IRRBB Model — BCBS 368",
    layout="wide",
    initial_sidebar_state="expanded",
)

if _CPR_IMPORT_ERROR:
    st.error("Failed to import src.cpr_calibration (full traceback below).")
    st.code(_CPR_IMPORT_ERROR)
    st.stop()

# ── Colour palette — light institutional theme ────────────────────────────────
BG        = "#f5f6fa"
BG2       = "#ffffff"
BG3       = "#eef0f7"
BORDER    = "#d0d5e8"
TEXT      = "#1a1f2e"
DIM       = "#6b7394"
NAVY      = "#1a3a6b"
BLUE      = "#1a56db"
RED       = "#c0392b"
GREEN     = "#1a7a4a"
AMBER     = "#b35c00"
ORANGE    = "#d4600a"
PURPLE    = "#6c3483"
CYAN      = "#0e7490"
TEAL      = "#0d6e6e"
YELLOW    = "#7d6608"

SIDEBAR_BG      = "#1a2744"
SIDEBAR_BORDER  = "#243156"
SIDEBAR_TEXT    = "#c8d4f0"
SIDEBAR_DIM     = "#6b88c4"
SIDEBAR_INPUT   = "#243156"
SIDEBAR_INPUT_BORDER = "#344573"

SCENARIO_COLORS = [RED, BLUE, PURPLE, ORANGE, TEAL, CYAN]

SCENARIO_SHORT = {
    "Parallel Shift Up": "Par Up",
    "Parallel Shift Down": "Par Dn",
    "Steepener": "Steep",
    "Flattener": "Flat",
    "Short Rates Up": "Short ↑",
    "Short Rates Down": "Short ↓",
}


def _is_retail_deposit(name: str) -> bool:
    """True for NMD / retail deposit instruments (excluded from comparison charts)."""
    n = name.lower()
    keys = (
        "core sticky", "rate sensitive", "non-core volatile", "non_core",
        "demand deposit", "retail nmd", "term deposit", "nmd /",
        "retail transactional", "retail non", "retail non_transactional",
        "retail non-transactional", "wholesale —", "wholesale operational",
        "/ wholesale", "sticky", "savings",
    )
    return any(k in n for k in keys)


def _filter_non_deposit_eve(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    return detail[~detail["instrument"].astype(str).map(_is_retail_deposit)].copy()


def _scenario_status_color(status: str) -> str:
    if status == "OUTLIER":
        return RED
    if status == "WATCH":
        return AMBER
    return GREEN

PLOTLY_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="'IBM Plex Sans', Arial, sans-serif", color=TEXT, size=11),
    margin=dict(l=20, r=20, t=44, b=20),
)

AXIS_STYLE = dict(
    gridcolor=BORDER,
    linecolor=BORDER,
    title_font=dict(color=TEXT, size=11),
    zerolinecolor=BORDER,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
}}
.stApp {{
    background-color: {BG};
}}
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {SIDEBAR_BORDER};
}}
[data-testid="stSidebar"] * {{
    color: {SIDEBAR_TEXT} !important;
}}
[data-testid="stSidebar"] input {{
    background: {SIDEBAR_INPUT} !important;
    border-color: {SIDEBAR_INPUT_BORDER} !important;
    color: {SIDEBAR_TEXT} !important;
}}
[data-testid="stSidebar"] [data-baseweb="radio"] label {{
    color: {SIDEBAR_TEXT} !important;
}}
[data-testid="metric-container"] {{
    background: {BG};
    border: 1px solid {BORDER};
    border-top: 3px solid {BLUE};
    padding: 14px 16px;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.4rem !important;
    color: {NAVY} !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    color: {DIM} !important;
    font-weight: 500 !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.75rem !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: {BG2};
    border-bottom: 2px solid {BORDER};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {DIM};
    border-radius: 0;
    padding: 10px 22px;
    font-size: 0.75rem;
    letter-spacing: 0.8px;
    font-weight: 500;
    text-transform: uppercase;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}}
.stTabs [aria-selected="true"] {{
    background: {BG2};
    color: {BLUE};
    border-bottom: 2px solid {BLUE};
    font-weight: 600;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 4px;
}}
[data-testid="stNumberInput"] input {{
    background: {SIDEBAR_INPUT} !important;
    border: 1px solid {SIDEBAR_INPUT_BORDER} !important;
    color: {SIDEBAR_TEXT} !important;
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 4px;
}}
h1, h2, h3 {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: {NAVY} !important;
    font-weight: 600 !important;
}}
hr {{
    border-color: {BORDER};
    margin: 12px 0;
}}
.section-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {DIM};
    margin-bottom: 8px;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"<div style='padding:4px 0 2px'>"
        f"<span style='font-size:13px;font-weight:700;color:{NAVY};"
        f"letter-spacing:1px'>IRRBB MODEL</span><br>"
        f"<span style='font-size:10px;color:{DIM}'>"
        f"Basel III / BCBS 368 · April 2016</span></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("<p class='section-label'>Regulatory Parameters</p>",
                unsafe_allow_html=True)
    tier1 = st.number_input(
        "Tier 1 Capital (USD M)",
        value=500, min_value=50, max_value=10000, step=50,
    )
    st.markdown(
    f"<div style='font-size:11px;line-height:1.8;margin-top:4px'>"
    f"<span style='color:{SIDEBAR_DIM}'>Outlier threshold: </span>"
    f"<span style='color:{SIDEBAR_TEXT};font-weight:600;font-family:monospace'>"
    f"${tier1 * 0.15:.0f}M</span>"
    f"<span style='color:{SIDEBAR_DIM}'> (15% of T1)</span><br>"
    f"<span style='color:{SIDEBAR_DIM}'>Watch threshold: </span>"
    f"<span style='color:{SIDEBAR_TEXT};font-weight:600;font-family:monospace'>"
    f"${tier1 * 0.10:.0f}M</span>"
    f"<span style='color:{SIDEBAR_DIM}'> (10% of T1)</span>"
    f"</div>",
    unsafe_allow_html=True,
)
    
    st.divider()

    st.markdown("<p class='section-label'>Balance Sheet</p>",
                unsafe_allow_html=True)
    bs_template_path = os.path.join(os.path.dirname(__file__), "data", "balance_sheet_template.csv")
    if not os.path.exists(bs_template_path):
        bs_template_path = os.path.join(os.path.dirname(__file__), "data", "sample_balance_sheet.csv")
    with open(bs_template_path, "rb") as f:
        template_bytes = f.read()
    st.download_button(
        "Download CSV template",
        template_bytes,
        file_name="balance_sheet_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    use_calibrated_bs = st.checkbox(
        "Load calibrated balance sheet sample",
        value=os.path.exists(bs_template_path),
        help="Uses balance_sheet_template.csv ($2.9B assets, aligned with deposit model).",
    )
    uploaded_csv = st.file_uploader(
        "Upload balance sheet CSV",
        type=["csv"],
        disabled=use_calibrated_bs,
        help="One row per instrument. Use the template for required columns.",
    )

    st.divider()

    st.markdown("<p class='section-label'>NMD Deposits</p>",
                unsafe_allow_html=True)
    nmd_template_path = os.path.join(os.path.dirname(__file__), "data", "comprehensive_deposit_template.xlsx")
    legacy_template = os.path.join(os.path.dirname(__file__), "data", "nmd_deposit_template.xlsx")
    deposit_template_path = nmd_template_path if os.path.exists(nmd_template_path) else legacy_template
    if os.path.exists(deposit_template_path):
        with open(deposit_template_path, "rb") as f:
            nmd_template_bytes = f.read()
        st.download_button(
            "Download comprehensive deposit template",
            nmd_template_bytes,
            file_name="comprehensive_deposit_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    v2_path = os.path.join(os.path.dirname(__file__), "data", "us_lcr_nsfr_deposit_model.xlsx")
    if not os.path.exists(v2_path):
        v2_path = os.path.join(os.path.dirname(__file__), "data", "comprehensive_deposit_template_v2.xlsx")
    if os.path.exists(v2_path):
        with open(v2_path, "rb") as f:
            v2_bytes = f.read()
        st.download_button(
            "Download deposit + LCR/NSFR model",
            v2_bytes,
            file_name="us_lcr_nsfr_deposit_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    use_nmd = st.checkbox(
        "Refine NMD deposits and feed IRRBB",
        value=True,
        help="Upload monthly deposit history to estimate core / non-core / sticky splits.",
    )
    use_v2_sample = st.checkbox(
        "Load calibrated deposit model sample",
        value=use_nmd and os.path.exists(v2_path),
        disabled=not use_nmd or not os.path.exists(v2_path),
        help="Uses us_lcr_nsfr_deposit_model.xlsx — NMD + HQLA + LCR/NSFR inputs.",
    )
    deposit_name = st.text_input("Deposit pool name", value="Retail NMD")
    uploaded_nmd = st.file_uploader(
        "Upload NMD customer deposit file",
        type=["csv", "xlsx", "xls"],
        disabled=not use_nmd or use_v2_sample,
        help=(
            "Excel: Customer Deposits + Deposit Rates (monthly product rates) "
            "+ Market Rates. Or enable 'Load v2 sample' above."
        ),
    )

    st.divider()

    st.markdown("<p class='section-label'>Yield Curve</p>",
                unsafe_allow_html=True)
    use_live_curve = st.checkbox(
        "Use live SOFR + USD IRS mid curve",
        value=True,
        help="0–12M: SOFR (NY Fed + SOFR swap tenors). 1Y–10Y: USD SOFR IRS mids.",
    )
    refresh_curve = st.button("Refresh live curve", use_container_width=True)

    st.divider()

    st.markdown("<p class='section-label'>Active Scenario</p>",
                unsafe_allow_html=True)
    selected_name = st.radio(
        "scenario", [s.name for s in SCENARIOS],
        label_visibility="collapsed",
    )
    selected_scenario = next(s for s in SCENARIOS if s.name == selected_name)

    st.divider()
    st.markdown("<p class='section-label'>Shock Profile</p>",
                unsafe_allow_html=True)
    for label, bp in zip(REF_LABELS, selected_scenario.ref_shocks_bp):
        color = GREEN if bp > 0 else (RED if bp < 0 else DIM)
        sign = "+" if bp > 0 else ""
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:11px;padding:2px 0;font-family:monospace'>"
            f"<span style='color:{DIM}'>{label}</span>"
            f"<span style='color:{color};font-weight:600'>{sign}{bp} bp</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("<p class='section-label'>Mortgage / MBS CPR (S-curve)</p>",
                unsafe_allow_html=True)
    st.caption(
        "Sidebar **WAC** and **PMMS** drive live EVE / KR01 for MBS and whole loans: "
        "WAC is applied to each OA pool; PMMS sets the primary-secondary spread so "
        "mortgage rate ≈ PMMS at the pool anchor. Refi incentive = WAC − mortgage rate. "
        "LCR/NSFR still use contractual maturity only."
    )

    cal_wac = st.number_input(
        "Portfolio WAC (%)",
        min_value=0.0, max_value=20.0, value=5.50, step=0.05,
        key="cpr_cal_wac",
        help="Weighted-average coupon on mortgages / MBS (manual input).",
    )
    cal_pmms = st.number_input(
        "Current PMMS / offering rate (%)",
        min_value=0.0, max_value=20.0, value=6.50, step=0.05,
        key="cpr_cal_pmms",
        help="Freddie Mac PMMS 30Y or current offering rate (manual input).",
    )
    cal_age = st.number_input(
        "Seasoning for PSA conversion (months)",
        min_value=0, max_value=360, value=30, step=1,
        key="cpr_cal_age",
    )
    apply_speed = st.radio(
        "Feed EVE / NII with",
        ["CPR %", "PSA %"],
        horizontal=True,
        key="cpr_apply_speed",
        help=(
            "Both are read off the S-curve at each scenario's refi incentive. "
            "CPR % = annual constant prepayment; PSA % = PSA multiple at seasoning."
        ),
    )

    _base_inc = incentive_bp(float(cal_wac), float(cal_pmms))
    st.markdown(
        f"<div style='font-size:12px;font-family:monospace;padding:6px 0;'>"
        f"<span style='color:{DIM}'>Base refi incentive</span> "
        f"<b style='color:{GREEN if _base_inc > 0 else (RED if _base_inc < 0 else DIM)}'>"
        f"{format_refi_incentive_bp(_base_inc)}</b>"
        f"<span style='color:{DIM}'> = WAC {cal_wac:.2f}% - PMMS {cal_pmms:.2f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _calib_preview = calibrate_scenario_cprs(
        CprCalibrationInputs(
            wac_pct=float(cal_wac),
            pmms_pct=float(cal_pmms),
            age_months=int(cal_age),
        )
    )
    shock_cpr_inputs, shock_psa_inputs = calibration_to_cpr_maps(_calib_preview)
    st.session_state["shock_cpr_store"] = dict(shock_cpr_inputs)
    st.session_state["shock_psa_store"] = dict(shock_psa_inputs)
    st.session_state["cpr_calib_df"] = _calib_preview
    use_custom_cpr = True

    st.dataframe(
        calibration_display_frame(_calib_preview),
        use_container_width=True,
        hide_index=True,
    )

    _sc = scurve_dataframe()
    fig_side = go.Figure()
    fig_side.add_trace(go.Scatter(
        x=_sc["incentive_bp"], y=_sc["cpr_pct"],
        mode="lines+markers", name="S-curve",
        line=dict(color=NAVY, width=2),
        marker=dict(size=5),
    ))
    fig_side.add_trace(go.Scatter(
        x=_calib_preview["incentive_bp"],
        y=_calib_preview["cpr_pct"],
        mode="markers+text",
        name="WAC/PMMS → scenarios",
        text=_calib_preview["key"],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=9, color=ORANGE),
        hovertemplate=(
            "%{text}<br>Refi incentive: %{customdata}<br>CPR: %{y:.1f}%<extra></extra>"
        ),
        customdata=_calib_preview["refi_incentive"],
    ))
    fig_side.add_vline(x=0, line_dash="dot", line_color=BORDER)
    fig_side.update_layout(
        **{**PLOTLY_BASE, "margin": dict(l=10, r=10, t=30, b=10)},
        height=280,
        xaxis=dict(**AXIS_STYLE, title="Refi incentive (WAC-PMMS), bp  (+ ITM / - OTM)"),
        yaxis=dict(**AXIS_STYLE, title="CPR %"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor=BG2, font=dict(size=9)),
    )
    st.plotly_chart(fig_side, use_container_width=True)

    with st.expander("S-curve knots (refi incentive → CPR %)"):
        st.dataframe(
            scurve_display_frame()[["Refi incentive", "CPR %"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Illustrative agency-style knots. Replace with Fannie/Freddie Cohort "
            "Analyzer or Clarity S-curve extracts for exam-ready calibration."
        )
    with st.expander("Historical regimes (relevance)"):
        st.dataframe(
            historical_regimes_dataframe(),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.caption("Upload balance sheet CSV and/or refine NMD deposits before IRRBB.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_market_curve(force_refresh: bool = False):
    """Fetch live SOFR / IRS curve (cached 5 minutes)."""
    _ = force_refresh  # changes cache key when Refresh is clicked
    return get_live_yield_curve(use_cache_on_failure=True)


@st.cache_data(ttl=60)
def run_model(
    tier1_cap: float,
    csv_bytes: bytes | None,
    nmd_bytes: bytes | None,
    nmd_name: str,
    use_nmd: bool,
    curve_tenors: tuple[float, ...] | None,
    curve_rates: tuple[float, ...] | None,
    use_custom_cpr: bool = False,
    apply_speed: str = "CPR %",
    cpr_pct_items: tuple[tuple[str, float], ...] = (),
    psa_pct_items: tuple[tuple[str, float], ...] = (),
    portfolio_wac_pct: float = 5.50,
    portfolio_pmms_pct: float = 6.50,
    _model_version: int = 13,
):
    if csv_bytes:
        assets, liabilities = load_instruments_from_csv(io.BytesIO(csv_bytes))
    else:
        assets, liabilities = get_instruments()

    nmd_result = None
    if use_nmd and nmd_bytes:
        nmd_result = refine_nmd_deposits(io.BytesIO(nmd_bytes), deposit_name=nmd_name)
        assets, liabilities = merge_nmd_into_balance_sheet(
            assets, liabilities, nmd_result
        )

    if curve_tenors and curve_rates:
        curve = YieldCurve(ref_tenors=list(curve_tenors), ref_rates=list(curve_rates))
    else:
        curve = YieldCurve()

    # Sidebar WAC + PMMS → live OA prepay (EVE / KR01)
    from src.mbs_pricing import apply_portfolio_wac_pmms
    apply_portfolio_wac_pmms(
        list(assets) + list(liabilities),
        curve,
        portfolio_wac_pct,
        portfolio_pmms_pct,
    )

    cpr_table = None
    psa_map: dict[str, float] = {}
    if use_custom_cpr:
        cpr_map = dict(cpr_pct_items)
        psa_map = dict(psa_pct_items)
        if apply_speed == "PSA %":
            cpr_table = ShockCprTable.from_psa_map(psa_map or DEFAULT_SHOCK_PSA_PCT)
        else:
            cpr_table = ShockCprTable.from_pct_map(cpr_map or DEFAULT_SHOCK_CPR_PCT)

    calc = IRRBBCalculator(
        assets,
        liabilities,
        tier1_capital=tier1_cap,
        yield_curve=curve,
        cpr_table=cpr_table,
    )
    results = calc.run_all(SCENARIOS)
    gap = calc.repricing_gap()
    maturity_gap = calc.bucket_maturity_gap()
    dv01_gap = calc.bucket_dv01_gap()
    return (
        calc, results, gap, maturity_gap, dv01_gap,
        assets, liabilities, nmd_result, curve, cpr_table, psa_map, apply_speed,
    )


# Live / stylised curve for EVE-NII discounting
curve_snap = None
curve_tenors_t: tuple[float, ...] | None = None
curve_rates_t: tuple[float, ...] | None = None
if use_live_curve:
    try:
        live_curve, curve_snap = load_market_curve(force_refresh=bool(refresh_curve))
        tenors = sorted(curve_snap.points.keys())
        curve_tenors_t = tuple(tenors)
        curve_rates_t = tuple(curve_snap.points[t] for t in tenors)
        st.sidebar.success(
            f"Live curve as of {curve_snap.as_of}"
        )
        for note in curve_snap.source_notes[:3]:
            st.sidebar.caption(note)
    except Exception as exc:
        st.sidebar.warning(f"Live curve unavailable — using stylised curve. ({exc})")
        use_live_curve = False

csv_payload = None
if use_calibrated_bs and os.path.exists(bs_template_path):
    with open(bs_template_path, "rb") as f:
        csv_payload = f.read()
elif uploaded_csv is not None:
    csv_payload = uploaded_csv.getvalue()
nmd_payload = None
if use_nmd:
    if use_v2_sample and os.path.exists(v2_path):
        with open(v2_path, "rb") as f:
            nmd_payload = f.read()
    elif uploaded_nmd is not None:
        nmd_payload = uploaded_nmd.getvalue()
if use_nmd and not nmd_payload:
    st.error(
        "NMD refinement is enabled. Upload a deposit file or enable "
        "'Load calibrated deposit model sample' in the sidebar."
    )
    st.stop()
try:
    (
        calc, results, gap, maturity_gap, dv01_gap,
        assets, liabilities, nmd_result, curve, cpr_table, psa_map, apply_speed_used,
    ) = run_model(
        float(tier1),
        csv_payload,
        nmd_payload,
        deposit_name.strip() or "Retail NMD",
        use_nmd,
        curve_tenors_t,
        curve_rates_t,
        use_custom_cpr,
        apply_speed,
        tuple(sorted(shock_cpr_inputs.items())) if use_custom_cpr else (),
        tuple(sorted(shock_psa_inputs.items())) if use_custom_cpr else (),
        float(cal_wac),
        float(cal_pmms),
    )
except UnicodeDecodeError:
    st.error(
        "File encoding error: the balance sheet must be a **CSV** file and the deposit "
        "file must be an **Excel workbook (.xlsx)**. Do not swap them in the uploaders."
    )
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
active = next(r for r in results if r.scenario.name == selected_name)
detail = calc.instrument_eve_detail(selected_scenario)
nmd_customers = None
if nmd_payload:
    try:
        nmd_customers = load_customer_nmd(io.BytesIO(nmd_payload)).customers
    except Exception:
        nmd_customers = None
workbook_src = io.BytesIO(nmd_payload) if nmd_payload else None
if workbook_src is None and os.path.exists(v2_path):
    with open(v2_path, "rb") as _wb_f:
        workbook_src = io.BytesIO(_wb_f.read())
liquidity = compute_liquidity_ratios_with_workbook(
    assets,
    liabilities,
    workbook_src,
    nmd_result=nmd_result,
    customers=nmd_customers,
    tier1_capital=float(tier1),
)
lcr_result = liquidity.lcr
nsfr_result = liquidity.nsfr


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"<h2 style='font-size:20px;letter-spacing:0.5px;margin-bottom:2px'>"
    f"Interest Rate Risk in the Banking Book</h2>"
    f"<p style='color:{DIM};font-size:11px;margin-top:0'>"
    f"BCBS 368 (April 2016) · 19 Repricing Buckets · "
    f"Full Cash Flow Discounting · 6 Prescribed Scenarios · "
    f"Mortgage CPR (S-curve / PSA)</p>",
    unsafe_allow_html=True,
)
st.divider()

total_a = sum(i.notional for i in assets)
total_l = sum(i.notional for i in liabilities)
total_cf = sum(len(i.cashflows) for i in assets + liabilities)
outliers = sum(1 for r in results if r.is_outlier)
watches = sum(1 for r in results if r.is_watch)
worst = max(results, key=lambda r: r.delta_eve_pct)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Assets",       f"${total_a:,.0f}M")
c2.metric("Total Liabilities",  f"${total_l:,.0f}M")
c3.metric("Scheduled CFs",      f"{total_cf:,}")
c4.metric("Tier 1 Capital",     f"${tier1:,.0f}M")
c5.metric(
    "Outlier Breaches", str(outliers),
    delta="None ✓" if outliers == 0 else f"{outliers} breach(es)",
    delta_color="normal" if outliers == 0 else "inverse",
)
c6.metric(
    "Worst |ΔEVE|/T1",
    f"{worst.delta_eve_pct:.1f}%",
    delta=worst.scenario.name,
    delta_color="off",
)

if cpr_table is not None:
    with st.expander(
        f"Diagnostic CPR / PSA from sidebar WAC+PMMS ({apply_speed_used})",
        expanded=False,
    ):
        st.caption(
            "Portfolio-level diagnostic S-curve (reporting). "
            "**EVE / KR01** reprice MBS and whole loans live from each pool’s "
            "curve anchor + WAC (option-adjusted). LCR/NSFR use contractual maturity only."
        )
        st.dataframe(
            cpr_table.as_pct_dataframe(psa_by_key=psa_map or None),
            use_container_width=True,
            hide_index=True,
        )

# Live option-adjusted prepayment diagnostics (spec §6)
try:
    from src.mbs_pricing import prepayment_diagnostics
    from src.key_rate_duration import shocked_yield_curve
    _oa_assets = [a for a in assets if getattr(a, "is_option_adjusted", False)]
    if _oa_assets:
        with st.expander(
            "Prepayment diagnostics — live OA (mortgage rate → CPR → WAL)",
            expanded=True,
        ):
            st.caption(
                "Per pool, re-derived from the active yield curve (Steps A/B/C). "
                "This is what drives EVE and KR01 for MBS / whole loans."
            )
            _diag_rows = []
            _diag_rows.append(
                prepayment_diagnostics(_oa_assets, curve, "Base")
            )
            for _sc in SCENARIOS:
                _scurve = shocked_yield_curve(curve, _sc.shocks_bp, scenario=_sc)
                _diag_rows.append(
                    prepayment_diagnostics(_oa_assets, _scurve, _sc.name)
                )
            _diag = pd.concat(_diag_rows, ignore_index=True)
            st.dataframe(_diag, use_container_width=True, hide_index=True)
except Exception:
    pass

if st.session_state.get("cpr_calib_df") is not None:
    with st.expander("CPR calibration detail (WAC/PMMS → S-curve)", expanded=False):
        from src.cpr_calibration import calibration_display_frame, scurve_dataframe
        _cdf = st.session_state["cpr_calib_df"]
        st.caption(
            "Refi incentive = (WAC − PMMS) × 100 bp. "
            "+ = in-the-money to refinance; − = lock-in. "
            "Points are plotted on the S-curve below."
        )
        st.dataframe(
            calibration_display_frame(_cdf),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇ Download CPR calibration CSV",
            _cdf.to_csv(index=False),
            file_name="cpr_calibration.csv",
            mime="text/csv",
            use_container_width=True,
        )
        _sc = scurve_dataframe()
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=_sc["incentive_bp"], y=_sc["cpr_pct"],
            mode="lines+markers", name="Agency-style S-curve",
            line=dict(color=NAVY, width=2.5),
        ))
        fig_sc.add_trace(go.Scatter(
            x=_cdf["incentive_bp"], y=_cdf["cpr_pct"],
            mode="markers+text", name="WAC/PMMS scenario points",
            text=_cdf["key"], textposition="top center",
            marker=dict(size=10, color=ORANGE),
            customdata=_cdf["refi_incentive"],
            hovertemplate=(
                "%{text}<br>Refi incentive: %{customdata}<br>"
                "CPR: %{y:.1f}%<extra></extra>"
            ),
        ))
        fig_sc.add_vline(x=0, line_dash="dot", line_color=BORDER)
        fig_sc.update_layout(
            **PLOTLY_BASE, height=340,
            xaxis=dict(
                **AXIS_STYLE,
                title="Refi incentive (WAC − PMMS), bp  (+ ITM / − OTM)",
            ),
            yaxis=dict(**AXIS_STYLE, title="CPR %"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor=BG2),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab0, tab_lcr, tab1, tab2, tab3, tab4, tab5, tab_kr, tab6 = st.tabs([
    "NMD Refinement",
    "LCR / NSFR",
    "All Scenarios",
    "Scenario Detail",
    "EVE Waterfall",
    "Scenario Comparison",
    "Repricing Gap",
    "ALCO / KR01 / Hedges",
    "Yield Curve",
])


# ── TAB 0: NMD refinement ─────────────────────────────────────────────────────
with tab0:
    st.markdown("<p class='section-label'>Behavioural NMD Refinement</p>",
                unsafe_allow_html=True)
    if not use_nmd:
        st.info(
            "Enable **Refine NMD deposits and feed IRRBB** in the sidebar, then upload "
            "the customer-level Excel template "
            "(customer_id, segment, start_date, deposit_rate, 36 EOM balances + market rates)."
        )
    elif nmd_result is None:
        st.warning("NMD refinement is enabled. Upload an NMD Excel/CSV file to continue.")
    else:
        st.success(
            f"Refined **{nmd_result.deposit_name}** from "
            f"**{nmd_result.customer_count} customers** × **{nmd_result.month_count} months**. "
            "Core / non-core / sticky balances are mapped into IRRBB liabilities. "
            "NII and EVE are calculated in the IRRBB tabs."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{nmd_result.customer_count}")
        c2.metric("Stable (core) %", f"{nmd_result.stable_pct * 100:.1f}%")
        c3.metric("Beta (pass-through)", f"{nmd_result.beta:.3f}")
        c4.metric("Behavioural WAL", f"{nmd_result.wal_years:.2f}Y")

        st.markdown("**By segment (BCBS 368 caps applied)**")
        if nmd_result.segment_summary is not None and not nmd_result.segment_summary.empty:
            st.dataframe(nmd_result.segment_summary, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Portfolio refinement summary**")
            st.dataframe(nmd_result.summary, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**Core deposit IRRBB buckets**")
            st.dataframe(nmd_result.irrbb_buckets, use_container_width=True)

        st.markdown("**IRRBB liability instruments created from NMD**")
        nmd_inst_rows = [{
            "Instrument": i.name,
            "Notional ($M)": i.notional,
            "Coupon (%)": i.coupon_pct,
            "Type": i.instrument_type,
            "Repricing (Y)": round(i.repricing_years, 3),
            "Eff. Duration (Y)": round(i.effective_duration, 2),
        } for i in nmd_result.instruments]
        st.dataframe(pd.DataFrame(nmd_inst_rows), use_container_width=True, hide_index=True)

        if nmd_result.liquidity_summary is not None and not nmd_result.liquidity_summary.empty:
            st.markdown("**LCR / NSFR bifurcation map (from deposit file)**")
            st.dataframe(nmd_result.liquidity_summary, use_container_width=True, hide_index=True)


# ── TAB LCR / NSFR: Liquidity ratios ──────────────────────────────────────────
with tab_lcr:
    st.markdown("<p class='section-label'>Basel III Liquidity Ratios — LCR & NSFR</p>",
                unsafe_allow_html=True)
    if liquidity.source == "workbook":
        st.info(
            "LCR / NSFR computed from **disclosure workbook inputs** "
            "(HQLA Stock, LCR Cash Outflows/Inflows, NSFR ASF/RSF). "
            "Deposit outflow rows with auto_from_nmd=Y are filled from NMD bifurcation."
        )
        if liquidity.workbook is not None:
            b = liquidity.workbook.nmd_buckets
            st.markdown("**NMD deposit buckets (auto-filled rows)**")
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            bc1.metric("Core sticky", f"${b.core_sticky_m:,.1f}M")
            bc2.metric("Rate sensitive", f"${b.rate_sensitive_m:,.1f}M")
            bc3.metric("Non-core retail", f"${b.non_core_volatile_m:,.1f}M")
            bc4.metric("Wholesale operational", f"${b.wholesale_operational_m:,.1f}M")
            bc5.metric("Wholesale non-op", f"${b.wholesale_non_op_m:,.1f}M")
    else:
        st.caption(
            "Using balance-sheet heuristics. Upload v2 template with liquidity sheets "
            "for HSBC-style disclosure-driven calculation."
        )
    st.dataframe(liquidity.summary, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("<p class='section-label'>Liquidity Coverage Ratio (LCR)</p>",
                unsafe_allow_html=True)
    st.caption(
        "LCR = HQLA Stock / Net Cash Outflows (30-day stress) ≥ 100%. "
        "Inflows capped at 75% of outflows."
    )

    lcr_color = GREEN if lcr_result.lcr_pass else RED
    lcr_status = "PASS ✓" if lcr_result.lcr_pass else "FAIL ✗"
    st.markdown(
        f"<h3 style='font-size:15px;color:{NAVY}'>"
        f"LCR = {lcr_result.lcr_pct:.1f}% &nbsp;"
        f"<span style='font-size:12px;color:{lcr_color};"
        f"background:{lcr_color}18;padding:3px 10px;"
        f"border-radius:3px;font-weight:600'>{lcr_status}</span></h3>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("HQLA Stock", f"${lcr_result.hqla_stock:,.1f}M")
    m2.metric("Gross Outflows", f"${lcr_result.total_outflows:,.1f}M")
    m3.metric("Capped Inflows", f"${lcr_result.capped_inflows:,.1f}M")
    m4.metric("Net Outflows", f"${lcr_result.net_cash_outflows:,.1f}M")
    m5.metric("Level 1 HQLA", f"${lcr_result.hqla_level1:,.1f}M")

    col_h, col_o = st.columns(2)
    with col_h:
        st.markdown("**HQLA breakdown**")
        if not lcr_result.hqla_breakdown.empty:
            st.dataframe(lcr_result.hqla_breakdown, use_container_width=True, hide_index=True)
        else:
            st.warning("No HQLA-eligible assets identified.")
    with col_o:
        st.markdown("**30-day cash outflows**")
        if not lcr_result.outflow_breakdown.empty:
            st.dataframe(lcr_result.outflow_breakdown, use_container_width=True, hide_index=True)

    if not lcr_result.inflow_breakdown.empty:
        st.markdown("**30-day cash inflows (before 75% cap)**")
        st.dataframe(lcr_result.inflow_breakdown, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("<p class='section-label'>Net Stable Funding Ratio (NSFR)</p>",
                unsafe_allow_html=True)
    st.caption(
        "NSFR = Available Stable Funding (ASF) / Required Stable Funding (RSF) ≥ 100%. "
        "ASF from capital (100%), retail deposits (95%/90%), wholesale by tenor. "
        "RSF from HQLA (0–50%), loans (65–85%), long-term assets."
    )

    nsfr_color = GREEN if nsfr_result.nsfr_pass else RED
    nsfr_status = "PASS ✓" if nsfr_result.nsfr_pass else "FAIL ✗"
    st.markdown(
        f"<h3 style='font-size:15px;color:{NAVY}'>"
        f"NSFR = {nsfr_result.nsfr_pct:.1f}% &nbsp;"
        f"<span style='font-size:12px;color:{nsfr_color};"
        f"background:{nsfr_color}18;padding:3px 10px;"
        f"border-radius:3px;font-weight:600'>{nsfr_status}</span></h3>",
        unsafe_allow_html=True,
    )

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("ASF Total", f"${nsfr_result.asf_total:,.1f}M")
    n2.metric("RSF Total", f"${nsfr_result.rsf_total:,.1f}M")
    n3.metric("ASF — Capital", f"${nsfr_result.asf_capital:,.1f}M")
    n4.metric("Funding Gap", f"${nsfr_result.asf_total - nsfr_result.rsf_total:+,.1f}M")

    if nmd_result is not None:
        st.info(
            f"NMD refinement active — LCR outflows and NSFR ASF use behavioural splits "
            f"({nmd_result.stable_pct * 100:.0f}% stable core, "
            f"{nmd_result.non_core_pct * 100:.0f}% non-core)."
        )

    col_asf, col_rsf = st.columns(2)
    with col_asf:
        st.markdown("**ASF breakdown (funding sources)**")
        st.dataframe(nsfr_result.asf_breakdown, use_container_width=True, hide_index=True)
    with col_rsf:
        st.markdown("**RSF breakdown (asset requirements)**")
        st.dataframe(nsfr_result.rsf_breakdown, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download Liquidity Ratios CSV",
        liquidity.summary.to_csv(index=False),
        file_name="liquidity_ratios_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── TAB 1: All scenarios ──────────────────────────────────────────────────────
with tab1:
    st.markdown("<p class='section-label'>BCBS 368 — Six Prescribed Scenarios</p>",
                unsafe_allow_html=True)
    st.caption(
        f"Tier 1 = ${tier1:,.0f}M · Outlier ≥ 15% (${tier1 * 0.15:,.0f}M) · "
        f"Watch ≥ 10% (${tier1 * 0.10:,.0f}M)"
    )

    card_cols = st.columns(3)
    for idx, r in enumerate(results):
        col = card_cols[idx % 3]
        sc_color = _scenario_status_color(r.status)
        short = SCENARIO_SHORT.get(r.scenario.name, r.scenario.name)
        with col:
            st.markdown(
                f"<div style='background:{BG2};border:1px solid {BORDER};"
                f"border-left:4px solid {SCENARIO_COLORS[idx]};border-radius:6px;"
                f"padding:12px 14px;margin-bottom:10px'>"
                f"<div style='font-size:12px;font-weight:600;color:{NAVY}'>"
                f"{short}</div>"
                f"<div style='font-size:10px;color:{DIM};margin:4px 0 8px'>"
                f"{r.scenario.description[:70]}{'…' if len(r.scenario.description) > 70 else ''}"
                f"</div>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:11px;font-family:monospace'>"
                f"<span style='color:{DIM}'>ΔNII</span>"
                f"<span style='color:{GREEN if r.delta_nii >= 0 else RED}'>"
                f"${r.delta_nii:+.1f}M</span></div>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:11px;font-family:monospace;margin-top:4px'>"
                f"<span style='color:{DIM}'>ΔEVE</span>"
                f"<span style='color:{GREEN if r.delta_eve >= 0 else RED}'>"
                f"${r.delta_eve:+.1f}M</span></div>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:11px;font-family:monospace;margin-top:4px'>"
                f"<span style='color:{DIM}'>|ΔEVE|/T1</span>"
                f"<span style='color:{sc_color};font-weight:600'>"
                f"{r.delta_eve_pct:.1f}%</span></div>"
                f"<div style='margin-top:8px;font-size:10px;font-weight:600;"
                f"color:{sc_color}'>{r.status}</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    col_tbl, col_chart = st.columns([1.1, 0.9])

    with col_tbl:
        st.markdown("<p class='section-label'>Summary Table</p>", unsafe_allow_html=True)
        df_summary = pd.DataFrame([{
            "Scenario": SCENARIO_SHORT.get(r.scenario.name, r.scenario.name),
            "ΔNII ($M)": r.delta_nii,
            "ΔEVE ($M)": r.delta_eve,
            "|ΔEVE|/T1 (%)": r.delta_eve_pct,
            "Status": r.status,
        } for r in results])

        def _style_status(val):
            if val == "OUTLIER":
                return f"color:{RED};font-weight:bold"
            if val == "WATCH":
                return f"color:{AMBER};font-weight:500"
            return f"color:{GREEN};font-weight:500"

        st.dataframe(
            df_summary.style.format({
                "ΔNII ($M)": "{:+.2f}",
                "ΔEVE ($M)": "{:+.2f}",
                "|ΔEVE|/T1 (%)": "{:.1f}",
            }).map(_style_status, subset=["Status"]),
            use_container_width=True, hide_index=True, height=260,
        )
        st.download_button(
            "⬇ Download CSV", df_summary.to_csv(index=False),
            file_name="irrbb_summary.csv", mime="text/csv",
            use_container_width=True,
        )

    with col_chart:
        st.markdown("<p class='section-label'>|ΔEVE| / Tier 1</p>", unsafe_allow_html=True)
        names = [SCENARIO_SHORT.get(r.scenario.name, r.scenario.name) for r in results]
        pcts = [r.delta_eve_pct for r in results]
        bar_colors = [
            RED if r.is_outlier else AMBER if r.is_watch else SCENARIO_COLORS[i]
            for i, r in enumerate(results)
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names, y=pcts, marker_color=bar_colors,
            marker_line_color=BORDER, marker_line_width=1,
            text=[f"{v:.1f}%" for v in pcts], textposition="outside",
            textfont=dict(size=10, color=TEXT),
        ))
        fig.add_hline(y=15, line_color=RED, line_dash="dash", line_width=1.5,
                      annotation_text="15% outlier", annotation_font=dict(color=RED, size=9))
        fig.add_hline(y=10, line_color=AMBER, line_dash="dot", line_width=1,
                      annotation_text="10% watch", annotation_font=dict(color=AMBER, size=9))
        fig.update_layout(
            **PLOTLY_BASE, height=300, showlegend=False,
            yaxis=dict(**AXIS_STYLE, ticksuffix="%", title="|ΔEVE| / T1 (%)"),
            xaxis=dict(**AXIS_STYLE),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 2: Scenario detail ────────────────────────────────────────────────────
with tab2:
    r = active
    status_color = RED if r.is_outlier else AMBER if r.is_watch else GREEN
    sc_color_idx = next(i for i, s in enumerate(SCENARIOS) if s.id == r.scenario.id)
    sc_color = SCENARIO_COLORS[sc_color_idx]

    st.markdown(
        f"<h3 style='font-size:15px;color:{NAVY}'>"
        f"{SCENARIO_SHORT.get(r.scenario.name, r.scenario.name)} — {r.scenario.name} &nbsp;"
        f"<span style='font-size:12px;color:{status_color};"
        f"background:{status_color}18;padding:3px 10px;"
        f"border-radius:3px;font-weight:600'>{r.status}</span></h3>"
        f"<p style='color:{DIM};font-size:11px'>{r.scenario.description}</p>",
        unsafe_allow_html=True,
    )

    if r.is_outlier:
        st.error(
            f"⚠ SUPERVISORY OUTLIER — |ΔEVE| = ${abs(r.delta_eve):.1f}M "
            f"exceeds 15% of Tier 1 (${tier1 * 0.15:.0f}M). "
            f"BCBS 368 §99: supervisor notification required."
        )
    elif r.is_watch:
        st.warning(
            f"|ΔEVE| = {r.delta_eve_pct:.1f}% — approaching the 15% outlier threshold."
        )
    else:
        st.success(
            f"✓ PASS — |ΔEVE| = {r.delta_eve_pct:.1f}% within the 15% threshold."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Δ NII (1Y)",  f"${r.delta_nii:+.2f}M")
    m2.metric("Δ EVE",       f"${r.delta_eve:+.2f}M")
    m3.metric("|ΔEVE| / T1", f"{r.delta_eve_pct:.1f}%")
    m4.metric("Asset EVE Δ", f"${r.eve_asset:+.2f}M")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<p class='section-label'>Rate Shock Profile</p>",
                    unsafe_allow_html=True)
        hex_c = sc_color.lstrip("#")
        r_int, g_int, b_int = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(N_BUCKETS)), y=selected_scenario.shocks_bp,
            mode="lines+markers",
            line=dict(color=sc_color, width=2.5),
            marker=dict(size=5, color=sc_color),
            fill="tozeroy",
            fillcolor=f"rgba({r_int},{g_int},{b_int},0.08)",
        ))
        fig2.add_hline(y=0, line_color=BORDER, line_width=1)
        fig2.update_layout(
            **PLOTLY_BASE, height=270, showlegend=False,
            xaxis=dict(
                **AXIS_STYLE,
                tickvals=list(range(0, N_BUCKETS, 2)),
                ticktext=[BUCKET_LABELS[i] for i in range(0, N_BUCKETS, 2)],
                tickangle=-35,
            ),
            yaxis=dict(**AXIS_STYLE, title="Shock (bp)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown("<p class='section-label'>NII Decomposition</p>",
                    unsafe_allow_html=True)
        fig3 = go.Figure(data=[
            go.Bar(
                name="Asset repricing",
                x=["Assets"], y=[r.nii_asset],
                marker_color=BLUE, marker_line_color=BORDER, marker_line_width=1,
                text=f"${r.nii_asset:+.1f}M", textposition="outside",
            ),
            go.Bar(
                name="Liability repricing",
                x=["Liabilities"], y=[r.nii_liability],
                marker_color=ORANGE, marker_line_color=BORDER, marker_line_width=1,
                text=f"${r.nii_liability:+.1f}M", textposition="outside",
            ),
            go.Bar(
                name="Net ΔNII",
                x=["Net"], y=[r.delta_nii],
                marker_color=GREEN if r.delta_nii >= 0 else RED,
                marker_line_color=BORDER, marker_line_width=1,
                text=f"${r.delta_nii:+.1f}M", textposition="outside",
            ),
        ])
        fig3.add_hline(y=0, line_color=BORDER, line_width=1)
        fig3.update_layout(
            **PLOTLY_BASE, height=270, barmode="group",
            yaxis=dict(**AXIS_STYLE, title="Δ NII (USD M)"),
            xaxis=dict(**AXIS_STYLE),
            legend=dict(font=dict(size=10), bgcolor=BG2,
                        bordercolor=BORDER, borderwidth=1),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── TAB 3: EVE Waterfall ──────────────────────────────────────────────────────
with tab3:
    st.markdown(
        f"<p class='section-label'>"
        f"Instrument-level ΔEVE Attribution — {selected_name}</p>",
        unsafe_allow_html=True,
    )

    df = detail.sort_values("delta_eve")
    bar_colors = [GREEN if v >= 0 else RED for v in df["delta_eve"]]
    labels = [
        f"{row['side'][:1]} · {row['instrument']}"
        for _, row in df.iterrows()
    ]

    fig4 = go.Figure(go.Bar(
        x=df["delta_eve"], y=labels,
        orientation="h",
        marker_color=bar_colors,
        marker_line_color=BORDER, marker_line_width=0.5,
        text=[f"${v:+.1f}M" for v in df["delta_eve"]],
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
    ))
    fig4.add_vline(x=0, line_color=BORDER, line_width=1)
    fig4.update_layout(
        **PLOTLY_BASE,
        height=max(440, len(df) * 32 + 80),
        title=dict(
            text=f"Total ΔEVE: ${df['delta_eve'].sum():+.1f}M  ·  {selected_name}",
            font=dict(color=DIM, size=11),
        ),
        xaxis=dict(**AXIS_STYLE, title="Δ EVE contribution (USD M)"),
        yaxis=dict(**AXIS_STYLE),
        showlegend=False,
    )
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("▼ Full attribution table"):
        st.dataframe(
            detail[["side", "instrument", "notional", "type",
                    "eff_duration", "pv_base", "pv_shocked", "delta_eve"]]
            .style.format({
                "notional":     "${:.0f}M",
                "eff_duration": "{:.2f}Y",
                "pv_base":      "${:.1f}M",
                "pv_shocked":   "${:.1f}M",
                "delta_eve":    "${:+.2f}M",
            }),
            use_container_width=True, hide_index=True,
        )

    st.download_button(
        "⬇ Download Attribution CSV", detail.to_csv(index=False),
        file_name=f"eve_attribution_{selected_scenario.id}.csv",
        mime="text/csv", use_container_width=True,
    )


# ── TAB 4: Scenario comparison ────────────────────────────────────────────────
with tab4:
    st.markdown(
        "<p class='section-label'>ΔEVE Attribution — Non-Deposit Instruments</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Retail / NMD deposit buckets excluded — focus on loans, bonds, and wholesale funding. "
        "Top 12 instruments by average |ΔEVE| across scenarios."
    )

    comp_rows = []
    for s_obj, color in zip(SCENARIOS, SCENARIO_COLORS):
        d = _filter_non_deposit_eve(calc.instrument_eve_detail(s_obj))
        for _, row in d.iterrows():
            comp_rows.append({
                "Scenario": SCENARIO_SHORT.get(s_obj.name, s_obj.name),
                "Instrument": row["instrument"],
                "Side": row["side"].title(),
                "ΔEVE ($M)": row["delta_eve"],
            })
    comp_df = pd.DataFrame(comp_rows)
    if comp_df.empty:
        st.info("No non-deposit instruments to display.")
    else:
        avg_impact = (
            comp_df.groupby("Instrument")["ΔEVE ($M)"]
            .apply(lambda s: float(s.abs().mean()))
            .sort_values(ascending=False)
        )
        top_inst = avg_impact.head(12).index.tolist()
        comp_df = comp_df[comp_df["Instrument"].isin(top_inst)]

        heat = comp_df.pivot_table(
            index="Instrument", columns="Scenario", values="ΔEVE ($M)", aggfunc="sum",
        ).reindex(top_inst)
        heat.index = [f"{i[:42]}{'…' if len(i) > 42 else ''}" for i in heat.index]

        fig5 = go.Figure(data=go.Heatmap(
            z=heat.values,
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            colorscale=[
                [0.0, RED], [0.5, BG2], [1.0, GREEN],
            ],
            zmid=0,
            text=[[f"${v:+.1f}" for v in row] for row in heat.values],
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="Instrument: %{y}<br>Scenario: %{x}<br>ΔEVE: %{z:+.2f}M<extra></extra>",
        ))
        fig5.update_layout(
            **PLOTLY_BASE,
            height=max(380, len(top_inst) * 34 + 100),
            title=dict(
                text="ΔEVE heatmap — loans, securities & wholesale funding only",
                font=dict(color=DIM, size=11),
            ),
            xaxis=dict(**AXIS_STYLE, side="top"),
            yaxis=dict(**AXIS_STYLE, autorange="reversed"),
        )
        st.plotly_chart(fig5, use_container_width=True)

        with st.expander("▼ Numeric comparison table"):
            st.dataframe(
                comp_df.pivot_table(
                    index=["Side", "Instrument"], columns="Scenario",
                    values="ΔEVE ($M)", aggfunc="sum",
                ).style.format("{:+.2f}"),
                use_container_width=True,
            )


# ── TAB 5: Repricing Gap ──────────────────────────────────────────────────────
with tab5:
    st.markdown("<p class='section-label'>Repricing Gap — DV01 by BCBS Bucket</p>",
                unsafe_allow_html=True)
    st.caption(
        "Maturing / repricing principal only (coupons excluded). "
        "Asset DV01 shown above zero, liability DV01 below zero (USD K per 1bp). "
        "Net DV01 line: positive = asset-heavy bucket, negative = liability-heavy bucket."
    )

    dv01_k = dv01_gap.assign(
        asset_dv01_k=dv01_gap["asset_dv01"] * 1000,
        liability_dv01_k=dv01_gap["liability_dv01"] * 1000,
        net_dv01_k=dv01_gap["net_dv01"] * 1000,
    )
    dv01_r = dv01_k.reset_index()
    x = list(range(len(dv01_r)))
    tick_step = 2
    tick_vals = list(range(0, N_BUCKETS, tick_step))
    tick_text = [BUCKET_LABELS[i] for i in tick_vals]

    total_asset_k = dv01_k["asset_dv01_k"].sum()
    total_liab_k = dv01_k["liability_dv01_k"].sum()
    total_net_k = dv01_k["net_dv01_k"].sum()
    m_dv1, m_dv2, m_dv3 = st.columns(3)
    m_dv1.metric("Total asset DV01", f"${total_asset_k:,.0f}K/bp")
    m_dv2.metric("Total liability DV01", f"${total_liab_k:,.0f}K/bp")
    m_dv3.metric("Total net DV01", f"${total_net_k:+,.0f}K/bp")

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=x,
        y=dv01_r["asset_dv01_k"],
        name="Asset DV01",
        marker_color=BLUE,
        opacity=0.85,
        marker_line_color=BORDER,
        marker_line_width=0.5,
        hovertemplate="Bucket: %{customdata}<br>Asset DV01: %{y:,.1f} $K/bp<extra></extra>",
        customdata=dv01_r["bucket"],
    ))
    fig6.add_trace(go.Bar(
        x=x,
        y=-dv01_r["liability_dv01_k"],
        name="Liability DV01",
        marker_color=ORANGE,
        opacity=0.85,
        marker_line_color=BORDER,
        marker_line_width=0.5,
        hovertemplate="Bucket: %{customdata}<br>Liability DV01: %{y:,.1f} $K/bp<extra></extra>",
        customdata=dv01_r["bucket"],
    ))
    fig6.add_trace(go.Scatter(
        x=x,
        y=dv01_r["net_dv01_k"],
        name="Net DV01",
        mode="lines+markers",
        line=dict(color=NAVY, width=2.5),
        marker=dict(size=6, color=NAVY, line=dict(color=BG2, width=1)),
        hovertemplate="Bucket: %{customdata}<br>Net DV01: %{y:+,.1f} $K/bp<extra></extra>",
        customdata=dv01_r["bucket"],
    ))
    fig6.add_hline(y=0, line_color=BORDER, line_width=1)
    fig6.update_layout(
        **PLOTLY_BASE,
        height=460,
        barmode="relative",
        xaxis=dict(
            **AXIS_STYLE,
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-40,
            title="BCBS 368 time bucket",
        ),
        yaxis=dict(**AXIS_STYLE, title="DV01 ($K per 1bp)", tickformat=",.0f"),
        legend=dict(
            font=dict(size=10), bgcolor=BG2,
            bordercolor=BORDER, borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02,
        ),
    )
    st.plotly_chart(fig6, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**DV01 summary by bucket ($K/bp)**")
        st.dataframe(
            dv01_k[["asset_dv01_k", "liability_dv01_k", "net_dv01_k"]]
            .rename(columns={
                "asset_dv01_k": "Asset ($K/bp)",
                "liability_dv01_k": "Liability ($K/bp)",
                "net_dv01_k": "Net ($K/bp)",
            })
            .style.format({
                "Asset ($K/bp)": "{:,.1f}",
                "Liability ($K/bp)": "{:,.1f}",
                "Net ($K/bp)": "{:+,.1f}",
            })
            .background_gradient(subset=["Net ($K/bp)"], cmap="RdYlGn"),
            use_container_width=True,
            height=360,
        )
    with col_t2:
        st.markdown("**Maturity notional gap ($M) — aligns with DV01**")
        st.caption("Principal + repricing cash flows only (no coupon double-count).")
        st.dataframe(
            maturity_gap.style.format({
                "assets": "{:.1f}",
                "liabilities": "{:.1f}",
                "net_gap": "{:+.1f}",
            }),
            use_container_width=True,
            height=360,
        )

    with st.expander("Why can notional gap and DV01 gap disagree? (e.g. 6M–9M bucket)"):
        st.markdown(
            f"""
The **legacy BCBS notional gap** (left reference table if shown separately) counts the
**full face value** of any instrument that has **any** cash flow in a bucket — including
**coupon-only** payments. Long-dated fixed bonds and mortgages therefore inflate asset
notional in intermediate buckets even when almost no principal matures there.

**DV01** and the **maturity notional gap** only use **principal and repricing** flows,
weighted by tenor (and discount factor for DV01).

**Example in your book (6M–9M):**
- Assets: gov bonds and mortgages appear in the notional gap because they **pay coupons**
  in this window, but contribute only **~\\$36M** of actual principal vs **\\$2,050M**
  counted under the legacy rule.
- Liabilities: **\\$300M** floating senior debt **reprices entirely** in this bucket,
  plus term deposits with **\\$104M** maturing — concentrated rate-sensitive outflows →
  **higher liability DV01** despite a smaller legacy notional total.

**Rule of thumb:** trust **DV01 / maturity notional** for hedging; use the legacy gap
only as a BCBS slotting diagnostic.
            """
        )

    col_ref1, col_ref2 = st.columns(2)
    with col_ref1:
        st.markdown("**Legacy BCBS notional gap ($M)**")
        st.caption("Full instrument notional if any cash flow hits bucket.")
        st.dataframe(
            gap.style.format({
                "assets": "{:.1f}",
                "liabilities": "{:.1f}",
                "net_gap": "{:+.1f}",
            }),
            use_container_width=True,
            height=280,
        )
    with col_ref2:
        st.markdown("**6M–9M bucket comparison**")
        _b6 = "6M – 9M"
        if _b6 in gap.index:
            st.table(pd.DataFrame([
                {"Measure": "Legacy net gap ($M)", "Value": f"{gap.loc[_b6, 'net_gap']:+.1f}"},
                {"Measure": "Maturity net gap ($M)", "Value": f"{maturity_gap.loc[_b6, 'net_gap']:+.1f}"},
                {"Measure": "Net DV01 ($K/bp)", "Value": f"{dv01_k.loc[_b6, 'net_dv01_k']:+,.1f}"},
            ]))
        else:
            st.caption("Bucket label not found in current gap table.")

    net_dv01_k = dv01_k["net_dv01_k"]
    asset_heavy = net_dv01_k[net_dv01_k > 0].index.tolist()
    liab_heavy = net_dv01_k[net_dv01_k < 0].index.tolist()
    if asset_heavy:
        st.success(
            f"Asset-heavy buckets (net DV01 > 0): {', '.join(asset_heavy[:5])}"
            f"{'…' if len(asset_heavy) > 5 else ''}"
        )
    if liab_heavy:
        st.warning(
            f"Liability-heavy buckets (net DV01 < 0): {', '.join(liab_heavy[:5])}"
            f"{'…' if len(liab_heavy) > 5 else ''}"
        )

    st.divider()
    st.markdown("<p class='section-label'>IRS hedging strategies to close the DV01 gap</p>",
                unsafe_allow_html=True)
    hedge_ratio = st.slider(
        "Target hedge ratio (%)",
        min_value=50, max_value=100, value=80, step=5,
        help="Share of bucket |net DV01| to neutralise with swaps.",
    ) / 100.0
    hedge_df = suggest_irs_hedges(dv01_gap, hedge_ratio=hedge_ratio)

    st.markdown(
        f"<div style='font-size:12px;color:{DIM};line-height:1.7;margin-bottom:12px'>"
        f"<strong>How to read:</strong> A <em>receiver</em> swap (receive-fixed / pay-floating) "
        f"adds asset-like DV01 in the bucket — use when liabilities dominate (negative net). "
        f"A <em>payer</em> swap (pay-fixed / receive-floating) offsets asset-heavy buckets. "
        f"Match swap tenor to the BCBS bucket midpoint. "
        f"Indicative notional uses DV01 ≈ N × tenor × 0.01bp.</div>",
        unsafe_allow_html=True,
    )

    if hedge_df.empty:
        st.info("No material DV01 gaps — book is approximately balanced by bucket.")
    else:
        st.dataframe(hedge_df, use_container_width=True, hide_index=True)

        st.markdown("**Portfolio-level playbook**")
        if total_net_k > 0:
            st.markdown(
                f"- **Overall asset-heavy** (net **${total_net_k:,.0f}K/bp**): "
                "layer **pay-fixed** swaps at 2Y–5Y to reduce rate-rise sensitivity; "
                "consider **receive-floating** on short-end to retain NII if cuts are expected."
            )
        elif total_net_k < 0:
            st.markdown(
                f"- **Overall liability-heavy** (net **${total_net_k:,.0f}K/bp**): "
                "add **receive-fixed** swaps at buckets with largest negative net DV01 "
                "(deposits / short funding); match tenor to behavioural repricing WAL."
            )
        else:
            st.markdown("- **Near flat** at portfolio level — focus on bucket-level mismatches above.")

        st.markdown(
            "- **Steepener / flattener risk**: if short-end net ≠ long-end net, use a "
            "**swap ladder** (O/N–1Y payer + 5Y–10Y receiver) rather than a single bullet.\n"
            "- **Basis & credit**: index selection (SOFR vs Fed Funds) and CSA terms affect "
            "effective hedge ratio — recalibrate after execution.\n"
            "- **IRRBB interaction**: re-run EVE/NII scenarios after adding swap notionals "
            "to confirm outlier thresholds remain inside policy."
        )

        st.download_button(
            "⬇ Download IRS hedge suggestions CSV",
            hedge_df.to_csv(index=False),
            file_name="irs_hedge_suggestions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    export_gap = dv01_k.join(gap, rsuffix="_notional_m")
    st.download_button(
        "⬇ Download DV01 Gap CSV", export_gap.to_csv(),
        file_name="repricing_gap_dv01.csv", mime="text/csv",
        use_container_width=True,
    )


# ── TAB: ALCO / Board / Treasury (KR01 ↔ EVE bridge) ─────────────────────────
with tab_kr:
    st.markdown(
        "<p class='section-label'>Risk Layers — Board · ALCO · Treasury</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Layer 1 Repricing Gap (structure) · Layer 2 EVE/NII vs limits · "
        "Layer 3 KR01 for trade execution. Bridge: "
        "ΔEVE from key i ≈ −KR01ᵢ × shockᵢ(bp) / 1000 ($M)."
    )

    kr_hedge_ratio = st.slider(
        "Target KR01 hedge ratio (%)",
        min_value=50, max_value=100, value=80, step=5,
        key="kr_hedge_ratio",
        help="Used for full multi-tenor hedge package B and spot IRS suggestions.",
    ) / 100.0

    with st.spinner("Building KR01 attribution & scenario hedges…"):
        actual_eve = {r.scenario.id: r.delta_eve for r in results}
        pack = build_treasury_alco_pack(
            assets,
            liabilities,
            curve,
            SCENARIOS,
            tier1_m=float(tier1),
            actual_delta_eve=actual_eve,
            cpr_for_scenario=calc._cpr_override_for,
            hedge_ratio=kr_hedge_ratio,
            breach_pct=15.0,
            amber_pct=10.0,
        )

    kr01_df = pack["kr01"]
    parallel_k = float(pack["parallel_dv01_k"])
    sum_kr = float(kr01_df["net_kr01_k"].sum())
    layer = st.radio(
        "Dashboard layer",
        ["Board / ALCO limits", "Treasury KR01 & attribution", "Hedge playbook"],
        horizontal=True,
        key="kr_layer",
    )

    # ── Board / ALCO ──────────────────────────────────────────────────────────
    if layer == "Board / ALCO limits":
        st.markdown(
            "<p class='section-label'>Board / ALCO — EVE limit dashboard</p>",
            unsafe_allow_html=True,
        )
        lim = pack["limits"]
        n_breach = int((lim["Status"] == "BREACH").sum())
        n_amber = int((lim["Status"] == "AMBER").sum())
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Tier 1", f"${tier1:,.0f}M")
        b2.metric("Breach (≥15%)", str(n_breach), delta_color="inverse")
        b3.metric("Amber (≥10%)", str(n_amber))
        b4.metric("Net KR01 (base)", f"${sum_kr:+,.0f}K/bp")

        def _lim_style(row):
            if row["Status"] == "BREACH":
                return ["background-color:#fde8e8"] * len(row)
            if row["Status"] == "AMBER":
                return ["background-color:#fff4e5"] * len(row)
            return [""] * len(row)

        st.dataframe(
            lim.style.apply(_lim_style, axis=1).format({
                "ΔEVE ($M)": "{:+.2f}",
                "% Tier 1": "{:+.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Why this matters for ALCO**")
        for note in pack["rationale"]:
            st.markdown(f"- {note}")

        st.markdown("**Hedge package comparison (% of Tier 1)** — linear KR01 bridge + convexity")
        st.dataframe(
            pack["packages_pct"].style.format("{:+.1f}%"),
            use_container_width=True,
        )
        st.dataframe(pack["packages_detail"], use_container_width=True, hide_index=True)
        st.caption(
            "Package A sizes a single pay-fixed at the Parallel-Up loss driver. "
            "B ladders all keys. C is a partial A. Hedged rows ≈ predicted+convexity."
        )

        fig_lim = go.Figure()
        colors_l = [
            RED if s == "BREACH" else AMBER if s == "AMBER" else GREEN
            for s in lim["Status"]
        ]
        fig_lim.add_trace(go.Bar(
            x=lim["Scenario"], y=lim["% Tier 1"],
            marker_color=colors_l, name="% Tier 1",
            hovertemplate="%{x}<br>%{y:+.1f}% Tier 1<extra></extra>",
        ))
        fig_lim.add_hline(y=-15, line_dash="dash", line_color=RED,
                          annotation_text="15% breach", annotation_font=dict(color=RED, size=9))
        fig_lim.add_hline(y=-10, line_dash="dot", line_color=AMBER,
                          annotation_text="10% amber", annotation_font=dict(color=AMBER, size=9))
        fig_lim.update_layout(
            **PLOTLY_BASE, height=360, showlegend=False,
            yaxis=dict(**AXIS_STYLE, title="ΔEVE / Tier 1 (%)"),
            xaxis=dict(**AXIS_STYLE, tickangle=-25),
        )
        st.plotly_chart(fig_lim, use_container_width=True)

    # ── Treasury KR01 ─────────────────────────────────────────────────────────
    elif layer == "Treasury KR01 & attribution":
        st.markdown(
            "<p class='section-label'>Treasury — monthly KR01 & EVE attribution</p>",
            unsafe_allow_html=True,
        )
        m_k1, m_k2, m_k3, m_k4 = st.columns(4)
        m_k1.metric("Asset KR01 (sum)", f"${kr01_df['asset_kr01_k'].sum():,.0f}K/bp")
        m_k2.metric("Liability KR01 (sum)", f"${kr01_df['liability_kr01_k'].sum():,.0f}K/bp")
        m_k3.metric("Net KR01 (sum)", f"${sum_kr:+,.0f}K/bp")
        m_k4.metric(
            "Parallel DV01",
            f"${parallel_k:+,.0f}K/bp",
            delta=f"Δ vs ΣKR01 {sum_kr - parallel_k:+,.1f}K",
            delta_color="off",
        )

        x_kr = list(kr01_df["label"])
        fig_kr = go.Figure()
        fig_kr.add_trace(go.Bar(
            x=x_kr, y=kr01_df["asset_kr01_k"], name="Asset KR01",
            marker_color=BLUE, opacity=0.85,
            hovertemplate="Key: %{x}<br>Asset: %{y:,.1f} $K/bp<extra></extra>",
        ))
        fig_kr.add_trace(go.Bar(
            x=x_kr, y=-kr01_df["liability_kr01_k"], name="Liability KR01",
            marker_color=ORANGE, opacity=0.85,
            hovertemplate="Key: %{x}<br>Liability: %{y:,.1f} $K/bp<extra></extra>",
        ))
        fig_kr.add_trace(go.Scatter(
            x=x_kr, y=kr01_df["net_kr01_k"], name="Net KR01",
            mode="lines+markers",
            line=dict(color=NAVY, width=2.5),
            marker=dict(size=8, color=NAVY),
            hovertemplate="Key: %{x}<br>Net: %{y:+,.1f} $K/bp<extra></extra>",
        ))
        fig_kr.add_hline(y=0, line_color=BORDER, line_width=1)
        fig_kr.update_layout(
            **PLOTLY_BASE, height=420, barmode="relative",
            xaxis=dict(**AXIS_STYLE, title="Tradeable key"),
            yaxis=dict(**AXIS_STYLE, title="KR01 ($K/bp)", tickformat=",.0f"),
            legend=dict(
                font=dict(size=10), bgcolor=BG2, bordercolor=BORDER, borderwidth=1,
                orientation="h", yanchor="bottom", y=1.02,
            ),
        )
        st.plotly_chart(fig_kr, use_container_width=True)

        st.markdown("**Step 1 — Base KR01 report ($K/bp)**")
        st.dataframe(
            kr01_df.set_index("label")[
                ["asset_kr01_k", "liability_kr01_k", "net_kr01_k"]
            ].rename(columns={
                "asset_kr01_k": "Asset",
                "liability_kr01_k": "Liability",
                "net_kr01_k": "Net KR01",
            }).style.format({
                "Asset": "{:,.1f}", "Liability": "{:,.1f}", "Net KR01": "{:+,.1f}",
            }),
            use_container_width=True,
        )

        st.markdown("**Step 2 — Shock at each key (bp)**")
        st.dataframe(
            pack["shock_bp"].style.format("{:+.0f}"),
            use_container_width=True,
        )

        st.markdown(
            "**Step 3 — Attribution matrix ($M)**  ·  "
            "cell = −KR01_base × shock_bp / 1000"
        )
        st.dataframe(
            pack["attribution"].style.format("{:+.2f}"),
            use_container_width=True,
        )
        st.caption(
            "Predicted key cells use **Base KR01 only** (BASE S-curve CPR) × rate shocks. "
            "Change WAC/PMMS in the sidebar to move **Actual ΔEVE**, **Convexity**, "
            "and Dynamic KR01 below."
        )

        st.markdown("**Dynamic KR01 under each BCBS shock ($K/bp net)**")
        st.caption(
            "Recomputed on the shocked curve with that scenario’s S-curve CPR "
            "(from sidebar WAC + PMMS)."
        )
        st.dataframe(
            pack["scenario_kr01"].style.format("{:+,.1f}"),
            use_container_width=True,
        )
        fig_sk = go.Figure()
        sk = pack["scenario_kr01"]
        for col in sk.columns:
            fig_sk.add_trace(go.Scatter(
                x=list(sk.index), y=sk[col], mode="lines+markers", name=col,
                line=dict(width=2 if col == "Base" else 1.5,
                          dash="solid" if col == "Base" else "dot"),
            ))
        fig_sk.add_hline(y=0, line_color=BORDER, line_width=1)
        fig_sk.update_layout(
            **PLOTLY_BASE, height=400,
            xaxis=dict(**AXIS_STYLE, title="Key"),
            yaxis=dict(**AXIS_STYLE, title="Net KR01 ($K/bp)"),
            legend=dict(
                font=dict(size=9), bgcolor=BG2, orientation="h",
                yanchor="bottom", y=1.02,
            ),
        )
        st.plotly_chart(fig_sk, use_container_width=True)

        with st.expander("▼ Instrument KR01 attribution"):
            attr = instrument_kr01_attribution(
                list(assets) + list(liabilities),
                curve,
                cpr_override=calc._cpr_override_for("BASE"),
            )
            st.dataframe(attr, use_container_width=True, height=360)

    # ── Hedge playbook ────────────────────────────────────────────────────────
    else:
        st.markdown(
            "<p class='section-label'>Treasury — IRS hedge playbook</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Proposed 2Y / 5Y / 10Y notionals close the target hedge ratio of net KR01 "
            "(positive = pay-fixed). Relief and post-swap EVE both use these sizes."
        )
        _ln = pack.get("ladder_notionals") or {}
        n2_def = float(_ln.get(2.0, 0.0))
        n5_def = float(_ln.get(5.0, 0.0))
        n10_def = float(_ln.get(10.0, 0.0))
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            n2 = st.number_input(
                "2Y signed notional ($M)",
                value=round(n2_def, 1), step=5.0, key="hedge_n_2y",
                help="Positive = pay-fixed; negative = receive-fixed.",
            )
        with hc2:
            n5 = st.number_input(
                "5Y signed notional ($M)",
                value=round(n5_def, 1), step=5.0, key="hedge_n_5y",
            )
        with hc3:
            n10 = st.number_input(
                "10Y signed notional ($M)",
                value=round(n10_def, 1), step=5.0, key="hedge_n_10y",
            )
        from src.key_rate_duration import hedge_efficiency_table, post_swap_eve_impact
        custom_notionals = {2.0: float(n2), 5.0: float(n5), 10.0: float(n10)}
        relief_df = hedge_efficiency_table(
            SCENARIOS, notionals=custom_notionals,
        )
        post = post_swap_eve_impact(
            kr01_df,
            SCENARIOS,
            float(tier1),
            {r.scenario.id: r.delta_eve for r in results},
            custom_notionals,
        )

        st.markdown(
            "**EVE relief from proposed 2Y / 5Y / 10Y swaps**  ·  "
            "cell = −swap_KR01 × shock_bp / 1000 (linked to notionals above)"
        )
        _relief_fmt = {
            c: "{:+.2f}" for c in relief_df.columns
            if c not in ("Instrument", "Notional ($M)", "KR01 ($K/bp)")
        }
        _relief_fmt["Notional ($M)"] = lambda v: "" if v == "" else f"{float(v):+.1f}"
        _relief_fmt["KR01 ($K/bp)"] = lambda v: "" if v == "" else f"{float(v):+,.1f}"
        st.dataframe(
            relief_df.style.format(_relief_fmt),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Ladder total = sum of relief across the three tickets under each scenario. "
            "Same notionals feed the before/after Tier 1 comparison below."
        )

        st.markdown("**Proposed swaps executed**")
        st.dataframe(post["notionals"], use_container_width=True, hide_index=True)

        st.markdown("**Total ΔEVE and % of Tier 1 — before vs after swaps**")
        summ = post["summary"]
        st.dataframe(
            summ.style.format({
                "ΔEVE before ($M)": "{:+.2f}",
                "ΔEVE after ($M)": "{:+.2f}",
                "ΔEVE change ($M)": "{:+.2f}",
                "% Tier 1 before": "{:+.1f}",
                "% Tier 1 after": "{:+.1f}",
                "pp change": "{:+.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig_t1 = go.Figure()
        fig_t1.add_trace(go.Bar(
            name="% Tier 1 before",
            x=summ["Scenario"], y=summ["% Tier 1 before"],
            marker_color=ORANGE, opacity=0.85,
        ))
        fig_t1.add_trace(go.Bar(
            name="% Tier 1 after",
            x=summ["Scenario"], y=summ["% Tier 1 after"],
            marker_color=BLUE, opacity=0.85,
        ))
        fig_t1.add_hline(
            y=-15, line_dash="dash", line_color=RED,
            annotation_text="15% breach", annotation_font=dict(color=RED, size=9),
        )
        fig_t1.add_hline(
            y=-10, line_dash="dot", line_color=AMBER,
            annotation_text="10% amber", annotation_font=dict(color=AMBER, size=9),
        )
        fig_t1.update_layout(
            **PLOTLY_BASE, height=380, barmode="group",
            yaxis=dict(**AXIS_STYLE, title="ΔEVE / Tier 1 (%)"),
            xaxis=dict(**AXIS_STYLE, tickangle=-25),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor=BG2),
        )
        st.plotly_chart(fig_t1, use_container_width=True)

        # Sanity: ladder-total Parallel-Up relief ≈ −ΔEVE change on Parallel Up (predicted)
        _par = next((s.name for s in SCENARIOS if s.id == "PS_UP"), None)
        if _par and "Ladder total" in set(relief_df["Instrument"]):
            _rel = float(relief_df.loc[relief_df["Instrument"] == "Ladder total", _par].iloc[0])
            _chg = float(summ.loc[summ["Scenario"] == _par, "ΔEVE change ($M)"].iloc[0])
            st.caption(
                f"Check (Parallel Up): ladder relief {_rel:+.2f} $M vs "
                f"ΔEVE change {_chg:+.2f} $M (differences = convexity / other keys)."
            )

        st.download_button(
            "⬇ Download post-swap EVE impact CSV",
            summ.to_csv(index=False),
            file_name="post_swap_eve_impact.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("**Why these IRS?**")
        for note in pack["rationale"]:
            st.markdown(f"- {note}")

        st.markdown("**Spot KR01 hedge tickets (existing book)**")
        kr_hedge_df = pack["hedges"]
        if kr_hedge_df.empty:
            st.info("No material key-rate gaps on the swap grid.")
        else:
            st.dataframe(kr_hedge_df, use_container_width=True, hide_index=True)
            desig = designate_key_rate_hedges(kr_hedge_df, assets, liabilities)
            if not desig.empty:
                st.markdown("**Indicative hedge-accounting tags** *(not advice)*")
                st.dataframe(desig, use_container_width=True, hide_index=True)

        st.markdown("**Package grid (% Tier 1)**")
        st.dataframe(
            pack["packages_pct"].style.format("{:+.1f}%"),
            use_container_width=True,
        )
        st.dataframe(pack["packages_detail"], use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download KR01 gap CSV",
            kr01_df.to_csv(index=False),
            file_name="key_rate_duration_gap.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download attribution matrix CSV",
            pack["attribution"].to_csv(),
            file_name="kr01_eve_attribution.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if not kr_hedge_df.empty:
            st.download_button(
                "⬇ Download hedge suggestions CSV",
                kr_hedge_df.to_csv(index=False),
                file_name="key_rate_hedge_suggestions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── TAB 6: Yield Curve ────────────────────────────────────────────────────────
with tab6:
    st.markdown("<p class='section-label'>Yield Curve — Base & Shocked</p>",
                unsafe_allow_html=True)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=list(range(N_BUCKETS)), y=curve.base_rates * 100,
        mode="lines+markers", name="Base curve",
        line=dict(color=NAVY, width=3),
        marker=dict(size=6, color=NAVY),
    ))
    for s_obj, color in zip(SCENARIOS, SCENARIO_COLORS):
        shocked = curve.shocked_rates(s_obj.shocks_bp) * 100
        fig7.add_trace(go.Scatter(
            x=list(range(N_BUCKETS)), y=shocked,
            mode="lines", name=s_obj.name,
            line=dict(color=color, width=1.5, dash="dash"),
            opacity=0.8,
        ))
    fig7.update_layout(
        **PLOTLY_BASE, height=440,
        xaxis=dict(
            **AXIS_STYLE,
            tickvals=list(range(0, N_BUCKETS, 2)),
            ticktext=[BUCKET_LABELS[i] for i in range(0, N_BUCKETS, 2)],
            tickangle=-35,
        ),
        yaxis=dict(
            **AXIS_STYLE,
            tickformat=".1f", ticksuffix="%",
            title="Rate (%)",
        ),
        legend=dict(
            font=dict(size=10), bgcolor=BG2,
            bordercolor=BORDER, borderwidth=1,
            orientation="h", yanchor="bottom", y=1.01,
        ),
    )
    st.plotly_chart(fig7, use_container_width=True)
    if curve_snap is not None:
        st.caption(
            f"Live curve as of **{curve_snap.as_of}**: "
            "0–12M SOFR (NY Fed overnight + SOFR swap tenors); "
            "1Y–10Y USD SOFR IRS mid (BlueGamma). "
            "BCBS shocks are added to this base curve for EVE revaluation; "
            "NII uses the same shock at each instrument’s repricing bucket."
        )
        pts = pd.DataFrame([
            {"Tenor (Y)": t, "Rate (%)": round(r * 100, 3), "Segment": (
                "SOFR" if t <= 1.0 else "USD IRS mid"
            )}
            for t, r in sorted(curve_snap.points.items()) if t <= 10.0
        ])
        st.dataframe(pts, use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Base curve is stylised (USD, late 2024) — enable "
            "**Use live SOFR + USD IRS mid curve** in the sidebar for market rates."
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<p style='color:{DIM};font-size:10px;text-align:center'>"
    f"BCBS 368 (April 2016) · Interest Rate Risk in the Banking Book · "
    f"Pillar 2 · Supervisory outlier: |ΔEVE| > 15% Tier 1 Capital</p>",
    unsafe_allow_html=True,
)
