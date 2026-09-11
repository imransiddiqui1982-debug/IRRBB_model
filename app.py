"""
app.py
------
IRRBB Model — Interactive Streamlit Dashboard
Basel III / BCBS 368 · 19 Buckets · Full Cash Flow Discounting

Run:  streamlit run app.py
"""

import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.balance_sheet import get_instruments  # noqa: E402
from src.load_balance_sheet import (  # noqa: E402
    instruments_to_dataframe,
    load_instruments_from_csv,
)
from src.nmd_refinement import refine_nmd_deposits, merge_nmd_into_balance_sheet, load_customer_nmd  # noqa: E402
from src.calculator import IRRBBCalculator  # noqa: E402
from src.key_rate_duration import (  # noqa: E402
    build_treasury_alco_pack,
    designate_key_rate_hedges,
    instrument_kr01_attribution,
)
from src.liquidity_ratios import compute_liquidity_ratios_with_workbook  # noqa: E402
from src.market_curve import get_live_yield_curve  # noqa: E402
from src.scenarios import SCENARIOS, REF_LABELS, Scenario  # noqa: E402
from src.time_buckets import BUCKET_LABELS, N_BUCKETS  # noqa: E402
from src.yield_curve import YieldCurve  # noqa: E402
from src.ui_helpers import (  # noqa: E402
    DOCS_MARKDOWN,
    available_packs,
    build_export_zip,
    custom_scenario,
    dataframe_to_csv_bytes,
    exception_rows,
    filter_instruments_df,
    scenario_heatmap_frame,
    style_delta_columns,
    validate_balance_sheet_bytes,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IRRBB Model — BCBS 368",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def _with_total_row(
    df: pd.DataFrame,
    *,
    label: str = "Total",
    sum_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Append a bottom Total row summing additive numeric columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if out.index.name is not None:
        out = out.reset_index()

    amount_hint = re.compile(
        r"\$M|notional|balance_mb|\bbalance\b|hqla value|"
        r"asf \(\$|rsf \(\$|outflow \(\$|inflow \(\$|customers|pct_of_core|"
        r"\basset\b|\bliability\b|\bnet\b|kr01",
        re.I,
    )
    skip = re.compile(
        r"(?:rate|beta|wal|coupon|duration|repricing|factor|haircut|"
        r"years?|pass|disclosure|%|pct|percent)",
        re.I,
    )

    if sum_cols is None:
        sum_cols = []
        for c in out.columns:
            name = str(c)
            series = pd.to_numeric(out[c], errors="coerce")
            if series.notna().sum() == 0:
                continue
            if amount_hint.search(name):
                sum_cols.append(c)
                continue
            if skip.search(name):
                continue
            sum_cols.append(c)

    row: dict = {}
    labeled = False
    for c in out.columns:
        if c in sum_cols:
            total = float(pd.to_numeric(out[c], errors="coerce").fillna(0).sum())
            row[c] = round(total, 4) if abs(total) < 1 else round(total, 2)
        elif not labeled and (
            out[c].dtype == object
            or pd.api.types.is_string_dtype(out[c])
            or str(c).lower() in (
                "bucket", "segment", "instrument", "metric", "item",
                "category", "line", "source", "label", "key",
            )
        ):
            row[c] = label
            labeled = True
        else:
            row[c] = ""
    if not labeled and len(out.columns):
        row[out.columns[0]] = label
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)


def _sum_row_after(
    df: pd.DataFrame,
    after: str = "10Y",
    *,
    label: str = "Total",
) -> pd.DataFrame:
    """Insert a numeric sum row immediately after ``after`` (e.g. after 10Y)."""
    if df is None or df.empty or after not in df.index:
        return _with_total_row(df, label=label)
    out = df.copy()
    skip_tok = ("predicted", "actual", "convexity", "total")
    keys = [i for i in out.index if not any(t in str(i).lower() for t in skip_tok)]
    # Sum key rows up to and including ``after``
    sum_keys = []
    for k in keys:
        sum_keys.append(k)
        if k == after:
            break
    if not sum_keys:
        return out
    total = out.loc[sum_keys].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
    total.name = label
    # Rebuild: rows through ``after``, then Total, then remaining
    head = list(out.index)
    pos = head.index(after)
    parts = [out.iloc[: pos + 1], total.to_frame().T]
    if pos + 1 < len(out):
        parts.append(out.iloc[pos + 1 :])
    return pd.concat(parts)


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
#  SIDEBAR — assumptions & uploads
# ══════════════════════════════════════════════════════════════════════════════

PACKS = available_packs()
PACK_BY_KEY = {p.key: p for p in PACKS}

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

    expert_mode = st.toggle("Expert mode (full tables)", value=False, key="expert_mode")

    st.markdown("<p class='section-label'>1 · Template pack</p>",
                unsafe_allow_html=True)
    pack_label = st.selectbox(
        "Starting book",
        [p.label for p in PACKS],
        index=0,
        help="Loads a sample balance sheet (and NMD workbook when available).",
    )
    pack = next(p for p in PACKS if p.label == pack_label)
    st.caption(pack.description)

    st.markdown("<p class='section-label'>2 · Regulatory</p>",
                unsafe_allow_html=True)
    tier1 = st.number_input(
        "Tier 1 Capital (USD M)",
        value=500, min_value=50, max_value=10000, step=50,
    )
    outlier_pct = st.slider("Outlier |ΔEVE|/T1 %", 10, 20, 15, 1)
    watch_pct = st.slider("Watch |ΔEVE|/T1 %", 5, 15, 10, 1)
    st.caption(
        f"Limits: watch ${tier1 * watch_pct / 100:.0f}M · "
        f"outlier ${tier1 * outlier_pct / 100:.0f}M"
    )

    st.markdown("<p class='section-label'>3 · Data uploads</p>",
                unsafe_allow_html=True)
    st.caption("Leave blank to use the selected template pack.")

    bs_template_path = str(pack.bs_path)
    lcr_nsfr_path = str(pack.nmd_path) if pack.nmd_path else ""

    uploaded_csv = st.file_uploader(
        "Balance sheet CSV",
        type=["csv"],
        key="upload_balance_sheet",
    )
    if pack.bs_path.exists():
        with open(pack.bs_path, "rb") as f:
            st.download_button(
                "⬇ Pack balance sheet",
                f.read(),
                file_name=pack.bs_path.name,
                mime="text/csv",
                use_container_width=True,
                key="dl_bs_template",
            )

    uploaded_lcr_nsfr = st.file_uploader(
        "LCR / NSFR + NMD workbook",
        type=["xlsx", "xls"],
        key="upload_lcr_nsfr",
    )
    if pack.nmd_path and pack.nmd_path.exists():
        with open(pack.nmd_path, "rb") as f:
            st.download_button(
                "⬇ LCR/NSFR + NMD template",
                f.read(),
                file_name=pack.nmd_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_lcr_nsfr_template",
            )

    deposit_name = "Retail NMD"

    st.markdown("<p class='section-label'>4 · Curve & PMMS</p>",
                unsafe_allow_html=True)
    use_live_curve = st.checkbox(
        "Live SOFR + USD IRS mid curve",
        value=True,
        help="0–12M SOFR; 1Y–10Y USD SOFR IRS mids.",
    )
    refresh_curve = st.button("Refresh live curve", use_container_width=True)
    curve_paste = st.text_area(
        "Optional curve override (tenor_years,rate_pct per line)",
        value="",
        height=70,
        help="Example:\n0.25,5.10\n1,4.80\n10,4.40",
        placeholder="Leave blank to use live / stylised curve",
    )

    st.markdown("<p class='section-label'>5 · NII assumptions</p>",
                unsafe_allow_html=True)
    nii_horizon = st.selectbox("NII horizon (months)", [12, 24], index=0)
    nii_constant_bs = st.checkbox(
        "Constant balance sheet (reinvest runoff)",
        value=True,
        help="BCBS default. Uncheck = static (no reinvestment).",
    )
    nii_us_mode = st.checkbox("US NII shock set (±100…400 + ramps)", value=False)

    st.markdown("<p class='section-label'>6 · Active scenario</p>",
                unsafe_allow_html=True)
    use_custom_shocks = st.checkbox("Custom pillar shocks", value=False)
    selected_name = st.radio(
        "scenario", [s.name for s in SCENARIOS],
        label_visibility="collapsed",
        disabled=use_custom_shocks,
    )
    selected_scenario = next(s for s in SCENARIOS if s.name == selected_name)

    custom_shocks: list[int] = []
    if use_custom_shocks:
        st.caption("Edit pillar shocks (bp) — used for detail / what-if views.")
        cols_sh = st.columns(3)
        for i, lab in enumerate(REF_LABELS):
            with cols_sh[i % 3]:
                custom_shocks.append(
                    int(st.number_input(
                        lab, value=int(selected_scenario.ref_shocks_bp[i]),
                        step=25, key=f"custom_sh_{lab}",
                    ))
                )
        selected_scenario = custom_scenario("Custom", custom_shocks)
        selected_name = selected_scenario.name

    st.divider()
    st.markdown("<p class='section-label'>Shock profile</p>",
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

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_market_curve(force_refresh: bool = False):
    _ = force_refresh
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
    pmms_rate_override: float | None,
    outlier_thr: float,
    watch_thr: float,
    _model_version: int = 16,
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

    from src.mbs_pricing import apply_pmms_anchor
    pmms_meta = apply_pmms_anchor(
        list(assets) + list(liabilities),
        curve,
        pmms_rate=pmms_rate_override,
    )

    calc = IRRBBCalculator(
        assets,
        liabilities,
        tier1_capital=tier1_cap,
        yield_curve=curve,
        cpr_table=None,
        outlier_threshold=outlier_thr,
        watch_threshold=watch_thr,
    )
    results = calc.run_all(SCENARIOS)
    gap = calc.repricing_gap()
    maturity_gap = calc.bucket_maturity_gap()
    dv01_gap = calc.bucket_dv01_gap()
    return (
        calc, results, gap, maturity_gap, dv01_gap,
        assets, liabilities, nmd_result, curve, pmms_meta,
    )


def _parse_curve_paste(text: str) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    text = (text or "").strip()
    if not text:
        return None
    tenors, rates = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,;\s]+", line)
        if len(parts) < 2:
            continue
        tenors.append(float(parts[0]))
        r = float(parts[1])
        rates.append(r / 100.0 if r > 1.0 else r)
    if len(tenors) < 2:
        return None
    order = np.argsort(tenors)
    return tuple(tenors[i] for i in order), tuple(rates[i] for i in order)


# Resolve curve
curve_snap = None
curve_tenors_t: tuple[float, ...] | None = None
curve_rates_t: tuple[float, ...] | None = None
curve_source_label = "stylised"
pasted = _parse_curve_paste(curve_paste)
if pasted is not None:
    curve_tenors_t, curve_rates_t = pasted
    curve_source_label = "manual paste"
elif use_live_curve:
    try:
        live_curve, curve_snap = load_market_curve(force_refresh=bool(refresh_curve))
        tenors = sorted(curve_snap.points.keys())
        curve_tenors_t = tuple(tenors)
        curve_rates_t = tuple(curve_snap.points[t] for t in tenors)
        curve_source_label = f"live @ {curve_snap.as_of}"
        st.sidebar.success(f"Live curve as of {curve_snap.as_of}")
        for note in curve_snap.source_notes[:3]:
            st.sidebar.caption(note)
    except Exception as exc:
        st.sidebar.warning(f"Live curve unavailable — stylised. ({exc})")
        use_live_curve = False
        curve_source_label = "stylised (fallback)"

# Resolve balance sheet bytes (pack / upload / session edit)
if "bs_csv_bytes" not in st.session_state:
    st.session_state.bs_csv_bytes = None
if "bs_editor_rev" not in st.session_state:
    st.session_state.bs_editor_rev = 0
if "pack_key_loaded" not in st.session_state:
    st.session_state.pack_key_loaded = None

# Reload pack when selection changes (unless user uploaded a file this session)
if uploaded_csv is not None:
    csv_payload = uploaded_csv.getvalue()
    bs_source_label = uploaded_csv.name
    st.session_state.bs_csv_bytes = csv_payload
    st.session_state.pack_key_loaded = f"upload:{uploaded_csv.name}"
elif (
    st.session_state.bs_csv_bytes is not None
    and st.session_state.pack_key_loaded == pack.key
):
    csv_payload = st.session_state.bs_csv_bytes
    bs_source_label = f"edited · {pack.bs_path.name}"
elif pack.bs_path.exists():
    csv_payload = pack.bs_path.read_bytes()
    bs_source_label = pack.bs_path.name
    st.session_state.bs_csv_bytes = csv_payload
    st.session_state.pack_key_loaded = pack.key
else:
    csv_payload = None
    bs_source_label = "built-in sample"

nmd_payload = None
nmd_source_label = "none"
if uploaded_lcr_nsfr is not None:
    nmd_payload = uploaded_lcr_nsfr.getvalue()
    nmd_source_label = uploaded_lcr_nsfr.name
elif pack.nmd_path and pack.nmd_path.exists():
    nmd_payload = pack.nmd_path.read_bytes()
    nmd_source_label = pack.nmd_path.name

use_nmd = nmd_payload is not None
st.sidebar.caption(f"BS: {bs_source_label}")
st.sidebar.caption(f"NMD/LCR: {nmd_source_label}")

# Validate
_bs_df_preview, _bs_issues = (None, [])
if csv_payload:
    _bs_df_preview, _bs_issues = validate_balance_sheet_bytes(csv_payload)
_hard_errors = [i for i in _bs_issues if i["severity"] == "error"]

pmms_override = None

pmms_meta: dict = {}
_run_ok = False
calc = results = gap = maturity_gap = dv01_gap = None
assets, liabilities, nmd_result = [], [], None
curve = YieldCurve()

if _hard_errors:
    st.warning(
        "Balance sheet validation issues — see **Inputs** tab. "
        "Attempting to run anyway if the CSV still loads."
    )
    for iss in _hard_errors[:5]:
        st.caption(f"• [{iss['column'] or 'file'}] {iss['message']}")

try:
    (
        calc, results, gap, maturity_gap, dv01_gap,
        assets, liabilities, nmd_result, curve, pmms_meta,
    ) = run_model(
        float(tier1),
        csv_payload,
        nmd_payload,
        deposit_name,
        use_nmd,
        curve_tenors_t,
        curve_rates_t,
        pmms_override,
        outlier_pct / 100.0,
        watch_pct / 100.0,
    )
    _run_ok = True
except UnicodeDecodeError:
    st.error("File encoding error: use CSV for BS and Excel for LCR/NSFR + NMD.")
    st.stop()
except Exception as exc:
    st.error(f"Model run failed: {exc}")
    # keep Inputs usable
    _run_ok = False

if pmms_meta.get("pmms_rate") is not None:
    st.sidebar.success(
        f"PMMS: {pmms_meta['pmms_rate'] * 100:.2f}% "
        f"({pmms_meta.get('as_of', 'n/a')}) · {pmms_meta.get('applied', 0)} OA pools"
    )
elif pmms_meta.get("error"):
    st.sidebar.caption(f"PMMS unavailable — sheet spreads ({pmms_meta['error']})")

active = None
detail = None
liquidity = None
lcr_result = None
nsfr_result = None
if _run_ok and calc is not None and results:
    # Map custom scenario onto nearest named result for dashboards that need SCENARIOS
    if selected_scenario.id == "CUSTOM":
        active = results[0]
        # Recompute detail under custom shocks
        try:
            detail = calc.instrument_eve_detail(selected_scenario)
        except Exception:
            detail = calc.instrument_eve_detail(results[0].scenario)
            selected_scenario = results[0].scenario
    else:
        active = next(r for r in results if r.scenario.name == selected_name)
        detail = calc.instrument_eve_detail(selected_scenario)

    nmd_customers = None
    if nmd_payload:
        try:
            nmd_customers = load_customer_nmd(io.BytesIO(nmd_payload)).customers
        except Exception:
            nmd_customers = None
    workbook_src = io.BytesIO(nmd_payload) if nmd_payload else None
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
#  RUN STATUS STRIP
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"<h2 style='font-size:20px;letter-spacing:0.5px;margin-bottom:2px'>"
    f"Interest Rate Risk in the Banking Book</h2>"
    f"<p style='color:{DIM};font-size:11px;margin-top:0'>"
    f"Upload → Validate → Edit → Assumptions → Run → Dashboard → Export</p>",
    unsafe_allow_html=True,
)

total_a = sum(i.notional for i in assets) if assets else 0.0
total_l = sum(i.notional for i in liabilities) if liabilities else 0.0
outliers = sum(1 for r in results if r.is_outlier) if results else 0
watches = sum(1 for r in results if r.is_watch) if results else 0
worst = max(results, key=lambda r: r.delta_eve_pct) if results else None
worst_nii = min(results, key=lambda r: r.delta_nii) if results else None

status_color = GREEN if _run_ok and outliers == 0 else (AMBER if _run_ok else RED)
status_txt = "READY" if _run_ok and outliers == 0 else ("WATCH / OUTLIER" if _run_ok else "BLOCKED")
pmms_txt = (
    f"{pmms_meta['pmms_rate'] * 100:.2f}%"
    if pmms_meta.get("pmms_rate") is not None else "n/a"
)

worst_eve_txt = f"{worst.delta_eve_pct:.1f}%" if worst else "—"
worst_nii_txt = f"${worst_nii.delta_nii:+.1f}M" if worst_nii else "—"
st.markdown(
    f"<div style='background:{BG3};border:1px solid {BORDER};border-radius:8px;"
    f"padding:10px 14px;margin:8px 0 12px;display:flex;flex-wrap:wrap;gap:18px;"
    f"align-items:center;font-size:12px'>"
    f"<span><b style='color:{status_color}'>{status_txt}</b></span>"
    f"<span style='color:{DIM}'>Curve</span> <b>{curve_source_label}</b>"
    f"<span style='color:{DIM}'>PMMS</span> <b>{pmms_txt}</b>"
    f"<span style='color:{DIM}'>BS</span> <b>{bs_source_label}</b>"
    f"<span style='color:{DIM}'>Assets</span> <b>${total_a:,.0f}M</b>"
    f"<span style='color:{DIM}'>Liab</span> <b>${total_l:,.0f}M</b>"
    f"<span style='color:{DIM}'>Outliers</span> <b>{outliers}</b>"
    f"<span style='color:{DIM}'>Worst |ΔEVE|/T1</span> <b>{worst_eve_txt}</b>"
    f"<span style='color:{DIM}'>Worst ΔNII</span> <b>{worst_nii_txt}</b>"
    f"</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Assets", f"${total_a:,.0f}M")
c2.metric("Total Liabilities", f"${total_l:,.0f}M")
c3.metric("Tier 1", f"${tier1:,.0f}M")
c4.metric("Outliers", str(outliers))
c5.metric(
    "Worst |ΔEVE|/T1",
    f"{worst.delta_eve_pct:.1f}%" if worst else "—",
    delta=worst.scenario.name if worst else None,
    delta_color="off",
)
c6.metric(
    "Worst ΔNII",
    f"${worst_nii.delta_nii:+.1f}M" if worst_nii else "—",
)

# Export + prepay diag
_exp_c1, _exp_c2 = st.columns([1, 3])
with _exp_c1:
    if _run_ok and results:
        try:
            from src.key_rate_duration import compute_kr01
            _kr_ex = compute_kr01(assets, liabilities, curve)
            _nii_ex = pd.DataFrame(
                calc.nii_sensitivity_grid(
                    us_mode=bool(nii_us_mode),
                    horizon_months=int(nii_horizon),
                    constant_balance_sheet=bool(nii_constant_bs),
                )
            )
            _zip = build_export_zip(
                instruments_df=instruments_to_dataframe(assets, liabilities),
                results_df=scenario_heatmap_frame(results),
                curve_df=pd.DataFrame({
                    "tenor_years": list(getattr(curve, "_ref_tenors", [])),
                    "rate": list(getattr(curve, "_ref_rates", [])),
                }),
                kr01_df=_kr_ex,
                nii_df=_nii_ex,
                meta={
                    "tier1": tier1,
                    "curve_source": curve_source_label,
                    "pmms": pmms_txt,
                    "outliers": outliers,
                    "worst_eve_pct": worst.delta_eve_pct if worst else None,
                    "pack": pack.label,
                },
            )
            st.download_button(
                "⬇ Export pack (ZIP)",
                _zip,
                file_name="irrbb_run_export.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as _ex:
            st.caption(f"Export unavailable: {_ex}")

if _run_ok and expert_mode:
    try:
        from src.mbs_pricing import prepayment_diagnostics
        from src.key_rate_duration import shocked_yield_curve
        _oa_assets = [a for a in assets if getattr(a, "is_option_adjusted", False)]
        if _oa_assets:
            with st.expander("Prepayment diagnostics (OA)", expanded=False):
                _diag_rows = [prepayment_diagnostics(_oa_assets, curve, "Base")]
                for _sc in SCENARIOS:
                    _scurve = shocked_yield_curve(curve, _sc.shocks_bp, scenario=_sc)
                    _diag_rows.append(prepayment_diagnostics(_oa_assets, _scurve, _sc.name))
                st.dataframe(pd.concat(_diag_rows, ignore_index=True),
                             use_container_width=True, hide_index=True)
    except Exception:
        pass

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

(
    tab_in, tab0, tab_lcr, tab1, tab2, tab3, tab4, tab5, tab_kr, tab6, tab_docs,
) = st.tabs([
    "Inputs",
    "NMD Refinement",
    "LCR / NSFR",
    "Dashboard",
    "Scenario Detail",
    "EVE Waterfall",
    "Scenario Comparison",
    "Repricing Gap",
    "ALCO / KR01 / Hedges",
    "Yield Curve",
    "Docs",
])

# ── TAB: Inputs (editor, validation, what-if) ─────────────────────────────────
with tab_in:
    st.markdown("<p class='section-label'>Balance sheet editor</p>",
                unsafe_allow_html=True)
    st.caption(
        "Edit notionals, coupons, WAC, OAS, resets. Filter, then **Apply edits & Run**. "
        "Optional columns: `book`, `currency` for filtering."
    )
    if _bs_issues:
        st.dataframe(pd.DataFrame(_bs_issues), use_container_width=True, hide_index=True)

    if csv_payload:
        _edit_df = pd.read_csv(io.BytesIO(csv_payload))
        f1, f2, f3 = st.columns(3)
        with f1:
            _side_f = st.selectbox("Side", ["All", "asset", "liability"], key="filt_side")
        with f2:
            _types = ["All"] + sorted(_edit_df["instrument_type"].astype(str).str.lower().unique())
            _type_f = st.selectbox("Type", _types, key="filt_type")
        with f3:
            _q = st.text_input("Name contains", "", key="filt_name")
        _view = filter_instruments_df(_edit_df, side=_side_f, itype=_type_f, name_query=_q)
        st.caption(f"Showing {len(_view)} / {len(_edit_df)} rows")
        edited = st.data_editor(
            _view,
            use_container_width=True,
            num_rows="dynamic",
            key=f"bs_editor_{st.session_state.bs_editor_rev}",
            height=360,
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Apply edits & Run", type="primary", use_container_width=True):
                # Merge edited rows back by name where possible
                base = _edit_df.copy()
                if "name" in edited.columns and "name" in base.columns:
                    for _, row in edited.iterrows():
                        m = base["name"].astype(str) == str(row["name"])
                        if m.any():
                            for col in edited.columns:
                                if col in base.columns:
                                    base.loc[m, col] = row[col]
                    # append new names
                    new_names = set(edited["name"].astype(str)) - set(base["name"].astype(str))
                    if new_names:
                        base = pd.concat(
                            [base, edited[edited["name"].astype(str).isin(new_names)]],
                            ignore_index=True,
                        )
                else:
                    base = edited
                st.session_state.bs_csv_bytes = dataframe_to_csv_bytes(base)
                st.session_state.bs_editor_rev += 1
                st.session_state.pack_key_loaded = pack.key
                st.rerun()
        with b2:
            if st.button("Reset to pack / upload", use_container_width=True):
                st.session_state.bs_csv_bytes = None
                st.session_state.pack_key_loaded = None
                st.session_state.bs_editor_rev += 1
                st.rerun()
    else:
        st.info("No balance sheet loaded.")

# ── TAB 0: NMD refinement ─────────────────────────────────────────────────────
with tab0:
    if not _run_ok:
        st.info("Run the model from **Inputs** first.")
    else:
        st.markdown("<p class='section-label'>Behavioural NMD Refinement</p>",
                    unsafe_allow_html=True)
    if nmd_result is None:
        st.info(
            "Upload the **LCR / NSFR + NMD workbook** in the sidebar "
            "(or use the built-in sample) to refine core / non-core deposits."
        )
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
            st.dataframe(
                _with_total_row(nmd_result.segment_summary),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Portfolio refinement summary**")
        st.caption(
            "Includes historical and long-run runoff / decay rates. "
            "Core sticky balance feeds IRRBB as WAL-based demand deposits "
            "(tenor bucket display removed — not used in EVE/NII pricing)."
        )
        st.dataframe(nmd_result.summary, use_container_width=True, hide_index=True)

        if nmd_result.liquidity_summary is not None and not nmd_result.liquidity_summary.empty:
            st.markdown("**LCR / NSFR bifurcation map (from deposit file)**")
            st.dataframe(
                _with_total_row(nmd_result.liquidity_summary),
                use_container_width=True,
                hide_index=True,
            )


# ── TAB LCR / NSFR: Liquidity ratios ──────────────────────────────────────────
with tab_lcr:
    if not _run_ok or liquidity is None:
        st.info("Run the model from **Inputs** first.")
    else:
        st.markdown("<p class='section-label'>Basel III Liquidity Ratios ??? LCR & NSFR</p>",
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
            "LCR = HQLA Stock / Net Cash Outflows (30-day stress) ??? 100%. "
            "Inflows capped at 75% of outflows."
        )

        lcr_color = GREEN if lcr_result.lcr_pass else RED
        lcr_status = "PASS ???" if lcr_result.lcr_pass else "FAIL ???"
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
                st.dataframe(
                    _with_total_row(lcr_result.hqla_breakdown),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("No HQLA-eligible assets identified.")
        with col_o:
            st.markdown("**30-day cash outflows**")
            if not lcr_result.outflow_breakdown.empty:
                st.dataframe(
                    _with_total_row(lcr_result.outflow_breakdown),
                    use_container_width=True,
                    hide_index=True,
                )

        if not lcr_result.inflow_breakdown.empty:
            st.markdown("**30-day cash inflows (before 75% cap)**")
            st.dataframe(
                _with_total_row(lcr_result.inflow_breakdown),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.markdown("<p class='section-label'>Net Stable Funding Ratio (NSFR)</p>",
                    unsafe_allow_html=True)
        st.caption(
            "NSFR = ASF / **tailored RSF** ??? minimum (workbook default 100%). "
            "ASF from capital (100%), retail deposits (95%/90%), wholesale by tenor. "
            "RSF line items are weighted first; Category IV banks then apply an "
            "**RSF reduction** (default 30%) before the ratio ??? so the breakdown "
            "Total is pre-adjustment, not the NSFR denominator."
        )

        nsfr_color = GREEN if nsfr_result.nsfr_pass else RED
        nsfr_status = "PASS ???" if nsfr_result.nsfr_pass else "FAIL ???"
        st.markdown(
            f"<h3 style='font-size:15px;color:{NAVY}'>"
            f"NSFR = {nsfr_result.nsfr_pct:.1f}% &nbsp;"
            f"<span style='font-size:12px;color:{nsfr_color};"
            f"background:{nsfr_color}18;padding:3px 10px;"
            f"border-radius:3px;font-weight:600'>{nsfr_status}</span></h3>",
            unsafe_allow_html=True,
        )

        _rsf_pre = 0.0
        if (
            nsfr_result.rsf_breakdown is not None
            and not nsfr_result.rsf_breakdown.empty
            and "RSF ($M)" in nsfr_result.rsf_breakdown.columns
        ):
            _rsf_pre = float(
                pd.to_numeric(nsfr_result.rsf_breakdown["RSF ($M)"], errors="coerce")
                .fillna(0)
                .sum()
            )
        if _rsf_pre <= 0:
            _rsf_pre = float(nsfr_result.rsf_total)
        _rsf_tailor_pct = (
            (1.0 - float(nsfr_result.rsf_total) / _rsf_pre) * 100.0
            if _rsf_pre > 1e-9
            else 0.0
        )

        n1, n2, n3, n4, n5 = st.columns(5)
        n1.metric("ASF Total", f"${nsfr_result.asf_total:,.1f}M")
        n2.metric("RSF (pre-adj)", f"${_rsf_pre:,.1f}M")
        n3.metric(
            "RSF (tailored)",
            f"${nsfr_result.rsf_total:,.1f}M",
            delta=f"???{_rsf_tailor_pct:.0f}% Category IV" if _rsf_tailor_pct > 0.5 else None,
            delta_color="off",
        )
        n4.metric("ASF ??? Capital", f"${nsfr_result.asf_capital:,.1f}M")
        n5.metric(
            "Funding Gap",
            f"${nsfr_result.asf_total - nsfr_result.rsf_total:+,.1f}M",
            help="ASF ??? tailored RSF (the NSFR denominator).",
        )

        if nmd_result is not None:
            st.info(
                f"NMD refinement active ??? LCR outflows and NSFR ASF use behavioural splits "
                f"({nmd_result.stable_pct * 100:.0f}% stable core, "
                f"{nmd_result.non_core_pct * 100:.0f}% non-core)."
            )

        col_asf, col_rsf = st.columns(2)
        with col_asf:
            st.markdown("**ASF breakdown (funding sources)**")
            st.dataframe(
                _with_total_row(nsfr_result.asf_breakdown),
                use_container_width=True,
                hide_index=True,
            )
        with col_rsf:
            st.markdown("**RSF breakdown (asset requirements)**")
            st.caption(
                f"Table Total = **pre-adjustment** RSF (${_rsf_pre:,.1f}M). "
                f"NSFR uses tailored RSF ${nsfr_result.rsf_total:,.1f}M "
                f"after ~{_rsf_tailor_pct:.0f}% Category IV reduction."
            )
            st.dataframe(
                _with_total_row(nsfr_result.rsf_breakdown),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "??? Download Liquidity Ratios CSV",
            liquidity.summary.to_csv(index=False),
            file_name="liquidity_ratios_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


    # ?????? TAB 1: All scenarios ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
with tab1:
    if not _run_ok:
        st.info("Fix balance-sheet errors on **Inputs**, then Apply & Run.")
    else:
        st.markdown("<p class='section-label'>Dashboard — EVE / NII heatmap & exceptions</p>",
                    unsafe_allow_html=True)
        st.caption(
            f"Tier 1 = ${tier1:,.0f}M · Outlier ≥ {outlier_pct}% · Watch ≥ {watch_pct}%"
        )

        heat = scenario_heatmap_frame(results)
        # Heatmap via plotly (ΔNII / ΔEVE only — no T1 % row)
        z = np.array([
            heat["ΔNII ($M)"].tolist(),
            heat["ΔEVE ($M)"].tolist(),
        ], dtype=float)
        fig_h = go.Figure(data=go.Heatmap(
            z=z,
            x=[SCENARIO_SHORT.get(s, s) for s in heat["Scenario"]],
            y=["ΔNII ($M)", "ΔEVE ($M)"],
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(z, 1),
            texttemplate="%{text}",
            hoverongaps=False,
        ))
        fig_h.update_layout(
            **{**PLOTLY_BASE, "height": 180, "margin": dict(l=80, r=20, t=20, b=40)},
        )
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("<p class='section-label'>Exception view</p>", unsafe_allow_html=True)
        exc = exception_rows(results)
        st.dataframe(
            style_delta_columns(exc, ["ΔEVE ($M)", "ΔNII ($M)"]),
            use_container_width=True,
            hide_index=True,
        )

        if expert_mode:
            st.markdown("<p class='section-label'>BCBS 368 — Six Prescribed Scenarios</p>",
                        unsafe_allow_html=True)

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
                    key="dl_dash_summary",
                )
        else:
            st.caption("Enable **Expert mode** in the sidebar for scenario cards and full summary chart.")

# ── TAB 2: Scenario detail ────────────────────────────────────────────────────
with tab2:
    if not _run_ok or active is None:
        st.info("Run the model from **Inputs** first.")
    else:
        r = active
        status_color = RED if r.is_outlier else AMBER if r.is_watch else GREEN
        try:
            sc_color_idx = next(i for i, s in enumerate(SCENARIOS) if s.id == r.scenario.id)
        except StopIteration:
            sc_color_idx = 0
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

        st.markdown("<p class='section-label'>NII Sensitivity Grid (accrual engine)</p>",
                    unsafe_allow_html=True)
        st.caption(
            f"Sidebar NII settings: {int(nii_horizon)}m horizon · "
            f"{'US shock set' if nii_us_mode else 'BCBS ±200'} · "
            f"{'constant BS' if nii_constant_bs else 'static BS'}. "
            "No discounting — distinct from EVE."
        )
        nii_rows = calc.nii_sensitivity_grid(
            us_mode=bool(nii_us_mode),
            horizon_months=int(nii_horizon),
            constant_balance_sheet=bool(nii_constant_bs),
        )
        nii_df = pd.DataFrame(nii_rows)
        if not nii_df.empty:
            show = nii_df.rename(columns={
                "scenario": "Scenario",
                "shock_bp": "Shock (bp)",
                "ramp": "Ramp",
                "nii": "NII ($M)",
                "d_nii": "ΔNII ($M)",
                "pct_tier1": "% Tier 1",
                "pct_base_nii": "% Base NII",
            })
            st.dataframe(
                show.style.format({
                    "NII ($M)": "{:.2f}",
                    "ΔNII ($M)": "{:+.2f}",
                    "% Tier 1": "{:+.2f}",
                    "% Base NII": "{:+.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Base 12m NII ≈ ${float(nii_df['nii'].iloc[0] - nii_df['d_nii'].iloc[0]):.2f}M · "
                "Deposits reprice 1:1 with the shock (floored at 0); floaters after next reset; "
                "MBS CPR is live under the shock."
            )


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
        _dv01_tbl = dv01_k[["asset_dv01_k", "liability_dv01_k", "net_dv01_k"]].rename(
            columns={
                "asset_dv01_k": "Asset ($K/bp)",
                "liability_dv01_k": "Liability ($K/bp)",
                "net_dv01_k": "Net ($K/bp)",
            }
        )
        st.dataframe(
            _with_total_row(_dv01_tbl),
            use_container_width=True,
            height=360,
            hide_index=True,
        )
    with col_t2:
        st.markdown("**Maturity notional gap ($M) — aligns with DV01**")
        st.caption("Principal + repricing cash flows only (no coupon double-count).")
        st.dataframe(
            _with_total_row(maturity_gap),
            use_container_width=True,
            height=360,
            hide_index=True,
        )

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

    export_gap = dv01_k.join(maturity_gap, rsuffix="_maturity_m")
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
        _kr01_step1 = kr01_df.set_index("label")[
            ["asset_kr01_k", "liability_kr01_k", "net_kr01_k"]
        ].rename(columns={
            "asset_kr01_k": "Asset",
            "liability_kr01_k": "Liability",
            "net_kr01_k": "Net KR01",
        })
        st.dataframe(
            _with_total_row(_kr01_step1),
            use_container_width=True,
            hide_index=True,
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
        _attr = _sum_row_after(pack["attribution"], after="10Y", label="Total (1Y–10Y)")
        st.dataframe(
            _attr.style.format("{:+.2f}"),
            use_container_width=True,
        )
        st.caption(
            "Predicted key cells use **Base KR01 only** × rate shocks. "
            "MBS / whole-loan Actual ΔEVE and Dynamic KR01 re-derive CPR from each "
            "instrument’s balance-sheet WAC on the shocked curve. "
            "**Total (1Y–10Y)** sums key rows after the 10Y bucket."
        )

        st.markdown("**Dynamic KR01 under each BCBS shock ($K/bp net)**")
        st.caption(
            "Recomputed on the shocked curve; OA pools reprice with live CPR from "
            "balance-sheet WAC. Total row = sum across keys."
        )
        st.dataframe(
            _with_total_row(pack["scenario_kr01"]),
            use_container_width=True,
            hide_index=True,
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

        # ── Mortgage prepay / CPR–PSA (live model vs ALCO reference) ──────────
        st.divider()
        st.markdown(
            "<p class='section-label'>MBS / mortgage prepay — how the model works</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""
**Live EVE / KR01 path (per MBS or whole loan)** — Steps A → B → C in `mbs_pricing`.
Re-run on every curve (including KR01 bumps). No portfolio sidebar CPR/PSA override.

**Step A — Mortgage rate & refi incentive**
- Live **FRED PMMS 30Y** anchors the mortgage-rate *level*: each OA pool’s
  `spread_to_curve` is set so \(\mathrm{curve}(\mathrm{anchor})+\mathrm{spread}\approx\mathrm{PMMS}\)
  on the base curve. BCBS / KR01 bumps then move the rate 1:1 with the anchor.
- \(\mathrm{refi\ incentive\ (pp)} = (\mathrm{WAC} - \mathrm{mortgage\ rate}) \times 100\)
  (WAC and rates are decimals; result is **percentage points**).
- Positive = in-the-money to refinance; negative = lock-in.
- Fallback: balance-sheet `spread_to_curve` if PMMS fetch fails.

**Step B — CPR (logistic × seasoning)**
- Logistic refi response (parameters from `data/calibrated_prepayment_params.json`
  when present — fit via `python -m src.calibrate_prepayment` on loan-level history;
  otherwise illustrative defaults):
  \(\mathrm{refi\_response} = \dfrac{\mathrm{max\_refi\_cpr}}{1 + e^{-k\,(\mathrm{incentive\_pp} - \mathrm{midpoint})}}\)
- **Seasoning ramp** (calibrated `seasoning_ramp_months`, else 30):
  \(\mathrm{seasoning} = \min\!\big((\mathrm{pool\_age} + m) / \mathrm{ramp\_months},\ 1\big)\)
  for forecast month \(m\). Young pools get lower CPR; at full seasoning the ramp is 1.
- \(\mathrm{CPR} = (\mathrm{base\_turnover} + \mathrm{refi\_response}) \times \mathrm{seasoning}\)
- Monthly SMM from CPR; drives scheduled principal + prepay cash flows.
- Seasoning is a linear age ramp on logistic CPR (fit separately near zero incentive).
  PSA is only a **reporting label** of that CPR.

**PSA (reporting only)**
- \(\mathrm{PSA} = \mathrm{CPR} / 0.06 \times 100\)  (convention: \(100\) PSA ≡ \(6\%\) seasoned CPR).
- PSA is **not** an input that prices the book; cash flows use the logistic CPR above.

**Step C — Price / EVE — where OAS is used**
- Discount each month’s CF at \(\mathrm{curve}(t) + \mathrm{OAS}\).
- **OAS does not change CPR or PSA.** It only shifts the discount rate (spread over the
  curve), so it moves PV, EVE, and KR01 for a given prepay schedule.
- Default OAS ≈ 50 bp if blank on the instrument.

**Agency knot S-curve tables below** are ALCO documentation / scenario mapping only
(WAC vs implied mortgage rate → illustrative CPR/PSA). They do **not** override
Steps A–C pricing. Live logistic parameters come from loan-level calibration when
`data/calibrated_prepayment_params.json` is present.
            """
        )

        from src.calibrate_prepayment import get_engine_prepay_defaults
        _pp = get_engine_prepay_defaults()
        _r2 = _pp.get("r_squared")
        _r2_txt = f"{float(_r2):.3f}" if _r2 is not None else "n/a"
        st.success(
            f"**Active Step-B parameters** (source: `{_pp.get('calibration_source', 'n/a')}` · "
            f"R²={_r2_txt}): "
            f"base_turnover={_pp['base_turnover']:.4f}, "
            f"max_refi_cpr={_pp['max_refi_cpr']:.4f}, "
            f"k={_pp['logistic_k']:.3f}, "
            f"midpoint={_pp['logistic_midpoint']:.3f}, "
            f"seasoning_ramp={_pp['seasoning_ramp_months']} mo. "
            f"Re-fit with `python -m src.calibrate_prepayment --data your_loans.csv "
            f"--out data/calibrated_prepayment_params.json`."
        )

        from src.cpr_calibration import (
            CprCalibrationInputs,
            calibrate_scenario_cprs,
            calibration_display_frame,
            scurve_dataframe,
            scurve_display_frame,
            historical_regimes_dataframe,
            format_refi_incentive_bp,
            incentive_bp,
        )
        from src.mbs_pricing import (
            DEFAULT_SEASONING_RAMP_MONTHS,
            step_a_mortgage_rate,
            step_b_cpr,
            cpr_to_psa,
            terms_from_instrument,
        )

        _oa = [a for a in assets if getattr(a, "is_option_adjusted", False)]
        if _oa:
            _w_not = sum(max(float(a.notional), 0.0) for a in _oa) or 1.0
            _wac_dec = sum(
                float(getattr(a, "wac", 0.0) or 0.0) * max(float(a.notional), 0.0)
                for a in _oa
            ) / _w_not
            _wac_pct = _wac_dec * 100.0 if _wac_dec <= 1.0 else _wac_dec
            _ages = [
                int(
                    getattr(a, "age_months", None)
                    or getattr(a, "pool_age_months", None)
                    or 0
                )
                for a in _oa
            ]
            _age = int(round(sum(_ages) / len(_ages)))
            _top = max(_oa, key=lambda a: float(a.notional))
            _terms = terms_from_instrument(_top)
            _mtg, _inc_pp = step_a_mortgage_rate(curve, _terms)
            _pmms_pct = _mtg * 100.0
            _cpr_live = step_b_cpr(_terms, _inc_pp, float(_terms.pool_age_months) + 1.0)
            _cpr_seas = step_b_cpr(_terms, _inc_pp, 60.0)
            _oas_bp = float(_terms.oas) * 10_000.0
            _ramp = int(_terms.seasoning_ramp_months or DEFAULT_SEASONING_RAMP_MONTHS)
            st.info(
                f"**Largest OA pool example (`{_terms.name or _top.name}`):** "
                f"WAC {_wac_pct:.2f}% · mortgage rate {_pmms_pct:.2f}% · "
                f"refi incentive {_inc_pp:+.2f} pp · pool age {_terms.pool_age_months} mo · "
                f"seasoning ramp {_ramp} mo · "
                f"CPR now {_cpr_live * 100:.1f}% / seasoned {_cpr_seas * 100:.1f}% "
                f"(≈ {cpr_to_psa(_cpr_seas):.0f} PSA) · "
                f"OAS {_oas_bp:.0f} bp (discount only)."
            )
        else:
            _wac_pct, _pmms_pct, _age = 5.50, 6.50, 30
            st.caption(
                "No option-adjusted MBS / whole loans on the sheet — "
                "illustrative WAC/mortgage defaults used for the documentation tables only."
            )

        _inc0 = incentive_bp(_wac_pct, _pmms_pct)
        st.caption(
            f"Documentation tables use book-weighted WAC **{_wac_pct:.2f}%** and "
            f"implied mortgage **{_pmms_pct:.2f}%** "
            f"(refi incentive {format_refi_incentive_bp(_inc0)} in bp for the knot chart). "
            f"Average pool age ≈ **{_age}** months."
        )

        _calib = calibrate_scenario_cprs(
            CprCalibrationInputs(
                wac_pct=float(_wac_pct),
                pmms_pct=float(_pmms_pct),
                age_months=max(int(_age), 1),
            )
        )
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**ALCO reference — agency S-curve knots (not live pricing)**")
            st.dataframe(
                scurve_display_frame()[["Refi incentive", "CPR %"]],
                use_container_width=True,
                hide_index=True,
            )
        with sc2:
            st.markdown("**ALCO reference — scenario CPR / PSA map (not live pricing)**")
            st.dataframe(
                calibration_display_frame(_calib),
                use_container_width=True,
                hide_index=True,
            )

        _sc = scurve_dataframe()
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=_sc["incentive_bp"], y=_sc["cpr_pct"],
            mode="lines+markers", name="Agency-style S-curve",
            line=dict(color=NAVY, width=2.5),
        ))
        fig_sc.add_trace(go.Scatter(
            x=_calib["incentive_bp"], y=_calib["cpr_pct"],
            mode="markers+text", name="WAC / mortgage → scenarios",
            text=_calib["key"], textposition="top center",
            marker=dict(size=10, color=ORANGE),
            customdata=_calib["refi_incentive"],
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
                title="Refi incentive (documentation, bp)  (+ ITM / − OTM)",
            ),
            yaxis=dict(**AXIS_STYLE, title="Illustrative CPR % (agency knots)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor=BG2),
            title=dict(
                text="ALCO reference only — live pricing uses logistic CPR × seasoning",
                font=dict(size=12, color=DIM),
            ),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        with st.expander("Historical regimes (ALCO relevance)"):
            st.dataframe(
                historical_regimes_dataframe(),
                use_container_width=True,
                hide_index=True,
            )

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



# ── TAB: Docs ─────────────────────────────────────────────────────────────────
with tab_docs:
    st.markdown(DOCS_MARKDOWN)
    st.markdown("<p class='section-label'>Required CSV columns</p>", unsafe_allow_html=True)
    st.code("name, side, notional, coupon_pct, instrument_type, maturity_years", language=None)
    st.caption("Optional: payment_freq, repricing_years, wac, wam_months, oas, anchor_tenor, mbs_level, current_rate, book, currency")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<p style='color:{DIM};font-size:10px;text-align:center'>"
    f"BCBS 368 (April 2016) · Interest Rate Risk in the Banking Book · "
    f"Pillar 2 · Supervisory outlier: |ΔEVE| > 15% Tier 1 Capital</p>",
    unsafe_allow_html=True,
)
