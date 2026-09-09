# IRRBB Production Engine (BCBS 368)

[![CI](https://github.com/imransiddiqui1982-debug/IRRBB_model/actions/workflows/ci.yaml/badge.svg)](https://github.com/imransiddiqui1982-debug/IRRBB_model/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end **Interest Rate Risk in the Banking Book (IRRBB)** engine with:

- **EVE / NII** under the six BCBS 368 shock scenarios  
- **19-bucket** cash-flow slotting and full discounting  
- **NMD behavioural deposit modelling** (HP filter + beta + BCBS caps)  
- **LCR / NSFR** from U.S. disclosure-style workbook inputs  
- **Live SOFR + USD IRS mid** yield curve (optional)  
- **DV01 gap** and indicative IRS hedge suggestions  
- **Key rate duration (KR01)** on the tradeable 1/2/3/5/7/10Y swap grid  
- **Option-adjusted MBS / whole-loan prepay** (per-instrument balance-sheet WAC)  

---

## Quick start

```bash
git clone https://github.com/imransiddiqui1982-debug/IRRBB_model.git
cd IRRBB_model
python -m pip install -r requirements.txt

# Interactive dashboard (recommended)
python -m streamlit run app.py

# CLI run (sample balance sheet)
python main.py --tier1 500

# Tests
python -m pytest tests/ -q
```

Open the Streamlit URL (default `http://localhost:8501`). In the sidebar:

1. Enable **Load calibrated balance sheet sample**  
2. Enable **Refine NMD deposits** + **Load calibrated deposit model sample**  
3. Enable **Use live SOFR + USD IRS mid curve** (optional)  
4. Review tabs: **NMD Refinement**, **LCR / NSFR**, **All Scenarios**, **Repricing Gap**, **ALCO / KR01 / Hedges**, **Yield Curve**

---

## Run on Streamlit Community Cloud (from GitHub)

GitHub stores the code; **Streamlit Community Cloud** runs the dashboard and gives you a public URL (e.g. `https://irrbb-model.streamlit.app`).

1. Sign in at **[share.streamlit.io](https://share.streamlit.io)** with your **GitHub** account (`imransiddiqui1982-debug`).
2. Click **Create app** → **From existing repo**.
3. Select repository **`imransiddiqui1982-debug/IRRBB_model`**, branch **`main`**, main file path **`app.py`**.
4. Click **Deploy**. Streamlit installs `requirements.txt` and starts the app (first build may take 2–5 minutes).
5. After deploy, open the app URL. The sidebar defaults load the calibrated balance sheet and deposit workbook from the `data/` folder in the repo.

**Notes**

- No API keys are required. The live yield curve uses public NY Fed / BlueGamma endpoints; if they fail, the app falls back to the bundled `data/live_curve_cache.json`.
- To redeploy after a `git push`, Streamlit Cloud rebuilds automatically (or use **Reboot app** in the app settings).
- Optional: under **Advanced settings**, set Python **3.11** if you want to pin the runtime.

---

## What this engine calculates

| Module | Output | Standard |
|--------|--------|----------|
| **EVE** | Δ Economic Value of Equity by scenario | BCBS 368 §119–121 |
| **NII** | Δ Net Interest Income (1Y horizon) | BCBS 368 §109–112 |
| **Outlier** | \|ΔEVE\| / Tier 1 ≥ 15% | BCBS 368 §99 |
| **NMD** | Core sticky / rate-sensitive / non-core | BCBS 368 Annex 1 caps |
| **LCR** | HQLA / net 30-day outflows | Basel LCR / 12 CFR 249.91 style |
| **NSFR** | ASF / RSF | Basel NSFR / 12 CFR 249.131 style |
| **DV01 gap** | Bucket asset vs liability DV01 + IRS hedges | Risk management |
| **Key rate duration** | KR01 on 1/2/3/5/7/10Y swap grid + hedge tags | Treasury / hedging |

---

## Data input files

Place these under `data/` (samples are included; replace with your bank data).

### 1. Balance sheet CSV — `data/balance_sheet_template.csv`

One row per instrument:

| Column | Required | Description |
|--------|----------|-------------|
| `name` | Yes | Instrument label |
| `side` | Yes | `asset` or `liability` |
| `notional` | Yes | USD millions |
| `coupon_pct` | Yes | Annual coupon % |
| `instrument_type` | Yes | `bullet_fixed` \| `bullet_floating` \| `amortising` \| `mbs` \| `whole_loan` \| `demand_deposit` |
| `maturity_years` | Yes | Contractual maturity (years) |
| `payment_freq` | No | Payments/year (default 2) |
| `repricing_years` | No | Next reset (floating / NMD) |
| `prepay_enabled` | No | `1`/`Y` enables CPR on amortising (auto-on if name contains "mortgage") |
| `base_cpr` | No | Legacy fixed CPR override; blank → option-adjusted path for MBS/whole loans |
| `age_months` / `pool_age_months` | No | Seasoning for OA ramp / PSA |
| `wac`, `wam_months`, `anchor_tenor`, `spread_to_curve`, `oas` | No | Option-adjusted MBS / whole-loan terms (Steps A–C) |
| `mbs_level` | No | For MBS: `ginnie` (L1) \| `agency` (L2A) \| `private` (not HQLA) |
| `hqla_level`, `nsfr_rsf_factor` | No | Static LCR/NSFR tags (never from CPR); whole loans → not HQLA / ~65% RSF |
| `credit_spread` | No | Whole-loan placeholder only (not in IRRBB pricing) |
| `encumbered` | No | `1`/`Y` → NSFR RSF 100% if maturity >1Y |

**MBS HQLA / NSFR:** Ginnie Mae → LCR Level 1 (0% haircut), NSFR RSF 5%. Fannie/Freddie agency → LCR Level 2A (15% haircut), NSFR RSF 15%. Private-label RMBS → not HQLA; NSFR RSF ~85%. Whole-loan residential mortgages ≥1Y → RSF 65%, **never HQLA**. Encumbered >1Y → RSF 100%. Residual maturity is **contractual** (CPR never feeds LCR/NSFR).

**Mortgage / MBS CPR:** Option-adjusted pricing (Steps A→B→C). **Step A** anchors the mortgage rate to live **FRED PMMS 30Y** (`apply_pmms_anchor`). **Step B** logistic parameters default from `data/calibrated_prepayment_params.json` — fit on Freddie SFLLD samples + PMMS history via `python scripts/calibrate_from_freddie_pmms.py` (place sample files in `data/freddie/`). OAS is discount-only.

**Tip:** When NMD is enabled, leave demand deposits out of the CSV (or they are replaced). Use the CSV for assets + wholesale funding only.

### 2. Deposit + liquidity workbook — `data/us_lcr_nsfr_deposit_model.xlsx`

Sheets used by the engine:

| Sheet | Purpose |
|-------|---------|
| **Customer Deposits** | Customer-level balances (36 months) + segment, product, rates |
| **Deposit Rates** | Monthly product rates by `rate_code` (for beta) |
| **Market Rates** | Monthly market / policy rate |
| **BCBS Segment Caps** | Max core % and WAL by segment |
| **HQLA Stock** | Level 1 / 2A / 2B amounts + haircuts |
| **LCR Cash Outflows / Inflows** | Disclosure-style LCR lines (`auto_from_nmd=Y` for deposits) |
| **NSFR ASF / RSF** | Stable funding lines |
| **Regulatory Parameters** | Tier 1/2, caps, Category IV tailoring |

Also provided:

- `data/sample_balance_sheet.csv` — built-in sample with demand deposits (CLI / tests)  
- `data/live_curve_cache.json` — last fetched SOFR/IRS curve (auto-updated)

Generate / refresh templates:

```bash
python scripts/calibrate_bank_data.py
python scripts/extend_v2_liquidity_sheets.py
python scripts/create_deposit_data_template.py
```

---

## How to use each module

### A. IRRBB — NII & EVE

1. Upload **balance sheet CSV** (or use calibrated sample).  
2. Optionally refine **NMD deposits** so behavioural tenors feed liabilities.  
3. Set **Tier 1 capital** in the sidebar.  
4. Open **All Scenarios** / **Scenario Detail** / **EVE Waterfall**.

**EVE method:** schedule CFs → slot into 19 BCBS buckets → discount on base curve and shocked curve →  
`ΔEVE = ΔPV(assets) − ΔPV(liabilities)`.

**NII method:** floating / demand instruments:  
`ΔNII = ± notional × shock_bp(bucket) / 10,000`.  
CPR mortgages: extra 1Y prepayments under the shock reinvest at the shocked short rate vs lost contractual coupon.

**Mortgage prepayment:** live OA path in `src/mbs_pricing.py`; calibrate Step-B params with `python -m src.calibrate_prepayment` → `data/calibrated_prepayment_params.json`. Optional HF Chronos via `requirements-hf.txt`.

**Curves:** with live curve on: **0–12M SOFR**, **1Y–10Y USD SOFR IRS mid**; then BCBS shocks are added.

### B. NMD deposit modelling

1. Fill **Customer Deposits** (monthly balances) + **Deposit Rates** + **Market Rates**.  
2. Enable NMD in Streamlit (or call the API below).  
3. Engine produces per segment: stable %, beta, core sticky, rate-sensitive, non-core, WAL.  
4. Instruments replace flat demand deposits for EVE/NII and auto-fill LCR/NSFR deposit rows.

```python
from src.nmd_refinement import refine_nmd_deposits, merge_nmd_into_balance_sheet
from src.load_balance_sheet import load_instruments_from_csv
from src.calculator import IRRBBCalculator
from src.scenarios import SCENARIOS

assets, liabilities = load_instruments_from_csv("data/balance_sheet_template.csv")
nmd = refine_nmd_deposits("data/us_lcr_nsfr_deposit_model.xlsx", deposit_name="Bank NMD")
assets, liabilities = merge_nmd_into_balance_sheet(assets, liabilities, nmd)
calc = IRRBBCalculator(assets, liabilities, tier1_capital=500)
results = calc.run_all(SCENARIOS)
```

### C. LCR & NSFR

1. Fill **HQLA Stock**, non-deposit outflows/inflows, **NSFR RSF** (and capital in Regulatory Parameters).  
2. Leave deposit outflow / ASF rows with `auto_from_nmd = Y`.  
3. Upload the same workbook; open **LCR / NSFR** tab.

```python
from src.liquidity_workbook import load_liquidity_workbook
from src.liquidity_ratios import compute_liquidity_from_workbook

wb = load_liquidity_workbook("data/us_lcr_nsfr_deposit_model.xlsx", nmd_result=nmd)
liq = compute_liquidity_from_workbook(wb)
print(liq.summary)
```

### D. DV01 gap & IRS hedges

Open **Repricing Gap**: asset DV01 (above zero), liability DV01 (below), net DV01 line, plus payer/receiver swap suggestions by bucket.

### D2. Key rate duration (treasury / ALCO)

Open **ALCO / KR01 / Hedges** for three layers:

1. **Board / ALCO** — EVE vs 15%/10% Tier 1 with key-rate **drivers**, hedge package grid  
2. **Treasury** — base KR01, shock vector, attribution `ΔEVE ≈ −KR01 × shock_bp`, **dynamic KR01 under all 6 shocks**  
3. **Hedge playbook** — EVE relief per $100m 2Y/5Y/10Y pay-fixed, IRS tickets, why that tenor

```python
from src.calculator import IRRBBCalculator

calc = IRRBBCalculator(assets, liabilities, tier1_capital=500, yield_curve=curve)
pack = calc.treasury_alco_pack()   # limits, attribution, scenario_kr01, hedges
```

### E. Live yield curve

```python
from src.market_curve import get_live_yield_curve
from src.calculator import IRRBBCalculator

curve, snap = get_live_yield_curve()
calc = IRRBBCalculator(assets, liabilities, tier1_capital=500, yield_curve=curve)
```

Sidebar: **Refresh live curve** to re-pull NY Fed SOFR + USD IRS mids.

---

## Project structure

```
IRRBB_model/
├── app.py                 # Streamlit production dashboard
├── main.py                # CLI entry point
├── requirements.txt
├── README.md
├── data/
│   ├── balance_sheet_template.csv
│   ├── sample_balance_sheet.csv
│   ├── us_lcr_nsfr_deposit_model.xlsx   # NMD + LCR/NSFR workbook
│   └── live_curve_cache.json
├── src/
│   ├── cashflows.py       # Instrument CF schedules (+ CPR amortisation)
│   ├── prepayment.py      # PSA / S-curve helpers (+ optional HF Chronos)
│   ├── mbs_pricing.py     # OA MBS / whole-loan Steps A→B→C
│   ├── calibrate_prepayment.py  # Fit Step-B params to loan-level history
│   ├── cpr_calibration.py # ALCO reference S-curve (documentation)
│   ├── time_buckets.py    # 19 BCBS buckets
│   ├── scenarios.py       # Six prescribed shocks
│   ├── yield_curve.py     # Discounting
│   ├── market_curve.py    # Live SOFR + IRS mid curve
│   ├── calculator.py      # EVE, NII, DV01, IRS suggestions
│   ├── key_rate_duration.py  # KR01 tent grid + hedge tags
│   ├── nmd_refinement.py  # Behavioural NMD engine
│   ├── lcr_calculator.py / nsfr_calculator.py / liquidity_*.py
│   ├── load_balance_sheet.py / balance_sheet.py / file_io.py / plots.py
├── scripts/               # Template builders & calibration
├── tests/                 # pytest suite
└── .github/workflows/ci.yaml
```

---

## Methodology (summary)

### EVE

```
PV_base(i)    = Σ_k  CF_i[k] / (1 + r_base[k])^t_k
PV_shocked(i) = Σ_k  CF_i[k] / (1 + r_shocked[k])^t_k
ΔEVE = Σ_assets (PV_shocked − PV_base) − Σ_liabilities (PV_shocked − PV_base)
```

### NII (1-year)

```
ΔNII = Σ floating assets notional × shock/10000
     − Σ floating liabilities notional × shock/10000
```

### NMD bifurcation

```
stable %     ← HP filter on aggregate balances (capped by BCBS segment max)
beta         ← regression of deposit rates vs market rates
core sticky  = balance × stable% × (1 − beta)   → WAL tenor
rate sens.   = balance × stable% × beta         → short reset
non-core     = balance × (1 − stable%)          → ~1M volatile
```

### BCBS / key-tenor shock grid (bp at 0.25Y, 1Y, 2Y, 5Y, 7Y, 10Y)

| Scenario | 0.25Y | 1Y | 2Y | 5Y | 7Y | 10Y |
|----------|-------|----|----|----|----|-----|
| Parallel Up | +200 | +200 | +200 | +200 | +200 | +200 |
| Parallel Down | −200 | −200 | −200 | −200 | −200 | −200 |
| Steepener | −175 | −122 | −65 | +40 | +76 | +108 |
| Flattener | +220 | +167 | +110 | +5 | −29 | −63 |
| Short Up | +282 | +234 | +182 | +86 | +49 | +25 |
| Short Down | −282 | −234 | −182 | −86 | −49 | −25 |

Shocks are interpolated from these pillars onto the 19 BCBS bucket midpoints for EVE/NII.

---

## Requirements

- Python 3.10+  
- See `requirements.txt` (numpy, pandas, scipy, openpyxl, streamlit, plotly, pytest, …)  
- Network access for live SOFR / IRS curve (optional; cache used offline)

---

## License & references

- Basel Committee on Banking Supervision. *Interest rate risk in the banking book.* BCBS 368, April 2016.  
  https://www.bis.org/bcbs/publ/d368.pdf  
- U.S. LCR / NSFR public disclosure templates (12 CFR 249.91 / 249.131)
