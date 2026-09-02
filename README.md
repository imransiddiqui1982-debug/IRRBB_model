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
4. Review tabs: **NMD Refinement**, **LCR / NSFR**, **All Scenarios**, **Repricing Gap**, **Yield Curve**

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
| `instrument_type` | Yes | `bullet_fixed` \| `bullet_floating` \| `amortising` \| `demand_deposit` |
| `maturity_years` | Yes | Contractual maturity (years) |
| `payment_freq` | No | Payments per year (default 2) |
| `repricing_years` | No | Next reset for floaters |

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

**NII method:** floating / demand instruments only:  
`ΔNII = ± notional × shock_bp(bucket) / 10,000`.

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
│   ├── cashflows.py       # Instrument CF schedules
│   ├── time_buckets.py    # 19 BCBS buckets
│   ├── scenarios.py       # Six prescribed shocks
│   ├── yield_curve.py     # Discounting
│   ├── market_curve.py    # Live SOFR + IRS mid curve
│   ├── calculator.py      # EVE, NII, DV01, IRS suggestions
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

### BCBS shock grid (bp at O/N, 1Y, 2Y, 5Y, 10Y, 20Y)

| Scenario | O/N | 1Y | 2Y | 5Y | 10Y | 20Y |
|----------|-----|----|----|----|-----|-----|
| Parallel Up | +200 | +200 | +200 | +200 | +200 | +200 |
| Parallel Down | −200 | −200 | −200 | −200 | −200 | −200 |
| Steepener | −100 | −75 | −50 | 0 | +100 | +150 |
| Flattener | +100 | +75 | +50 | 0 | −100 | −150 |
| Short Up | +250 | +200 | +150 | +75 | 0 | 0 |
| Short Down | −250 | −200 | −150 | −75 | 0 | 0 |

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
