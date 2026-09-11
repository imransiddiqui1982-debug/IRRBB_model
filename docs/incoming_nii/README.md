# alm_engine

Banking-book interest rate risk engine. Balance sheet in, hedge with its
accounting designation out.

```
cash flows (monthly)  ->  EVE, six BCBS scenarios
                      ->  KR01 on a tradeable key grid
                      ->  hedge solved against that grid
                      ->  each trade tagged fair-value / cash-flow
```

## Run

```bash
python3 demo.py
python3 -m pytest test_alm_engine.py -q     # 28 tests
```

## Four design rules

**Cash flows are fine, key rates are coarse.** Flows are generated monthly.
The key grid is 1/2/3/5/7/10y — standard swap tenors. A key you cannot trade
is a row in the hedge matrix with no column, and adjacent keys are near
collinear, which makes the solve blow up into huge offsetting notionals.

**Behavioural models re-run inside every bump.** `MBS.flows()` reads the
mortgage rate off the curve on each call, so prepayment responds when you bump.
Freezing the schedule overstates KR01 by ~25% on a mortgage book, concentrated
at exactly the tenors you would hedge.

**Scenario magnitudes are arguments, never constants.** `bcbs_shock(...,
r_par=, r_short=, r_long=)`. The Committee recalibrates and currencies differ.

**Fair value hedges go to P&L; cash flow hedges go to OCI.** These are not
interchangeable. `designate()` tags each trade and flags the NMD problem —
demand deposits generally cannot be the hedged item in a fair value hedge, so
the largest behavioural exposure sits outside the framework.

## Instruments

| Class | Notes |
|---|---|
| `FixedBullet` | Coupon plus principal at maturity |
| `FixedAmortising` | Level payment; KR01 is a ramp, not a spike |
| `Floater` | Risk stops at next reset. Fixing is snapshotted, not re-read |
| `NMD` | `core = balance x stable_pct x (1-beta)`, capped; exponential decay |
| `MBS` | Prepayment S-curve driven off `anchor_tenor` |
| `Swap` | Short fixed bond + long floater |

## Invariants the tests enforce

- Tent weights sum to 1.0 at every maturity (partition of unity)
- `kr01().sum() == dv01()` within 2%
- Par bond prices to 100, par swap to zero
- 10y floater DV01 < 1/15th of a 10y bond
- MBS convexity negative, bullet positive
- MBS KR01 drifts >30% across states; a bullet's does not
- Live CPR gives lower KR01 than a frozen schedule

## Known gaps

Deterministic curve, no Monte Carlo OAS. No basis or credit spread risk. No
swaptions, so convexity cannot actually be hedged — only measured. NII is not
implemented; EVE only. Hedge accounting effectiveness is not tested.

Not accounting or investment advice.

## Hedge effectiveness testing

`hedge_effectiveness.py` adds ASC 815 regression and dollar-offset testing
for any `Instrument` pair (hedged item vs. hedge instrument).

```python
from alm_engine import Curve, FixedBullet, Swap
from hedge_effectiveness import full_report

loan = FixedBullet("10y fixed mortgage", 100.0, coupon=0.06, years=10, freq=2)
swap = Swap("10y pay-fixed swap", 100.0, side=1, years=10, fixed_rate=0.04, freq=2)
print(full_report(loan, swap, Curve.flat(0.04)))
```

Runs four tests: dollar-offset and regression, each prospectively (across
the six BCBS scenarios) and retrospectively (backtested against a simulated
or actual historical rate path).

**Why both methods, and why regression is usually the better one to trust.**
On a well-matched 10y loan / 10y swap pair, the prospective dollar-offset
ratio comes out at 136% -- a FAIL -- while regression shows slope -0.915,
R-squared 0.9995 -- a clear PASS. This isn't a bug: with only six scenario
observations, several of which nearly cancel (net sum close to zero), the
dollar-offset ratio is dividing two small numbers and is numerically
unstable. Each individual scenario offset is actually ~90%. Regression uses
the full six-point relationship rather than a single cumulative ratio, which
is why FASB guidance treats it as the more robust ongoing test. Both are
implemented because auditors typically want to see both.

**Sign convention.** Slope near -1 and dollar-offset ratio near +100% both
indicate a properly offsetting hedge. A hedge that fails with a *positive*
slope (moving the same direction as the hedged item, not against it) is a
strong diagnostic of a curve-shape mismatch -- exactly what the deliberately
undersized single-swap-against-an-NMD test case in
`test_hedge_effectiveness.py` demonstrates.

Not accounting advice -- confirm thresholds and methodology with your
auditor.

## NII engine (`nii_engine.py`)

Distinct mechanism from EVE, not just a different report: EVE discounts every
cash flow at its own time (no horizon, no reinvestment). NII has no
discounting anywhere — it accrues interest income/expense month by month
over a 12- or 24-month horizon under a constant (or static) balance sheet.

```python
from alm_engine import Curve, FixedBullet, Floater, NMD, MBS
from nii_engine import NIIPortfolio, BCBS_NII_SCENARIOS, US_NII_SCENARIOS

book = NIIPortfolio([...], Curve.flat(0.04), horizon_months=12)
book.grid(BCBS_NII_SCENARIOS)        # Basel: 2 scenarios, 12m, constant BS
book.grid(US_NII_SCENARIOS)          # US practice: 8 shocks + 2 ramps
```

**BCBS 368 compliance** — `BCBS_NII_SCENARIOS` is exactly the two parallel
shocks (±200bp), instantaneous, 12-month horizon, constant balance sheet.
Basel sets **no mandated NII threshold** — `limit_status()` defaults to
showing % of Tier 1 (the EU/EBA's 5% convention, included as a reference
point, not applied as if it were a Basel rule) alongside % of base NII, since
the two can tell very different stories (see the earlier worked example:
−17.4% of base NII vs −2.0% of Tier 1 for the identical dollar move).

**US extension** — `US_NII_SCENARIOS` adds ±100/300/400bp and two 12-month
linear ramps, reflecting practice (not regulation — the US has no IRRBB
rule; see the 2010 Interagency Advisory and FFIEC handbook). `horizon_months`
and `constant_balance_sheet` are both configurable per portfolio.

**Where beta lives, and where it doesn't.** `NMD` gained `current_rate` and
`beta_down` fields for NII use only — `beta` drives NII (how fast deposit
cost reprices), `decay`/`wal_cap` drive EVE slotting only. A regression
test (`test_nmd_decay_and_wal_do_not_affect_nii`) confirms two NMDs with
identical beta but very different decay show identical NII sensitivity —
this is the exact separation flagged as "the most common modelling error"
earlier in this build, now enforced as a permanent test rather than a
one-off observation.

**A genuine finding from the demo, not a design goal:** static balance
sheet (no reinvestment) doesn't just understate sensitivity — it can flip
the *sign*. On the demo's balance sheet, constant-balance-sheet NII falls
−$1.56m under +200bp (liability-sensitive, as the earlier gap/EVE analysis
of a similar book predicted). Static NII *rises* +$0.93m under the same
shock, because a maturing costly CD simply vanishes from the expense line
instead of being correctly replaced at the new, higher market rate. A bank
relying on static modelling here wouldn't just misjudge the *size* of its
rate risk — it would misjudge the *direction*.

**Known limitations**, consistent with the rest of this engine's approach
of stating gaps rather than hiding them: `Floater`/`Swap` support only a
single reset date (matching the underlying EVE classes), not a recurring
schedule; `NMD` NII models rate repricing via beta but not volume runoff
under stress (non-core balances leaving during a shock) — that's a genuine
next addition, not currently in scope; reinvestment pricing uses one spread
assumption per instrument type rather than a full replacement-product curve.
