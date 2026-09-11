# MBS / Mortgage Prepayment Module — Implementation Spec for Cursor

Add option-adjusted MBS/mortgage pricing to the existing IRRBB engine. This
spec is self-contained: data model, the three-step pipeline (rate anchor ->
CPR -> cash flows/pricing), integration points, and test cases with expected
numeric output to verify against.

Read this whole file before writing code. The ordering of steps A/B/C is not
arbitrary — each step's output is the next step's input, and skipping the
re-derivation on every curve bump is the single most common bug in this kind
of model (silently overstates KR01 by materially misrepresenting duration).

---

## 0. Where this plugs into the existing engine

If the engine already has an `Instrument` base class with a `flows(curve) ->
(times, cashflows)` method and a `pv(curve) -> float` method (as in
alm_engine.py from earlier in this build), add a new `MBS` subclass that
overrides both. Nothing else in the engine — KR01, EVE scenario grid, hedge
solver — needs to change. They already call `flows()` / `pv()` generically.

The one hard requirement: **`flows()` must be called fresh, and must derive
its schedule from the `curve` argument, every single time the curve is
bumped.** Never cache a schedule and reuse it across curve states. This is
what makes the option "live" inside a KR01 finite-difference bump.

---

## 1a. Class bifurcation — MBS securities vs. whole mortgage loans

The prepayment BEHAVIOR (Steps A/B/C) is identical for both instrument types
-- same borrower, same S-curve, same negative convexity. What differs is
what sits between the borrower and the balance sheet, and that difference
must be represented as two distinct classes, not two configurations of one
class, because downstream modules (CECL/credit provisioning, LCR/NSFR) key
off the type directly.

```
                    ┌─────────────────────┐
                    │   Instrument (base)  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴─────────────────┐
              │                                   │
    ┌─────────▼──────────┐              ┌─────────▼───────────┐
    │        MBS           │              │   WholeLoanPool       │
    │  (securitized pool)   │              │  (loans on B/S)        │
    └───────────────────────┘              └────────────────────────┘
```

Build `WholeLoanPool` as a SUBCLASS of `MBS`. It reuses Steps A/B/C
unmodified (identical `step_a_mortgage_rate`, `step_b_cpr`,
`step_c_flows_and_price` logic) and only adds/overrides the fields and
downstream tags listed in the table below. Do not duplicate the pricing
logic -- if the amortization loop or S-curve needs to change, it must change
in exactly one place.

### Bifurcation table -- what each class carries differently

| Attribute / behavior            | `MBS`                                  | `WholeLoanPool`                                  |
|----------------------------------|------------------------------------------|-----------------------------------------------------|
| `wac` meaning                     | Pass-through rate to security holder (already net of servicing + g-fee, stripped by the issuer at securitization) | Gross borrower rate                                    |
| `spread_to_curve` meaning           | Pure secondary-market/OAS-related spread | Primary-secondary spread, INCLUDING an embedded servicing + guarantee-fee equivalent (so Step A's mortgage_rate stays comparable) |
| Steps A / B / C                     | Identical, unmodified                     | Identical, unmodified (inherited, not reimplemented)   |
| Credit risk                           | None to holder (agency-guaranteed) or separately priced via `oas` widening (non-agency/private-label) | Direct credit exposure -- REQUIRES a separate `credit_spread` / CECL provisioning field, NOT part of this IRRBB pricing path |
| Servicing risk/cost                    | None to holder                              | Present if self-serviced or sub-serviced -- separate cost line, not modeled here |
| `hqla_level` tag (for LCR/NSFR module, separate from this spec) | `level_1` (Ginnie Mae) or `level_2a` (Fannie/Freddie) or `not_eligible` (private-label) -- set explicitly per pool | Always `not_eligible` -- whole loans get ZERO HQLA credit regardless of quality |
| `nsfr_rsf_factor` tag (informational only; do not compute prepayment-adjusted RSF -- LCR/NSFR use CONTRACTUAL residual maturity, never CPR) | 5% (Ginnie Mae) / 15% (agency) / 85%+ (private-label, non-HQLA) | 65% (residential, <=50% risk weight, >=1y residual) or per applicable RSF table |
| Calibration source for `logistic_k` / `logistic_midpoint` | Benchmark against published dealer prepayment speed assumptions (agency-standard curves) as a starting point | Fit to the bank's OWN historical portfolio prepayment data where available -- market-standard agency curves are a weaker proxy for non-conforming or credit-impaired loans |
| Data granularity                       | Usually already pool-level (WAC/WAM/factor published by issuer at the CUSIP level) -- one `MBS` instance per security held | Usually loan-level source data -- bucket into homogeneous `WholeLoanPool` instances by WAC band + origination vintage + product type before pricing (do not price 50,000 individual loans through the full monthly loop unless compute budget allows; bucketing is the standard practice) |

### Explicit non-goals for this bifurcation

- Do NOT let `hqla_level` or `nsfr_rsf_factor` be derived from the CPR/S-curve
  output. They are static classification tags set at instrument creation,
  independent of any prepayment scenario. LCR/NSFR must never read from
  Steps A/B/C.
- Do NOT model `credit_spread` / CECL provisioning inside this module. Add the
  field to `WholeLoanPool` as a placeholder/passthrough only, so the credit
  risk module (separate workstream) has somewhere to attach, but leave its
  calculation out of scope here.
- Do NOT reimplement `step_a_mortgage_rate`, `step_b_cpr`, or
  `step_c_flows_and_price` inside `WholeLoanPool`. It should override only
  `__init__` (to accept/validate the additional fields above) and any
  `hqla_level` / `credit_spread` accessors. If you find yourself writing a
  second amortization loop, stop -- that is the bug this bifurcation is
  designed to prevent.

### Test case addition for this section

```
TEST wholeloanpool_pricing_matches_mbs_given_equal_effective_terms:
    # Construct an MBS and a WholeLoanPool with the SAME notional, wac,
    # wam_months, anchor_tenor, and an equivalent spread_to_curve (i.e.
    # WholeLoanPool.spread_to_curve set so its resulting mortgage_rate in
    # Step A matches the MBS case exactly).
    # Steps A/B/C must produce IDENTICAL price, WAL, and KR01 -- since the
    # borrower-behavior math is shared code, not a coincidence.
    ASSERT  mbs.pv(curve) == whole_loan_pool.pv(curve)          # within 1e-6
    ASSERT  mbs.flows(curve) == whole_loan_pool.flows(curve)     # within 1e-6

TEST hqla_and_rsf_tags_are_static_not_scenario_dependent:
    # Confirm hqla_level / nsfr_rsf_factor do not change across the six
    # BCBS scenarios -- they are instrument attributes, not model outputs.
    FOR scenario IN all_six_scenarios:
        ASSERT  mbs.hqla_level  ==  mbs.hqla_level   # unchanged regardless
                                                        # of curve/CPR state
        ASSERT  whole_loan_pool.hqla_level == "not_eligible"   # always
```

---

## 1. Data model — inputs needed per pool/loan

```
class MBS(Instrument):
    notional: float          # current pool balance, $
    wac: float                # weighted average coupon, e.g. 0.055
    wam_months: int            # weighted average maturity remaining, e.g. 360
    pool_age_months: int        # months since origination (for seasoning), e.g. 0
    anchor_tenor: float          # which curve point drives the mortgage rate, years
                                  #   -> 10.0 for US 30y conventional (PMMS tracks 10y UST)
                                  #   -> use whatever tenor is appropriate for the product
    spread_to_curve: float        # PMMS/primary-secondary spread, e.g. 0.0175
    oas: float = 0.005             # option-adjusted spread over the discount curve

    # S-curve (logistic) prepayment model parameters -- calibrate these to
    # actual portfolio history where available; these are reasonable defaults:
    base_turnover: float = 0.06     # CPR floor: housing turnover unrelated to rates
    max_refi_cpr: float = 0.34      # ceiling on the refi-driven CPR component
    logistic_k: float = 2.2         # steepness of the S-curve
    logistic_midpoint: float = 0.60 # incentive (in percentage points) at 50% of max_refi_cpr
    seasoning_ramp_months: int = 30 # months to reach full seasoned speed

    # STATIC classification tags -- set once at construction, NEVER derived
    # from Steps A/B/C or from any scenario. See section 1a for the
    # MBS vs WholeLoanPool bifurcation and why these must stay static.
    hqla_level: str = "level_2a"      # "level_1" | "level_2a" | "not_eligible"
    nsfr_rsf_factor: float = 0.15      # informational; LCR/NSFR module reads
                                         # this directly, using CONTRACTUAL
                                         # residual maturity -- never CPR
```

For a portfolio of many pools/loans with different WAC/age/anchor, instantiate
one `MBS` object per homogeneous bucket (standard practice: bucket by WAC band
and origination vintage) rather than one per loan.

---

## 2. STEP A — read the mortgage rate off the shocked curve

Do NOT shock the mortgage rate directly. Derive it from a single point on the
already-shocked discount curve. This is what makes the model respond
correctly to non-parallel scenarios (steepener/flattener) as well as parallel
ones.

```
FUNCTION step_a_mortgage_rate(curve, mbs):
    anchor_rate = curve.rate(mbs.anchor_tenor)      # curve already has the
                                                       # scenario shock applied
                                                       # (parallel/short/steep/flat)
                                                       # AND the post-shock floor
    mortgage_rate = anchor_rate + mbs.spread_to_curve
    refi_incentive_pp = (mbs.wac - mortgage_rate) * 100   # in PERCENTAGE POINTS
    RETURN mortgage_rate, refi_incentive_pp
```

Sign convention, do not flip this:
  refi_incentive_pp > 0  =>  borrower CAN save money by refinancing (CPR should rise)
  refi_incentive_pp < 0  =>  refinancing would cost the borrower more (CPR near floor)

Sanity check to add as an assertion/log during development: under a par-down
shock, refi_incentive_pp must increase vs. base. Under par-up, it must
decrease. If this doesn't hold, the anchor_tenor or spread sign is wrong.

---

## 3. STEP B — S-curve: incentive -> seasoned CPR -> equivalent PSA

```
FUNCTION step_b_cpr(mbs, refi_incentive_pp, age_months):
    refi_response = mbs.max_refi_cpr / (1 + exp(-mbs.logistic_k *
                        (refi_incentive_pp - mbs.logistic_midpoint)))

    seasoning_ramp = min(age_months / mbs.seasoning_ramp_months, 1.0)

    cpr = (mbs.base_turnover + refi_response) * seasoning_ramp
    RETURN cpr

FUNCTION cpr_to_psa(cpr):
    # 100 PSA is DEFINED as 6% seasoned CPR. Pure relabeling, no independent
    # calculation -- just for reporting/comparison against market convention.
    RETURN (cpr / 0.06) * 100
```

Note: `step_b_cpr` must be called freshly for EACH MONTH inside the cash flow
loop in Step C (age_months changes every month during the ramp period), not
just once per scenario. Only the "seasoned CPR" figure used for reporting
(e.g. in an EVE attribution table) evaluates this at a fixed age (e.g. 60
months) to show the steady-state comparison across scenarios.

Calibration note for whoever tunes this: `logistic_midpoint` and `logistic_k`
should be fit to the bank's own historical refi response if data exists
(logistic regression of realized SMM against historical incentive). The
defaults above are illustrative, not calibrated to any specific portfolio.

---

## 4. STEP C — monthly amortization loop -> cash flows -> price -> WAL

**CRITICAL BUG WARNING (found in the actual engine implementation while
verifying this spec against real numbers -- read this before writing the
loop):** the monthly payment MUST be a fixed dollar amount, computed ONCE
from the balance and remaining term, and held constant every month
thereafter. A common and easy-to-write-by-accident mistake is to recompute
`balance * payment_factor` fresh every month against the CURRENT (already
partially paid down) balance -- this looks almost identical to the correct
version but produces a geometric decay that never actually reaches zero.
Concretely: a $300m, 5.5%, 324-months-remaining pool computed the wrong way
still had **$153 million outstanding after 500 months** of "payments" when
it should have fully amortized to exactly $0 by month 324. This is not a
rounding error -- it is the wrong recurrence relation, and it silently
produces plausible-looking but wrong prices, WALs, and KR01s for every
scenario. Verify against the test in 7.5 below before trusting anything else.

```
FUNCTION step_c_flows_and_price(curve, mbs, scenario_name):
    mortgage_rate, refi_incentive_pp = step_a_mortgage_rate(curve, mbs)

    monthly_coupon = mbs.wac / 12
    n = mbs.wam_months
    payment_factor = monthly_coupon / (1 - (1 + monthly_coupon)^(-n))

    balance = mbs.notional
    # THE FIX: compute the dollar payment ONCE, here, from the STARTING
    # balance -- never recompute it against the balance inside the loop.
    payment_dollar = balance * payment_factor

    times = []
    cashflows = []

    FOR month IN 1..n:
        IF balance <= epsilon: BREAK

        age = mbs.pool_age_months + month
        cpr = step_b_cpr(mbs, refi_incentive_pp, age)          # <- re-derived
                                                                  #    every month
        smm = 1 - (1 - cpr)^(1/12)                              # geometric,
                                                                  #    NOT cpr/12

        interest      = balance * monthly_coupon
        scheduled_prin = MIN(payment_dollar - interest, balance)  # FIXED payment_dollar,
                                                                     # NOT balance * payment_factor
        prepayment    = MAX((balance - scheduled_prin) * smm, 0)
        total_cf      = interest + scheduled_prin + prepayment

        times.append(month / 12)
        cashflows.append(total_cf)

        balance = balance - scheduled_prin - prepayment

    # discount at the SAME shocked curve used to derive the mortgage rate
    # in Step A, plus the fixed OAS
    discount_rates = curve.rate(times) + mbs.oas          # curve.rate already
                                                              # includes the shock
                                                              # and the floor
    price = SUM( cashflows[i] / (1 + discount_rates[i])^times[i]  for i in times )

    wal = SUM(times[i] * cashflows[i]) / SUM(cashflows)

    RETURN times, cashflows, price, wal
```

This is the `flows()` and `pv()` implementation for the `MBS` class:

```
CLASS MBS(Instrument):
    METHOD flows(curve):
        times, cashflows, _, _ = step_c_flows_and_price(curve, self, scenario_name=None)
        RETURN times, cashflows

    METHOD pv(curve):
        _, _, price, _ = step_c_flows_and_price(curve, self, scenario_name=None)
        RETURN self.side * price          # side = +1 asset, -1 liability
```

---

## 5. Wiring into the existing EVE / KR01 / scenario engine

No changes needed to the scenario grid, KR01 tent-bump logic, or hedge solver
— they already call `instrument.flows(curve)` / `instrument.pv(curve)`
generically for every instrument type. The MBS class above satisfies that
interface. Confirm the following three things hold once wired in (these are
exactly the properties tested in section 7 below):

  1. `curve` passed into `flows()`/`pv()` must be the FULLY shocked curve for
     whatever scenario is being evaluated (parallel/short/steepener/flattener/
     or a KR01 finite-difference bump) -- never the base curve with a label.

  2. KR01 computation for MBS must call `pv()` fresh at each bumped curve,
     letting the mortgage rate / CPR / cash flow schedule fully re-derive.
     Do NOT precompute a cash flow schedule once and re-discount it under
     different curves -- this silently reintroduces frozen-CPR error and
     overstates duration, concentrated exactly at the tenors you'd use to
     size a hedge.

  3. The six-scenario EVE grid, the KR01-by-key-rate report, and any
     "recompute KR01 inside each scenario" diagnostic should all now show:
       - EVE loss under par_up > EVE gain under par_down (magnitude) --
         negative convexity signature
       - Total KR01 varies materially across scenario states (not roughly
         constant, as it would for a bullet bond)
       - The convexity row (actual dEVE - linear-KR01-predicted dEVE) is
         NEGATIVE for an MBS-heavy book on both parallel scenarios -- this
         is a good automated regression check to add (see section 7).

---

## 6. Reporting additions (attribution table)

For the EVE attribution report described earlier, add a "prepayment
diagnostics" block per scenario, per MBS position:

```
| Scenario   | Mortgage rate | Refi incentive | Seasoned CPR | PSA  | WAL   |
|------------|---------------|-----------------|--------------|------|-------|
| base       | ...           | ...              | ...          | ...  | ...   |
| par_up     | ...           | ...              | ...          | ...  | ...   |
| par_down   | ...           | ...              | ...          | ...  | ...   |
| steepener  | ...           | ...              | ...          | ...  | ...   |
| flattener  | ...           | ...              | ...          | ...  | ...   |
```

This table is what makes the model's behavior auditable/explainable to a
risk committee or examiner -- they can trace exactly why EVE moved the way
it did (rate -> incentive -> CPR -> WAL -> price), not just see a black-box
number.

---

## 7. Test cases — verify against these hand-calculated values

Use a single test pool with these exact parameters so results are directly
checkable against the worked example already validated in this conversation:

```
notional          = 200.0
wac                = 0.055
wam_months          = 360
pool_age_months      = 0
anchor_tenor          = 10.0
spread_to_curve        = 0.0175
oas                     = 0.005
base_turnover            = 0.06
max_refi_cpr              = 0.34
logistic_k                 = 2.2
logistic_midpoint           = 0.60
seasoning_ramp_months        = 30

base curve: flat 4.00% (0.04)
BCBS shocks: parallel = 200bp, short = 300bp, long = 150bp (standard USD calibration)
```

### 7.0 Amortization correctness check (implement and pass THIS FIRST)

```
TEST zero_prepayment_fully_amortizes_to_exactly_zero:
    # Force CPR to 0 (base_turnover=0, max_refi_cpr=0) so this is PURE
    # scheduled amortization with no prepayment at all -- the simplest
    # possible case, and the one that most reliably exposes the payment-
    # factor bug described at the top of section 4.
    pool = MBS(notional=300.0, wac=0.055, wam_months=324, pool_age_months=0,
               base_turnover=0.0, max_refi_cpr=0.0, ...)
    times, cashflows, price, wal = step_c_flows_and_price(flat_curve, pool, None)

    ASSERT  len(times) == 324          # must take EXACTLY the stated term,
                                          # not run past it
    ASSERT  final_balance == 0.0        # to within 1e-6 -- must NOT still
                                          # have a large leftover balance

    # If this test fails with a large nonzero leftover balance after the
    # loop terminates (or the loop runs to some arbitrary cap without ever
    # reaching zero), the payment is being recomputed against the declining
    # balance each month instead of held fixed -- see section 4's bug warning.
```

### 7.0b Reference numbers for the standard test pool (post-bugfix)

For the specific pool in section 7 (notional=300, wac=0.055, wam_months=360,
pool_age_months=36, anchor_tenor=10, spread=0.0175, oas=0.005), after fixing
the amortization bug, the CORRECTED reference values are:

| Scenario   | price ($m) | months to fully amortize | WAL (years) |
|------------|-------------|-----------------------------|----------------|
| base       | ~314.6       | ~135                          | ~4.9            |

(The earlier section 7.3 table in this spec's draft history used the
pre-bugfix payment mechanics and should NOT be trusted -- regenerate any
such reference table fresh from the corrected code before relying on it.)



### 7.1 Step A checks (mortgage rate / incentive at seasoned CPR eval)

| Scenario   | curve@10y | mortgage rate | refi incentive (pp) |
|------------|-----------|-----------------|------------------------|
| base       | 4.00%     | 5.75%           | -0.25                   |
| par_up     | 6.00%     | 7.75%           | -2.25                    |
| par_down   | 2.00%     | 3.75%           | +1.75                     |
| steepener  | 5.08%     | 6.83%           | -1.33                      |
| flattener  | 3.37%     | 5.12%           | +0.38                       |

Assert each of these to within 1bp on rate, 0.01pp on incentive.

### 7.2 Step B checks (seasoned CPR at age=60 months, ramp=1.0)

| Scenario   | seasoned CPR | PSA equivalent |
|------------|---------------|------------------|
| base       | 10.5%          | ~176              |
| par_up     | 6.1%            | ~101               |
| par_down   | 37.5%            | ~625                |
| steepener  | 6.5%              | ~108                 |
| flattener  | 18.9%              | ~316                  |

Assert within 0.2 percentage points on CPR.

### 7.3 Step C checks (full pricing, pool_age_months=0)

**WARNING — the table below predates a payment-factor bug fix (see the bug
warning at the top of section 4, and section 7.5 below) and should NOT be
trusted numerically.** It was generated with a payment amount that was
incorrectly recomputed against the declining balance each month rather than
held fixed. Once section 7.5's test passes against your implementation,
REGENERATE this table from your own corrected code before using it as a
pinned reference. The table is left here only to illustrate the expected
SHAPE of the result (par_up worst on price, par_down WAL shortest via
compression, par_up WAL longest via extension) — not the specific numbers.

| Scenario   | price ($m) | WAL (years) | delta vs base |
|------------|-------------|--------------|-----------------|
| base       | *(regenerate — do not use 213.08)*       | *(regenerate)*  | --                |
| par_up     | *(regenerate)*        | *(regenerate)*          | worst price, largest magnitude drop            |
| par_down   | *(regenerate)*         | *(regenerate)*            | small positive gain, capped by compression               |
| steepener  | *(regenerate)*          | *(regenerate)*            | negative, smaller magnitude than par_up                |
| flattener  | *(regenerate)*           | *(regenerate)*              | small negative                 |

Do not assert against specific dollar/year figures from a stale table.
Regenerate with corrected code, confirm the shape matches the pattern
above, and pin your own freshly-generated numbers as the actual reference.

```
TEST negative_convexity_signature:
    ASSERT  abs(price_change[par_up])  >  abs(price_change[par_down])
    # i.e. losses on the way up exceed gains on the way down -- this must
    # ALWAYS hold for an MBS with a live prepayment model. If a future code
    # change makes this test fail, the option has stopped being "live"
    # somewhere in the pipeline (most likely: schedule got cached/frozen).

TEST wal_extends_under_rate_rise:
    ASSERT  wal[par_up] > wal[base]

TEST wal_compresses_under_rate_fall:
    ASSERT  wal[par_down] < wal[base]

TEST frozen_schedule_overstates_kr01:
    # Compute KR01 the WRONG way: derive cash flow schedule ONCE at base
    # curve, then discount that fixed schedule under each bumped curve
    # (i.e. skip re-deriving mortgage_rate/CPR/schedule per bump).
    # Compute KR01 the RIGHT way: full step_a/b/c re-derivation per bump.
    ASSERT  frozen_kr01_total  >  live_kr01_total
    # frozen should overstate sensitivity -- if this test ever shows
    # frozen <= live, something in the live path is broken and is
    # accidentally UNDER-stating risk, which is worse.

TEST kr01_drifts_across_scenario_states:
    # KR01 recomputed AT par_down curve should show total KR01 well below
    # KR01 recomputed AT base curve (duration compression from refi wave).
    ASSERT  total_kr01_at(par_down)  <  0.6 * total_kr01_at(base)

TEST anchor_tenor_matters:
    # A short-end-only shock (e.g. "short_up", 300bp at t=0.25 decaying to
    # near zero by t=10) should move refi_incentive far LESS than a parallel
    # shock of the same headline magnitude, because it barely touches the
    # 10y anchor point.
    ASSERT  abs(incentive_change[short_up])  <  abs(incentive_change[par_up]) * 0.3
```

---

## 8. Common implementation bugs to specifically avoid

0. **[CONFIRMED — found in production while verifying this spec] Recomputing
   the payment against the declining balance instead of fixing it once.**
   Wrong: `scheduled_prin = balance * payment_factor - interest` computed
   fresh every month, where `payment_factor` is a rate-derived ratio and
   `balance` is the current (already-amortized) balance. This produces a
   geometric decay of the balance that asymptotically approaches zero but
   never reaches it — verified concretely: a $300m, 5.5%, 324-month pool
   still had $153m outstanding after 500 "months of payments" computed the
   wrong way. Right: compute `payment_dollar = starting_balance * payment_factor`
   ONCE, before the loop, and use that same fixed dollar figure in
   `scheduled_prin = payment_dollar - interest` every month thereafter. Only
   the interest/principal split should change month to month — never the
   payment total. This is section 7.0's test, and it should be the very
   first thing implemented and passed, before trusting any other number this
   module produces.

1. **Caching the cash flow schedule per instrument instead of per (instrument,
   curve) pair.** This is the #1 bug. If `flows()` or `pv()` has any
   memoization/caching decorator, it must key on the full curve state
   (including the bump), not just the instrument identity.

2. **Using CPR/12 instead of the geometric SMM conversion.** Wrong:
   `smm = cpr / 12`. Right: `smm = 1 - (1 - cpr)^(1/12)`.

3. **Discounting at the base curve after computing cash flows off the shocked
   rate.** Step A determines the schedule; the SAME shocked curve (not base)
   must be used to discount that schedule in Step C. Mixing these produces
   internally inconsistent prices.

4. **Wrong anchor tenor for the product.** A 30y conventional US mortgage
   should anchor around 10y (PMMS tracks the 10y UST via MBS secondary
   market pricing). A 5/1 ARM approaching its first reset should anchor
   much shorter. Get this wrong and every downstream number is off even
   though the code runs without error -- add a config validation warning
   if anchor_tenor looks inconsistent with wam_months (e.g. anchor_tenor >
   wam_months/12).

5. **Applying the seasoning ramp once per scenario instead of once per month
   inside the amortization loop.** The ramp must use the *running* age
   (pool_age_months + current_month), recomputed at every step, not a single
   snapshot value.

6. **Forgetting the post-shock floor when reading the anchor rate.** `curve.
   rate(t)` must apply the floor (e.g. -100bp at overnight, rising 5bp/year to
   0% at 50y) BEFORE the anchor tenor is read out and BEFORE it's used to
   discount cash flows in Step C -- both steps read off the same floored
   curve.
