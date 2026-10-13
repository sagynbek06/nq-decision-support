# Intraday Data Layer: Contracts, a Source Seam and a Synthetic 1-Minute Source

**Why this exists.** Every signal in the system so far runs on daily bars, which allows at most one
decision a day. The target use is 1 to 5 trades per day on 1-minute bars. This layer defines what an
intraday pipeline receives, and it provides one seam that real data will plug into. It does not make
any signal intraday, and nothing in it is evidence of an edge.

Code: [src/data/schema.py](../../src/data/schema.py), [src/data/source.py](../../src/data/source.py),
[src/data/synthetic_intraday.py](../../src/data/synthetic_intraday.py). Tests:
[tests/test_data_schema.py](../../tests/test_data_schema.py),
[tests/test_data_source.py](../../tests/test_data_source.py).

## 1. The contracts

Four tables cross the seam. Each has a validator in [schema.py](../../src/data/schema.py). A validator
returns its input unchanged, or raises `ContractError`, which names the table, the rule and the first
offending row index. Validators never repair data.

- **Bars.** `timestamp` (New York time, start-labelled), `open`, `high`, `low`, `close` (on the 0.25
  grid), `volume` (whole contracts), and optionally `bid_volume` and `ask_volume` (seller-initiated and
  buyer-initiated). When present, the two sum to `volume`.
- **Quotes (depth).** `timestamp`, `bid`, `ask` (on the 0.25 grid, not crossed), `bid_size`, `ask_size`
  (whole numbers).
- **Option chain.** `as_of` (New York time), `expiry` (a naive calendar date), `strike`,
  `call_open_interest`, `put_open_interest`, `implied_vol` (in (0, 5]).
- **Event calendar.** `timestamp` (New York time), `event_type`, `surprise` (NaN until released).

`validate_bars` rejects non-monotonic or duplicate timestamps; bars outside Globex hours (weekends, the
17:00 to 18:00 halt, after the Friday close, before the Sunday open); prices off the 0.25 grid; high below
low; high below open or close; low above open or close; negative volume; and an aggressor split that does
not sum to volume. The OHLC checks go beyond the brief, because a bar that breaks them is internally
inconsistent. The schema's eight numbered assumptions (time zone, tick tolerance, Globex hours, unmodelled
holidays, OHLC consistency, aggressor convention, option expiries and volatility range, NaN surprises) are
in the module docstring.

## 2. The seam

`MarketDataSource` has four methods: `bars(start, end, freq)`, `depth(start, end)`,
`option_chain(as_of)` and `events(start, end)`. Windows are half-open, and their bounds must be
New York-aware. Bar frequencies are 1, 5 and 15 minutes. The option chain is causal: it is the chain as
it stood at the last bar at or before `as_of`.

Downstream code imports `MarketDataSource`, and nothing else from the data layer. It does not import a
vendor library or read a file. A test parses every module in `src/data/` and fails if one imports
anything outside a short list (a few standard-library modules, numpy, pandas and `src`), or calls a
file-reading function.

### Where a real adapter attaches

A real source is one new class in `src/data/`. It subclasses `MarketDataSource`, and its methods return
tables that pass the validators. Constructing it at the call site, in place of `SyntheticIntradaySource`,
is the only change downstream. The adapter has five jobs:

1. Convert the vendor's timestamps to America/New_York, and reject naive ones.
2. Map the vendor's contract symbols and rolls to one continuous series, and document the roll rule.
3. Fill `bid_volume` and `ask_volume` from the trade tape's aggressor flag, if the vendor supplies it.
   Otherwise leave those columns out.
4. Return a NaN surprise for releases that have not happened yet.
5. Raise when a validator fails. Never repair the data.

Vendor libraries and credentials stay inside that module. Real-data integration is Phase 6 in
ROADMAP.md, and this seam is where it plugs in.

## 3. The synthetic source

[synthetic_intraday.py](../../src/data/synthetic_intraday.py) generates 1-minute cash-session bars,
depth snapshots, a weekly option chain and a scheduled-event calendar. Every parameter is a documented
constant. The assumptions are numbered in the module docstring. The main ones:

1. Cash session only: 09:30 to 15:59 New York time, 390 bars a day, no overnight gap.
2. Regimes come from `generate_regime_switching_returns`, scaled to minutes.
3. Volatility is U-shaped, high at the open and close and low at midday. The profile is normalised so each
   day's variance equals its regime's daily variance.
4. Innovations are Student-t with 5 degrees of freedom, scaled to unit variance.
5. Volume is U-shaped too, and linked to volatility after Clark (1973).
6. The aggressor split follows the sign and size of the bar's return.
7. ISM releases at 10:00 on the first session of each month, and FOMC at 14:00 every 31 sessions. Each
   surprise is standard normal. At the release minute, volatility rises with the surprise's size, and the
   return gets a jump proportional to the surprise.
8. Depth is two ticks either side of the close, with size shrinking when volatility is high.
9. A hidden AR(1) order pressure predicts the next minute's normalised return, with slope `planted_edge`.
   The default is 0.0.
10. Weekly Friday expiries, a quadratic volatility skew, and deterministic open interest, with no term
    structure.
11. One seed. Random draws are taken session by session in a fixed order, so the first K sessions are
    identical whatever the horizon.

### What a default run measures

Results for 120 sessions with seed 18, the setting the tests use:

| Property | Measured | Check |
|---|---|---|
| Construction | 0.2 s for 46,800 bars | none |
| U-shape: per-minute mean squared return, open (09:30 to 10:00) over midday (12:00 to 13:00) | 2.68 observed, 2.87 expected from the profile | within 3 cluster-robust SEs (SE 0.16, z = −1.15) |
| Mean absolute return, open and midday | 0.000466 and 0.000283 | open is higher |
| Lag-1 autocorrelation of returns | −0.0038 | within 3 SEs of zero (0.0151) |
| Lag-1 autocorrelation of absolute returns | +0.043 | iid standard deviation is 0.0046; test requires > 0.02 |
| Planted-edge identity, `planted_edge` = 0.05 minus 0 | error 9 × 10⁻¹⁶ | exact |
| Planted slope, `planted_edge` = 0 | −0.0010 with SE 0.0047 | within 3 SEs of zero |
| Planted slope, `planted_edge` = 0.05 | 0.0490 with SE 0.0047 | within 3 SEs of 0.05 |
| Log volume vs log volatility | correlation 0.58 | test requires > 0.5 |
| Aggressor imbalance vs standardised return | correlation 0.91 | test requires > 0.3 |
| Mean daily volume | 246,989 contracts, 1.23 × `DAILY_VOLUME` | not normalised (section 5) |
| Events in 120 sessions | 6 ISM, 4 FOMC | schedule checked against the calendar |

The planted-edge identity is the strongest of these checks. The two sources differ by exactly
0.05 × (the previous bar's hidden pressure) at every bar. The slopes are checked against their standard error, which is
the same for both.

## 4. What the synthetic source can and cannot validate

**Can validate:**

- The plumbing: contract compliance, half-open windows, time zones, resampling, the causal option-chain
  rule, determinism and prefix consistency.
- Whether an analysis recovers structure that was built in on purpose: the U-shape, volatility clustering,
  the Clark volume link, the aggressor link, the event schedule and jump, and the planted edge.
- Whether a pipeline behaves under a known null. With `planted_edge` = 0 the only structure is volatility
  clustering, the intraday profile, and same-minute links between volume, the aggressor split and the
  return. None of these predicts the next minute's return.

**Cannot validate:**

- Anything about real markets. Volatility and volume levels, the price path (this run drifts from 17,997
  to 16,940), the size of event reactions, queue dynamics, cancellations and hidden liquidity, option
  prices, and the realism of the chain are all invented.
- Whether a real edge exists, or how large one is. A later acceptance test (AC-7) asks whether the
  pipeline can detect a planted edge. Passing it on synthetic data would show that the detector works on
  this generator. It would say nothing about a market.
- Trading costs, fills or slippage.

## 5. Limits and open items

- Cash session only. The validator accepts Globex-hours bars, but none are generated.
- Exchange holidays are not modelled. The validator accepts a holiday bar that falls inside Globex hours.
- There is no overnight gap, so the price path is continuous across sessions.
- There is one continuous series, with no contract roll.
- Volume is not normalised to `DAILY_VOLUME`. The default run averages 1.23 times that constant. The constant
  is a scale, not a target. Normalising over the whole horizon would make early sessions depend on later
  draws, which would break the prefix consistency that the tests require.
- The option chain has no term structure, its open interest has no noise, and it is not priced from the bars.
- Nothing downstream uses this layer yet. Signals still consume daily data.
- Numbering: the request asked for `07_intraday_data_layer.md`. `07_overfitting_audit.md` is on the
  walk-forward branch (PR #4), so one of the two should be renumbered before both reach `main`.

## 6. Tests

There are 77 tests, which run in about 3 seconds. Each rule that `validate_bars` names has its own test, and
each checks that the error names the first offending row. The session calendar has a truth table. The
synthetic source's tests check known answers, not only that the code runs: the exact planted identity, the
event jump, the scheduled dates, the snapping rule and the strike ladder. Tolerances are standard errors
computed from the data. Three thresholds are structural rather than statistical: 0.02 for the
autocorrelation of absolute returns, 0.5 for the Clark correlation, and 0.3 for the aggressor correlation.
Section 3 gives the measured values behind each.

Run them with:

```bash
python -m pytest tests/test_data_schema.py tests/test_data_source.py -v
```
