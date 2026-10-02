# Consensus Engine: Combining Programs 1-4 Into One View

**Phase 5 of the NQ decision-support system.** This writeup covers the
3-vote directional design, why Program 4's Gamma Exposure is deliberately
*not* a fourth vote, an interpretive choice made in Program 3's vote that
is currently untested, and a worked example on the project's synthetic
data.

Code: [`src/consensus_engine.py`](../../src/consensus_engine.py). Tests:
[`tests/test_consensus_engine.py`](../../tests/test_consensus_engine.py).

## 1. The design: three votes, one separate context field

`compute_consensus` takes four inputs — a regime label, an order-flow
signal, a kernel-regression deviation, and a total GEX — and combines the
first three into a single weighted directional score in `[-1, 1]`:

- **Regime** (Program 1): `bull → +1`, `bear → -1`, `sideways → 0`.
- **Order flow** (Program 2): the MP-filtered composite signal, squashed
  through `tanh` so large magnitudes saturate toward ±1 rather than
  dominating the average outright.
- **Kernel deviation** (Program 3): same `tanh` squashing, with a sign
  flip — see the interpretive choice below.

Each vote is weighted (equal weights by default) and averaged into a score,
which maps to a `"bullish"` / `"bearish"` / `"neutral"` label (anything
within ±0.15 of zero reads as neutral). GEX is *not* part of that average —
it comes back as a separate `volatility_regime` field
(`"dampening"`/`"amplifying"`).

## 2. Why GEX isn't a fourth vote

Programs 1-3 each produce something that's naturally a directional read: a
regime label, a signed order-flow pressure, a signed deviation from a price
trend. GEX is not that kind of signal. It describes how dealer hedging
flows are expected to *behave* — suppressing realized volatility when
dealers are net long gamma, amplifying it when net short (see
[docs/writeups/04_greeks_dashboard.md](04_greeks_dashboard.md)) — which is
a statement about expected volatility, not about which way price is likely
to go. A market can be GEX-dampened and heading up, or GEX-dampened and
heading down; the sign of GEX doesn't say which.

Forcing it into the directional vote anyway — say, treating positive GEX as
"bullish" because dampened markets feel calmer — would manufacture a
directional claim the signal doesn't actually make, just to make the
aggregation look tidier. Keeping it as separate context is less tidy and
more honest: a consumer of this engine's output sees "the three directional
signals lean bullish, and separately, the market is currently in a
volatility-dampening regime" rather than a single number that quietly
blends two different kinds of claims together.

## 3. An interpretive choice, flagged as untested

Program 3's vote reads the kernel-regression deviation as **mean-reversion**:
price trading above its local trend votes bearish, below votes bullish —
the conventional reading for a distance-from-smoothed-trend indicator (in
the spirit of Bollinger-Bands-style mean reversion). The opposite
(**momentum**: above trend votes bullish, extending the move) is equally
defensible in principle, and would just flip that one vote's sign.

This module picks mean-reversion and documents that it's a choice, not a
derived fact. Whichever reading actually has predictive value on real price
action is an empirical question — nothing in this codebase has backtested
it either way. That validation belongs to Phase 7's walk-forward framework,
the same place the kernel regression writeup's open bandwidth and causality
questions land. Until then, "the consensus engine treats kernel deviation
as mean-reversion" should be read as "this is the convention currently
wired up," not "this is the validated right answer."

## 4. Worked example

Using the project's synthetic NQ series, as of its last date
(2024-10-30, spot = 16,152.5), running all four programs and feeding their
real outputs into `compute_consensus`:

| Program | Output | Vote |
|---|---:|---:|
| 1 — Regime | `sideways` | 0.000 |
| 2 — Order flow | signal = -0.505 | -0.247 |
| 3 — Kernel deviation (h=40) | price 16,152.5 vs. trend 16,242.4 → deviation = -0.109 | +0.054 |
| 4 — GEX | +$46,746,024 | *(not a vote — see §2)* |

```
score  = (0.000 + -0.247 + 0.054) / 3 = -0.064
label  = "neutral"                      (within +/-0.15 of zero)
volatility_regime = "dampening"         (GEX > 0)
```

Reading this the way a user would: Program 1 sees no clear regime (flat
vote), Program 2 sees mild net selling pressure, and Program 3 sees price
sitting just slightly *below* its smoothed trend, which — under the
mean-reversion reading — counts as a (small) bullish signal. Those three
roughly cancel out, landing on `neutral` rather than a confident call in
either direction, while GEX separately says the market is currently in a
dampened, lower-realized-volatility regime. That's the intended shape of
the output: not a single confident number manufactured by force-averaging
four different kinds of signal, but a small, legible vote plus explicit
context — including on a day, like this one, where the honest answer is
"no strong lean."

## 5. Where this leaves things

The aggregation mechanics are tested directly — vote bounds and
monotonicity, the weighted average matching an independent manual
computation, weight sensitivity, GEX's independence from the directional
label — and a real end-to-end test wires up the actual Programs 1-4 outputs
rather than only testing `compute_consensus` in isolation. What isn't
tested is whether the resulting consensus is *good*: the mean-reversion
convention, the equal default weights, and the ±0.15 neutral band are all
reasonable-looking choices, not validated ones. That's consistent with
every other program in this system at this stage — Phase 7 is where
"reasonable-looking" is supposed to become "backtested," for this program
and the three it depends on.
